
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset import StandardDataset, FGCenteredDataset, CopyPasteDataset
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_model
from shared.training import run_training, compute_class_weights, train_epoch, validate, plot_training_history, plot_predictions
from shared.postprocessing import optimize_threshold, test_time_augmentation, connected_component_filter, apply_threshold
from shared.metrics import compute_all_metrics, compute_dice_score, compute_recall, compute_precision
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp12_combo_all_training"
    DESCRIPTION = "Combo: All training improvements (aggressive tversky + FG sampling + boundary + copy-paste)"

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8

    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 15

    COPY_PASTE_PROB = 0.3


def get_transforms(train=True, img_size=384):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(img_size, img_size), ToTensorV2()])


def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                labeled = get_labeled_slice_indices(seg)
                if len(labeled) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations


def compute_batch_distance_maps(masks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    batch_maps = []
    masks_np = masks.cpu().numpy()
    for i in range(masks_np.shape[0]):
        dm = BoundaryLoss.compute_distance_map(masks_np[i], num_classes=num_classes)
        batch_maps.append(dm)
    return torch.from_numpy(np.stack(batch_maps, axis=0)).float()


def train_epoch_boundary(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    num_classes: int = 3,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_recall = 0.0
    total_precision = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        if images.dtype != torch.float32:
            images = images.float()
        masks = masks.long()

        distance_maps = compute_batch_distance_maps(masks, num_classes=num_classes)
        distance_maps = distance_maps.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss, loss_dict = criterion(
            outputs, masks, epoch=epoch, distance_map=distance_maps
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
        optimizer.step()

        with torch.no_grad():
            dice, _ = compute_dice_score(outputs, masks)
            recall, _ = compute_recall(outputs, masks)
            precision, _ = compute_precision(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        total_recall += recall
        total_precision += precision
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}",
            "recall": f"{recall:.2f}",
            "prec": f"{precision:.2f}",
        })

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "recall": total_recall / n,
        "precision": total_precision / n,
    }


def validate_boundary(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    device: torch.device,
    epoch: int = 0,
    num_classes: int = 3,
) -> Dict:
    model.eval()

    total_loss = 0.0
    num_batches = 0
    smooth = 1e-7

    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Val epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            distance_maps = compute_batch_distance_maps(masks, num_classes=num_classes)
            distance_maps = distance_maps.to(device)

            outputs = model(images)

            loss, _ = criterion(
                outputs, masks, epoch=epoch, distance_map=distance_maps
            )

            total_loss += loss.item()
            num_batches += 1

            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            targetNp = masks.cpu().numpy()

            for c in range(num_classes):
                pred_c = (pred_labels == c)
                tgt_c = (targetNp == c)
                tp[c] += float(np.logical_and(pred_c, tgt_c).sum())
                fp[c] += float(np.logical_and(pred_c, ~tgt_c).sum())
                fn[c] += float(np.logical_and(~pred_c, tgt_c).sum())

    n = max(num_batches, 1)

    class_dices: Dict[int, float] = {}
    class_recalls: Dict[int, float] = {}

    for c in range(num_classes):
        dice_denom = 2 * tp[c] + fp[c] + fn[c]
        class_dices[c] = float('nan') if dice_denom < smooth else float((2 * tp[c] + smooth) / (dice_denom + smooth))

        rec_denom = tp[c] + fn[c]
        class_recalls[c] = float('nan') if rec_denom < smooth else float((tp[c] + smooth) / (rec_denom + smooth))

    fg_dices = [class_dices[c] for c in range(1, num_classes)]
    fg_recalls = [class_recalls[c] for c in range(1, num_classes)]
    fg_precs = []
    for c in range(1, num_classes):
        prec_denom = tp[c] + fp[c]
        fg_precs.append(float('nan') if prec_denom < smooth else float((tp[c] + smooth) / (prec_denom + smooth)))

    mean_dice = float(np.nanmean(fg_dices)) if fg_dices else 0.0
    mean_recall = float(np.nanmean(fg_recalls)) if fg_recalls else 0.0
    mean_precision = float(np.nanmean(fg_precs)) if fg_precs else 0.0

    if np.isnan(mean_dice): mean_dice = 0.0
    if np.isnan(mean_recall): mean_recall = 0.0
    if np.isnan(mean_precision): mean_precision = 0.0

    return {
        "loss": total_loss / n,
        "dice": mean_dice,
        "recall": mean_recall,
        "precision": mean_precision,
        "class_dices": class_dices,
        "class_recalls": class_recalls,
    }


def main():
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

    print("\nLoading data...")
    volumes, segmentations = load_data(cfg.DATA_DIR)
    print(f"Loaded {len(volumes)} patients")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Exiting.")
        return

    indices = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )

    train_volumes = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    print(f"Train: {len(train_volumes)} patients, Val: {len(val_volumes)} patients")

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    train_dataset = CopyPasteDataset(
        train_volumes, train_segs,
        transform=train_transform,
        copy_paste_prob=cfg.COPY_PASTE_PROB,
        oversample=cfg.OVERSAMPLE_FACTOR,
    )

    val_dataset = StandardDataset(
        val_volumes, val_segs,
        transform=val_transform,
        oversample=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Copy-paste probability: {cfg.COPY_PASTE_PROB}")
    print(f"Donor pool size: {len(train_dataset.donors)} slices with FG")

    device = torch.device(cfg.DEVICE)
    model = create_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME,
        attention_type=cfg.ATTENTION_TYPE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = CompoundLoss(
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        boundary_weight=cfg.BOUNDARY_WEIGHT,
        class_weights=class_weights.to(device),
        tversky_alpha=cfg.TVERSKY_ALPHA,
        tversky_beta=cfg.TVERSKY_BETA,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        use_boundary=cfg.USE_BOUNDARY,
        epoch_for_boundary_rampup=cfg.EPOCH_FOR_BOUNDARY_RAMPUP,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.SCHEDULER_T0,
        T_mult=cfg.SCHEDULER_TMULT,
        eta_min=cfg.SCHEDULER_ETA_MIN,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"training_{timestamp}.log"

    logger = logging.getLogger(f"training.{cfg.EXPERIMENT_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(formatter)

    logger.info(cfg.summary())
    logger.info(f"Output dir         : {output_dir}")
    logger.info(f"Device             : {device}")
    logger.info(f"Tversky alpha/beta : {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    logger.info(f"Copy-paste prob    : {cfg.COPY_PASTE_PROB}")
    logger.info(f"Boundary weight    : {cfg.BOUNDARY_WEIGHT}")
    logger.info(f"Boundary ramp-up   : {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs")

    history: Dict[str, list] = {
        "train_loss": [],
        "train_dice": [],
        "train_recall": [],
        "val_loss": [],
        "val_dice": [],
        "val_recall": [],
    }

    best_dice = 0.0
    best_epoch = 0
    best_val_metrics: Dict = {}

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")

        train_metrics = train_epoch_boundary(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        val_metrics = validate_boundary(
            model, val_loader, criterion, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        scheduler.step()

        logger.info(
            f"TRAIN  - loss: {train_metrics['loss']:.4f} | "
            f"dice: {train_metrics['dice']:.4f} | "
            f"recall: {train_metrics['recall']:.4f} | "
            f"precision: {train_metrics['precision']:.4f}"
        )
        logger.info(
            f"VAL    - loss: {val_metrics['loss']:.4f} | "
            f"dice: {val_metrics['dice']:.4f} | "
            f"recall: {val_metrics['recall']:.4f} | "
            f"precision: {val_metrics['precision']:.4f}"
        )
        class_dices = val_metrics.get("class_dices", {})
        class_recalls = val_metrics.get("class_recalls", {})
        logger.info(
            "  Per-class Dice  : "
            + ", ".join(f"{c}={d:.3f}" for c, d in sorted(class_dices.items()))
        )
        logger.info(
            "  Per-class Recall: "
            + ", ".join(f"{c}={r:.3f}" for c, r in sorted(class_recalls.items()))
        )

        ramp = min(1.0, epoch / cfg.EPOCH_FOR_BOUNDARY_RAMPUP)
        eff_bw = cfg.BOUNDARY_WEIGHT * ramp
        logger.info(f"  Boundary ramp   : {ramp:.3f} -> effective weight = {eff_bw:.4f}")

        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_recall"].append(val_metrics["recall"])

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            best_model_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_recall": val_metrics["recall"],
                    "val_precision": val_metrics["precision"],
                    "class_dices": class_dices,
                    "class_recalls": class_recalls,
                },
                best_model_path,
            )
            logger.info(f">>> NEW BEST MODEL  dice={val_metrics['dice']:.4f}")

        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_metrics["dice"],
                },
                ckpt_path,
            )

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best val dice : {best_dice:.4f}  (epoch {best_epoch})")

    history_plot_path = output_dir / "training_history.png"
    plot_training_history(history, history_plot_path)
    logger.info(f"Saved training curves -> {history_plot_path}")

    best_ckpt = torch.load(
        output_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    predictions_plot_path = output_dir / "predictions.png"
    plot_predictions(model, val_loader, device, predictions_plot_path)
    logger.info(f"Saved predictions     -> {predictions_plot_path}")

    results = {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "description": cfg.DESCRIPTION,
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_val_recall": best_val_metrics.get("recall", 0.0),
        "best_val_precision": best_val_metrics.get("precision", 0.0),
        "best_class_dices": {
            str(k): v for k, v in best_val_metrics.get("class_dices", {}).items()
        },
        "best_class_recalls": {
            str(k): v for k, v in best_val_metrics.get("class_recalls", {}).items()
        },
        "num_epochs": cfg.NUM_EPOCHS,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "copy_paste_prob": cfg.COPY_PASTE_PROB,
        "boundary_weight": cfg.BOUNDARY_WEIGHT,
        "boundary_rampup_epochs": cfg.EPOCH_FOR_BOUNDARY_RAMPUP,
        "history": history,
        "timestamp": timestamp,
        "config": {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and not callable(v)
        },
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON    -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"  Tversky alpha/beta: {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    print(f"  Copy-paste prob   : {cfg.COPY_PASTE_PROB}")
    print(f"  Boundary weight   : {cfg.BOUNDARY_WEIGHT} (ramp-up: {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs)")
    print(f"  Best val Dice     : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"  Best val Recall   : {best_val_metrics.get('recall', 0.0):.4f}")
    print(f"  Best val Prec     : {best_val_metrics.get('precision', 0.0):.4f}")
    print(f"  Output            : {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
