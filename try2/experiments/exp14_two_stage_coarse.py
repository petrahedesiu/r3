
import sys, os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from albumentations.pytorch import ToTensorV2
import albumentations as A

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_coarse import CoarseFullSliceDataset
from shared.losses import CompoundLoss
from shared.models import create_coarse_model
from shared.training import compute_class_weights, plot_training_history, plot_predictions
from shared.metrics import compute_dice_score, compute_recall, compute_precision
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp14_two_stage_coarse"
    DESCRIPTION = "Two-stage pipeline Stage 1: binary coarse localizer on full slices"

    NUM_CLASSES = 2
    CLASS_NAMES = ("BG", "FG")
    IMG_SIZE = 256
    BATCH_SIZE = 8
    LR = 1e-4
    NUM_EPOCHS = 30

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8

    OVERSAMPLE_FACTOR = 5



def get_transforms(train=True, img_size=256):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3),
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


def compute_binary_class_weights(segmentations, device):
    fgPixels = 0
    bgPixels = 0
    for seg in segmentations:
        fgPixels += (seg > 0).sum()
        bgPixels += (seg == 0).sum()
    total = fgPixels + bgPixels
    if fgPixels == 0:
        return torch.tensor([1.0, 1.0], device=device)
    w_bg = total/(2 * bgPixels)
    w_fg = total / (2*fgPixels)
    weights = torch.tensor([w_bg, w_fg], dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    return weights



def train_epoch_coarse(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    num_classes: int = 2,
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

        optimizer.zero_grad()

        outputs = model(images)

        loss, loss_dict = criterion(outputs, masks, epoch=epoch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
        optimizer.step()

        with torch.no_grad():
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)

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


def validate_coarse(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompoundLoss,
    device: torch.device,
    epoch: int = 0,
    num_classes: int = 2,
) -> Dict:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_recall = 0.0
    total_precision = 0.0

    class_dice_sums: Dict[int, float] = {}
    class_dice_counts: Dict[int, int] = {}
    class_recall_sums: Dict[int, float] = {}
    class_recall_counts: Dict[int, int] = {}
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Val epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            outputs = model(images)

            loss, _ = criterion(outputs, masks, epoch=epoch)

            dice, class_dices = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, class_recalls = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)

            total_loss += loss.item()
            total_dice += dice
            total_recall += recall
            total_precision += precision

            for c, d in class_dices.items():
                class_dice_sums[c] = class_dice_sums.get(c, 0.0) + d
                class_dice_counts[c] = class_dice_counts.get(c, 0) + 1

            for c, r in class_recalls.items():
                class_recall_sums[c] = class_recall_sums.get(c, 0.0) + r
                class_recall_counts[c] = class_recall_counts.get(c, 0) + 1

            num_batches += 1

    n = max(num_batches, 1)

    avg_class_dices = {
        c: class_dice_sums[c] / max(1, class_dice_counts[c])
        for c in sorted(class_dice_sums)
    }
    avg_class_recalls = {
        c: class_recall_sums[c] / max(1, class_recall_counts[c])
        for c in sorted(class_recall_sums)
    }

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "recall": total_recall / n,
        "precision": total_precision / n,
        "class_dices": avg_class_dices,
        "class_recalls": avg_class_recalls,
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

    idxs = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        idxs, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )
    train_volumes = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    print(f"Train: {len(train_volumes)} patients, Val: {len(val_volumes)} patients")

    total_train_slices = sum(s.shape[2] for s in train_segs)
    fg_train_slices = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0)
        for s in train_segs
    )
    print(f"Train slices: {total_train_slices} total, {fg_train_slices} with foreground "
          f"({100*fg_train_slices/total_train_slices:.1f}%)")

    device = torch.device(cfg.DEVICE)
    class_weights = compute_binary_class_weights(train_segs, device)
    print(f"Binary class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    train_dataset = CoarseFullSliceDataset(
        train_volumes, train_segs,
        transform=train_transform,
        oversample=cfg.OVERSAMPLE_FACTOR,
    )

    val_dataset = CoarseFullSliceDataset(
        val_volumes, val_segs,
        transform=val_transform,
        oversample=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = create_coarse_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {nParams:,}")

    # the loss function
    criterion = CompoundLoss(
        focal_weight=0.35, tversky_weight=0.35, lovasz_weight=0.30,
        class_weights=class_weights,
        tversky_alpha=cfg.TVERSKY_ALPHA,
        tversky_beta=cfg.TVERSKY_BETA,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        use_boundary=False,
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
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Num classes : {cfg.NUM_CLASSES} (binary)")
    logger.info(f"Tversky     : alpha={cfg.TVERSKY_ALPHA}, beta={cfg.TVERSKY_BETA}")
    logger.info(f"Oversample  : {cfg.OVERSAMPLE_FACTOR}x")

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

    print("\n" + "=" * 70)
    print("STAGE 1: COARSE LOCALIZER TRAINING")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")

        train_metrics = train_epoch_coarse(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        val_metrics = validate_coarse(
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
                    "num_classes": cfg.NUM_CLASSES,
                    "img_size": cfg.IMG_SIZE,
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
    plot_predictions(model, val_loader, device, predictions_plot_path,
                     num_classes=cfg.NUM_CLASSES)
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
        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "oversample": cfg.OVERSAMPLE_FACTOR,
        "history": history,
        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON    -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"  Best val Dice  : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"  Best val Recall: {best_val_metrics.get('recall', 0.0):.4f}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
