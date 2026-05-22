
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
    EXPERIMENT_NAME = "exp13_full_pipeline"
    DESCRIPTION = "Full pipeline: All training + threshold tuning + TTA + connected components"

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8

    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 15

    FG_RATIO = 0.5

    CC_MIN_SIZE = 3
    CC_MAX_SIZE = 1000



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
                lbl = get_labeled_slice_indices(seg)
                if len(lbl) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations


def compute_batch_distance_maps(masks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    batch_maps = []
    masksNp = masks.cpu().numpy()
    for i in range(masksNp.shape[0]):
        dm = BoundaryLoss.compute_distance_map(masksNp[i], num_classes=num_classes)
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

            distance_maps = compute_batch_distance_maps(masks, num_classes=num_classes)
            distance_maps = distance_maps.to(device)

            outputs = model(images)

            loss, _ = criterion(
                outputs, masks, epoch=epoch, distance_map=distance_maps
            )

            dice, class_dices = compute_dice_score(outputs, masks)
            recall, class_recalls = compute_recall(outputs, masks)
            precision, _ = compute_precision(outputs, masks)

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



def evaluate_raw_predictions(model, val_loader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Collecting raw predictions"):
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def evaluate_full_pipeline(
    model,
    val_loader,
    device,
    threshold,
    cc_min_size=3,
    cc_max_size=1000,
    num_classes=3,
):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Full pipeline evaluation"):
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            bs = images.shape[0]

            for i in range(bs):
                single_image = images[i:i+1]

                _, tta_probs = test_time_augmentation(
                    model, single_image, device, merge_mode="max"
                )

                fg_probs = tta_probs[1:, :, :]
                max_fg_prob, max_fg_class = fg_probs.max(dim=0)
                pred = torch.zeros(tta_probs.shape[1], tta_probs.shape[2], dtype=torch.long)
                fgMask = max_fg_prob > threshold
                pred[fgMask] = (max_fg_class[fgMask] + 1).long()

                predNp = pred.numpy()
                predNp = connected_component_filter(
                    predNp, min_size=cc_min_size, max_size=cc_max_size
                )

                all_preds.append(predNp)
                all_targets.append(masks[i].numpy())

    all_preds = np.stack(all_preds, axis=0)
    all_targets = np.stack(all_targets, axis=0)
    return all_preds, all_targets


def _serialize_metrics(m):
    return {
        "mean_fg_dice": m["mean_fg_dice"],
        "mean_fg_recall": m["mean_fg_recall"],
        "mean_fg_precision": m["mean_fg_precision"],
        "mean_fg_f2": m["mean_fg_f2"],
        "dice_per_class": {str(k): v for k, v in m["dice_per_class"].items()},
        "recall_per_class": {str(k): v for k, v in m["recall_per_class"].items()},
        "precision_per_class": {str(k): v for k, v in m["precision_per_class"].items()},
        "f2_per_class": {str(k): v for k, v in m["f2_per_class"].items()},
    }


def print_pipeline_comparison(raw_metrics, pipeline_metrics):
    print("\n" + "=" * 70)
    print("COMPARISON: Raw Predictions vs Full Pipeline")
    print("=" * 70)

    header = f"{'Mode':<25s} {'Dice':>8s} {'Recall':>8s} {'Precision':>10s} {'F2':>8s}"
    sep = "-" * len(header)

    print(header)
    print(sep)

    for label, m in [("Raw (argmax)", raw_metrics), ("Full Pipeline", pipeline_metrics)]:
        print(
            f"{label:<25s} "
            f"{m['mean_fg_dice']:>8.4f} "
            f"{m['mean_fg_recall']:>8.4f} "
            f"{m['mean_fg_precision']:>10.4f} "
            f"{m['mean_fg_f2']:>8.4f}"
        )

    print(sep)

    for label, m in [("Raw (argmax)", raw_metrics), ("Full Pipeline", pipeline_metrics)]:
        print(f"\n  {label} per-class:")
        for c in sorted(m['dice_per_class'].keys()):
            print(
                f"    Class {c}: "
                f"Dice={m['dice_per_class'][c]:.4f}  "
                f"Recall={m['recall_per_class'][c]:.4f}  "
                f"Precision={m['precision_per_class'][c]:.4f}  "
                f"F2={m['f2_per_class'][c]:.4f}"
            )

    print(f"\n  --- Improvement from Full Pipeline ---")
    delta_dice = pipeline_metrics['mean_fg_dice'] - raw_metrics['mean_fg_dice']
    delta_recall = pipeline_metrics['mean_fg_recall'] - raw_metrics['mean_fg_recall']
    delta_prec = pipeline_metrics['mean_fg_precision'] - raw_metrics['mean_fg_precision']
    delta_f2 = pipeline_metrics['mean_fg_f2'] - raw_metrics['mean_fg_f2']
    print(f"    Dice      : {delta_dice:+.4f}")
    print(f"    Recall    : {delta_recall:+.4f}")
    print(f"    Precision : {delta_prec:+.4f}")
    print(f"    F2        : {delta_f2:+.4f}")

    print("=" * 70)



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

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    train_dataset = FGCenteredDataset(
        train_volumes, train_segs,
        transform=train_transform,
        fg_ratio=cfg.FG_RATIO,
        patch_size=cfg.IMG_SIZE,
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

    device = torch.device(cfg.DEVICE)
    model = create_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME,
        attention_type=cfg.ATTENTION_TYPE,
    ).to(device)

    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {nParams:,}")
    # build the loss
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
    logger.info(f"Output dir        : {output_dir}")
    logger.info(f"Device            : {device}")
    logger.info(f"Tversky alpha/beta: {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    logger.info(f"FG sampling ratio : {cfg.FG_RATIO}")
    logger.info(f"Boundary weight   : {cfg.BOUNDARY_WEIGHT}")
    logger.info(f"Boundary ramp-up  : {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs")
    logger.info(f"CC filter         : min={cfg.CC_MIN_SIZE}, max={cfg.CC_MAX_SIZE}")

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
    print("PHASE 1: TRAINING")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")
        # run one training epoch
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
    logger.info("PHASE 1 COMPLETE: TRAINING")
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

    print("\n" + "=" * 70)
    print("PHASE 2: POST-PROCESSING PIPELINE")
    print("=" * 70)

    print("\nStep 2a: Collecting raw predictions...")
    raw_preds, raw_targets = evaluate_raw_predictions(
        model, val_loader, device, num_classes=cfg.NUM_CLASSES
    )
    raw_metrics = compute_all_metrics(raw_preds, raw_targets, cfg.NUM_CLASSES)
    print(f"  Raw Dice  : {raw_metrics['mean_fg_dice']:.4f}")
    print(f"  Raw Recall: {raw_metrics['mean_fg_recall']:.4f}")
    print(f"  Raw F2    : {raw_metrics['mean_fg_f2']:.4f}")

    print("\nStep 2b: Optimizing decision threshold...")
    best_threshold = optimize_threshold(
        model, val_loader, device, num_classes=cfg.NUM_CLASSES
    )
    print(f"  Optimal threshold: {best_threshold:.2f}")
    logger.info(f"Optimal threshold: {best_threshold:.2f}")

    print(f"\nStep 2c: Running full pipeline (TTA max + threshold={best_threshold:.2f} + CC filter)...")
    pipeline_preds, pipeline_targets = evaluate_full_pipeline(
        model, val_loader, device,
        threshold=best_threshold,
        cc_min_size=cfg.CC_MIN_SIZE,
        cc_max_size=cfg.CC_MAX_SIZE,
        num_classes=cfg.NUM_CLASSES,
    )
    pipeline_metrics = compute_all_metrics(pipeline_preds, pipeline_targets, cfg.NUM_CLASSES)

    print_pipeline_comparison(raw_metrics, pipeline_metrics)

    logger.info(
        f"\nRaw predictions      : Dice={raw_metrics['mean_fg_dice']:.4f} "
        f"Recall={raw_metrics['mean_fg_recall']:.4f} "
        f"F2={raw_metrics['mean_fg_f2']:.4f}"
    )
    logger.info(
        f"Full pipeline        : Dice={pipeline_metrics['mean_fg_dice']:.4f} "
        f"Recall={pipeline_metrics['mean_fg_recall']:.4f} "
        f"F2={pipeline_metrics['mean_fg_f2']:.4f}"
    )

    delta_f2 = (pipeline_metrics['mean_fg_f2'] - raw_metrics['mean_fg_f2'])
    delta_dice = pipeline_metrics['mean_fg_dice'] - raw_metrics['mean_fg_dice']
    delta_recall = pipeline_metrics['mean_fg_recall'] - raw_metrics['mean_fg_recall']
    logger.info(
        f"Pipeline improvement : Dice={delta_dice:+.4f} "
        f"Recall={delta_recall:+.4f} "
        f"F2={delta_f2:+.4f}"
    )

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
        "fg_ratio": cfg.FG_RATIO,
        "boundary_weight": cfg.BOUNDARY_WEIGHT,
        "boundary_rampup_epochs": cfg.EPOCH_FOR_BOUNDARY_RAMPUP,

        "optimal_threshold": best_threshold,
        "cc_min_size": cfg.CC_MIN_SIZE,
        "cc_max_size": cfg.CC_MAX_SIZE,

        "raw_metrics": _serialize_metrics(raw_metrics),
        "pipeline_metrics": _serialize_metrics(pipeline_metrics),
        "pipeline_improvement": {
            "dice": delta_dice,
            "recall": delta_recall,
            "precision": pipeline_metrics['mean_fg_precision'] - raw_metrics['mean_fg_precision'],
            "f2": delta_f2,
        },

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
    print(f"")
    print(f"  TRAINING:")
    print(f"    Tversky alpha/beta: {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    print(f"    FG sampling ratio : {cfg.FG_RATIO}")
    print(f"    Boundary weight   : {cfg.BOUNDARY_WEIGHT} (ramp-up: {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs)")
    print(f"    Best val Dice     : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"    Best val Recall   : {best_val_metrics.get('recall', 0.0):.4f}")
    print(f"")
    print(f"  POST-PROCESSING:")
    print(f"    Optimal threshold : {best_threshold:.2f}")
    print(f"    TTA mode          : max")
    print(f"    CC filter         : min={cfg.CC_MIN_SIZE}, max={cfg.CC_MAX_SIZE}")
    print(f"")
    print(f"  FINAL METRICS (full pipeline):")
    print(f"    Dice              : {pipeline_metrics['mean_fg_dice']:.4f}")
    print(f"    Recall            : {pipeline_metrics['mean_fg_recall']:.4f}")
    print(f"    Precision         : {pipeline_metrics['mean_fg_precision']:.4f}")
    print(f"    F2                : {pipeline_metrics['mean_fg_f2']:.4f}")
    print(f"")
    print(f"  IMPROVEMENT OVER RAW:")
    print(f"    Dice              : {delta_dice:+.4f}")
    print(f"    Recall            : {delta_recall:+.4f}")
    print(f"    F2                : {delta_f2:+.4f}")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
