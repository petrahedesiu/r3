
import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_fine import FineBBoxCropDataset
from shared.losses import TverskyLoss, FocalLoss
from shared.training import compute_class_weights, plot_training_history, plot_predictions
from shared.metrics import compute_dice_score, compute_recall, compute_precision
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp19_combined_b0_unet"
    DESCRIPTION = "Combined data + EfficientNet-B0 + plain Unet + Tversky+Focal loss + strong augmentation"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    BATCH_SIZE = 4
    LR = 5e-5
    NUM_EPOCHS = 30

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8

    USE_BOUNDARY = False

    BBOX_PADDING = 50
    BBOX_JITTER_TRAIN = 15
    BBOX_JITTER_VAL = 0

    OVERSAMPLE_FACTOR = 3

    EARLY_STOPPING_PATIENCE = 7



DATA_DIRS = [
    os.path.join(_PROJECT_ROOT, "CROP1"),
    os.path.join(_PROJECT_ROOT, "CROP - februarie 2026"),
]



class SimpleCombinedLoss(nn.Module):
    def __init__(self, class_weights, tversky_alpha=0.2, tversky_beta=0.8,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                               class_weights=class_weights)

    def forward(self, pred, target, **kwargs):
        tLoss = self.tversky(pred, target)
        fLoss = self.focal(pred, target)
        return 0.5*tLoss + 0.5*fLoss



def get_transforms(train=True, img_size=384):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(img_size, img_size), ToTensorV2()])


def load_data():
    volumes, segmentations = [], []
    for data_dir in DATA_DIRS:
        print(f"  Scanning: {data_dir}")
        if not os.path.isdir(data_dir):
            print(f"    WARNING: directory not found, skipping: {data_dir}")
            continue
        pts = discover_patients(data_dir)
        print(f"    Found {len(pts)} patient folders")
        for p in tqdm(pts, desc=f"Loading from {os.path.basename(data_dir)}"):
            try:
                vol, seg, meta = load_patient_data(
                    p['dicom_dir'], p['nrrd_path'], verbose=False
                )
                if meta['alignment_success']:
                    lbl = get_labeled_slice_indices(seg)
                    if len(lbl) >= 2:
                        volumes.append(vol)
                        segmentations.append(seg)
            except Exception:
                pass
    return volumes, segmentations



def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SimpleCombinedLoss,
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

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

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


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: SimpleCombinedLoss,
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

            outputs = model(images)
            loss = criterion(outputs, masks)

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

    print("\nLoading data from combined directories...")
    volumes, segmentations = load_data()
    print(f"Loaded {len(volumes)} patients total")

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

    fg_train_slices = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0)
        for s in train_segs
    )
    fg_val_slices = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0)
        for s in val_segs
    )
    print(f"Foreground slices: train={fg_train_slices}, val={fg_val_slices}")

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    train_dataset = FineBBoxCropDataset(
        train_volumes, train_segs,
        transform=train_transform,
        padding=cfg.BBOX_PADDING,
        jitter=cfg.BBOX_JITTER_TRAIN,
        oversample=cfg.OVERSAMPLE_FACTOR,
    )

    val_dataset = FineBBoxCropDataset(
        val_volumes, val_segs,
        transform=val_transform,
        padding=cfg.BBOX_PADDING,
        jitter=cfg.BBOX_JITTER_VAL,
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
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    ).to(device)

    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {nParams:,}")
    # combined tversky+focal loss
    criterion = SimpleCombinedLoss(
        class_weights=class_weights.to(device),
        tversky_alpha=cfg.TVERSKY_ALPHA,
        tversky_beta=cfg.TVERSKY_BETA,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
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
    logger.info(f"Data dirs         : {DATA_DIRS}")
    logger.info(f"Total patients    : {len(volumes)}")
    logger.info(f"Tversky alpha/beta: {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    logger.info(f"Loss              : 0.5*Tversky + 0.5*Focal (no boundary, no Lovasz)")
    logger.info(f"Model             : Unet + EfficientNet-B0 (no SCSE)")
    logger.info(f"Bbox padding      : {cfg.BBOX_PADDING} +/- {cfg.BBOX_JITTER_TRAIN} jitter")
    logger.info(f"Early stopping    : patience={cfg.EARLY_STOPPING_PATIENCE}")

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
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("EXP19: COMBINED B0 UNET TRAINING")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        val_metrics = validate(
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
            epochs_without_improvement = 0
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
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement for {epochs_without_improvement}/"
                f"{cfg.EARLY_STOPPING_PATIENCE} epochs"
            )

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

        if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"EARLY STOPPING triggered at epoch {epoch} "
                f"(no improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs)"
            )
            break

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
        "num_epochs_run": len(history["train_loss"]),
        "num_epochs_max": cfg.NUM_EPOCHS,
        "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE,
        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "loss": "0.5*Tversky + 0.5*Focal",
        "model": "Unet + EfficientNet-B0 (no SCSE)",
        "data_dirs": DATA_DIRS,
        "total_patients": len(volumes),
        "bbox_padding": cfg.BBOX_PADDING,
        "bbox_jitter": cfg.BBOX_JITTER_TRAIN,
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
