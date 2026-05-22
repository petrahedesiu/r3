
import os
import sys
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_fine_patches import FinePatchDataset
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_coarse_model
from shared.training import compute_class_weights, plot_training_history, plot_predictions
from shared.metrics import compute_all_metrics, compute_dice_score, compute_recall, compute_precision
from shared.two_stage_inference import native_patch_predict_slice
from shared.dataset import _normalize
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp22_small_model_patches"
    DESCRIPTION = "B0 + Unet (~6M params) + native 128x128 patches (anti-overfitting)"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    BATCH_SIZE = 4
    LR = 3e-5
    NUM_EPOCHS = 30

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8
    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 15

    PATCH_SIZE = 128
    PATCH_JITTER_TRAIN = 10
    PATCH_JITTER_VAL = 0

    OVERSAMPLE_FACTOR = 3

    EARLY_STOPPING_PATIENCE = 5



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


def create_small_model(in_channels: int = 1, num_classes: int = 3) -> nn.Module:
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )


def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for pat in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(pat['dicom_dir'], pat['nrrd_path'], verbose=False)
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


def find_latest_model_dir(experiment_name: str) -> Optional[str]:
    results_base = Config.OUTPUT_BASE
    exp_dir = os.path.join(results_base, experiment_name)
    if not os.path.isdir(exp_dir):
        return None

    subdirs = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        reverse=True,
    )

    for subdir in subdirs:
        model_path = os.path.join(exp_dir, subdir, "best_model.pth")
        if os.path.exists(model_path):
            return os.path.join(exp_dir, subdir)

    return None


def load_coarse_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 2)
    model = create_coarse_model(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded coarse model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model


def load_small_fine_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 3)
    model = create_small_model(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded small fine model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model


def plot_two_stage_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    prediction: np.ndarray,
    info: dict,
    save_path: str,
    title: str = "",
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Full CT Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    if info.get("bbox") is not None:
        rmin, rmax, cmin, cmax = info["bbox"]
        rect = mpatches.Rectangle(
            (cmin, rmin), cmax - cmin, rmax - rmin,
            linewidth=2, edgecolor='lime', facecolor='none',
        )
        axes[2].add_patch(rect)
        status = "detected"
    elif info.get("detected"):
        status = "fallback (full)"
    else:
        status = "nothing detected"
    axes[2].set_title(f"Stage 1: {status}")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="tab10", vmin=0, vmax=2)
    axes[3].set_title("Final Prediction")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def train_epoch_fine(
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
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)

        total_loss += loss.item()
        total_dice += dice
        total_recall += recall
        total_precision += precision
        num_batches += 1

        if num_batches % 50 == 0 and device.type == "mps":
            torch.mps.empty_cache()

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


def validate_fine(
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

    indices = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
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

    train_dataset = FinePatchDataset(
        train_volumes, train_segs,
        transform=train_transform,
        patch_size=cfg.PATCH_SIZE,
        jitter=cfg.PATCH_JITTER_TRAIN,
        oversample=cfg.OVERSAMPLE_FACTOR,
    )

    val_dataset = FinePatchDataset(
        val_volumes, val_segs,
        transform=val_transform,
        patch_size=cfg.PATCH_SIZE,
        jitter=cfg.PATCH_JITTER_VAL,
        oversample=1,
    )

    use_pin_memory = cfg.DEVICE != "mps"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    device = torch.device(cfg.DEVICE)
    model = create_small_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
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
    logger.info(f"Model              : B0 + Unet ({num_params:,} params)")
    logger.info(f"Tversky alpha/beta : {cfg.TVERSKY_ALPHA}/{cfg.TVERSKY_BETA}")
    logger.info(f"Boundary weight    : {cfg.BOUNDARY_WEIGHT}")
    logger.info(f"Boundary ramp-up   : {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs")
    logger.info(f"Patch size         : {cfg.PATCH_SIZE}")
    logger.info(f"Patch jitter train : {cfg.PATCH_JITTER_TRAIN}")
    logger.info(f"Early stopping     : patience={cfg.EARLY_STOPPING_PATIENCE}")

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
    print("STAGE 2: SMALL MODEL (B0+Unet) NATIVE-PATCH TRAINING")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")

        train_metrics = train_epoch_fine(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        val_metrics = validate_fine(
            model, val_loader, criterion, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        scheduler.step()

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

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
        effBw = cfg.BOUNDARY_WEIGHT * ramp
        logger.info(f"  Boundary ramp   : {ramp:.3f} -> effective weight = {effBw:.4f}")

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
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")

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
                f"EARLY STOPPING at epoch {epoch} "
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

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION WITH NATIVE PATCHES (SMALL MODEL)")
    print("=" * 70)

    coarse_dir = find_latest_model_dir("exp14_two_stage_coarse")
    if coarse_dir is None:
        logger.warning("No trained coarse model found (exp14). Skipping E2E evaluation.")
        print("WARNING: No trained coarse model found. Run exp14 first.")
        print("Skipping E2E evaluation -- saving training-only results.")

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
            "num_epochs": epoch,
            "num_classes": cfg.NUM_CLASSES,
            "img_size": cfg.IMG_SIZE,
            "patch_size": cfg.PATCH_SIZE,
            "model_params": num_params,
            "model_type": "B0 + Unet",
            "early_stopped": epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE,
            "history": history,
            "e2e_evaluation": "SKIPPED -- no coarse model found",
            "timestamp": timestamp,
        }
        results_json_path = output_dir / "results.json"
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nOutput: {output_dir}")
        return results

    print(f"Coarse model dir: {coarse_dir}")
    coarse_model = load_coarse_model(coarse_dir, device)

    fine_model = load_small_fine_model(str(output_dir), device)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    total_fg_slices = 0
    detected_fg_slices = 0
    total_bg_slices = 0
    false_positive_bg_slices = 0
    fallback_count = 0

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0
    max_vis = 20

    for patient_idx, (vol, seg) in enumerate(
        tqdm(list(zip(val_volumes, val_segs)), desc="E2E evaluating patients")
    ):
        n_slices = vol.shape[2]

        for slice_idx in range(n_slices):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = native_patch_predict_slice(
                image,
                coarse_model,
                fine_model,
                device,
                coarse_size=256,
                patch_size=cfg.PATCH_SIZE,
                fine_size=cfg.IMG_SIZE,
                coarse_threshold=0.3,
                use_tta=False,
                use_cc_filter=False,
            )

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if has_fg:
                total_fg_slices += 1
                if info["detected"]:
                    detected_fg_slices += 1
                if info["fallback_full"]:
                    fallback_count += 1
            else:
                total_bg_slices += 1
                if info["detected"]:
                    false_positive_bg_slices += 1

            if has_fg and vis_count < max_vis:
                image_norm = _normalize(image.astype(np.float32))
                plot_two_stage_visualization(
                    image_norm, gt_mask, pred, info,
                    save_path=str(vis_dir / f"patient{patient_idx}_slice{slice_idx}.png"),
                    title=f"Patient {patient_idx}, Slice {slice_idx}",
                )
                vis_count += 1

    print("\n" + "=" * 70)
    print("E2E RESULTS")
    print("=" * 70)

    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_targets_flat = np.concatenate([t.ravel() for t in all_targets])

    all_metrics = compute_all_metrics(all_preds_flat, all_targets_flat, num_classes=3)

    fg_mask_indices = [i for i in range(len(all_targets)) if all_targets[i].max() > 0]
    if fg_mask_indices:
        fg_preds_flat = np.concatenate([all_preds[i].ravel() for i in fg_mask_indices])
        fg_targets_flat = np.concatenate([all_targets[i].ravel() for i in fg_mask_indices])
        fg_metrics = compute_all_metrics(fg_preds_flat, fg_targets_flat, num_classes=3)
    else:
        fg_metrics = all_metrics

    detection_rate = detected_fg_slices / max(1, total_fg_slices)
    false_positive_rate = false_positive_bg_slices / max(1, total_bg_slices)

    print(f"\nStage 1 Detection Statistics:")
    print(f"  Foreground slices: {total_fg_slices}")
    print(f"  Detected         : {detected_fg_slices} ({100*detection_rate:.1f}%)")
    print(f"  Missed           : {total_fg_slices - detected_fg_slices}")
    print(f"  Fallback (full)  : {fallback_count}")
    print(f"  BG slices        : {total_bg_slices}")
    print(f"  False positives  : {false_positive_bg_slices} ({100*false_positive_rate:.1f}%)")

    print(f"\nEnd-to-End Metrics (ALL slices, full resolution):")
    print(f"  Dice      : {all_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {all_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {all_metrics['mean_fg_precision']:.4f}")
    print(f"  F2        : {all_metrics['mean_fg_f2']:.4f}")

    print(f"\nEnd-to-End Metrics (FG slices only, full resolution):")
    print(f"  Dice      : {fg_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {fg_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {fg_metrics['mean_fg_precision']:.4f}")
    print(f"  F2        : {fg_metrics['mean_fg_f2']:.4f}")

    print(f"\nPer-class breakdown (all slices):")
    for c in sorted(all_metrics['dice_per_class'].keys()):
        name = ["BG", "AEAL", "AEAR"][c] if c < 3 else f"Class{c}"
        print(
            f"  {name}: "
            f"Dice={all_metrics['dice_per_class'][c]:.4f}  "
            f"Recall={all_metrics['recall_per_class'][c]:.4f}  "
            f"Precision={all_metrics['precision_per_class'][c]:.4f}  "
            f"F2={all_metrics['f2_per_class'][c]:.4f}"
        )

    logger.info(f"\nStage 1 detection rate: {100*detection_rate:.1f}% "
                f"({detected_fg_slices}/{total_fg_slices})")
    logger.info(f"Stage 1 FP rate: {100*false_positive_rate:.1f}% "
                f"({false_positive_bg_slices}/{total_bg_slices})")
    logger.info(f"E2E all-slices Dice={all_metrics['mean_fg_dice']:.4f} "
                f"Recall={all_metrics['mean_fg_recall']:.4f} "
                f"F2={all_metrics['mean_fg_f2']:.4f}")
    logger.info(f"E2E fg-only Dice={fg_metrics['mean_fg_dice']:.4f} "
                f"Recall={fg_metrics['mean_fg_recall']:.4f} "
                f"F2={fg_metrics['mean_fg_f2']:.4f}")
    logger.info(f"Saved {vis_count} visualizations to {vis_dir}")

    def _serialize(m):
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

        "num_epochs_run": epoch,
        "num_epochs_max": cfg.NUM_EPOCHS,
        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "patch_size": cfg.PATCH_SIZE,
        "patch_jitter_train": cfg.PATCH_JITTER_TRAIN,
        "patch_jitter_val": cfg.PATCH_JITTER_VAL,
        "lr": cfg.LR,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "boundary_weight": cfg.BOUNDARY_WEIGHT,

        "model_type": "B0 + Unet",
        "model_params": num_params,
        "early_stopped": epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE,

        "coarse_model_dir": coarse_dir,
        "fine_model_dir": str(output_dir),

        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
            "false_positive_rate": false_positive_rate,
            "fallback_count": fallback_count,
        },

        "e2e_metrics_all_slices": _serialize(all_metrics),
        "e2e_metrics_fg_only": _serialize(fg_metrics),

        "total_slices": len(all_preds),
        "fg_slices": len(fg_mask_indices),

        "history": history,

        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"")
    print(f"  Training (B0 + Unet, {num_params:,} params):")
    print(f"    Best val Dice  : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"    Best val Recall: {best_val_metrics.get('recall', 0.0):.4f}")
    if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
        print(f"    Early stopped at epoch {epoch}")
    print(f"")
    print(f"  End-to-End (native patches):")
    print(f"    Stage 1 detection rate: {100*detection_rate:.1f}%")
    print(f"    E2E Dice (all)  : {all_metrics['mean_fg_dice']:.4f}")
    print(f"    E2E Recall (all): {all_metrics['mean_fg_recall']:.4f}")
    print(f"    E2E F2 (all)    : {all_metrics['mean_fg_f2']:.4f}")
    print(f"    E2E Dice (FG)   : {fg_metrics['mean_fg_dice']:.4f}")
    print(f"    E2E Recall (FG) : {fg_metrics['mean_fg_recall']:.4f}")
    print(f"    E2E F2 (FG)     : {fg_metrics['mean_fg_f2']:.4f}")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
