
import sys
import os
import gc
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_fine import FineBBoxCropDataset
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_model, create_coarse_model
from shared.training import compute_class_weights, plot_training_history, plot_predictions
from shared.metrics import compute_all_metrics, compute_dice_score, compute_recall, compute_precision
from shared.two_stage_inference import two_stage_predict_slice
from shared.dataset import _normalize
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp18_tight_crops"
    DESCRIPTION = "Stage 2 fine segmenter retrained with tighter bbox crops (padding=15, jitter=5)"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    BATCH_SIZE = 4
    LR = 5e-5
    NUM_EPOCHS = 30

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8
    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 15

    BBOX_PADDING = 15
    BBOX_JITTER_TRAIN = 5
    BBOX_JITTER_VAL = 0

    OVERSAMPLE_FACTOR = 3

    COARSE_SIZE = 256
    FINE_SIZE = 384
    COARSE_THRESHOLD = 0.3
    E2E_BBOX_PADDING = 10

    USE_TTA = True
    USE_CC_FILTER = True
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
                labeled = get_labeled_slice_indices(seg)
                if len(labeled) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations


def compute_batch_distance_maps(masks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    bmaps = []
    mNp = masks.cpu().numpy()
    for i in range(mNp.shape[0]):
        dm = BoundaryLoss.compute_distance_map(mNp[i], num_classes=num_classes)
        bmaps.append(dm)
    return torch.from_numpy(np.stack(bmaps, axis=0)).float()


def find_latest_model_dir(experiment_name: str) -> Optional[str]:
    results_base = Config.OUTPUT_BASE
    exp_dir = os.path.join(results_base, experiment_name)
    if not os.path.isdir(exp_dir):
        return None
    subDirs = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        reverse=True,
    )
    for sd in subDirs:
        mp = os.path.join(exp_dir, sd, "best_model.pth")
        if os.path.exists(mp):
            return os.path.join(exp_dir, sd)
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


def load_fine_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 3)
    model = create_model(
        in_channels=1,
        num_classes=num_classes,
        encoder_name="efficientnet-b4",
        attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded fine model from {model_path}")
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
        rect = patches.Rectangle(
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

    # pinned memory does not play nice with mps
    use_pin_memory = (cfg.DEVICE != "mps")
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
    model = create_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME,
        attention_type=cfg.ATTENTION_TYPE,
    ).to(device)

    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {nParams:,}")

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
    logger.info(f"Boundary weight   : {cfg.BOUNDARY_WEIGHT}")
    logger.info(f"Boundary ramp-up  : {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs")
    logger.info(f"Bbox padding      : {cfg.BBOX_PADDING} +/- {cfg.BBOX_JITTER_TRAIN} jitter")

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
    print("STAGE 2: FINE SEGMENTER TRAINING (TIGHT CROPS)")
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

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION ON FULL-RESOLUTION IMAGES")
    print("=" * 70)

    coarse_dir = find_latest_model_dir("exp14_two_stage_coarse")
    if coarse_dir is None:
        print("WARNING: No trained coarse model found for exp14. Skipping E2E evaluation.")
        print(f"  Searched in: {cfg.OUTPUT_BASE}/exp14_two_stage_coarse/")
        logger.warning("Skipping E2E evaluation -- no coarse model found.")

        results = _build_training_results(cfg, output_dir, best_epoch, best_dice,
                                          best_val_metrics, history, timestamp)
        results_json_path = output_dir / "results.json"
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results JSON (training only) -> {results_json_path}")
        _print_summary(cfg, best_dice, best_epoch, best_val_metrics, output_dir,
                       e2e_metrics=None)
        return results

    print(f"\nCoarse model dir: {coarse_dir}")
    coarse_model = load_coarse_model(coarse_dir, device)

    fine_model = model
    fine_model.eval()
    fine_dir = str(output_dir)
    print(f"Fine model dir: {fine_dir} (just trained)")

    logger.info(f"Coarse model: {coarse_dir}")
    logger.info(f"Fine model  : {fine_dir} (this experiment)")
    logger.info(f"E2E bbox padding: {cfg.E2E_BBOX_PADDING}")
    logger.info(f"Coarse threshold: {cfg.COARSE_THRESHOLD}")
    logger.info(f"Use TTA         : {cfg.USE_TTA}")
    logger.info(f"Use CC filter   : {cfg.USE_CC_FILTER}")

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
            hasFg = gt_mask.max() > 0
            pred, info = two_stage_predict_slice(
                image,
                coarse_model,
                fine_model,
                device,
                coarse_size=cfg.COARSE_SIZE,
                fine_size=cfg.FINE_SIZE,
                coarse_threshold=cfg.COARSE_THRESHOLD,
                bbox_padding=cfg.E2E_BBOX_PADDING,
                use_tta=cfg.USE_TTA,
                use_cc_filter=cfg.USE_CC_FILTER,
                cc_min_size=cfg.CC_MIN_SIZE,
                cc_max_size=cfg.CC_MAX_SIZE,
            )

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if hasFg:
                total_fg_slices += 1
                if info["detected"]:
                    detected_fg_slices += 1
                if info["fallback_full"]:
                    fallback_count += 1
            else:
                total_bg_slices += 1
                if info["detected"]:
                    false_positive_bg_slices += 1

            if hasFg and vis_count < max_vis:
                imgNorm = _normalize(image.astype(np.float32))
                plot_two_stage_visualization(
                    imgNorm, gt_mask, pred, info,
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

        "training": {
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
            "bbox_padding": cfg.BBOX_PADDING,
            "bbox_jitter": cfg.BBOX_JITTER_TRAIN,
            "history": history,
        },

        "coarse_model_dir": coarse_dir,
        "fine_model_dir": fine_dir,

        "inference": {
            "coarse_threshold": cfg.COARSE_THRESHOLD,
            "e2e_bbox_padding": cfg.E2E_BBOX_PADDING,
            "use_tta": cfg.USE_TTA,
            "use_cc_filter": cfg.USE_CC_FILTER,
        },

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

        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "boundary_weight": cfg.BOUNDARY_WEIGHT,

        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON -> {results_json_path}")

    _print_summary(cfg, best_dice, best_epoch, best_val_metrics, output_dir,
                   e2e_metrics={
                       "detection_rate": detection_rate,
                       "all_metrics": all_metrics,
                       "fg_metrics": fg_metrics,
                   })

    return results


def _build_training_results(cfg, output_dir, best_epoch, best_dice,
                            best_val_metrics, history, timestamp):
    return {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "description": cfg.DESCRIPTION,
        "output_dir": str(output_dir),
        "training": {
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
            "bbox_padding": cfg.BBOX_PADDING,
            "bbox_jitter": cfg.BBOX_JITTER_TRAIN,
            "history": history,
        },
        "e2e_metrics_all_slices": None,
        "e2e_metrics_fg_only": None,
        "num_classes": cfg.NUM_CLASSES,
        "img_size": cfg.IMG_SIZE,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "boundary_weight": cfg.BOUNDARY_WEIGHT,
        "timestamp": timestamp,
    }


def _print_summary(cfg, best_dice, best_epoch, best_val_metrics, output_dir,
                   e2e_metrics=None):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"")
    print(f"  Training:")
    print(f"    Best val Dice  : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"    Best val Recall: {best_val_metrics.get('recall', 0.0):.4f}")
    if e2e_metrics is not None:
        print(f"")
        print(f"  End-to-End:")
        print(f"    Stage 1 detection rate: {100*e2e_metrics['detection_rate']:.1f}%")
        am = e2e_metrics["all_metrics"]
        fm = e2e_metrics["fg_metrics"]
        print(f"    E2E Dice (all)  : {am['mean_fg_dice']:.4f}")
        print(f"    E2E Recall (all): {am['mean_fg_recall']:.4f}")
        print(f"    E2E F2 (all)    : {am['mean_fg_f2']:.4f}")
        print(f"    E2E Dice (FG)   : {fm['mean_fg_dice']:.4f}")
        print(f"    E2E Recall (FG) : {fm['mean_fg_recall']:.4f}")
        print(f"    E2E F2 (FG)     : {fm['mean_fg_f2']:.4f}")
    else:
        print(f"")
        print(f"  End-to-End: SKIPPED (no coarse model available)")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
