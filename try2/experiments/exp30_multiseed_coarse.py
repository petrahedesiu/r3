
import sys
import os
import gc
import random
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_coarse import CoarseFullSliceDataset
from shared.losses import CompoundLoss
from shared.models import create_coarse_model
from shared.training import compute_class_weights, plot_training_history
from shared.metrics import compute_dice_score, compute_recall, compute_precision
from shared.two_stage_inference import _normalize, _resize_image, _image_to_tensor
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp30_multiseed_coarse"
    DESCRIPTION = "Multi-seed coarse ensemble: 5 seeds x 5 epochs, threshold sweep"

    NUM_CLASSES = 2
    CLASS_NAMES = ("BG", "FG")
    IMG_SIZE = 256
    BATCH_SIZE = 8
    LR = 1e-4

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8

    OVERSAMPLE_FACTOR = 5

    NUM_EPOCHS = 5
    SEEDS = [42, 43, 44, 45, 46]

    THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]



def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    fg_pixels = 0
    bg_pixels = 0
    for seg in segmentations:
        fg_pixels += (seg > 0).sum()
        bg_pixels += (seg == 0).sum()
    total = fg_pixels + bg_pixels
    if fg_pixels == 0:
        return torch.tensor([1.0, 1.0], device=device)
    wBg = total / (2 * bg_pixels)
    wFg = total / (2 * fg_pixels)
    weights = torch.tensor([wBg, wFg], dtype=torch.float32, device=device)
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

            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)

            total_loss += loss.item()
            total_dice += dice
            total_recall += recall
            total_precision += precision
            num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "recall": total_recall / n,
        "precision": total_precision / n,
    }



def evaluate_ensemble_detection(
    models: List[nn.Module],
    val_volumes: List[np.ndarray],
    val_segs: List[np.ndarray],
    device: torch.device,
    coarse_size: int = 256,
    thresholds: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
) -> Dict:
    results = {}

    for threshold in thresholds:
        total_fg_slices = 0
        detected_fg_slices = 0
        total_bg_slices = 0
        false_positive_bg_slices = 0

        for vol, seg in tqdm(
            list(zip(val_volumes, val_segs)),
            desc=f"Detection sweep threshold={threshold:.2f}",
        ):
            n_slices = vol.shape[2]

            for slice_idx in range(n_slices):
                image = vol[:, :, slice_idx].copy()
                gt_mask = seg[:, :, slice_idx].copy()
                has_fg = gt_mask.max() > 0

                image_norm = _normalize(image.astype(np.float32))
                coarse_input = _resize_image(image_norm, coarse_size)
                coarse_tensor = _image_to_tensor(coarse_input, device)

                fg_probs = []
                for model in models:
                    with torch.no_grad():
                        logits = model(coarse_tensor)
                        probs = F.softmax(logits, dim=1)
                        fg_prob = probs[0, 1].cpu().numpy()
                        fg_probs.append(fg_prob)

                avg_fg_prob = np.mean(fg_probs, axis=0)
                detected = (avg_fg_prob > threshold).any()
                if has_fg:
                    total_fg_slices += 1
                    if detected:
                        detected_fg_slices += 1
                else:
                    total_bg_slices += 1
                    if detected:
                        false_positive_bg_slices += 1

        detection_rate = detected_fg_slices / max(1, total_fg_slices)
        fp_rate = false_positive_bg_slices / max(1, total_bg_slices)

        results[threshold] = {
            "detection_rate": detection_rate,
            "fp_rate": fp_rate,
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "missed_fg_slices": total_fg_slices - detected_fg_slices,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
        }

    return results



def main():
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

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
    device = torch.device(cfg.DEVICE)
    # compute weights for the binary coarse task
    class_weights = compute_binary_class_weights(train_segs, device)
    print(f"Binary class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    all_checkpoints = {}

    for seed in cfg.SEEDS:
        print(f"\n{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")
        logger.info(f"\n{'=' * 70}")
        logger.info(f"TRAINING SEED {seed}")
        logger.info(f"{'=' * 70}")

        set_seed(seed)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

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
            train_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

        model = create_coarse_model(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
        ).to(device)

        criterion = CompoundLoss(
            focal_weight=0.35,
            tversky_weight=0.35,
            lovasz_weight=0.30,
            class_weights=class_weights,
            tversky_alpha=cfg.TVERSKY_ALPHA,
            tversky_beta=cfg.TVERSKY_BETA,
            focal_alpha=cfg.FOCAL_ALPHA,
            focal_gamma=cfg.FOCAL_GAMMA,
            use_boundary=False,
            num_classes=cfg.NUM_CLASSES,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT,
            eta_min=cfg.SCHEDULER_ETA_MIN,
        )

        best_dice = 0.0
        best_epoch = 0
        seed_checkpoints = []

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            logger.info(f"  Seed {seed} -- Epoch {epoch}/{cfg.NUM_EPOCHS}")

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
                f"    TRAIN  loss={train_metrics['loss']:.4f} "
                f"dice={train_metrics['dice']:.4f} "
                f"recall={train_metrics['recall']:.4f}"
            )
            logger.info(
                f"    VAL    loss={val_metrics['loss']:.4f} "
                f"dice={val_metrics['dice']:.4f} "
                f"recall={val_metrics['recall']:.4f} "
                f"precision={val_metrics['precision']:.4f}"
            )

            ckpt_path = seed_dir / f"epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_recall": val_metrics["recall"],
                    "val_precision": val_metrics["precision"],
                    "val_loss": val_metrics["loss"],
                    "num_classes": cfg.NUM_CLASSES,
                    "img_size": cfg.IMG_SIZE,
                },
                ckpt_path,
            )

            ckpt_info = {
                "path": str(ckpt_path),
                "seed": seed,
                "epoch": epoch,
                "val_dice": val_metrics["dice"],
                "val_recall": val_metrics["recall"],
                "val_precision": val_metrics["precision"],
                "val_loss": val_metrics["loss"],
            }
            seed_checkpoints.append(ckpt_info)

            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                best_epoch = epoch

        best_ckpt_src = seed_dir / f"epoch_{best_epoch}.pth"
        best_ckpt_dst = seed_dir / "best_model.pth"
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        logger.info(f"  Seed {seed} best: epoch {best_epoch}, dice={best_dice:.4f}")

        all_checkpoints[f"seed_{seed}"] = seed_checkpoints

        del model, criterion, optimizer, scheduler
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    ckpt_index_path = output_dir / "all_checkpoints.json"
    with open(ckpt_index_path, "w") as f:
        json.dump(all_checkpoints, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ENSEMBLE DETECTION EVALUATION")
    print("=" * 70)

    ensemble_models = []
    for seed in cfg.SEEDS:
        seed_dir = output_dir / f"seed_{seed}"
        model_path = seed_dir / "best_model.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = create_coarse_model(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        ensemble_models.append(model)
        print(f"Loaded seed {seed} best model (epoch {checkpoint['epoch']}, "
              f"dice={checkpoint['val_dice']:.4f})")

    detection_results = evaluate_ensemble_detection(
        ensemble_models, val_volumes, val_segs, device,
        coarse_size=cfg.IMG_SIZE,
        thresholds=cfg.THRESHOLDS,
    )

    print(f"\n{'Threshold':>10s} {'Det Rate':>10s} {'FP Rate':>10s} {'Missed':>8s}")
    print("-" * 42)
    best_threshold = None
    best_det_rate = 0.0
    for thr in sorted(detection_results.keys()):
        r = detection_results[thr]
        print(f"{thr:>10.2f} {r['detection_rate']:>10.1%} {r['fp_rate']:>10.1%} "
              f"{r['missed_fg_slices']:>8d}")
        logger.info(f"Threshold {thr:.2f}: det={r['detection_rate']:.1%}, "
                     f"FP={r['fp_rate']:.1%}, missed={r['missed_fg_slices']}")

        if r['fp_rate'] < 0.40 and r['detection_rate'] > best_det_rate:
            best_det_rate = r['detection_rate']
            best_threshold = thr

    if best_threshold is None:
        best_threshold = 0.30
    print(f"\nBest threshold: {best_threshold:.2f} "
          f"(det={best_det_rate:.1%})")
    logger.info(f"Best threshold: {best_threshold:.2f} (det={best_det_rate:.1%})")

    del ensemble_models
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    results = {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "description": cfg.DESCRIPTION,
        "output_dir": str(output_dir),
        "seeds": cfg.SEEDS,
        "num_epochs_per_seed": cfg.NUM_EPOCHS,
        "total_checkpoints": sum(len(v) for v in all_checkpoints.values()),
        "all_checkpoints": all_checkpoints,
        "detection_sweep": {
            str(k): v for k, v in detection_results.items()
        },
        "best_threshold": best_threshold,
        "best_detection_rate": best_det_rate,
        "config": {
            "num_classes": cfg.NUM_CLASSES,
            "img_size": cfg.IMG_SIZE,
            "lr": cfg.LR,
            "tversky_alpha": cfg.TVERSKY_ALPHA,
            "tversky_beta": cfg.TVERSKY_BETA,
            "oversample": cfg.OVERSAMPLE_FACTOR,
        },
        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"  Seeds trained: {cfg.SEEDS}")
    print(f"  Epochs per seed: {cfg.NUM_EPOCHS}")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Best detection rate: {best_det_rate:.1%}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
