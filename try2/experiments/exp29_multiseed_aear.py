
import os
import sys
import gc
import random
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_fine_patches import FinePatchDataset
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_model
from shared.training import compute_class_weights, plot_training_history
from shared.metrics import compute_dice_score, compute_recall, compute_precision
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp29_multiseed_aear"
    DESCRIPTION = "Multi-seed AEAR ensemble: 5 seeds x 3 epochs + sliding window, exp19-style"

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8
    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 15

    PATCH_SIZE = 128
    PATCH_JITTER_TRAIN = 10
    PATCH_JITTER_VAL = 0

    OVERSAMPLE_FACTOR = 3
    NUM_EPOCHS = 3
    SEEDS = [42, 43, 44, 45, 46]

    SLIDE_OFFSETS = [-32, 0, 32]



def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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



def sliding_window_predict_patch(
    image_norm: np.ndarray,
    centroid_r: int,
    centroid_c: int,
    models: List[nn.Module],
    device: torch.device,
    patch_size: int = 128,
    fine_size: int = 384,
    slide_offsets: List[int] = [-32, 0, 32],
    num_classes: int = 3,
) -> np.ndarray:
    import cv2
    H, W = image_norm.shape[:2]
    half = patch_size//2
    prob_accum = np.zeros((num_classes, H, W), dtype=np.float32)
    for dr in slide_offsets:
        for dc in slide_offsets:
            cr = centroid_r+dr
            cc = centroid_c+dc

            rmin = max(0, cr - half)
            rmax = min(H, cr + half)
            cmin = max(0, cc - half)
            cmax = min(W, cc + half)

            if rmax - rmin < patch_size:
                if rmin == 0:
                    rmax = min(H, patch_size)
                else:
                    rmin = max(0, rmax - patch_size)
            if cmax - cmin < patch_size:
                if cmin == 0:
                    cmax = min(W, patch_size)
                else:
                    cmin = max(0, cmax - patch_size)

            crop = image_norm[rmin:rmax, cmin:cmax]
            crop_h, crop_w = crop.shape[:2]
            if crop_h == 0 or crop_w == 0:
                continue

            fine_input = cv2.resize(crop, (fine_size, fine_size),
                                    interpolation=cv2.INTER_LINEAR)
            fine_tensor = torch.from_numpy(
                np.ascontiguousarray(fine_input)
            ).float().unsqueeze(0).unsqueeze(0).to(device)

            for model in models:
                with torch.no_grad():
                    logits = model(fine_tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                probs_resized = np.zeros((num_classes, crop_h, crop_w), dtype=np.float32)
                for c in range(num_classes):
                    probs_resized[c] = cv2.resize(
                        probs[c], (crop_w, crop_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                prob_accum[:, rmin:rmax, cmin:cmax] = np.maximum(
                    prob_accum[:, rmin:rmax, cmin:cmax],
                    probs_resized,
                )

    # collapse accumulated probs to a label map
    prediction = prob_accum.argmax(axis=0).astype(np.int64)
    return prediction



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

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    device = torch.device(cfg.DEVICE)

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
            train_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=use_pin_memory,
        )

        model = create_model(
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
            encoder_name=cfg.ENCODER_NAME,
            attention_type=cfg.ATTENTION_TYPE,
        ).to(device)

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
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT,
            eta_min=cfg.SCHEDULER_ETA_MIN,
        )

        seed_checkpoints = []

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            logger.info(f"  Seed {seed} -- Epoch {epoch}/{cfg.NUM_EPOCHS}")

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
                    "patch_size": cfg.PATCH_SIZE,
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
            logger.info(f"    Saved checkpoint -> {ckpt_path}")

        all_checkpoints[f"seed_{seed}"] = seed_checkpoints

        del model, criterion, optimizer, scheduler
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    ckpt_index_path = output_dir / "all_checkpoints.json"
    with open(ckpt_index_path, "w") as f:
        json.dump(all_checkpoints, f, indent=2, default=str)
    logger.info(f"Saved checkpoint index -> {ckpt_index_path}")

    epoch1_dices = []
    for seed_key, ckpts in all_checkpoints.items():
        for ckpt in ckpts:
            if ckpt["epoch"] == 1:
                epoch1_dices.append(ckpt["val_dice"])
    if epoch1_dices:
        dice_var = np.var(epoch1_dices)
        dice_mean = np.mean(epoch1_dices)
        logger.info(f"\nSeed diversity at epoch 1:")
        logger.info(f"  Val Dice values: {epoch1_dices}")
        logger.info(f"  Mean: {dice_mean:.4f}, Variance: {dice_var:.6f}")

    results = {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "description": cfg.DESCRIPTION,
        "output_dir": str(output_dir),
        "seeds": cfg.SEEDS,
        "num_epochs_per_seed": cfg.NUM_EPOCHS,
        "total_checkpoints": sum(len(v) for v in all_checkpoints.values()),
        "all_checkpoints": all_checkpoints,
        "seed_diversity_epoch1": {
            "dices": epoch1_dices,
            "mean": float(np.mean(epoch1_dices)) if epoch1_dices else 0.0,
            "variance": float(np.var(epoch1_dices)) if epoch1_dices else 0.0,
        },
        "sliding_window": {
            "offsets": cfg.SLIDE_OFFSETS,
            "grid_size": "3x3",
            "effective_coverage": "192x192",
        },
        "config": {
            "tversky_alpha": cfg.TVERSKY_ALPHA,
            "tversky_beta": cfg.TVERSKY_BETA,
            "boundary_weight": cfg.BOUNDARY_WEIGHT,
            "patch_size": cfg.PATCH_SIZE,
            "patch_jitter_train": cfg.PATCH_JITTER_TRAIN,
            "lr": cfg.LR,
            "img_size": cfg.IMG_SIZE,
            "encoder_name": cfg.ENCODER_NAME,
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
    print(f"  Total checkpoints: {sum(len(v) for v in all_checkpoints.values())}")
    print(f"  Sliding window: 3x3 grid, offsets={cfg.SLIDE_OFFSETS}")
    if epoch1_dices:
        print(f"  Epoch 1 val dice variance: {dice_var:.6f} (diversity check)")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
