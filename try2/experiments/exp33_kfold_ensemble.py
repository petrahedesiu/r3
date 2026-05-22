
import os
import sys
import gc
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset_coarse import CoarseFullSliceDataset
from shared.dataset import FGCenteredDataset, StandardDataset, _normalize
from shared.dataset_fine_patches import FinePatchDataset
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_model, create_coarse_model
from shared.training import compute_class_weights, plot_training_history
from shared.metrics import (
    compute_all_metrics, compute_dice_score, compute_recall, compute_precision,
)
from shared.two_stage_inference import (
    _normalize as _infer_normalize,
    _resize_image, _resize_mask, _image_to_tensor,
    _extract_bbox_from_binary_mask,
)
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp33_kfold_ensemble"
    DESCRIPTION = "5-fold CV ensemble on 52 combined patients"

    N_FOLDS = 5
    NUM_EPOCHS = 5

    COARSE_IMG_SIZE = 256
    COARSE_BATCH_SIZE = 8
    COARSE_LR = 1e-4
    COARSE_OVERSAMPLE = 5

    AEAL_IMG_SIZE = 384
    AEAL_BATCH_SIZE = 4
    AEAL_LR = 5e-5
    AEAL_FG_RATIO = 0.5
    AEAL_BOUNDARY_WEIGHT = 0.15
    AEAL_BOUNDARY_RAMPUP = 15

    AEAR_IMG_SIZE = 384
    AEAR_BATCH_SIZE = 4
    AEAR_LR = 5e-5
    AEAR_PATCH_SIZE = 128
    AEAR_JITTER_TRAIN = 10
    AEAR_JITTER_VAL = 0
    AEAR_BOUNDARY_WEIGHT = 0.15
    AEAR_BOUNDARY_RAMPUP = 15

    TVERSKY_ALPHA = 0.2
    TVERSKY_BETA = 0.8



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


def load_data_combined(data_dirs):
    volumes, segmentations = [], []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"WARNING: Directory not found: {data_dir}")
            continue
        patients = discover_patients(data_dir)
        for p in tqdm(patients, desc=f"Loading from {os.path.basename(data_dir)}"):
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


def compute_binary_class_weights(segmentations, device):
    fg_pixels = bg_pixels = 0
    for s in segmentations:
        fg_pixels += (s > 0).sum()
        bg_pixels += (s == 0).sum()
    total = fg_pixels + bg_pixels
    if fg_pixels == 0:
        return torch.tensor([1.0, 1.0], device=device)
    w_bg = total / (2 * bg_pixels)
    w_fg = total / (2 * fg_pixels)
    weights = torch.tensor([w_bg, w_fg], dtype=torch.float32, device=device)
    return weights / weights.mean()


def compute_batch_distance_maps(masks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    batch_maps = []
    masks_np = masks.cpu().numpy()
    for i in range(masks_np.shape[0]):
        dm = BoundaryLoss.compute_distance_map(masks_np[i], num_classes=num_classes)
        batch_maps.append(dm)
    return torch.from_numpy(np.stack(batch_maps, axis=0)).float()



def train_epoch_coarse(model, dataloader, criterion, optimizer, device, epoch=0, num_classes=2):
    model.train()
    total_loss = total_dice = total_recall = total_precision = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"    Coarse train ep {epoch}", leave=False)
    for images, masks in pbar:
        images = images.to(device).float()
        masks = masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = criterion(outputs, masks, epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)
        total_loss += loss.item(); total_dice += dice
        total_recall += recall; total_precision += precision
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
    n = max(num_batches, 1)
    return {"loss": total_loss/n, "dice": total_dice/n,
            "recall": total_recall/n, "precision": total_precision/n}


def validate_coarse(model, dataloader, criterion, device, epoch=0, num_classes=2):
    model.eval()
    total_loss = total_dice = total_recall = total_precision = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"    Coarse val ep {epoch}", leave=False):
            images = images.to(device).float()
            masks = masks.to(device).long()
            outputs = model(images)
            loss, _ = criterion(outputs, masks, epoch=epoch)
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)
            total_loss += loss.item(); total_dice += dice
            total_recall += recall; total_precision += precision
            num_batches += 1
    n = max(num_batches, 1)
    return {"loss": total_loss/n, "dice": total_dice/n,
            "recall": total_recall/n, "precision": total_precision/n}


def train_epoch_fine(model, dataloader, criterion, optimizer, device, epoch=0, num_classes=3):
    model.train()
    total_loss = total_dice = total_recall = total_precision = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"    Fine train ep {epoch}", leave=False)
    for images, masks in pbar:
        images = images.to(device).float()
        masks = masks.to(device).long()
        distance_maps = compute_batch_distance_maps(masks, num_classes=num_classes).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = criterion(outputs, masks, epoch=epoch, distance_map=distance_maps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
        optimizer.step()
        with torch.no_grad():
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)
        total_loss += loss.item(); total_dice += dice
        total_recall += recall; total_precision += precision
        num_batches += 1
        if num_batches % 50 == 0 and device.type == "mps":
            torch.mps.empty_cache()
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
    n = max(num_batches, 1)
    return {"loss": total_loss/n, "dice": total_dice/n,
            "recall": total_recall/n, "precision": total_precision/n}


def validate_fine(model, dataloader, criterion, device, epoch=0, num_classes=3):
    model.eval()
    total_loss = total_dice = total_recall = total_precision = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"    Fine val ep {epoch}", leave=False):
            images = images.to(device).float()
            masks = masks.to(device).long()
            distance_maps = compute_batch_distance_maps(masks, num_classes=num_classes).to(device)
            outputs = model(images)
            loss, _ = criterion(outputs, masks, epoch=epoch, distance_map=distance_maps)
            dice, _ = compute_dice_score(outputs, masks, num_classes=num_classes)
            recall, _ = compute_recall(outputs, masks, num_classes=num_classes)
            precision, _ = compute_precision(outputs, masks, num_classes=num_classes)
            total_loss += loss.item(); total_dice += dice
            total_recall += recall; total_precision += precision
            num_batches += 1
    n = max(num_batches, 1)
    return {"loss": total_loss/n, "dice": total_dice/n,
            "recall": total_recall/n, "precision": total_precision/n}



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, save_dir, model_name, num_epochs, num_classes,
                train_fn, val_fn, logger):
    best_dice = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_fn(model, train_loader, criterion, optimizer, device,
                                 epoch=epoch, num_classes=num_classes)
        val_metrics = val_fn(model, val_loader, criterion, device,
                             epoch=epoch, num_classes=num_classes)
        scheduler.step()

        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        logger.info(
            f"    [{model_name}] ep {epoch}: "
            f"train dice={train_metrics['dice']:.4f} "
            f"val dice={val_metrics['dice']:.4f} recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_recall": val_metrics["recall"],
                    "val_precision": val_metrics["precision"],
                    "num_classes": num_classes,
                },
                save_dir / "best_model.pth",
            )

    logger.info(f"    [{model_name}] best: epoch {best_epoch}, dice={best_dice:.4f}")
    return best_epoch, best_dice



def ensemble_coarse_stage(image_norm, coarse_models, device, coarse_size=256, threshold=0.3):
    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    avg_fg_prob = np.zeros((coarse_size, coarse_size), dtype=np.float32)
    with torch.no_grad():
        for model in coarse_models:
            logits = model(coarse_tensor)
            probs = F.softmax(logits, dim=1)
            avg_fg_prob += probs[0, 1].cpu().numpy()
    avg_fg_prob /= len(coarse_models)

    coarse_binary = (avg_fg_prob > threshold).astype(np.uint8)
    info = {
        "detected": coarse_binary.sum() > 0,
        "coarse_fg_fraction": float(coarse_binary.sum()) / coarse_binary.size,
    }
    if coarse_binary.sum() == 0:
        return None, info
    return coarse_binary, info


def ensemble_aeal_bbox_path(image_norm, coarse_binary, aeal_models, device,
                            coarse_size=256, fine_size=384, bbox_padding=50):
    H, W = image_norm.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)

    scale_r = H / coarse_size
    scale_c = W / coarse_size

    coarse_bbox = _extract_bbox_from_binary_mask(coarse_binary, padding=0, max_fraction=0.6)
    if coarse_bbox is None and coarse_binary.sum() > 0:
        bbox_orig = (0, H, 0, W)
    elif coarse_bbox is None:
        return prediction
    else:
        rmin_orig = max(0, int(coarse_bbox[0] * scale_r) - bbox_padding)
        rmax_orig = min(H, int(coarse_bbox[1] * scale_r) + bbox_padding)
        cmin_orig = max(0, int(coarse_bbox[2] * scale_c) - bbox_padding)
        cmax_orig = min(W, int(coarse_bbox[3] * scale_c) + bbox_padding)
        bbox_orig = (rmin_orig, rmax_orig, cmin_orig, cmax_orig)

    rmin, rmax, cmin, cmax = bbox_orig
    crop = image_norm[rmin:rmax, cmin:cmax]
    crop_h, crop_w = crop.shape[:2]

    fine_input = _resize_image(crop, fine_size)
    fine_tensor = _image_to_tensor(fine_input, device)

    avg_probs = None
    with torch.no_grad():
        for model in aeal_models:
            logits = model(fine_tensor)
            probs = F.softmax(logits, dim=1)
            if avg_probs is None:
                avg_probs = probs.cpu().numpy()
            else:
                avg_probs += probs.cpu().numpy()
    avg_probs /= len(aeal_models)

    fine_pred = avg_probs[0].argmax(axis=0)
    fine_pred_resized = _resize_mask(fine_pred.astype(np.int64), (crop_h, crop_w))
    aeal_mask = (fine_pred_resized == 1)
    prediction[rmin:rmax, cmin:cmax][aeal_mask] = 1
    return prediction


def ensemble_aear_patch_path(image_norm, coarse_binary, aear_models, device,
                             coarse_size=256, patch_size=128, fine_size=384):
    H, W = image_norm.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)

    rows_c, cols_c = np.where(coarse_binary > 0)
    cr = int(rows_c.mean() * H / coarse_size)
    cc = int(cols_c.mean() * W / coarse_size)

    half = patch_size // 2
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

    fine_input = _resize_image(crop, fine_size)
    fine_tensor = _image_to_tensor(fine_input, device)

    avg_probs = None
    with torch.no_grad():
        for model in aear_models:
            logits = model(fine_tensor)
            probs = F.softmax(logits, dim=1)
            if avg_probs is None:
                avg_probs = probs.cpu().numpy()
            else:
                avg_probs += probs.cpu().numpy()
    avg_probs /= len(aear_models)

    fine_pred = avg_probs[0].argmax(axis=0)
    fine_pred_resized = _resize_mask(fine_pred.astype(np.int64), (crop_h, crop_w))
    aear_mask = (fine_pred_resized == 2)
    prediction[rmin:rmax, cmin:cmax][aear_mask] = 2
    return prediction


def ensemble_predict_slice(image, coarse_models, aeal_models, aear_models, device,
                           coarse_size=256, fine_size=384, patch_size=128,
                           coarse_threshold=0.3, bbox_padding=50):
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {"detected": False, "coarse_fg_fraction": 0.0}

    image_norm = _infer_normalize(image.astype(np.float32))

    coarse_binary, coarse_info = ensemble_coarse_stage(
        image_norm, coarse_models, device,
        coarse_size=coarse_size, threshold=coarse_threshold,
    )
    info["coarse_fg_fraction"] = coarse_info["coarse_fg_fraction"]

    if coarse_binary is None:
        return prediction, info

    info["detected"] = True

    aeal_pred = ensemble_aeal_bbox_path(
        image_norm, coarse_binary, aeal_models, device,
        coarse_size=coarse_size, fine_size=fine_size, bbox_padding=bbox_padding,
    )
    aear_pred = ensemble_aear_patch_path(
        image_norm, coarse_binary, aear_models, device,
        coarse_size=coarse_size, patch_size=patch_size, fine_size=fine_size,
    )

    prediction[aeal_pred == 1] = 1
    prediction[aear_pred == 2] = 2
    return prediction, info



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

    print("\nLoading combined data...")
    data_dirs = [
        cfg.DATA_DIR,
        os.path.join(os.path.dirname(cfg.DATA_DIR), "CROP - februarie 2026"),
    ]
    volumes, segmentations = load_data_combined(data_dirs)
    print(f"Loaded {len(volumes)} patients total")
    logger.info(f"Loaded {len(volumes)} patients total")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Exiting.")
        return

    device = torch.device(cfg.DEVICE)

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    all_indices = np.arange(len(volumes))
    fold_results = []
    patient_fold_map = {}  # maps patient index -> fold it was validated in

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx}/{cfg.N_FOLDS - 1} -- "
              f"Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")
        print(f"{'=' * 70}")
        logger.info(f"{'=' * 70}")
        logger.info(f"FOLD {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

        for idx in val_idx:
            patient_fold_map[int(idx)] = fold_idx

        train_vols = [volumes[i] for i in train_idx]
        train_segs = [segmentations[i] for i in train_idx]
        val_vols = [volumes[i] for i in val_idx]
        val_segs = [segmentations[i] for i in val_idx]

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Training coarse model...")
        coarse_dir = fold_dir / "coarse"
        coarse_dir.mkdir(exist_ok=True)

        coarse_cw = compute_binary_class_weights(train_segs, device)
        coarse_train_ds = CoarseFullSliceDataset(
            train_vols, train_segs,
            transform=get_transforms(train=True, img_size=cfg.COARSE_IMG_SIZE),
            oversample=cfg.COARSE_OVERSAMPLE,
        )
        coarse_val_ds = CoarseFullSliceDataset(
            val_vols, val_segs,
            transform=get_transforms(train=False, img_size=cfg.COARSE_IMG_SIZE),
            oversample=1,
        )
        coarse_train_loader = DataLoader(coarse_train_ds, batch_size=cfg.COARSE_BATCH_SIZE,
                                         shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        coarse_val_loader = DataLoader(coarse_val_ds, batch_size=cfg.COARSE_BATCH_SIZE,
                                       shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

        coarse_model = create_coarse_model(in_channels=cfg.IN_CHANNELS, num_classes=2).to(device)
        coarse_criterion = CompoundLoss(
            focal_weight=0.35, tversky_weight=0.35, lovasz_weight=0.30,
            class_weights=coarse_cw, tversky_alpha=cfg.TVERSKY_ALPHA, tversky_beta=cfg.TVERSKY_BETA,
            focal_alpha=cfg.FOCAL_ALPHA, focal_gamma=cfg.FOCAL_GAMMA,
            use_boundary=False, num_classes=2,
        ).to(device)
        coarse_opt = torch.optim.AdamW(coarse_model.parameters(), lr=cfg.COARSE_LR, weight_decay=cfg.WEIGHT_DECAY)
        coarse_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            coarse_opt, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT, eta_min=cfg.SCHEDULER_ETA_MIN)

        coarse_best_ep, coarse_best_dice = train_model(
            coarse_model, coarse_train_loader, coarse_val_loader,
            coarse_criterion, coarse_opt, coarse_sched,
            device, coarse_dir, "coarse", cfg.NUM_EPOCHS, 2,
            train_epoch_coarse, validate_coarse, logger,
        )

        del coarse_train_ds, coarse_val_ds, coarse_train_loader, coarse_val_loader
        del coarse_criterion, coarse_opt, coarse_sched, coarse_model
        if device.type == "mps": torch.mps.empty_cache()
        gc.collect()

        print(f"  Training AEAL model...")
        aeal_dir = fold_dir / "aeal"
        aeal_dir.mkdir(exist_ok=True)

        aeal_cw = compute_class_weights(train_segs, num_classes=3)
        aeal_train_ds = FGCenteredDataset(
            train_vols, train_segs,
            transform=get_transforms(train=True, img_size=cfg.AEAL_IMG_SIZE),
            fg_ratio=cfg.AEAL_FG_RATIO, patch_size=cfg.AEAL_IMG_SIZE,
            oversample=cfg.OVERSAMPLE_FACTOR,
        )
        aeal_val_ds = StandardDataset(
            val_vols, val_segs,
            transform=get_transforms(train=False, img_size=cfg.AEAL_IMG_SIZE),
            oversample=1,
        )
        aeal_train_loader = DataLoader(aeal_train_ds, batch_size=cfg.AEAL_BATCH_SIZE,
                                        shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        aeal_val_loader = DataLoader(aeal_val_ds, batch_size=cfg.AEAL_BATCH_SIZE,
                                      shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

        aeal_model = create_model(
            in_channels=cfg.IN_CHANNELS, num_classes=3,
            encoder_name=cfg.ENCODER_NAME, attention_type=cfg.ATTENTION_TYPE,
        ).to(device)
        aeal_criterion = CompoundLoss(
            focal_weight=0.35, tversky_weight=0.35, lovasz_weight=0.30,
            boundary_weight=cfg.AEAL_BOUNDARY_WEIGHT,
            class_weights=aeal_cw.to(device),
            tversky_alpha=cfg.TVERSKY_ALPHA, tversky_beta=cfg.TVERSKY_BETA,
            focal_alpha=cfg.FOCAL_ALPHA, focal_gamma=cfg.FOCAL_GAMMA,
            use_boundary=True, epoch_for_boundary_rampup=cfg.AEAL_BOUNDARY_RAMPUP,
            num_classes=3,
        ).to(device)
        aeal_opt = torch.optim.AdamW(aeal_model.parameters(), lr=cfg.AEAL_LR, weight_decay=cfg.WEIGHT_DECAY)
        aeal_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            aeal_opt, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT, eta_min=cfg.SCHEDULER_ETA_MIN)

        aeal_best_ep, aeal_best_dice = train_model(
            aeal_model, aeal_train_loader, aeal_val_loader,
            aeal_criterion, aeal_opt, aeal_sched,
            device, aeal_dir, "aeal", cfg.NUM_EPOCHS, 3,
            train_epoch_fine, validate_fine, logger,
        )

        del aeal_train_ds, aeal_val_ds, aeal_train_loader, aeal_val_loader
        del aeal_criterion, aeal_opt, aeal_sched, aeal_model
        if device.type == "mps": torch.mps.empty_cache()
        gc.collect()

        print(f"  Training AEAR model...")
        aear_dir = fold_dir / "aear"
        aear_dir.mkdir(exist_ok=True)

        aear_train_ds = FinePatchDataset(
            train_vols, train_segs,
            transform=get_transforms(train=True, img_size=cfg.AEAR_IMG_SIZE),
            patch_size=cfg.AEAR_PATCH_SIZE, jitter=cfg.AEAR_JITTER_TRAIN,
            oversample=cfg.OVERSAMPLE_FACTOR,
        )
        aear_val_ds = FinePatchDataset(
            val_vols, val_segs,
            transform=get_transforms(train=False, img_size=cfg.AEAR_IMG_SIZE),
            patch_size=cfg.AEAR_PATCH_SIZE, jitter=cfg.AEAR_JITTER_VAL,
            oversample=1,
        )
        aear_train_loader = DataLoader(aear_train_ds, batch_size=cfg.AEAR_BATCH_SIZE,
                                        shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        pin_memory=(cfg.DEVICE != "mps"))
        aear_val_loader = DataLoader(aear_val_ds, batch_size=cfg.AEAR_BATCH_SIZE,
                                      shuffle=False, num_workers=cfg.NUM_WORKERS,
                                      pin_memory=(cfg.DEVICE != "mps"))

        aear_model = create_model(
            in_channels=cfg.IN_CHANNELS, num_classes=3,
            encoder_name=cfg.ENCODER_NAME, attention_type=cfg.ATTENTION_TYPE,
        ).to(device)
        aear_criterion = CompoundLoss(
            focal_weight=0.35, tversky_weight=0.35, lovasz_weight=0.30,
            boundary_weight=cfg.AEAR_BOUNDARY_WEIGHT,
            class_weights=aeal_cw.to(device),
            tversky_alpha=cfg.TVERSKY_ALPHA, tversky_beta=cfg.TVERSKY_BETA,
            focal_alpha=cfg.FOCAL_ALPHA, focal_gamma=cfg.FOCAL_GAMMA,
            use_boundary=True, epoch_for_boundary_rampup=cfg.AEAR_BOUNDARY_RAMPUP,
            num_classes=3,
        ).to(device)
        aear_opt = torch.optim.AdamW(aear_model.parameters(), lr=cfg.AEAR_LR, weight_decay=cfg.WEIGHT_DECAY)
        aear_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            aear_opt, T_0=cfg.SCHEDULER_T0, T_mult=cfg.SCHEDULER_TMULT, eta_min=cfg.SCHEDULER_ETA_MIN)

        aear_best_ep, aear_best_dice = train_model(
            aear_model, aear_train_loader, aear_val_loader,
            aear_criterion, aear_opt, aear_sched,
            device, aear_dir, "aear", cfg.NUM_EPOCHS, 3,
            train_epoch_fine, validate_fine, logger,
        )

        del aear_train_ds, aear_val_ds, aear_train_loader, aear_val_loader
        del aear_criterion, aear_opt, aear_sched, aear_model
        if device.type == "mps": torch.mps.empty_cache()
        gc.collect()

        fold_results.append({
            "fold": fold_idx,
            "train_patients": len(train_idx),
            "val_patients": len(val_idx),
            "coarse_best_epoch": coarse_best_ep, "coarse_best_dice": coarse_best_dice,
            "aeal_best_epoch": aeal_best_ep, "aeal_best_dice": aeal_best_dice,
            "aear_best_epoch": aear_best_ep, "aear_best_dice": aear_best_dice,
        })

    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION (all 5 folds)")
    print("=" * 70)
    logger.info("=" * 70)
    logger.info("ENSEMBLE EVALUATION")

    coarse_models = []
    aeal_models = []
    aear_models = []
    for fold_idx in range(cfg.N_FOLDS):
        fold_dir = output_dir / f"fold_{fold_idx}"

        cm = create_coarse_model(in_channels=cfg.IN_CHANNELS, num_classes=2).to(device)
        ckpt = torch.load(fold_dir / "coarse" / "best_model.pth", map_location=device, weights_only=False)
        cm.load_state_dict(ckpt["model_state_dict"])
        cm.eval()
        coarse_models.append(cm)

        am = create_model(in_channels=cfg.IN_CHANNELS, num_classes=3,
                          encoder_name=cfg.ENCODER_NAME, attention_type=cfg.ATTENTION_TYPE).to(device)
        ckpt = torch.load(fold_dir / "aeal" / "best_model.pth", map_location=device, weights_only=False)
        am.load_state_dict(ckpt["model_state_dict"])
        am.eval()
        aeal_models.append(am)

        rm = create_model(in_channels=cfg.IN_CHANNELS, num_classes=3,
                          encoder_name=cfg.ENCODER_NAME, attention_type=cfg.ATTENTION_TYPE).to(device)
        ckpt = torch.load(fold_dir / "aear" / "best_model.pth", map_location=device, weights_only=False)
        rm.load_state_dict(ckpt["model_state_dict"])
        rm.eval()
        aear_models.append(rm)

    print(f"Loaded {len(coarse_models)} coarse + {len(aeal_models)} AEAL + {len(aear_models)} AEAR models")
    logger.info(f"Loaded {cfg.N_FOLDS} models per type for ensemble")

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_fg_slices = detected_fg_slices = 0
    total_bg_slices = false_positive_bg_slices = 0

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0
    max_vis = 30

    for patient_idx in tqdm(range(len(volumes)), desc="Ensemble eval all patients"):
        vol = volumes[patient_idx]
        seg = segmentations[patient_idx]
        n_slices = vol.shape[2]

        for slice_idx in range(n_slices):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = ensemble_predict_slice(
                image, coarse_models, aeal_models, aear_models, device,
                coarse_size=cfg.COARSE_IMG_SIZE, fine_size=cfg.AEAL_IMG_SIZE,
                patch_size=cfg.AEAR_PATCH_SIZE,
                coarse_threshold=0.3, bbox_padding=50,
            )

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if has_fg:
                total_fg_slices += 1
                if info["detected"]:
                    detected_fg_slices += 1
            else:
                total_bg_slices += 1
                if info["detected"]:
                    false_positive_bg_slices += 1

            if has_fg and vis_count < max_vis:
                image_norm = _infer_normalize(image.astype(np.float32))
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_norm, cmap="gray"); axes[0].set_title("CT"); axes[0].axis("off")
                axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2); axes[1].set_title("GT"); axes[1].axis("off")
                axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=2); axes[2].set_title("Ensemble Pred"); axes[2].axis("off")
                fig.suptitle(f"Patient {patient_idx}, Slice {slice_idx}")
                plt.tight_layout()
                plt.savefig(vis_dir / f"patient{patient_idx}_slice{slice_idx}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                vis_count += 1

        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

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

    print(f"\nStage 1 Detection (ensemble of {cfg.N_FOLDS} coarse models):")
    print(f"  FG slices: {total_fg_slices}, detected: {detected_fg_slices} ({100*detection_rate:.1f}%)")
    print(f"  BG slices: {total_bg_slices}, FP: {false_positive_bg_slices} ({100*false_positive_rate:.1f}%)")

    print(f"\nEnsemble E2E Metrics (all {len(volumes)} patients, all slices):")
    print(f"  Dice={all_metrics['mean_fg_dice']:.4f}  Recall={all_metrics['mean_fg_recall']:.4f}  "
          f"Precision={all_metrics['mean_fg_precision']:.4f}  F2={all_metrics['mean_fg_f2']:.4f}")

    print(f"\nEnsemble E2E Metrics (FG slices only):")
    print(f"  Dice={fg_metrics['mean_fg_dice']:.4f}  Recall={fg_metrics['mean_fg_recall']:.4f}  "
          f"Precision={fg_metrics['mean_fg_precision']:.4f}  F2={fg_metrics['mean_fg_f2']:.4f}")

    print(f"\nPer-class (all slices):")
    for c in [1, 2]:
        name = ["BG", "AEAL", "AEAR"][c]
        print(f"  {name}: Dice={all_metrics['dice_per_class'][c]:.4f}  "
              f"Recall={all_metrics['recall_per_class'][c]:.4f}  "
              f"Precision={all_metrics['precision_per_class'][c]:.4f}")

    exp23_dice = 0.636
    delta = all_metrics['mean_fg_dice'] - exp23_dice
    print(f"\nComparison to exp23 baseline (0.636):")
    print(f"  exp33 Ensemble Dice: {all_metrics['mean_fg_dice']:.4f} ({delta:+.4f})")

    logger.info(f"Detection: {100*detection_rate:.1f}%")
    logger.info(f"E2E all: Dice={all_metrics['mean_fg_dice']:.4f} Recall={all_metrics['mean_fg_recall']:.4f}")
    logger.info(f"E2E FG:  Dice={fg_metrics['mean_fg_dice']:.4f} Recall={fg_metrics['mean_fg_recall']:.4f}")

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
        "total_patients": len(volumes),
        "n_folds": cfg.N_FOLDS,
        "epochs_per_fold": cfg.NUM_EPOCHS,

        "fold_results": fold_results,

        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
        },

        "e2e_metrics_all_slices": _serialize(all_metrics),
        "e2e_metrics_fg_only": _serialize(fg_metrics),

        "comparison": {
            "exp23_baseline_dice": exp23_dice,
            "exp33_dice": all_metrics['mean_fg_dice'],
            "delta": delta,
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
    print(f"  Patients: {len(volumes)}, Folds: {cfg.N_FOLDS}, Epochs/fold: {cfg.NUM_EPOCHS}")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: coarse={fr['coarse_best_dice']:.4f} "
              f"aeal={fr['aeal_best_dice']:.4f} aear={fr['aear_best_dice']:.4f}")
    print(f"  E2E Ensemble Dice (all): {all_metrics['mean_fg_dice']:.4f}")
    print(f"  E2E Ensemble Dice (FG):  {fg_metrics['mean_fg_dice']:.4f}")
    print(f"  AEAL: {all_metrics['dice_per_class'][1]:.4f}  AEAR: {all_metrics['dice_per_class'][2]:.4f}")
    print(f"  vs exp23: {delta:+.4f}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    del coarse_models, aeal_models, aear_models
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return results


if __name__ == "__main__":
    main()
