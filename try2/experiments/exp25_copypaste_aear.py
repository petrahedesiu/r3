
import sys
import os
import gc
import random
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.losses import CompoundLoss, BoundaryLoss
from shared.models import create_model, create_coarse_model
from shared.training import compute_class_weights, plot_training_history, plot_predictions
from shared.metrics import compute_all_metrics, compute_dice_score, compute_recall, compute_precision
from shared.two_stage_inference import native_patch_predict_slice
from shared.dataset import _normalize, _to_tensors
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp25_copypaste_aear"
    DESCRIPTION = "Copy-paste AEAR specialist with anti-overfitting interventions"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    BATCH_SIZE = 4
    LR = 1e-5
    NUM_EPOCHS = 50

    TVERSKY_ALPHA = 0.15
    TVERSKY_BETA = 0.85

    USE_BOUNDARY = True
    BOUNDARY_WEIGHT = 0.15
    EPOCH_FOR_BOUNDARY_RAMPUP = 20

    PATCH_SIZE = 128
    PATCH_JITTER_TRAIN = 15
    PATCH_JITTER_VAL = 0

    OVERSAMPLE_FACTOR = 5

    DROPOUT_P = 0.2

    WARMUP_EPOCHS = 5

    PATIENCE = 10

    COPY_PASTE_PROB = 0.5
    MAX_DONORS = 3



class DropoutModel(nn.Module):

    def __init__(self, base_model: nn.Module, dropout_p: float = 0.2):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encoder(x)
        decoder_output = self.base_model.decoder(features)
        decoder_output = self.dropout(decoder_output)
        return self.base_model.segmentation_head(decoder_output)



class CopyPasteFinePatchDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        patch_size: int = 128,
        jitter: int = 15,
        oversample: int = 5,
        copy_paste_prob: float = 0.5,
        max_donors: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.patch_size = patch_size
        self.jitter = jitter
        self.copy_paste_prob = copy_paste_prob
        self.max_donors = max_donors

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                if seg[:, :, slice_idx].max() > 0:
                    for _ in range(oversample):
                        self.indices.append((patient_idx, slice_idx))

        self.donors: List[Tuple[int, int, int, int, int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                s = seg[:, :, slice_idx]
                if s.max() > 0:
                    rows = np.any(s > 0, axis=1)
                    cols = np.any(s > 0, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        self.donors.append(
                            (patient_idx, slice_idx, rmin, rmax + 1, cmin, cmax + 1)
                        )

    def __len__(self) -> int:
        return len(self.indices)

    def _patch_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return image, mask
        cr, cc = int(rows.mean()), int(cols.mean())
        if self.jitter > 0:
            cr += random.randint(-self.jitter, self.jitter)
            cc += random.randint(-self.jitter, self.jitter)
        H, W = image.shape[:2]
        half = self.patch_size//2

        rmin = max(0, cr - half)
        rmax = min(H, cr + half)
        cmin = max(0, cc - half)
        cmax = min(W, cc + half)

        if rmax - rmin < self.patch_size:
            if rmin == 0:
                rmax = min(H, self.patch_size)
            else:
                rmin = max(0, rmax - self.patch_size)
        if cmax - cmin < self.patch_size:
            if cmin == 0:
                cmax = min(W, self.patch_size)
            else:
                cmin = max(0, cmax - self.patch_size)

        return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]

    def _apply_copy_paste(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.donors:
            return image, mask

        n_paste = random.randint(1, self.max_donors)
        H, W = image.shape[:2]

        for _ in range(n_paste):
            donor = self.donors[random.randint(0, len(self.donors) - 1)]
            d_pidx, d_sidx, d_rmin, d_rmax, d_cmin, d_cmax = donor

            donor_img = self.volumes[d_pidx][:, :, d_sidx]
            donor_msk = self.segmentations[d_pidx][:, :, d_sidx]

            patch_img = donor_img[d_rmin:d_rmax, d_cmin:d_cmax].copy().astype(np.float32)
            patch_msk = donor_msk[d_rmin:d_rmax, d_cmin:d_cmax].copy()

            patch_img = _normalize(patch_img)

            ph, pw = patch_img.shape[:2]
            if ph > H or pw > W:
                continue

            paste_r = random.randint(0, max(0, H - ph))
            paste_c = random.randint(0, max(0, W - pw))

            fg_mask = patch_msk > 0
            region_img = image[paste_r:paste_r + ph, paste_c:paste_c + pw]
            region_msk = mask[paste_r:paste_r + ph, paste_c:paste_c + pw]

            actual_h = min(ph, region_img.shape[0])
            actual_w = min(pw, region_img.shape[1])
            fg_sub = fg_mask[:actual_h, :actual_w]

            region_img[:actual_h, :actual_w][fg_sub] = patch_img[:actual_h, :actual_w][fg_sub]
            region_msk[:actual_h, :actual_w][fg_sub] = patch_msk[:actual_h, :actual_w][fg_sub]

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)

        if random.random() < self.copy_paste_prob:
            image, mask = self._apply_copy_paste(image, mask)

        image, mask = self._patch_crop(image, mask)

        return _to_tensors(image, mask, self.transform)



def get_transforms(train=True, img_size=384):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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
    return model


def load_fine_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 3)
    model = create_model(
        in_channels=1, num_classes=num_classes,
        encoder_name="efficientnet-b4", attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded fine model from {model_path}")
    return model



class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            factor = epoch / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.eta_min, base_lr * factor)



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
        loss, _ = criterion(outputs, masks, epoch=epoch, distance_map=distance_maps)
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
            loss, _ = criterion(outputs, masks, epoch=epoch, distance_map=distance_maps)

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
        print("ERROR: No valid patients found.")
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

    fg_train = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0) for s in train_segs
    )
    fg_val = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0) for s in val_segs
    )
    print(f"Foreground slices: train={fg_train}, val={fg_val}")

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    train_dataset = CopyPasteFinePatchDataset(
        train_volumes, train_segs,
        transform=train_transform,
        patch_size=cfg.PATCH_SIZE,
        jitter=cfg.PATCH_JITTER_TRAIN,
        oversample=cfg.OVERSAMPLE_FACTOR,
        copy_paste_prob=cfg.COPY_PASTE_PROB,
        max_donors=cfg.MAX_DONORS,
    )

    from shared.dataset_fine_patches import FinePatchDataset
    val_dataset = FinePatchDataset(
        val_volumes, val_segs,
        transform=val_transform,
        patch_size=cfg.PATCH_SIZE,
        jitter=cfg.PATCH_JITTER_VAL,
        oversample=1,
    )

    use_pin_memory = cfg.DEVICE != "mps"
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=use_pin_memory,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    device = torch.device(cfg.DEVICE)
    base_model = create_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME,
        attention_type=cfg.ATTENTION_TYPE,
    )
    model = DropoutModel(base_model, dropout_p=cfg.DROPOUT_P).to(device)

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
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=cfg.WARMUP_EPOCHS,
        total_epochs=cfg.NUM_EPOCHS, eta_min=1e-7,
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
    logger.info(f"Boundary weight    : {cfg.BOUNDARY_WEIGHT}")
    logger.info(f"Boundary ramp-up   : {cfg.EPOCH_FOR_BOUNDARY_RAMPUP} epochs")
    logger.info(f"Dropout2d p        : {cfg.DROPOUT_P}")
    logger.info(f"Warmup epochs      : {cfg.WARMUP_EPOCHS}")
    logger.info(f"Copy-paste prob    : {cfg.COPY_PASTE_PROB}")
    logger.info(f"Max donors         : {cfg.MAX_DONORS}")
    logger.info(f"Oversample factor  : {cfg.OVERSAMPLE_FACTOR}")
    logger.info(f"Patience           : {cfg.PATIENCE}")

    history: Dict[str, list] = {
        "train_loss": [], "train_dice": [], "train_recall": [],
        "val_loss": [], "val_dice": [], "val_recall": [],
    }

    best_dice = 0.0
    best_epoch = 0
    best_val_metrics: Dict = {}
    patience_counter = 0

    print("\n" + "=" * 70)
    print("TRAINING: COPY-PASTE AEAR SPECIALIST")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{cfg.NUM_EPOCHS}")
        logger.info(f"{'=' * 70}")

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning rate: {current_lr:.2e}")

        train_metrics = train_epoch_fine(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        val_metrics = validate_fine(
            model, val_loader, criterion, device,
            epoch=epoch, num_classes=cfg.NUM_CLASSES,
        )

        scheduler.step(epoch)

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

        logger.info(
            f"TRAIN  - loss: {train_metrics['loss']:.4f} | "
            f"dice: {train_metrics['dice']:.4f} | "
            f"recall: {train_metrics['recall']:.4f}"
        )
        logger.info(
            f"VAL    - loss: {val_metrics['loss']:.4f} | "
            f"dice: {val_metrics['dice']:.4f} | "
            f"recall: {val_metrics['recall']:.4f}"
        )
        class_dices = val_metrics.get("class_dices", {})
        class_recalls = val_metrics.get("class_recalls", {})
        logger.info(
            "  Per-class Dice  : "
            + ", ".join(f"{c}={d:.3f}" for c, d in sorted(class_dices.items()))
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
            patience_counter = 0
            best_model_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.base_model.state_dict(),
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
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{cfg.PATIENCE})")

        if patience_counter >= cfg.PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.base_model.state_dict(),
                "val_dice": val_metrics["dice"],
            }, ckpt_path)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best val dice: {best_dice:.4f} (epoch {best_epoch})")

    history_plot_path = output_dir / "training_history.png"
    plot_training_history(history, history_plot_path)

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION")
    print("=" * 70)

    coarse_dir = find_latest_model_dir("exp14_two_stage_coarse")
    if coarse_dir is None:
        logger.warning("No coarse model found. Skipping E2E evaluation.")
        results = {
            "experiment_name": cfg.EXPERIMENT_NAME,
            "description": cfg.DESCRIPTION,
            "output_dir": str(output_dir),
            "best_epoch": best_epoch,
            "best_val_dice": best_dice,
            "history": history,
            "e2e_evaluation": "SKIPPED",
            "timestamp": timestamp,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    coarse_model = load_coarse_model(coarse_dir, device)
    fine_model = load_fine_model(str(output_dir), device)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_fg_slices = 0
    detected_fg_slices = 0
    total_bg_slices = 0
    false_positive_bg_slices = 0

    for patient_idx, (vol, seg) in enumerate(
        tqdm(list(zip(val_volumes, val_segs)), desc="E2E evaluating")
    ):
        for slice_idx in range(vol.shape[2]):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = native_patch_predict_slice(
                image, coarse_model, fine_model, device,
                coarse_size=256, patch_size=cfg.PATCH_SIZE,
                fine_size=cfg.IMG_SIZE, coarse_threshold=0.3,
                use_tta=False, use_cc_filter=False,
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

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_targets_flat = np.concatenate([t.ravel() for t in all_targets])
    all_metrics = compute_all_metrics(all_preds_flat, all_targets_flat, num_classes=3)

    detection_rate = detected_fg_slices / max(1, total_fg_slices)
    fp_rate = false_positive_bg_slices / max(1, total_bg_slices)

    print(f"\nE2E Results:")
    print(f"  Dice      : {all_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {all_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {all_metrics['mean_fg_precision']:.4f}")
    print(f"  Detection : {100*detection_rate:.1f}%")

    print(f"\nPer-class breakdown:")
    for c in [1, 2]:
        name = ["BG", "AEAL", "AEAR"][c]
        print(f"  {name}: Dice={all_metrics['dice_per_class'][c]:.4f}  "
              f"Recall={all_metrics['recall_per_class'][c]:.4f}")

    logger.info(f"E2E Dice={all_metrics['mean_fg_dice']:.4f} "
                f"AEAL={all_metrics['dice_per_class'].get(1, 0):.4f} "
                f"AEAR={all_metrics['dice_per_class'].get(2, 0):.4f}")

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
        "num_epochs_run": len(history["train_loss"]),
        "num_classes": cfg.NUM_CLASSES,
        "patch_size": cfg.PATCH_SIZE,
        "tversky_alpha": cfg.TVERSKY_ALPHA,
        "tversky_beta": cfg.TVERSKY_BETA,
        "dropout_p": cfg.DROPOUT_P,
        "warmup_epochs": cfg.WARMUP_EPOCHS,
        "copy_paste_prob": cfg.COPY_PASTE_PROB,
        "max_donors": cfg.MAX_DONORS,
        "oversample_factor": cfg.OVERSAMPLE_FACTOR,
        "coarse_model_dir": coarse_dir,
        "fine_model_dir": str(output_dir),
        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
            "false_positive_rate": fp_rate,
        },
        "e2e_metrics_all_slices": _serialize(all_metrics),
        "history": history,
        "timestamp": timestamp,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  Best val Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"  E2E AEAR Dice: {all_metrics['dice_per_class'].get(2, 0):.4f}")
    print(f"  vs exp19 baseline AEAR Dice: 0.569")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
