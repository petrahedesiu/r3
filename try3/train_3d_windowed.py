
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import config
from shared.data import load_all_patients, patient_split
from shared.dataset_3d import VolumetricPatchDataset
from shared.model_3d import create_3d_model
from shared.training import compute_class_weights


class FocalLoss3D(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce_loss).mean()


class TverskyLoss3D(nn.Module):

    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        predSoft = F.softmax(pred, dim=1)
        num_classes = predSoft.shape[1]
        targetOh = F.one_hot(target, num_classes).float()
        targetOh = targetOh.permute(0, 4, 1, 2, 3)

        losses = []
        for c in range(1, num_classes):
            pred_c = predSoft[:, c].reshape(-1)
            tgt_c = targetOh[:, c].reshape(-1)
            tp = (pred_c * tgt_c).sum()
            fp = (pred_c * (1.0 - tgt_c)).sum()
            fn = ((1.0 - pred_c) * tgt_c).sum()
            ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            losses.append(1.0 - ti)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


class CompoundLoss3D(nn.Module):

    def __init__(self, focal_weight=0.4, tversky_weight=0.6,
                 class_weights=None, tversky_alpha=0.2, tversky_beta=0.8):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal = FocalLoss3D(alpha=0.25, gamma=2.0, class_weights=class_weights)
        self.tversky = TverskyLoss3D(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        total = self.focal_weight * focal_loss + self.tversky_weight * tversky_loss
        return total, {"focal": focal_loss.item(), "tversky": tversky_loss.item()}


def train_epoch_3d(model, dataloader, criterion, optimizer, device, epoch=0):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_recall = 0.0
    num_batches = 0

    # deep supervision weights
    dsWeightMain = 1.0
    dsWeightAux = 0.3

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device).float()
        masks = masks.to(device).long()
        optimizer.zero_grad()

        outputs = model(images)

        if isinstance(outputs, tuple):
            main_out, ds2, ds3 = outputs
            loss_main, _ = criterion(main_out, masks)
            loss_ds2, _ = criterion(ds2, masks)
            loss_ds3, _ = criterion(ds3, masks)
            loss = dsWeightMain * loss_main + dsWeightAux * (loss_ds2 + loss_ds3)
            pred_logits = main_out
        else:
            loss_out = criterion(outputs, masks)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            pred_logits = outputs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_labels = pred_logits.argmax(dim=1)
            num_classes = pred_logits.shape[1]
            batch_dices = []
            batch_recalls = []
            for c in range(1, num_classes):
                pred_c = (pred_labels == c).float()
                tgt_c = (masks == c).float()
                tp = (pred_c * tgt_c).sum()
                fp = (pred_c * (1.0 - tgt_c)).sum()
                fn = ((1.0 - pred_c) * tgt_c).sum()
                denomD = 2 * tp + fp + fn
                denomR = tp + fn
                if denomD > 0:
                    batch_dices.append((2 * tp / denomD).item())
                if denomR > 0:
                    batch_recalls.append((tp / denomR).item())

        batch_dice = np.mean(batch_dices) if batch_dices else 0.0
        batch_recall = np.mean(batch_recalls) if batch_recalls else 0.0

        total_loss += loss.item()
        total_dice += batch_dice
        total_recall += batch_recall
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{batch_dice:.4f}",
            "recall": f"{batch_recall:.2f}",
        })

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "recall": total_recall / n,
        "precision": 0.0,
    }


def validate_3d(model, dataloader, criterion, device, epoch=0, num_classes=3):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Val epoch {epoch}"):
            images = images.to(device).float()
            masks = masks.to(device).long()

            outputs = model(images)
            loss_out = criterion(outputs, masks)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            total_loss += loss.item()
            num_batches += 1

            predLabels = outputs.argmax(dim=1).cpu().numpy()
            targetNp = masks.cpu().numpy()
            for c in range(num_classes):
                pred_c = predLabels == c
                tgt_c = targetNp == c
                tp[c] += float(np.logical_and(pred_c, tgt_c).sum())
                fp[c] += float(np.logical_and(pred_c, ~tgt_c).sum())
                fn[c] += float(np.logical_and(~pred_c, tgt_c).sum())

    n = max(num_batches, 1)
    smooth = 1e-7

    class_dices, class_recalls, class_precisions = {}, {}, {}
    for c in range(num_classes):
        d_den = 2 * tp[c] + fp[c] + fn[c]
        class_dices[c] = float((2 * tp[c] + smooth) / (d_den + smooth)) if d_den >= smooth else float("nan")
        r_den = tp[c] + fn[c]
        class_recalls[c] = float((tp[c] + smooth) / (r_den + smooth)) if r_den >= smooth else float("nan")
        p_den = tp[c] + fp[c]
        class_precisions[c] = float((tp[c] + smooth) / (p_den + smooth)) if p_den >= smooth else float("nan")

    fg_d = [class_dices[c] for c in range(1, num_classes)]
    fg_r = [class_recalls[c] for c in range(1, num_classes)]
    fg_p = [class_precisions[c] for c in range(1, num_classes)]

    def _safe(vals):
        v = float(np.nanmean(vals)) if vals else 0.0
        return 0.0 if np.isnan(v) else v

    return {
        "loss": total_loss / n,
        "dice": _safe(fg_d),
        "recall": _safe(fg_r),
        "precision": _safe(fg_p),
        "class_dices": class_dices,
        "class_recalls": class_recalls,
    }


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_title("Dice (FG)"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(epochs, history["train_recall"], "b-", label="Train")
    axes[2].plot(epochs, history["val_recall"], "r-", label="Val")
    axes[2].set_title("Recall"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_3d(model, dataloader, device, save_path, num_samples=4, num_classes=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
            images = images.to(device).float()
            preds = model(images).argmax(dim=1)

            midD = images.shape[2] // 2
            img_slice = images[0, 0, midD].cpu().numpy()
            gt_slice = masks[0, midD].numpy()
            pred_slice = preds[0, midD].cpu().numpy()

            axes[i, 0].imshow(img_slice, cmap="gray")
            axes[i, 0].set_title(f"Input (bone, z={midD})")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt_slice, cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[i, 1].set_title("GT")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_slice, cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[i, 2].set_title("Pred")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("3D WINDOWED SEGMENTATION EXPERIMENT")
    print("CT windowing (bone/soft/narrow) + 3D UNet")
    print("=" * 70)

    volumes, segmentations, infos = load_all_patients()
    n_patients = len(volumes)
    print(f"Loaded {n_patients} patients")

    train_idx, val_idx = patient_split(n_patients)
    print(f"Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")

    train_vols = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_vols = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    class_weights = compute_class_weights(train_segs, num_classes=3)
    print(f"Class weights: {class_weights}")

    patch_size = (96, 96, 32)
    train_ds = VolumetricPatchDataset(
        train_vols, train_segs,
        patch_size=patch_size,
        jitter_hw=15, jitter_d=5,
        oversample=8, augment=True,
    )
    val_ds = VolumetricPatchDataset(
        val_vols, val_segs,
        patch_size=patch_size,
        jitter_hw=0, jitter_d=0,
        oversample=1, augment=False,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device(config.DEVICE)
    model = create_3d_model(
        in_channels=3, num_classes=3,
        base_filters=16, deep_supervision=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = CompoundLoss3D(
        focal_weight=0.4,
        tversky_weight=0.6,
        class_weights=class_weights.to(device),
        tversky_alpha=0.2,
        tversky_beta=0.8,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7,
    )

    num_epochs = 50
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_BASE) / "3d_windowed" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"training_{timestamp}.log"
    logger = logging.getLogger(f"try3.3d_windowed.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info("=" * 70)
    logger.info("EXPERIMENT: 3D Windowed Segmentation")
    logger.info("=" * 70)
    logger.info(f"Output dir   : {output_dir}")
    logger.info(f"Device       : {device}")
    logger.info(f"Num epochs   : {num_epochs}")
    logger.info(f"Patch size   : {patch_size}")
    logger.info(f"Batch size   : 2")
    logger.info(f"Model params : {num_params:,}")
    logger.info(f"Train patients: {len(train_idx)}, Val patients: {len(val_idx)}")
    logger.info(f"Train samples : {len(train_ds)}, Val samples: {len(val_ds)}")
    logger.info(f"Class weights : {class_weights.tolist()}")
    logger.info("Windows: Bone(700/4000) + Soft(50/400) + Narrow(40/80)")
    logger.info("=" * 70)

    history = {k: [] for k in
               ["train_loss", "train_dice", "train_recall",
                "val_loss", "val_dice", "val_recall"]}

    best_dice = 0.0
    best_epoch = 0
    best_val_metrics = {}

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'=' * 70}\nEPOCH {epoch}/{num_epochs}\n{'=' * 70}")

        train_m = train_epoch_3d(model, train_loader, criterion, optimizer, device, epoch=epoch)
        val_m = validate_3d(model, val_loader, criterion, device, epoch=epoch, num_classes=3)
        scheduler.step()

        logger.info(f"TRAIN - loss:{train_m['loss']:.4f} dice:{train_m['dice']:.4f} "
                     f"recall:{train_m['recall']:.4f}")
        logger.info(f"VAL   - loss:{val_m['loss']:.4f} dice:{val_m['dice']:.4f} "
                     f"recall:{val_m['recall']:.4f} prec:{val_m['precision']:.4f}")
        cd = val_m.get("class_dices", {})
        cr = val_m.get("class_recalls", {})
        logger.info("  Dice  : " + ", ".join(f"{c}={d:.3f}" for c, d in sorted(cd.items())))
        logger.info("  Recall: " + ", ".join(f"{c}={r:.3f}" for c, r in sorted(cr.items())))

        history["train_loss"].append(train_m["loss"])
        history["train_dice"].append(train_m["dice"])
        history["train_recall"].append(train_m["recall"])
        history["val_loss"].append(val_m["loss"])
        history["val_dice"].append(val_m["dice"])
        history["val_recall"].append(val_m["recall"])

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            best_epoch = epoch
            best_val_metrics = val_m.copy()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_m["dice"],
                "val_recall": val_m["recall"],
                "val_precision": val_m["precision"],
                "class_dices": cd,
                "class_recalls": cr,
            }, output_dir / "best_model.pth")
            logger.info(f">>> NEW BEST MODEL  dice={val_m['dice']:.4f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_m["dice"],
            }, output_dir / f"checkpoint_epoch_{epoch}.pth")

    logger.info(f"\n{'=' * 70}\nTRAINING COMPLETE\n{'=' * 70}")
    logger.info(f"Best val dice: {best_dice:.4f} (epoch {best_epoch})")

    plot_training_history(history, output_dir / "training_history.png")

    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    plot_predictions_3d(model, val_loader, device, output_dir / "predictions.png", num_classes=3)

    results = {
        "experiment_name": "3d_windowed",
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_val_recall": best_val_metrics.get("recall", 0.0),
        "best_val_precision": best_val_metrics.get("precision", 0.0),
        "best_class_dices": {str(k): v for k, v in best_val_metrics.get("class_dices", {}).items()},
        "best_class_recalls": {str(k): v for k, v in best_val_metrics.get("class_recalls", {}).items()},
        "num_epochs": num_epochs,
        "history": history,
        "timestamp": timestamp,
        "config": {
            "patch_size": list(patch_size),
            "batch_size": 2,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingWarmRestarts(T_0=15, T_mult=2)",
            "loss": "0.4*Focal + 0.6*Tversky",
            "deep_supervision": True,
            "windows": "bone(700/4000) + soft(50/400) + narrow(40/80)",
            "base_filters": 16,
            "oversample_train": 8,
            "jitter_hw": 15,
            "jitter_d": 5,
            "num_params": num_params,
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n3D Windowed training complete. Best dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
