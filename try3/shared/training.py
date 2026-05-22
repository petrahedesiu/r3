
import os
import sys
import json
import inspect
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from .metrics import compute_dice_score, compute_recall, compute_precision


def compute_class_weights(segmentations, num_classes=3):
    class_counts = Counter()
    for seg in segmentations:
        unique, counts = np.unique(seg, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[int(u)] += c
    total = sum(class_counts.values())
    weights = []
    for c in range(num_classes):
        if class_counts[c] > 0:
            freq = class_counts[c] / total
            weight = min(100.0, np.sqrt(1.0 / (freq * num_classes)))
        else:
            weight = 100.0
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch=0,
                scaler=None):
    model.train()
    use_amp = scaler is not None and getattr(device, "type", device) == "cuda"
    total_loss = 0.0
    total_dice = 0.0
    total_recall = 0.0
    total_precision = 0.0
    num_batches = 0

    _sig = inspect.signature(criterion.forward if hasattr(criterion, "forward") else criterion)
    _takes_epoch = "epoch" in _sig.parameters

    num_classes = None

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device).float()
        masks = masks.to(device).long()
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_amp):
            outputs = model(images)
            if num_classes is None:
                num_classes = outputs.shape[1]
            if _takes_epoch:
                loss_out = criterion(outputs, masks, epoch=epoch)
            else:
                loss_out = criterion(outputs, masks)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}",
                          "recall": f"{recall:.2f}", "prec": f"{precision:.2f}"})

    n = max(num_batches, 1)
    return {"loss": total_loss / n, "dice": total_dice / n,
            "recall": total_recall / n, "precision": total_precision / n}


def validate(model, dataloader, criterion, device, epoch=0, num_classes=3):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    _sig = inspect.signature(criterion.forward if hasattr(criterion, "forward") else criterion)
    _takes_epoch = "epoch" in _sig.parameters

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Val epoch {epoch}"):
            images = images.to(device).float()
            masks = masks.to(device).long()

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(getattr(device, "type", device) == "cuda")):
                outputs = model(images)
            outputs = outputs.float()

            if _takes_epoch:
                loss_out = criterion(outputs, masks, epoch=epoch)
            else:
                loss_out = criterion(outputs, masks)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            total_loss += loss.item()
            num_batches += 1

            predLabels = outputs.argmax(dim=1).cpu().numpy()
            target_np = masks.cpu().numpy()
            for c in range(num_classes):
                pred_c = (predLabels == c)
                tgt_c = (target_np == c)
                inter = float(np.logical_and(pred_c, tgt_c).sum())
                pred_sum = float(pred_c.sum())
                tgt_sum = float(tgt_c.sum())
                tp[c] += inter
                fp[c] += pred_sum - inter
                fn[c] += tgt_sum - inter

    n = max(num_batches, 1)
    smooth = 1e-7
    class_dices, class_recalls, class_precisions = {}, {}, {}
    for c in range(num_classes):
        d_den = 2 * tp[c] + fp[c] + fn[c]
        class_dices[c] = float((2 * tp[c] + smooth) / (d_den + smooth)) if d_den >= smooth else float('nan')
        r_den = tp[c] + fn[c]
        class_recalls[c] = float((tp[c] + smooth) / (r_den + smooth)) if r_den >= smooth else float('nan')
        p_den = tp[c] + fp[c]
        class_precisions[c] = float((tp[c] + smooth) / (p_den + smooth)) if p_den >= smooth else float('nan')

    fg_d = [class_dices[c] for c in range(1, num_classes)]
    fg_r = [class_recalls[c] for c in range(1, num_classes)]
    fg_p = [class_precisions[c] for c in range(1, num_classes)]

    def _safe(vals):
        v = float(np.nanmean(vals)) if vals else 0.0
        return 0.0 if np.isnan(v) else v

    return {
        "loss": total_loss / n,
        "dice": _safe(fg_d), "recall": _safe(fg_r), "precision": _safe(fg_p),
        "class_dices": class_dices, "class_recalls": class_recalls,
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


def plot_predictions(model, dataloader, device, save_path, num_samples=4, num_classes=3):
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
            axes[i, 0].imshow(images[0, 0].cpu().numpy(), cmap="gray"); axes[i, 0].set_title("Input"); axes[i, 0].axis("off")
            axes[i, 1].imshow(masks[0].numpy(), cmap="tab10", vmin=0, vmax=num_classes-1); axes[i, 1].set_title("GT"); axes[i, 1].axis("off")
            axes[i, 2].imshow(preds[0].cpu().numpy(), cmap="tab10", vmin=0, vmax=num_classes-1); axes[i, 2].set_title("Pred"); axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_training(config_dict):
    experiment_name = config_dict["experiment_name"]
    output_dir = Path(config_dict["output_dir"])
    model = config_dict["model"]
    train_loader = config_dict["train_loader"]
    val_loader = config_dict["val_loader"]
    criterion = config_dict["criterion"]
    optimizer = config_dict["optimizer"]
    scheduler = config_dict["scheduler"]
    num_epochs = config_dict["num_epochs"]
    device = config_dict["device"]
    num_classes = config_dict.get("num_classes", 3)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"training_{timestamp}.log"
    logger = logging.getLogger(f"try3.{experiment_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 70)
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Num epochs  : {num_epochs}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {num_params:,}")
    logger.info("=" * 70)

    history = {k: [] for k in
               ["train_loss", "train_dice", "train_recall",
                "val_loss", "val_dice", "val_recall"]}

    best_dice = 0.0
    best_epoch = 0
    best_val_metrics = {}

    _dev_type = getattr(device, "type", device)
    scaler = torch.amp.GradScaler("cuda", enabled=(_dev_type == "cuda"))
    if _dev_type == "cuda":
        logger.info("Mixed precision: ENABLED (fp16 autocast + GradScaler)")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*70}\nEPOCH {epoch}/{num_epochs}\n{'='*70}")

        train_m = train_epoch(model, train_loader, criterion, optimizer, device,
                              epoch=epoch, scaler=scaler)
        val_m = validate(model, val_loader, criterion, device, epoch=epoch, num_classes=num_classes)
        scheduler.step()

        logger.info(f"TRAIN - loss:{train_m['loss']:.4f} dice:{train_m['dice']:.4f} "
                     f"recall:{train_m['recall']:.4f} prec:{train_m['precision']:.4f}")
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

    logger.info(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")
    logger.info(f"Best val dice: {best_dice:.4f} (epoch {best_epoch})")

    plot_training_history(history, output_dir / "training_history.png")

    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    plot_predictions(model, val_loader, device, output_dir / "predictions.png", num_classes=num_classes)

    results = {
        "experiment_name": experiment_name,
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
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
