
import os
import sys
import json
import inspect
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.metrics import compute_dice_score, compute_recall, compute_precision


def compute_class_weights(
    segmentations: List[np.ndarray],
    num_classes: int = 3,
) -> torch.Tensor:
    class_counts: Counter = Counter()

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


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    totalRecall = 0.0
    total_precision = 0.0
    num_batches = 0

    _criterion_sig = inspect.signature(criterion.forward if hasattr(criterion, "forward") else criterion)
    _criterion_takes_epoch = "epoch" in _criterion_sig.parameters

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        if images.dtype != torch.float32:
            images = images.float()
        masks = masks.long()

        optimizer.zero_grad()

        outputs = model(images)

        if _criterion_takes_epoch:
            loss_out = criterion(outputs, masks, epoch=epoch)
        else:
            loss_out = criterion(outputs, masks)

        if isinstance(loss_out, tuple):
            loss = loss_out[0]
        else:
            loss = loss_out

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            dice, _ = compute_dice_score(outputs, masks)
            recall, _ = compute_recall(outputs, masks)
            precision, _ = compute_precision(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        totalRecall += recall
        total_precision += precision
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "recall": f"{recall:.2f}",
                "prec": f"{precision:.2f}",
            }
        )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "recall": totalRecall / n,
        "precision": total_precision / n,
    }


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    epoch: int = 0,
    num_classes: int = 3,
) -> Dict:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    _criterion_sig = inspect.signature(criterion.forward if hasattr(criterion, "forward") else criterion)
    _criterion_takes_epoch = "epoch" in _criterion_sig.parameters

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Val epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            outputs = model(images)

            if _criterion_takes_epoch:
                loss_out = criterion(outputs, masks, epoch=epoch)
            else:
                loss_out = criterion(outputs, masks)

            if isinstance(loss_out, tuple):
                loss = loss_out[0]
            else:
                loss = loss_out

            total_loss += loss.item()
            num_batches += 1

            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            target_np = masks.cpu().numpy()

            for c in range(num_classes):
                pred_c = (pred_labels == c)
                tgt_c = (target_np == c)
                tp[c] += float(np.logical_and(pred_c, tgt_c).sum())
                fp[c] += float(np.logical_and(pred_c, ~tgt_c).sum())
                fn[c] += float(np.logical_and(~pred_c, tgt_c).sum())

    n = max(num_batches, 1)
    smooth = 1e-7

    class_dices: Dict[int, float] = {}
    class_recalls: Dict[int, float] = {}
    class_precisions: Dict[int, float] = {}

    for c in range(num_classes):
        dice_denom = 2 * tp[c] + fp[c] + fn[c]
        if dice_denom < smooth:
            class_dices[c] = float('nan')
        else:
            class_dices[c] = float((2 * tp[c] + smooth) / (dice_denom + smooth))

        rec_denom = tp[c] + fn[c]
        if rec_denom < smooth:
            class_recalls[c] = float('nan')
        else:
            class_recalls[c] = float((tp[c] + smooth) / (rec_denom + smooth))

        prec_denom = tp[c] + fp[c]
        if prec_denom < smooth:
            class_precisions[c] = float('nan')
        else:
            class_precisions[c] = float((tp[c] + smooth) / (prec_denom + smooth))

    fg_dices = [class_dices[c] for c in range(1, num_classes)]
    fg_recalls = [class_recalls[c] for c in range(1, num_classes)]
    fg_precisions = [class_precisions[c] for c in range(1, num_classes)]

    mean_dice = float(np.nanmean(fg_dices)) if fg_dices else 0.0
    mean_recall = float(np.nanmean(fg_recalls)) if fg_recalls else 0.0
    mean_precision = float(np.nanmean(fg_precisions)) if fg_precisions else 0.0

    if np.isnan(mean_dice):
        mean_dice = 0.0
    if np.isnan(mean_recall):
        mean_recall = 0.0
    if np.isnan(mean_precision):
        mean_precision = 0.0

    return {
        "loss": total_loss / n,
        "dice": mean_dice,
        "recall": mean_recall,
        "precision": mean_precision,
        "class_dices": class_dices,
        "class_recalls": class_recalls,
    }


def plot_training_history(history: Dict, save_path) -> None:
    save_path = Path(save_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_title("Dice Score (foreground)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["train_recall"], "b-", label="Train")
    axes[2].plot(epochs, history["val_recall"], "r-", label="Val")
    axes[2].set_title("Recall (Sensitivity)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    save_path,
    num_samples: int = 4,
    num_classes: int = 3,
) -> None:
    save_path = Path(save_path)
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            img = images[0, 0].cpu().numpy()
            mask = masks[0].cpu().numpy()
            pred = preds[0].cpu().numpy()

            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask, cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="tab10", vmin=0, vmax=num_classes - 1)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_training(config_dict: Dict) -> Dict:
    experiment_name: str = config_dict["experiment_name"]
    output_dir = Path(config_dict["output_dir"])
    model: nn.Module = config_dict["model"]
    train_loader = config_dict["train_loader"]
    val_loader = config_dict["val_loader"]
    criterion = config_dict["criterion"]
    optimizer = config_dict["optimizer"]
    scheduler = config_dict["scheduler"]
    num_epochs: int = config_dict["num_epochs"]
    device: torch.device = config_dict["device"]

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"training_{timestamp}.log"

    logger = logging.getLogger(f"training.{experiment_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(formatter)

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 70)
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Device      : {device}")
    logger.info(f"Num epochs  : {num_epochs}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {num_params:,}")
    logger.info("=" * 70)

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

    # loop over epochs
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch}/{num_epochs}")
        logger.info(f"{'=' * 70}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch
        )

        val_metrics = validate(
            model, val_loader, criterion, device, epoch=epoch
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
            f"  Per-class Dice  : "
            + ", ".join(f"{c}={d:.3f}" for c, d in sorted(class_dices.items()))
        )
        logger.info(
            f"  Per-class Recall: "
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
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best val dice : {best_dice:.4f}  (epoch {best_epoch})")

    history_plot_path = output_dir / "training_history.png"
    plot_training_history(history, history_plot_path)
    logger.info(f"Saved training curves -> {history_plot_path}")

    best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    predictions_plot_path = output_dir / "predictions.png"
    plot_predictions(model, val_loader, device, predictions_plot_path)
    logger.info(f"Saved predictions     -> {predictions_plot_path}")

    results = {
        "experiment_name": experiment_name,
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
        "num_epochs": num_epochs,
        "history": history,
        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results JSON    -> {results_json_path}")

    return results
