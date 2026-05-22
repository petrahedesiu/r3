
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _nanmean_safe(values) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = float(np.nanmean(values))
    return 0.0 if np.isnan(result) else result


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_label_map(pred, num_classes: int) -> np.ndarray:
    pred = _to_numpy(pred)
    if pred.ndim == 4 and pred.shape[1] == num_classes:
        return pred.argmax(axis=1)
    return pred


def _ensure_target(target) -> np.ndarray:
    return _to_numpy(target)


def compute_dice_score(
    pred,
    target,
    num_classes: int = 3,
    smooth: float = 1e-7,
) -> Tuple[float, Dict[int, float]]:
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _ensure_target(target)

    per_class: Dict[int, float] = {}

    for c in range(num_classes):
        pred_c = (pred_labels == c).astype(np.float64)
        tgt_c = (target_np == c).astype(np.float64)

        intersection = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()

        if denom < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((2.0 * intersection + smooth) / (denom + smooth))

    fg_scores = [per_class[c] for c in range(1, num_classes)]
    mean_fg_dice = _nanmean_safe(fg_scores) if fg_scores else 0.0
    return mean_fg_dice, per_class


def compute_recall(
    pred,
    target,
    num_classes: int = 3,
    smooth: float = 1e-7,
) -> Tuple[float, Dict[int, float]]:
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _ensure_target(target)

    per_class: Dict[int, float] = {}

    for c in range(num_classes):
        pred_c = (pred_labels == c)
        tgt_c = (target_np == c)

        tp = float(np.logical_and(pred_c, tgt_c).sum())
        fn = float(np.logical_and(~pred_c, tgt_c).sum())

        if (tp + fn) < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((tp + smooth) / (tp + fn + smooth))

    fg_scores = [per_class[c] for c in range(1, num_classes)]
    mean_fg_recall = _nanmean_safe(fg_scores) if fg_scores else 0.0
    return mean_fg_recall, per_class


def compute_precision(
    pred,
    target,
    num_classes: int = 3,
    smooth: float = 1e-7,
) -> Tuple[float, Dict[int, float]]:
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _ensure_target(target)

    per_class: Dict[int, float] = {}

    for c in range(num_classes):
        pred_c = (pred_labels == c)
        tgt_c = (target_np == c)

        tp = float(np.logical_and(pred_c, tgt_c).sum())
        fp = float(np.logical_and(pred_c, ~tgt_c).sum())

        if (tp + fp) < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((tp + smooth) / (tp + fp + smooth))

    fg_scores = [per_class[c] for c in range(1, num_classes)]
    mean_fg_precision = _nanmean_safe(fg_scores) if fg_scores else 0.0
    return mean_fg_precision, per_class


def compute_f2_score(
    pred,
    target,
    num_classes: int = 3,
    beta: float = 2.0,
    smooth: float = 1e-7,
) -> Tuple[float, Dict[int, float]]:
    _, prec_per_class = compute_precision(pred, target, num_classes, smooth)
    _, rec_per_class = compute_recall(pred, target, num_classes, smooth)

    beta_sq = beta * beta
    per_class: Dict[int, float] = {}

    for c in range(num_classes):
        p = prec_per_class[c]
        r = rec_per_class[c]
        if np.isnan(p) or np.isnan(r):
            per_class[c] = float('nan')
        else:
            denom = beta_sq * p + r
            if denom < smooth:
                per_class[c] = 0.0
            else:
                per_class[c] = float((1.0 + beta_sq) * p * r / denom)

    fg_scores = [per_class[c] for c in range(1, num_classes)]
    mean_fg_f2 = _nanmean_safe(fg_scores) if fg_scores else 0.0
    return mean_fg_f2, per_class


def compute_all_metrics(
    pred,
    target,
    num_classes: int = 3,
) -> Dict:
    dice_mean, dice_cls = compute_dice_score(pred, target, num_classes)
    rec_mean, rec_cls = compute_recall(pred, target, num_classes)
    prec_mean, prec_cls = compute_precision(pred, target, num_classes)
    f2_mean, f2_cls = compute_f2_score(pred, target, num_classes)

    return {
        "mean_fg_dice": dice_mean,
        "dice_per_class": dice_cls,
        "mean_fg_recall": rec_mean,
        "recall_per_class": rec_cls,
        "mean_fg_precision": prec_mean,
        "precision_per_class": prec_cls,
        "mean_fg_f2": f2_mean,
        "f2_per_class": f2_cls,
    }


def optimize_threshold(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 3,
    threshold_range: Optional[Tuple[float, float, float]] = None,
) -> float:
    if threshold_range is None:
        threshold_range = (0.05, 0.55, 0.05)

    start, stop, step = threshold_range
    thresholds = list(np.arange(start, stop, step))

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    best_f2 = -1.0
    best_threshold = 0.5

    # loop over thresholds
    for thr in thresholds:
        pred_labels = torch.zeros_like(all_targets)

        fg_probs = all_probs[:, 1:, :, :]

        max_fg_prob, max_fg_class = fg_probs.max(dim=1)

        fg_mask = max_fg_prob > thr
        pred_labels[fg_mask] = (max_fg_class[fg_mask] + 1).long()

        f2_mean, _ = compute_f2_score(
            pred_labels.numpy(),
            all_targets.numpy(),
            num_classes=num_classes,
        )

        if f2_mean > best_f2:
            best_f2 = f2_mean
            best_threshold = float(thr)

    return best_threshold
