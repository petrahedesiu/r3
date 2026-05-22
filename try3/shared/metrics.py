
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _nanmean_safe(values) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = float(np.nanmean(values))
    return 0.0 if np.isnan(result) else result


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_label_map(pred, num_classes):
    pred = _to_numpy(pred)
    if pred.ndim == 4 and pred.shape[1] == num_classes:
        return pred.argmax(axis=1)
    return pred


def compute_dice_score(pred, target, num_classes=3, smooth=1e-7):
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _to_numpy(target)
    per_class = {}
    for c in range(num_classes):
        pred_c = (pred_labels == c).astype(np.float64)
        tgt_c = (target_np == c).astype(np.float64)
        intersection = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        if denom < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((2.0 * intersection + smooth) / (denom + smooth))
    fg = [per_class[c] for c in range(1, num_classes)]
    return _nanmean_safe(fg), per_class


def compute_recall(pred, target, num_classes=3, smooth=1e-7):
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _to_numpy(target)
    per_class = {}
    for c in range(num_classes):
        pred_c = (pred_labels == c)
        tgt_c = (target_np == c)
        tp = float(np.logical_and(pred_c, tgt_c).sum())
        fn = float(np.logical_and(~pred_c, tgt_c).sum())
        if (tp + fn) < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((tp + smooth) / (tp + fn + smooth))
    fg = [per_class[c] for c in range(1, num_classes)]
    return _nanmean_safe(fg), per_class


def compute_precision(pred, target, num_classes=3, smooth=1e-7):
    pred_labels = _to_label_map(pred, num_classes)
    target_np = _to_numpy(target)
    per_class = {}
    for c in range(num_classes):
        pred_c = (pred_labels == c)
        tgt_c = (target_np == c)
        tp = float(np.logical_and(pred_c, tgt_c).sum())
        fp = float(np.logical_and(pred_c, ~tgt_c).sum())
        if (tp + fp) < smooth:
            per_class[c] = float('nan')
        else:
            per_class[c] = float((tp + smooth) / (tp + fp + smooth))
    fg = [per_class[c] for c in range(1, num_classes)]
    return _nanmean_safe(fg), per_class


def compute_f2_score(pred, target, num_classes=3, beta=2.0, smooth=1e-7):
    _, prec_cls = compute_precision(pred, target, num_classes, smooth)
    _, rec_cls = compute_recall(pred, target, num_classes, smooth)
    beta_sq = beta * beta
    per_class = {}
    for c in range(num_classes):
        p, r = prec_cls[c], rec_cls[c]
        if np.isnan(p) or np.isnan(r):
            per_class[c] = float('nan')
        else:
            denom = beta_sq * p + r
            per_class[c] = float((1.0 + beta_sq) * p * r / denom) if denom >= smooth else 0.0
    fg = [per_class[c] for c in range(1, num_classes)]
    return _nanmean_safe(fg), per_class


def compute_all_metrics(pred, target, num_classes=3):
    # gather every metric into one dict
    d_m, d_c = compute_dice_score(pred, target, num_classes)
    r_m, r_c = compute_recall(pred, target, num_classes)
    p_m, p_c = compute_precision(pred, target, num_classes)
    f_m, f_c = compute_f2_score(pred, target, num_classes)
    return {
        "mean_fg_dice": d_m, "dice_per_class": d_c,
        "mean_fg_recall": r_m, "recall_per_class": r_c,
        "mean_fg_precision": p_m, "precision_per_class": p_c,
        "mean_fg_f2": f_m, "f2_per_class": f_c,
    }
