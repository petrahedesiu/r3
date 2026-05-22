
import warnings
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask, size_hw):
    return cv2.resize(
        mask.astype(np.float32), (size_hw[1], size_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def _normalize(image):
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32)


def _to_tensor(image, device):
    t = torch.from_numpy(np.ascontiguousarray(image)).float()
    return t.unsqueeze(0).unsqueeze(0).to(device)


def _extract_bbox(binary_mask, padding=30, max_fraction=0.6):
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = binary_mask.shape
    rmin = max(0, rmin - padding)
    rmax = min(H, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(W, cmax + padding + 1)
    if (rmax - rmin) * (cmax - cmin) > max_fraction * H * W:
        return None
    return (rmin, rmax, cmin, cmax)


def coarse_predict(image_norm, coarse_model, device, coarse_size=256, threshold=0.3):
    H, W = image_norm.shape[:2]
    coarse_input = _resize(image_norm, coarse_size)
    coarse_tensor = _to_tensor(coarse_input, device)

    coarse_model.eval()
    with torch.no_grad():
        logits = coarse_model(coarse_tensor)
        probs = F.softmax(logits, dim=1)
        fg_prob = probs[0, 1].cpu().numpy()

    binary = (fg_prob > threshold).astype(np.uint8)
    return binary, fg_prob


def bbox_predict(image_norm, binary_coarse, fine_model, device,
                 coarse_size=256, fine_size=384, bbox_padding=30):
    H, W = image_norm.shape[:2]

    scale_r = H / coarse_size
    scale_c = W / coarse_size

    coarse_bbox = _extract_bbox(binary_coarse, padding=0, max_fraction=0.6)

    if coarse_bbox is None and binary_coarse.sum() == 0:
        return np.zeros((H, W), dtype=np.int64), {"detected": False, "bbox": None}

    if coarse_bbox is None:
        bbox_orig = (0, H, 0, W)
    else:
        rmin = max(0, int(coarse_bbox[0]*scale_r) - bbox_padding)
        rmax = min(H, int(coarse_bbox[1] * scale_r) + bbox_padding)
        cmin = max(0, int(coarse_bbox[2] * scale_c) - bbox_padding)
        cmax = min(W, int(coarse_bbox[3]*scale_c) + bbox_padding)
        bbox_orig = (rmin, rmax, cmin, cmax)

    rmin, rmax, cmin, cmax = bbox_orig
    crop = image_norm[rmin:rmax, cmin:cmax]
    crop_h, crop_w = crop.shape[:2]

    fine_input = _resize(crop, fine_size)
    fine_tensor = _to_tensor(fine_input, device)

    fine_model.eval()
    with torch.no_grad():
        fine_logits = fine_model(fine_tensor)
        fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))
    fine_pred_resized[fine_pred_resized != 1] = 0

    prediction = np.zeros((H, W), dtype=np.int64)
    prediction[rmin:rmax, cmin:cmax] = fine_pred_resized
    return prediction, {"detected": True, "bbox": bbox_orig}


def patch_predict(image_norm, binary_coarse, fine_model, device,
                  coarse_size=256, patch_size=128, fine_size=384):
    H, W = image_norm.shape[:2]

    if binary_coarse.sum() == 0:
        return np.zeros((H, W), dtype=np.int64), {"detected": False, "bbox": None}

    rows_c, cols_c = np.where(binary_coarse > 0)
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

    fine_input = _resize(crop, fine_size)
    fine_tensor = _to_tensor(fine_input, device)

    fine_model.eval()
    with torch.no_grad():
        fine_logits = fine_model(fine_tensor)
        fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))
    fine_pred_resized[fine_pred_resized != 2] = 0

    prediction = np.zeros((H, W), dtype=np.int64)
    prediction[rmin:rmax, cmin:cmax] = fine_pred_resized
    return prediction, {"detected": True, "bbox": (rmin, rmax, cmin, cmax)}


def ensemble_predict_slice(
    image: np.ndarray,
    coarse_model: nn.Module,
    aeal_model: nn.Module,
    aear_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    patch_size: int = 128,
    coarse_threshold: float = 0.3,
    bbox_padding: int = 30,
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    image_norm = _normalize(image.astype(np.float32))

    binary_coarse, fg_prob = coarse_predict(
        image_norm, coarse_model, device,
        coarse_size=coarse_size, threshold=coarse_threshold,
    )

    info = {
        "coarse_fg_fraction": float(binary_coarse.sum()) / binary_coarse.size,
        "detected": binary_coarse.sum() > 0,
    }

    if binary_coarse.sum() == 0:
        return np.zeros((H, W), dtype=np.int64), info

    aeal_pred, aeal_info = bbox_predict(
        image_norm, binary_coarse, aeal_model, device,
        coarse_size=coarse_size, fine_size=fine_size, bbox_padding=bbox_padding,
    )
    info["aeal_bbox"] = aeal_info.get("bbox")

    aear_pred, aear_info = patch_predict(
        image_norm, binary_coarse, aear_model, device,
        coarse_size=coarse_size, patch_size=patch_size, fine_size=fine_size,
    )
    info["aear_bbox"] = aear_info.get("bbox")

    # merge the two predictions into one label map
    merged = np.zeros((H, W), dtype=np.int64)
    merged[aeal_pred == 1] = 1
    merged[aear_pred == 2] = 2
    return merged, info
