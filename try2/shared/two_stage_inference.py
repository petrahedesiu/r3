
import warnings
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.postprocessing import (
    test_time_augmentation,
    connected_component_filter,
)


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.float32), (size[1], size[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def _normalize(image: np.ndarray) -> np.ndarray:
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32)


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(image)).float()
    return t.unsqueeze(0).unsqueeze(0).to(device)


def _extract_bbox_from_binary_mask(
    binary_mask: np.ndarray,
    padding: int = 30,
    max_fraction: float = 0.6,
) -> Optional[Tuple[int, int, int, int]]:
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

    bbox_area = (rmax - rmin) * (cmax - cmin)
    image_area = H * W
    if bbox_area > max_fraction * image_area:
        return None

    return (rmin, rmax, cmin, cmax)


def two_stage_predict_slice(
    image: np.ndarray,
    coarse_model: nn.Module,
    fine_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    coarse_threshold: float = 0.3,
    bbox_padding: int = 30,
    use_tta: bool = False,
    use_cc_filter: bool = False,
    cc_min_size: int = 3,
    cc_max_size: int = 1000,
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {
        "detected": False,
        "bbox": None,
        "fallback_full": False,
        "coarse_fg_fraction": 0.0,
    }

    coarse_model.eval()
    image_norm = _normalize(image.astype(np.float32))
    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    with torch.no_grad():
        coarse_logits = coarse_model(coarse_tensor)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_fg_prob = coarse_probs[0, 1].cpu().numpy()

    coarse_binary = (coarse_fg_prob > coarse_threshold).astype(np.uint8)

    info["coarse_fg_fraction"] = float(coarse_binary.sum()) / coarse_binary.size

    scaleR = H / coarse_size
    scaleC = W / coarse_size

    coarse_bbox = _extract_bbox_from_binary_mask(
        coarse_binary, padding=0, max_fraction=0.6,
    )

    if coarse_bbox is None and coarse_binary.sum() == 0:
        warnings.warn("Stage 1 detected no foreground -- returning all-zeros.")
        return prediction, info

    if coarse_bbox is None and coarse_binary.sum() > 0:
        info["fallback_full"] = True
        info["detected"] = True
        bbox_orig = (0, H, 0, W)
    else:
        info["detected"] = True
        rmin_orig = max(0, int(coarse_bbox[0] * scaleR) - bbox_padding)
        rmax_orig = min(H, int(coarse_bbox[1] * scaleR) + bbox_padding)
        cmin_orig = max(0, int(coarse_bbox[2] * scaleC) - bbox_padding)
        cmax_orig = min(W, int(coarse_bbox[3] * scaleC) + bbox_padding)
        bbox_orig = (rmin_orig, rmax_orig, cmin_orig, cmax_orig)

    info["bbox"] = bbox_orig

    fine_model.eval()

    rmin, rmax, cmin, cmax = bbox_orig
    crop = image_norm[rmin:rmax, cmin:cmax]
    crop_h, crop_w = crop.shape[:2]

    fine_input = _resize_image(crop, fine_size)
    fine_tensor = _image_to_tensor(fine_input, device)

    if use_tta:
        fine_pred, _ = test_time_augmentation(
            fine_model, fine_tensor, device, merge_mode="max"
        )
        fine_pred = fine_pred.numpy()
    else:
        with torch.no_grad():
            fine_logits = fine_model(fine_tensor)
            fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))

    if use_cc_filter:
        fine_pred_resized = connected_component_filter(
            fine_pred_resized, min_size=cc_min_size, max_size=cc_max_size
        )

    prediction[rmin:rmax, cmin:cmax] = fine_pred_resized

    return prediction, info


def native_patch_predict_slice(
    image: np.ndarray,
    coarse_model: nn.Module,
    fine_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    patch_size: int = 128,
    fine_size: int = 384,
    coarse_threshold: float = 0.3,
    use_tta: bool = False,
    use_cc_filter: bool = False,
    cc_min_size: int = 3,
    cc_max_size: int = 1000,
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {
        "detected": False,
        "bbox": None,
        "fallback_full": False,
        "coarse_fg_fraction": 0.0,
    }

    coarse_model.eval()
    image_norm = _normalize(image.astype(np.float32))
    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    with torch.no_grad():
        coarse_logits = coarse_model(coarse_tensor)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_fg_prob = coarse_probs[0, 1].cpu().numpy()

    coarse_binary = (coarse_fg_prob > coarse_threshold).astype(np.uint8)
    info["coarse_fg_fraction"] = float(coarse_binary.sum()) / coarse_binary.size

    if coarse_binary.sum() == 0:
        warnings.warn("Stage 1 detected no foreground -- returning all-zeros.")
        return prediction, info

    info["detected"] = True

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

    info["bbox"] = (rmin, rmax, cmin, cmax)

    fine_model.eval()
    crop = image_norm[rmin:rmax, cmin:cmax]
    crop_h, crop_w = crop.shape[:2]

    fine_input = _resize_image(crop, fine_size)
    fine_tensor = _image_to_tensor(fine_input, device)

    if use_tta:
        fine_pred, _ = test_time_augmentation(
            fine_model, fine_tensor, device, merge_mode="max"
        )
        fine_pred = fine_pred.numpy()
    else:
        with torch.no_grad():
            fine_logits = fine_model(fine_tensor)
            fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))

    if use_cc_filter:
        fine_pred_resized = connected_component_filter(
            fine_pred_resized, min_size=cc_min_size, max_size=cc_max_size
        )

    prediction[rmin:rmax, cmin:cmax] = fine_pred_resized
    return prediction, info


def two_stage_predict_volume_slice(
    volume: np.ndarray,
    slice_idx: int,
    coarse_model: nn.Module,
    fine_model: nn.Module,
    device: torch.device,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    image = volume[:, :, slice_idx].copy()
    return two_stage_predict_slice(
        image, coarse_model, fine_model, device, **kwargs
    )
