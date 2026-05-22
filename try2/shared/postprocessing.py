
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure
from scipy.ndimage import label as scipy_label


def optimize_threshold(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    num_classes: int = 3,
) -> float:
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    thresholds = [round(t * 0.05, 2) for t in range(1, 12)]
    best_threshold = 0.5
    best_mean_f2 = -1.0

    print(f"\n{'Thresh':>8s}", end="")
    for c in range(1, num_classes):
        print(f"  {'F2_c' + str(c):>8s}", end="")
    print(f"  {'Mean F2':>8s}")
    print("-" * (8 + (num_classes - 1) * 10 + 10))

    for thresh in thresholds:
        tp = np.zeros(num_classes, dtype=np.float64)
        fp = np.zeros(num_classes, dtype=np.float64)
        fn = np.zeros(num_classes, dtype=np.float64)

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                probs = F.softmax(logits, dim=1)

                preds = _threshold_to_labels(probs, thresh)

                masksNp = masks.cpu().numpy()
                predsNp = preds.cpu().numpy()

                for c in range(num_classes):
                    pred_c = (predsNp == c)
                    true_c = (masksNp == c)
                    tp[c] += np.sum(pred_c & true_c)
                    fp[c] += np.sum(pred_c & ~true_c)
                    fn[c] += np.sum(~pred_c & true_c)

        f2_scores = []
        for c in range(1, num_classes):
            f2 = _f2_from_counts(tp[c], fp[c], fn[c])
            f2_scores.append(f2)

        mean_f2 = float(np.mean(f2_scores))

        print(f"{thresh:>8.2f}", end="")
        for f2 in f2_scores:
            print(f"  {f2:>8.4f}", end="")
        print(f"  {mean_f2:>8.4f}")

        if mean_f2 > best_mean_f2:
            best_mean_f2 = mean_f2
            best_threshold = thresh

    print(f"\n>>> Best threshold = {best_threshold:.2f}  "
          f"(mean foreground F2 = {best_mean_f2:.4f})\n")
    return best_threshold


def apply_threshold(
    model_output: torch.Tensor,
    threshold: float = 0.5,
    num_classes: int = 3,
) -> torch.Tensor:
    probs = F.softmax(model_output, dim=1)
    return _threshold_to_labels(probs, threshold)


def test_time_augmentation(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device | str,
    merge_mode: str = "max",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device
    image = image.to(device, non_blocking=True)

    def _identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def _hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])

    def _vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])

    def _rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=[-2, -1])

    def _rot90_inv(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=-1, dims=[-2, -1])

    augmentations = [
        (_identity, _identity),
        (_hflip,    _hflip),
        (_vflip,    _vflip),
        (_rot90,    _rot90_inv),
    ]

    all_probs = []

    with torch.no_grad():
        for fwd, inv in augmentations:
            aug_img = fwd(image)
            logits = model(aug_img)
            probs = F.softmax(logits, dim=1)
            probs = inv(probs)
            all_probs.append(probs)

    stacked = torch.cat(all_probs, dim=0)

    if merge_mode == "max":
        merged, _ = stacked.max(dim=0)
    elif merge_mode == "mean":
        merged = stacked.mean(dim=0)
    else:
        raise ValueError(f"Unknown merge_mode '{merge_mode}'. "
                         f"Choose 'max' or 'mean'.")

    pred = merged.argmax(dim=0)
    return pred.cpu(), merged.cpu()


def connected_component_filter(
    pred_mask: np.ndarray,
    min_size: int = 3,
    max_size: int = 1000,
) -> np.ndarray:
    pred_mask = np.asarray(pred_mask, dtype=np.int64)
    filtered = np.zeros_like(pred_mask)

    fg_classes = np.unique(pred_mask)
    fg_classes = fg_classes[fg_classes > 0]

    for cls in fg_classes:
        binary = (pred_mask == cls).astype(np.int32)
        labelled, nComponents = scipy_label(binary)

        for comp_id in range(1, nComponents + 1):
            comp_mask = (labelled == comp_id)
            comp_size = int(comp_mask.sum())
            if min_size <= comp_size <= max_size:
                filtered[comp_mask] = cls

    return filtered


def morphological_postprocess(
    pred_mask: np.ndarray,
    close_iter: int = 1,
    dilate_iter: int = 0,
) -> np.ndarray:
    pred_mask = np.asarray(pred_mask, dtype=np.int64)

    if close_iter == 0 and dilate_iter == 0:
        return pred_mask.copy()

    struct = generate_binary_structure(2, 1)
    result = np.zeros_like(pred_mask)

    fg_classes = np.unique(pred_mask)
    fg_classes = fg_classes[fg_classes > 0]

    # loop over fg classes
    for cls in fg_classes:
        binary = (pred_mask == cls)

        if close_iter > 0:
            binary = binary_closing(binary, structure=struct,
                                    iterations=close_iter)
        if dilate_iter > 0:
            binary = binary_dilation(binary, structure=struct,
                                     iterations=dilate_iter)

        result[binary] = cls

    return result


def full_postprocess_pipeline(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device | str,
    threshold: float = 0.3,
    use_tta: bool = True,
    use_cc: bool = True,
    min_cc_size: int = 3,
    max_cc_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_tta:
        _, probs = test_time_augmentation(
            model, image, device, merge_mode="max",
        )
    else:
        model.eval()
        device = torch.device(device) if isinstance(device, str) else device
        image = image.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(image)
            probs = F.softmax(logits, dim=1).squeeze(0)
            probs = probs.cpu()

    probs_batch = probs.unsqueeze(0)
    pred = _threshold_to_labels(probs_batch, threshold)
    pred = pred.squeeze(0)

    if use_cc:
        pred_np = pred.numpy()
        pred_np = connected_component_filter(
            pred_np, min_size=min_cc_size, max_size=max_cc_size,
        )
        pred = torch.from_numpy(pred_np)

    return pred, probs


def _threshold_to_labels(
    probs: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    max_probs, max_indices = probs.max(dim=1)

    fg_probs = probs[:, 1:, :, :]
    fg_max_probs, fg_max_indices = fg_probs.max(dim=1)
    fg_labels = fg_max_indices + 1

    fg_wins = (fg_max_probs > threshold) & (fg_max_probs >= probs[:, 0, :, :])

    preds = torch.zeros_like(max_indices)
    preds[fg_wins] = fg_labels[fg_wins]

    return preds


def _f2_from_counts(
    tp: float,
    fp: float,
    fn: float,
    beta: float = 2.0,
) -> float:
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * tp
    denominator = (1.0 + beta_sq) * tp + beta_sq * fn + fp
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
