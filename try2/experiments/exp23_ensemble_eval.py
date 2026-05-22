
import sys
import os
import json
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.models import create_model, create_coarse_model
from shared.metrics import compute_all_metrics
from shared.two_stage_inference import (
    _normalize, _resize_image, _resize_mask, _image_to_tensor,
    _extract_bbox_from_binary_mask,
)
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp23_ensemble_eval"
    DESCRIPTION = "Ensemble: exp13 (AEAL) + exp19 (AEAR) -- eval only"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    PATCH_SIZE = 128

    COARSE_MODEL_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp14_two_stage_coarse", "20260224_143036",
    )
    EXP13_MODEL_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp13_full_pipeline", "20260224_131543",
    )
    EXP19_MODEL_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp19_native_patches", "20260224_210846",
    )



def load_coarse_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 2)
    model = create_coarse_model(in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded coarse model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model


def load_exp13_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 3)
    model = create_model(
        in_channels=1,
        num_classes=num_classes,
        encoder_name="efficientnet-b4",
        attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded exp13 (AEAL) model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model


def load_exp19_model(model_dir: str, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint.get("num_classes", 3)
    model = create_model(
        in_channels=1,
        num_classes=num_classes,
        encoder_name="efficientnet-b4",
        attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded exp19 (AEAR) model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model



def load_data(data_dir):
    patients = discover_patients(data_dir)
    vols, segs = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                labeled = get_labeled_slice_indices(seg)
                if len(labeled) >= 2:
                    vols.append(vol)
                    segs.append(seg)
        except Exception:
            pass
    return vols, segs


def run_coarse_stage(
    image_norm: np.ndarray,
    coarse_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    coarse_threshold: float = 0.3,
) -> Tuple[Optional[np.ndarray], dict]:
    H, W = image_norm.shape[:2]
    info = {
        "detected": False,
        "coarse_fg_fraction": 0.0,
    }

    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    with torch.no_grad():
        coarse_logits = coarse_model(coarse_tensor)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_fg_prob = coarse_probs[0, 1].cpu().numpy()

    coarse_binary = (coarse_fg_prob > coarse_threshold).astype(np.uint8)
    cbsum = coarse_binary.sum()
    info["coarse_fg_fraction"] = float(cbsum) / coarse_binary.size

    if cbsum == 0:
        return None, info

    info["detected"] = True
    return coarse_binary, info


def run_exp13_bbox_path(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    exp13_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    bbox_padding: int = 50,
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)

    scale_r = H / coarse_size
    scale_c = W / coarse_size

    coarse_bbox = _extract_bbox_from_binary_mask(
        coarse_binary, padding=0, max_fraction=0.6,
    )

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

    with torch.no_grad():
        fine_logits = exp13_model(fine_tensor)
        fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))

    aeal_mask = (fine_pred_resized == 1)
    prediction[rmin:rmax, cmin:cmax][aeal_mask] = 1

    return prediction


def run_exp19_patch_path(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    exp19_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    patch_size: int = 128,
    fine_size: int = 384,
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    rows_c, cols_c = np.where(coarse_binary > 0)
    cr = int(rows_c.mean()*H/coarse_size)
    cc = int(cols_c.mean()*W/coarse_size)
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

    with torch.no_grad():
        fine_logits = exp19_model(fine_tensor)
        fine_pred = fine_logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))

    aear_mask = (fine_pred_resized == 2)
    prediction[rmin:rmax, cmin:cmax][aear_mask] = 2

    return prediction


def ensemble_predict_slice(
    image: np.ndarray,
    coarse_model: nn.Module,
    exp13_model: nn.Module,
    exp19_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    patch_size: int = 128,
    coarse_threshold: float = 0.3,
    bbox_padding: int = 50,
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {
        "detected": False,
        "exp13_bbox": None,
        "exp19_bbox": None,
        "fallback_full": False,
        "coarse_fg_fraction": 0.0,
    }

    image_norm = _normalize(image.astype(np.float32))

    coarse_binary, coarse_info = run_coarse_stage(
        image_norm, coarse_model, device,
        coarse_size=coarse_size,
        coarse_threshold=coarse_threshold,
    )
    info["coarse_fg_fraction"] = coarse_info["coarse_fg_fraction"]

    if coarse_binary is None:
        return prediction, info

    info["detected"] = True

    aeal_pred = run_exp13_bbox_path(
        image_norm, coarse_binary, exp13_model, device,
        coarse_size=coarse_size, fine_size=fine_size,
        bbox_padding=bbox_padding,
    )

    aear_pred = run_exp19_patch_path(
        image_norm, coarse_binary, exp19_model, device,
        coarse_size=coarse_size, patch_size=patch_size,
        fine_size=fine_size,
    )

    prediction[aeal_pred == 1] = 1
    prediction[aear_pred == 2] = 2

    return prediction, info


def plot_ensemble_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    prediction: np.ndarray,
    info: dict,
    save_path: str,
    title: str = "",
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Full CT Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="tab10", vmin=0, vmax=2)
    detected = info.get("detected", False)
    axes[2].set_title(f"Ensemble Prediction ({'detected' if detected else 'no detection'})")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

    for name, path in [
        ("exp14 coarse", cfg.COARSE_MODEL_DIR),
        ("exp13 fine (AEAL)", cfg.EXP13_MODEL_DIR),
        ("exp19 fine (AEAR)", cfg.EXP19_MODEL_DIR),
    ]:
        model_file = os.path.join(path, "best_model.pth")
        if not os.path.exists(model_file):
            print(f"ERROR: {name} model not found at {model_file}")
            print("Cannot run ensemble evaluation. Exiting.")
            return
        print(f"Found {name} model: {model_file}")

    device = torch.device(cfg.DEVICE)
    print(f"\nDevice: {device}")

    coarse_model = load_coarse_model(cfg.COARSE_MODEL_DIR, device)
    exp13_model = load_exp13_model(cfg.EXP13_MODEL_DIR, device)
    exp19_model = load_exp19_model(cfg.EXP19_MODEL_DIR, device)

    print("\nLoading data...")
    volumes, segmentations = load_data(cfg.DATA_DIR)
    print(f"Loaded {len(volumes)} patients")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Exiting.")
        return

    indices = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )

    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    print(f"Val: {len(val_volumes)} patients")

    fg_val_slices = sum(
        sum(1 for sl in range(s.shape[2]) if s[:, :, sl].max() > 0)
        for s in val_segs
    )
    print(f"Foreground val slices: {fg_val_slices}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"eval_{timestamp}.log"

    logger = logging.getLogger(f"eval.{cfg.EXPERIMENT_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(formatter)

    logger.info(cfg.summary())
    logger.info(f"Output dir       : {output_dir}")
    logger.info(f"Device           : {device}")
    logger.info(f"Coarse model     : {cfg.COARSE_MODEL_DIR}")
    logger.info(f"Exp13 model (AEAL): {cfg.EXP13_MODEL_DIR}")
    logger.info(f"Exp19 model (AEAR): {cfg.EXP19_MODEL_DIR}")

    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION: exp13 (AEAL) + exp19 (AEAR)")
    print("=" * 70)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    total_fg_slices = 0
    detected_fg_slices = 0
    total_bg_slices = 0
    false_positive_bg_slices = 0

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0
    max_vis = 20

    for patient_idx, (vol, seg) in enumerate(
        tqdm(list(zip(val_volumes, val_segs)), desc="Ensemble evaluating patients")
    ):
        n_slices = vol.shape[2]

        for slice_idx in range(n_slices):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = ensemble_predict_slice(
                image,
                coarse_model,
                exp13_model,
                exp19_model,
                device,
                coarse_size=256,
                fine_size=cfg.IMG_SIZE,
                patch_size=cfg.PATCH_SIZE,
                coarse_threshold=0.3,
                bbox_padding=50,
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
                image_norm = _normalize(image.astype(np.float32))
                plot_ensemble_visualization(
                    image_norm, gt_mask, pred, info,
                    save_path=str(vis_dir / f"patient{patient_idx}_slice{slice_idx}.png"),
                    title=f"Patient {patient_idx}, Slice {slice_idx}",
                )
                vis_count += 1

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    print("\n" + "=" * 70)
    print("ENSEMBLE E2E RESULTS")
    print("=" * 70)

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

    print(f"\nStage 1 Detection Statistics:")
    print(f"  Foreground slices: {total_fg_slices}")
    print(f"  Detected         : {detected_fg_slices} ({100*detection_rate:.1f}%)")
    print(f"  Missed           : {total_fg_slices - detected_fg_slices}")
    print(f"  BG slices        : {total_bg_slices}")
    print(f"  False positives  : {false_positive_bg_slices} ({100*false_positive_rate:.1f}%)")

    print(f"\nEnsemble E2E Metrics (ALL slices, full resolution):")
    print(f"  Dice      : {all_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {all_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {all_metrics['mean_fg_precision']:.4f}")
    print(f"  F2        : {all_metrics['mean_fg_f2']:.4f}")

    print(f"\nEnsemble E2E Metrics (FG slices only, full resolution):")
    print(f"  Dice      : {fg_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {fg_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {fg_metrics['mean_fg_precision']:.4f}")
    print(f"  F2        : {fg_metrics['mean_fg_f2']:.4f}")

    print(f"\nPer-class breakdown (all slices):")
    for c in sorted(all_metrics['dice_per_class'].keys()):
        name = ["BG", "AEAL", "AEAR"][c] if c < 3 else f"Class{c}"
        print(
            f"  {name}: "
            f"Dice={all_metrics['dice_per_class'][c]:.4f}  "
            f"Recall={all_metrics['recall_per_class'][c]:.4f}  "
            f"Precision={all_metrics['precision_per_class'][c]:.4f}  "
            f"F2={all_metrics['f2_per_class'][c]:.4f}"
        )

    logger.info(f"\nStage 1 detection rate: {100*detection_rate:.1f}% "
                f"({detected_fg_slices}/{total_fg_slices})")
    logger.info(f"Stage 1 FP rate: {100*false_positive_rate:.1f}% "
                f"({false_positive_bg_slices}/{total_bg_slices})")
    logger.info(f"Ensemble all-slices Dice={all_metrics['mean_fg_dice']:.4f} "
                f"Recall={all_metrics['mean_fg_recall']:.4f} "
                f"F2={all_metrics['mean_fg_f2']:.4f}")
    logger.info(f"Ensemble fg-only Dice={fg_metrics['mean_fg_dice']:.4f} "
                f"Recall={fg_metrics['mean_fg_recall']:.4f} "
                f"F2={fg_metrics['mean_fg_f2']:.4f}")
    logger.info(f"Saved {vis_count} visualizations to {vis_dir}")

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

        "training": "NONE -- eval only",

        "coarse_model_dir": cfg.COARSE_MODEL_DIR,
        "exp13_model_dir": cfg.EXP13_MODEL_DIR,
        "exp19_model_dir": cfg.EXP19_MODEL_DIR,

        "ensemble_strategy": "AEAL from exp13 (bbox crop), AEAR from exp19 (native patch), priority: AEAR > AEAL > BG",

        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
            "false_positive_rate": false_positive_rate,
        },

        "e2e_metrics_all_slices": _serialize(all_metrics),
        "e2e_metrics_fg_only": _serialize(fg_metrics),

        "total_slices": len(all_preds),
        "fg_slices": len(fg_mask_indices),

        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results JSON -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  {cfg.DESCRIPTION}")
    print(f"")
    print(f"  Ensemble Strategy:")
    print(f"    AEAL: exp13 (bbox crop, padding=50)")
    print(f"    AEAR: exp19 (native 128x128 patch)")
    print(f"    Priority: AEAR > AEAL > BG")
    print(f"")
    print(f"  End-to-End Results:")
    print(f"    Stage 1 detection rate: {100*detection_rate:.1f}%")
    print(f"    E2E Dice (all)  : {all_metrics['mean_fg_dice']:.4f}")
    print(f"    E2E Recall (all): {all_metrics['mean_fg_recall']:.4f}")
    print(f"    E2E F2 (all)    : {all_metrics['mean_fg_f2']:.4f}")
    print(f"    E2E Dice (FG)   : {fg_metrics['mean_fg_dice']:.4f}")
    print(f"    E2E Recall (FG) : {fg_metrics['mean_fg_recall']:.4f}")
    print(f"    E2E F2 (FG)     : {fg_metrics['mean_fg_f2']:.4f}")
    print(f"")
    print(f"  Per-class (all slices):")
    for c in [1, 2]:
        name = ["BG", "AEAL", "AEAR"][c]
        print(f"    {name}: Dice={all_metrics['dice_per_class'][c]:.4f}  "
              f"Recall={all_metrics['recall_per_class'][c]:.4f}")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
