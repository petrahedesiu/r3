
import sys, os
import gc
import json
import logging
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.models import create_model, create_coarse_model
from shared.metrics import compute_all_metrics
from shared.postprocessing import (
    test_time_augmentation,
    connected_component_filter,
    morphological_postprocess,
)
from shared.two_stage_inference import (
    _normalize, _resize_image, _resize_mask, _image_to_tensor,
    _extract_bbox_from_binary_mask,
)
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp27_super_ensemble"
    DESCRIPTION = "Super ensemble: best AEAL + best AEAR + full post-processing"

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
    return model


def load_fine_model(model_dir: str, device: torch.device, label: str = "") -> nn.Module:
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 3)
    model = create_model(
        in_channels=1, num_classes=num_classes,
        encoder_name="efficientnet-b4", attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {label} model from {model_path}")
    return model


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



def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                labeled = get_labeled_slice_indices(seg)
                n_labeled = len(labeled)
                if n_labeled >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations



def run_coarse_stage(
    image_norm: np.ndarray,
    coarse_model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    coarse_threshold: float = 0.2,
) -> Tuple[Optional[np.ndarray], dict]:
    info = {"detected": False, "coarse_fg_fraction": 0.0}
    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    with torch.no_grad():
        coarse_logits = coarse_model(coarse_tensor)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_fg_prob = coarse_probs[0, 1].cpu().numpy()

    coarse_binary = (coarse_fg_prob > coarse_threshold).astype(np.uint8)
    info["coarse_fg_fraction"] = float(coarse_binary.sum()) / coarse_binary.size

    if coarse_binary.sum() == 0:
        return None, info

    info["detected"] = True
    return coarse_binary, info


def run_bbox_probs(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    bbox_padding: int = 50,
    use_tta: bool = True,
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    full_probs = np.zeros((3, H, W), dtype=np.float32)
    full_probs[0] = 1.0

    scale_r = H / coarse_size
    scale_c = W / coarse_size

    coarse_bbox = _extract_bbox_from_binary_mask(
        coarse_binary, padding=0, max_fraction=0.6,
    )

    if coarse_bbox is None and coarse_binary.sum() > 0:
        bbox_orig = (0, H, 0, W)
    elif coarse_bbox is None:
        return full_probs
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

    if use_tta:
        _, tta_probs = test_time_augmentation(
            model, fine_tensor, device, merge_mode="mean"
        )
        probs_np = tta_probs.numpy()
    else:
        with torch.no_grad():
            fine_logits = model(fine_tensor)
            probs_np = F.softmax(fine_logits, dim=1)[0].cpu().numpy()

    for c in range(3):
        resized = cv2.resize(probs_np[c], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        full_probs[c, rmin:rmax, cmin:cmax] = resized

    return full_probs


def run_patch_probs(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    model: nn.Module,
    device: torch.device,
    coarse_size: int = 256,
    patch_size: int = 128,
    fine_size: int = 384,
    use_tta: bool = True,
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    full_probs = np.zeros((3, H, W), dtype=np.float32)
    full_probs[0] = 1.0

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

    crop = image_norm[rmin:rmax, cmin:cmax]
    crop_h, crop_w = crop.shape[:2]

    fine_input = _resize_image(crop, fine_size)
    fine_tensor = _image_to_tensor(fine_input, device)

    if use_tta:
        _, tta_probs = test_time_augmentation(
            model, fine_tensor, device, merge_mode="mean"
        )
        probs_np = tta_probs.numpy()
    else:
        with torch.no_grad():
            fine_logits = model(fine_tensor)
            probs_np = F.softmax(fine_logits, dim=1)[0].cpu().numpy()

    for c in range(3):
        resized = cv2.resize(probs_np[c], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        full_probs[c, rmin:rmax, cmin:cmax] = resized

    return full_probs



def super_ensemble_predict_slice(
    image: np.ndarray,
    coarse_model: nn.Module,
    aeal_model: nn.Module,
    aear_model: nn.Module,
    device: torch.device,
    aeal_path: str,
    aear_path: str,
    coarse_size: int = 256,
    fine_size: int = 384,
    patch_size: int = 128,
    coarse_threshold: float = 0.2,
    bbox_padding: int = 50,
    use_tta: bool = True,
    aeal_threshold: float = 0.5,
    aear_threshold: float = 0.5,
    cc_min_size: int = 5,
    use_cc: bool = True,
    use_morph: bool = True,
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {"detected": False, "coarse_fg_fraction": 0.0}
    image_norm = _normalize(image.astype(np.float32))
    coarse_binary, coarse_info = run_coarse_stage(
        image_norm, coarse_model, device,
        coarse_size=coarse_size, coarse_threshold=coarse_threshold,
    )
    info["coarse_fg_fraction"] = coarse_info["coarse_fg_fraction"]
    if coarse_binary is None:
        return prediction, info
    info["detected"] = True

    if aeal_path == "bbox":
        aeal_probs = run_bbox_probs(
            image_norm, coarse_binary, aeal_model, device,
            coarse_size=coarse_size, fine_size=fine_size,
            bbox_padding=bbox_padding, use_tta=use_tta,
        )
    else:
        aeal_probs = run_patch_probs(
            image_norm, coarse_binary, aeal_model, device,
            coarse_size=coarse_size, patch_size=patch_size,
            fine_size=fine_size, use_tta=use_tta,
        )

    if aear_path == "bbox":
        aear_probs = run_bbox_probs(
            image_norm, coarse_binary, aear_model, device,
            coarse_size=coarse_size, fine_size=fine_size,
            bbox_padding=bbox_padding, use_tta=use_tta,
        )
    else:
        aear_probs = run_patch_probs(
            image_norm, coarse_binary, aear_model, device,
            coarse_size=coarse_size, patch_size=patch_size,
            fine_size=fine_size, use_tta=use_tta,
        )

    merged_probs = np.maximum(aeal_probs, aear_probs)

    aeal_mask = merged_probs[1] > aeal_threshold
    aear_mask = merged_probs[2] > aear_threshold

    prediction[aeal_mask] = 1
    prediction[aear_mask] = 2

    if use_morph and prediction.max() > 0:
        prediction = morphological_postprocess(prediction, close_iter=1, dilate_iter=0)

    if use_cc and prediction.max() > 0:
        prediction = connected_component_filter(prediction, min_size=cc_min_size, max_size=1000)

    return prediction, info



def optimize_thresholds_for_combo(
    val_volumes, val_segs,
    coarse_model, aeal_model, aear_model,
    device, aeal_path, aear_path,
    coarse_threshold, use_tta, use_cc, cc_min_size, use_morph,
    fine_size=384, patch_size=128,
):
    thresholds = [round(t * 0.05, 2) for t in range(1, 12)]

    all_merged = []
    all_targets = []

    for vol, seg in tqdm(
        list(zip(val_volumes, val_segs)), desc="Collecting probs"
    ):
        for slice_idx in range(vol.shape[2]):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            image_norm = _normalize(image.astype(np.float32))

            coarse_binary, _ = run_coarse_stage(
                image_norm, coarse_model, device,
                coarse_size=256, coarse_threshold=coarse_threshold,
            )

            if coarse_binary is None:
                H, W = image.shape[:2]
                merged = np.zeros((3, H, W), dtype=np.float32)
                merged[0] = 1.0
            else:
                if aeal_path == "bbox":
                    aeal_p = run_bbox_probs(
                        image_norm, coarse_binary, aeal_model, device,
                        fine_size=fine_size, bbox_padding=50, use_tta=use_tta,
                    )
                else:
                    aeal_p = run_patch_probs(
                        image_norm, coarse_binary, aeal_model, device,
                        patch_size=patch_size, fine_size=fine_size, use_tta=use_tta,
                    )

                if aear_path == "bbox":
                    aear_p = run_bbox_probs(
                        image_norm, coarse_binary, aear_model, device,
                        fine_size=fine_size, bbox_padding=50, use_tta=use_tta,
                    )
                else:
                    aear_p = run_patch_probs(
                        image_norm, coarse_binary, aear_model, device,
                        patch_size=patch_size, fine_size=fine_size, use_tta=use_tta,
                    )

                merged = np.maximum(aeal_p, aear_p)

            all_merged.append(merged)
            all_targets.append(gt_mask)

        if Config.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    best_aeal_t = 0.5
    best_aear_t = 0.5
    best_mean_f2 = -1.0

    for aeal_t, aear_t in product(thresholds, thresholds):
        tp = np.zeros(3, dtype=np.float64)
        fp = np.zeros(3, dtype=np.float64)
        fn = np.zeros(3, dtype=np.float64)

        for merged, gt in zip(all_merged, all_targets):
            pred = np.zeros_like(gt, dtype=np.int64)
            pred[merged[1] > aeal_t] = 1
            pred[merged[2] > aear_t] = 2

            if use_morph and pred.max() > 0:
                pred = morphological_postprocess(pred, close_iter=1, dilate_iter=0)
            if use_cc and pred.max() > 0:
                pred = connected_component_filter(pred, min_size=cc_min_size, max_size=1000)

            for c in range(3):
                pred_c = (pred == c)
                true_c = (gt == c)
                tp[c] += np.sum(pred_c & true_c)
                fp[c] += np.sum(pred_c & ~true_c)
                fn[c] += np.sum(~pred_c & true_c)

        f2_scores = []
        for c in [1, 2]:
            beta_sq = 4.0
            num = (1 + beta_sq) * tp[c]
            den = (1 + beta_sq) * tp[c] + beta_sq * fn[c] + fp[c]
            f2_scores.append(float(num / den) if den > 0 else 0.0)

        mean_f2 = np.mean(f2_scores)
        if mean_f2 > best_mean_f2:
            best_mean_f2 = mean_f2
            best_aeal_t = aeal_t
            best_aear_t = aear_t

    return best_aeal_t, best_aear_t, best_mean_f2



def evaluate_combo(
    val_volumes, val_segs,
    coarse_model, aeal_model, aear_model,
    device, aeal_path, aear_path,
    coarse_threshold, use_tta, aeal_t, aear_t,
    use_cc, cc_min_size, use_morph,
    fine_size=384, patch_size=128,
):
    all_preds = []
    all_targets = []
    total_fg = 0
    detected_fg = 0

    for vol, seg in tqdm(
        list(zip(val_volumes, val_segs)), desc="Evaluating"
    ):
        for slice_idx in range(vol.shape[2]):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = super_ensemble_predict_slice(
                image, coarse_model, aeal_model, aear_model, device,
                aeal_path=aeal_path, aear_path=aear_path,
                coarse_size=256, fine_size=fine_size,
                patch_size=patch_size,
                coarse_threshold=coarse_threshold,
                bbox_padding=50, use_tta=use_tta,
                aeal_threshold=aeal_t, aear_threshold=aear_t,
                cc_min_size=cc_min_size, use_cc=use_cc,
                use_morph=use_morph,
            )

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if has_fg:
                total_fg += 1
                if info["detected"]:
                    detected_fg += 1

        if Config.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    preds_flat = np.concatenate([p.ravel() for p in all_preds])
    targets_flat = np.concatenate([t.ravel() for t in all_targets])
    metrics = compute_all_metrics(preds_flat, targets_flat, num_classes=3)

    return metrics, detected_fg / max(1, total_fg)



def main():
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

    coarse_file = os.path.join(cfg.COARSE_MODEL_DIR, "best_model.pth")
    if not os.path.exists(coarse_file):
        print(f"ERROR: Coarse model not found at {coarse_file}")
        return

    device = torch.device(cfg.DEVICE)
    coarse_model = load_coarse_model(cfg.COARSE_MODEL_DIR, device)

    aeal_candidates = {}
    for exp_name, label, path_type in [
        ("exp13_full_pipeline", "exp13", "bbox"),
        ("exp26_copypaste_aeal", "exp26", "bbox"),
    ]:
        model_dir = find_latest_model_dir(exp_name)
        if model_dir:
            model = load_fine_model(model_dir, device, label=f"AEAL-{label}")
            if model is not None:
                aeal_candidates[label] = {"model": model, "path": path_type, "dir": model_dir}

    aear_candidates = {}
    for exp_name, label, path_type in [
        ("exp19_native_patches", "exp19", "patch"),
        ("exp25_copypaste_aear", "exp25", "patch"),
    ]:
        model_dir = find_latest_model_dir(exp_name)
        if model_dir:
            model = load_fine_model(model_dir, device, label=f"AEAR-{label}")
            if model is not None:
                aear_candidates[label] = {"model": model, "path": path_type, "dir": model_dir}

    if not aeal_candidates or not aear_candidates:
        print("ERROR: Need at least one AEAL and one AEAR candidate.")
        print(f"  AEAL found: {list(aeal_candidates.keys())}")
        print(f"  AEAR found: {list(aear_candidates.keys())}")
        return

    print(f"\nAEAL candidates: {list(aeal_candidates.keys())}")
    print(f"AEAR candidates: {list(aear_candidates.keys())}")

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
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]
    print(f"Val: {len(val_volumes)} patients")

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

    combo_results = []

    print("\n" + "=" * 70)
    print("SUPER ENSEMBLE: Testing all model combinations")
    print("=" * 70)

    for aeal_label, aeal_info in aeal_candidates.items():
        for aear_label, aear_info in aear_candidates.items():
            combo_name = f"AEAL={aeal_label}_AEAR={aear_label}"
            print(f"\n--- {combo_name} ---")

            aeal_t, aear_t, opt_f2 = optimize_thresholds_for_combo(
                val_volumes, val_segs,
                coarse_model, aeal_info["model"], aear_info["model"],
                device,
                aeal_path=aeal_info["path"],
                aear_path=aear_info["path"],
                coarse_threshold=0.2,
                use_tta=True,
                use_cc=True,
                cc_min_size=5,
                use_morph=True,
                fine_size=cfg.IMG_SIZE,
                patch_size=cfg.PATCH_SIZE,
            )

            print(f"  Optimized thresholds: AEAL={aeal_t:.2f}, AEAR={aear_t:.2f}")

            metrics, det_rate = evaluate_combo(
                val_volumes, val_segs,
                coarse_model, aeal_info["model"], aear_info["model"],
                device,
                aeal_path=aeal_info["path"],
                aear_path=aear_info["path"],
                coarse_threshold=0.2,
                use_tta=True,
                aeal_t=aeal_t,
                aear_t=aear_t,
                use_cc=True,
                cc_min_size=5,
                use_morph=True,
                fine_size=cfg.IMG_SIZE,
                patch_size=cfg.PATCH_SIZE,
            )

            entry = {
                "combo_name": combo_name,
                "aeal_model": aeal_label,
                "aear_model": aear_label,
                "aeal_model_dir": aeal_info["dir"],
                "aear_model_dir": aear_info["dir"],
                "aeal_path": aeal_info["path"],
                "aear_path": aear_info["path"],
                "aeal_threshold": aeal_t,
                "aear_threshold": aear_t,
                "mean_fg_dice": metrics["mean_fg_dice"],
                "mean_fg_recall": metrics["mean_fg_recall"],
                "mean_fg_precision": metrics["mean_fg_precision"],
                "mean_fg_f2": metrics["mean_fg_f2"],
                "aeal_dice": metrics["dice_per_class"].get(1, 0.0),
                "aear_dice": metrics["dice_per_class"].get(2, 0.0),
                "aeal_recall": metrics["recall_per_class"].get(1, 0.0),
                "aear_recall": metrics["recall_per_class"].get(2, 0.0),
                "detection_rate": det_rate,
            }
            combo_results.append(entry)

            print(f"  Dice={metrics['mean_fg_dice']:.4f}  "
                  f"AEAL={metrics['dice_per_class'].get(1, 0):.4f}  "
                  f"AEAR={metrics['dice_per_class'].get(2, 0):.4f}  "
                  f"Det={100*det_rate:.1f}%")

            logger.info(f"{combo_name}: Dice={metrics['mean_fg_dice']:.4f} "
                        f"AEAL={metrics['dice_per_class'].get(1, 0):.4f} "
                        f"AEAR={metrics['dice_per_class'].get(2, 0):.4f}")

    # sort combos best-first
    combo_results.sort(key=lambda x: x["mean_fg_dice"], reverse=True)
    best = combo_results[0]

    print("\n" + "=" * 70)
    print("COMBINATION RESULTS (sorted by Mean FG Dice)")
    print("=" * 70)
    print(f"{'Combo':<35s} {'Dice':>6s} {'AEAL':>6s} {'AEAR':>6s} {'Recall':>6s}")
    print("-" * 65)
    for r in combo_results:
        print(f"{r['combo_name']:<35s} "
              f"{r['mean_fg_dice']:>6.4f} "
              f"{r['aeal_dice']:>6.4f} "
              f"{r['aear_dice']:>6.4f} "
              f"{r['mean_fg_recall']:>6.4f}")

    print(f"\nBEST COMBINATION: {best['combo_name']}")
    print(f"  Mean FG Dice: {best['mean_fg_dice']:.4f}")
    print(f"  AEAL Dice:    {best['aeal_dice']:.4f}")
    print(f"  AEAR Dice:    {best['aear_dice']:.4f}")

    print(f"\n{'='*70}")
    print("IMPROVEMENT OVER BASELINES")
    print(f"{'='*70}")
    print(f"{'Metric':<20s} {'exp23':>8s} {'exp27':>8s} {'Delta':>8s}")
    print("-" * 48)
    baselines = {"mean_fg_dice": 0.636, "aeal_dice": 0.703, "aear_dice": 0.569}
    for metric, baseline in baselines.items():
        val = best[metric]
        delta = val - baseline
        label = {"mean_fg_dice": "Mean FG Dice", "aeal_dice": "AEAL Dice", "aear_dice": "AEAR Dice"}[metric]
        print(f"{label:<20s} {baseline:>8.4f} {val:>8.4f} {delta:>+8.4f}")

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
        "best_combination": best,
        "all_combinations": combo_results,
        "inference_settings": {
            "coarse_threshold": 0.2,
            "use_tta": True,
            "tta_merge_mode": "mean",
            "use_cc": True,
            "cc_min_size": 5,
            "use_morph": True,
            "morph_close_iter": 1,
        },
        "baseline_comparison": {
            "exp23_mean_fg_dice": 0.636,
            "exp23_aeal_dice": 0.703,
            "exp23_aear_dice": 0.569,
            "improvement_dice": best["mean_fg_dice"] - 0.636,
            "improvement_aeal": best["aeal_dice"] - 0.703,
            "improvement_aear": best["aear_dice"] - 0.569,
        },
        "timestamp": timestamp,
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {cfg.EXPERIMENT_NAME}")
    print(f"  Best combo: {best['combo_name']}")
    print(f"  Mean FG Dice: {best['mean_fg_dice']:.4f} (target >= 0.80)")
    print(f"  AEAL: {best['aeal_dice']:.4f}  AEAR: {best['aear_dice']:.4f}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
