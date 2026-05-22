
import sys
import os
import gc
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    EXPERIMENT_NAME = "exp31_grand_ensemble"
    DESCRIPTION = "Grand multi-seed ensemble: coarse(exp30) + AEAL(exp28) + AEAR(exp29) vs exp23"

    NUM_CLASSES = 3
    IMG_SIZE = 384
    PATCH_SIZE = 128

    SLIDE_OFFSETS = [-32, 0, 32]

    MAX_MODELS_PER_STRUCTURE = 7

    EXP23_COARSE_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp14_two_stage_coarse", "20260224_143036",
    )
    EXP23_EXP13_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp13_full_pipeline", "20260224_131543",
    )
    EXP23_EXP19_DIR = os.path.join(
        ExperimentConfig.OUTPUT_BASE,
        "exp19_native_patches", "20260224_210846",
    )



def find_latest_result_dir(experiment_name: str) -> Optional[str]:
    exp_dir = os.path.join(Config.OUTPUT_BASE, experiment_name)
    if not os.path.isdir(exp_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        reverse=True,
    )
    for subdir in subdirs:
        ckpt_file = os.path.join(exp_dir, subdir, "all_checkpoints.json")
        if os.path.exists(ckpt_file):
            return os.path.join(exp_dir, subdir)
    return None


def load_fine_model_from_checkpoint(
    ckpt_path: str, device: torch.device, num_classes: int = 3,
) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nc = checkpoint.get("num_classes", num_classes)
    model = create_model(
        in_channels=1, num_classes=nc,
        encoder_name="efficientnet-b4", attention_type="scse",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_coarse_model_from_checkpoint(
    ckpt_path: str, device: torch.device, num_classes: int = 2,
) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    nc = checkpoint.get("num_classes", num_classes)
    model = create_coarse_model(in_channels=1, num_classes=nc)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model



def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                lbl = get_labeled_slice_indices(seg)
                if len(lbl) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations



def run_coarse_ensemble(
    image_norm: np.ndarray,
    coarse_models: List[nn.Module],
    device: torch.device,
    coarse_size: int = 256,
    coarse_threshold: float = 0.3,
) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    info = {"detected": False, "coarse_fg_fraction": 0.0}

    coarse_input = _resize_image(image_norm, coarse_size)
    coarse_tensor = _image_to_tensor(coarse_input, device)

    fg_probs = []
    for model in coarse_models:
        with torch.no_grad():
            logits = model(coarse_tensor)
            probs = F.softmax(logits, dim=1)
            fg_prob = probs[0, 1].cpu().numpy()
            fg_probs.append(fg_prob)

    avg_fg_prob = np.mean(fg_probs, axis=0)
    coarse_binary = (avg_fg_prob > coarse_threshold).astype(np.uint8)
    cb_sum = coarse_binary.sum()
    info["coarse_fg_fraction"] = float(cb_sum) / coarse_binary.size
    if cb_sum == 0:
        return None, avg_fg_prob, info
    info["detected"] = True
    return coarse_binary, avg_fg_prob, info


def run_aeal_ensemble(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    aeal_models: List[nn.Module],
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    bbox_padding: int = 50,
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    num_classes = 3

    scale_r = H / coarse_size
    scale_c = W / coarse_size

    coarse_bbox = _extract_bbox_from_binary_mask(
        coarse_binary, padding=0, max_fraction=0.6,
    )

    if coarse_bbox is None and coarse_binary.sum() > 0:
        bbox_orig = (0, H, 0, W)
    elif coarse_bbox is None:
        return np.zeros((H, W), dtype=np.int64)
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

    prob_accum = np.zeros((num_classes, fine_size, fine_size), dtype=np.float32)
    for model in aeal_models:
        with torch.no_grad():
            logits = model(fine_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            prob_accum += probs

    prob_avg = prob_accum / len(aeal_models)
    fine_pred = prob_avg.argmax(axis=0).astype(np.int64)

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))

    prediction = np.zeros((H, W), dtype=np.int64)
    aeal_mask = (fine_pred_resized == 1)
    prediction[rmin:rmax, cmin:cmax][aeal_mask] = 1

    return prediction


def run_aear_sliding_window_ensemble(
    image_norm: np.ndarray,
    coarse_binary: np.ndarray,
    aear_models: List[nn.Module],
    device: torch.device,
    coarse_size: int = 256,
    patch_size: int = 128,
    fine_size: int = 384,
    slide_offsets: List[int] = [-32, 0, 32],
) -> np.ndarray:
    H, W = image_norm.shape[:2]
    num_classes = 3
    half = patch_size // 2

    rows_c, cols_c = np.where(coarse_binary > 0)
    cr = int(rows_c.mean() * H / coarse_size)
    cc = int(cols_c.mean() * W / coarse_size)

    prob_accum = np.zeros((num_classes, H, W), dtype=np.float32)

    for dr in slide_offsets:
        for dc in slide_offsets:
            ctr_r = cr + dr
            ctr_c = cc + dc

            rmin = max(0, ctr_r - half)
            rmax = min(H, ctr_r + half)
            cmin = max(0, ctr_c - half)
            cmax = min(W, ctr_c + half)

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
            if crop_h == 0 or crop_w == 0:
                continue

            fine_input = cv2.resize(crop, (fine_size, fine_size),
                                    interpolation=cv2.INTER_LINEAR)
            fine_tensor = torch.from_numpy(
                np.ascontiguousarray(fine_input)
            ).float().unsqueeze(0).unsqueeze(0).to(device)

            for model in aear_models:
                with torch.no_grad():
                    logits = model(fine_tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                probs_resized = np.zeros((num_classes, crop_h, crop_w), dtype=np.float32)
                for c in range(num_classes):
                    probs_resized[c] = cv2.resize(
                        probs[c], (crop_w, crop_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                prob_accum[:, rmin:rmax, cmin:cmax] = np.maximum(
                    prob_accum[:, rmin:rmax, cmin:cmax],
                    probs_resized,
                )

    full_pred = prob_accum.argmax(axis=0).astype(np.int64)
    prediction = np.zeros((H, W), dtype=np.int64)
    prediction[full_pred == 2] = 2

    return prediction


def grand_ensemble_predict_slice(
    image: np.ndarray,
    coarse_models: List[nn.Module],
    aeal_models: List[nn.Module],
    aear_models: List[nn.Module],
    device: torch.device,
    coarse_size: int = 256,
    fine_size: int = 384,
    patch_size: int = 128,
    coarse_threshold: float = 0.3,
    bbox_padding: int = 50,
    slide_offsets: List[int] = [-32, 0, 32],
) -> Tuple[np.ndarray, dict]:
    H, W = image.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)
    info = {
        "detected": False,
        "fallback_full": False,
        "coarse_fg_fraction": 0.0,
    }

    image_norm = _normalize(image.astype(np.float32))

    coarse_binary, avg_fg_prob, coarse_info = run_coarse_ensemble(
        image_norm, coarse_models, device,
        coarse_size=coarse_size,
        coarse_threshold=coarse_threshold,
    )
    info["coarse_fg_fraction"] = coarse_info["coarse_fg_fraction"]

    if coarse_binary is None:
        return prediction, info

    info["detected"] = True

    aeal_pred = run_aeal_ensemble(
        image_norm, coarse_binary, aeal_models, device,
        coarse_size=coarse_size, fine_size=fine_size,
        bbox_padding=bbox_padding,
    )

    aear_pred = run_aear_sliding_window_ensemble(
        image_norm, coarse_binary, aear_models, device,
        coarse_size=coarse_size, patch_size=patch_size,
        fine_size=fine_size, slide_offsets=slide_offsets,
    )

    prediction[aeal_pred == 1] = 1
    prediction[aear_pred == 2] = 2

    return prediction, info



def greedy_select_models(
    checkpoint_infos: List[dict],
    load_fn,
    eval_fn,
    device: torch.device,
    max_models: int = 7,
    logger=None,
) -> List[str]:
    # rank candidate checkpoints by their validation dice
    sorted_ckpts = sorted(checkpoint_infos, key=lambda x: x["val_dice"], reverse=True)
    selected_paths = [sorted_ckpts[0]["path"]]
    selected_models = [load_fn(sorted_ckpts[0]["path"], device)]
    best_score = eval_fn(selected_models)

    msg = (f"Greedy selection: start with {sorted_ckpts[0]['path']} "
           f"(val_dice={sorted_ckpts[0]['val_dice']:.4f}), score={best_score:.4f}")
    print(msg)
    if logger:
        logger.info(msg)

    for ckpt in sorted_ckpts[1:]:
        if len(selected_paths) >= max_models:
            break

        candidate_model = load_fn(ckpt["path"], device)
        candidate_models = selected_models + [candidate_model]
        candidate_score = eval_fn(candidate_models)

        if candidate_score > best_score:
            selected_paths.append(ckpt["path"])
            selected_models.append(candidate_model)
            best_score = candidate_score
            msg = (f"  + Added {ckpt['path']} "
                   f"(val_dice={ckpt['val_dice']:.4f}), "
                   f"ensemble score={best_score:.4f}")
            print(msg)
            if logger:
                logger.info(msg)
        else:
            del candidate_model
            if device.type == "mps":
                torch.mps.empty_cache()

    msg = f"Selected {len(selected_paths)} models, final score={best_score:.4f}"
    print(msg)
    if logger:
        logger.info(msg)

    del selected_models
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return selected_paths



def run_exp23_baseline(
    val_volumes, val_segs, device, cfg,
) -> Dict:
    from shared.two_stage_inference import _normalize

    coarse_path = os.path.join(cfg.EXP23_COARSE_DIR, "best_model.pth")
    exp13_path = os.path.join(cfg.EXP23_EXP13_DIR, "best_model.pth")
    exp19_path = os.path.join(cfg.EXP23_EXP19_DIR, "best_model.pth")

    for name, path in [("coarse", coarse_path), ("exp13", exp13_path), ("exp19", exp19_path)]:
        if not os.path.exists(path):
            print(f"WARNING: exp23 {name} model not found at {path}")
            return None

    coarse_model = load_coarse_model_from_checkpoint(coarse_path, device)
    exp13_model = load_fine_model_from_checkpoint(exp13_path, device)
    exp19_model = load_fine_model_from_checkpoint(exp19_path, device)

    all_preds = []
    all_targets = []
    total_fg = 0
    detected_fg = 0

    for vol, seg in tqdm(list(zip(val_volumes, val_segs)), desc="exp23 baseline"):
        for slice_idx in range(vol.shape[2]):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            image_norm = _normalize(image.astype(np.float32))
            H, W = image_norm.shape[:2]

            coarse_binary, _, coarse_info = run_coarse_ensemble(
                image_norm, [coarse_model], device,
                coarse_size=256, coarse_threshold=0.3,
            )

            if coarse_binary is None:
                pred = np.zeros((H, W), dtype=np.int64)
            else:
                if has_fg:
                    detected_fg += 1

                aeal_pred = run_aeal_ensemble(
                    image_norm, coarse_binary, [exp13_model], device,
                    coarse_size=256, fine_size=384, bbox_padding=50,
                )

                aear_pred = _run_single_patch_aear(
                    image_norm, coarse_binary, exp19_model, device,
                    coarse_size=256, patch_size=128, fine_size=384,
                )

                pred = np.zeros((H, W), dtype=np.int64)
                pred[aeal_pred == 1] = 1
                pred[aear_pred == 2] = 2

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if has_fg:
                total_fg += 1

    del coarse_model, exp13_model, exp19_model
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_targets_flat = np.concatenate([t.ravel() for t in all_targets])
    metrics = compute_all_metrics(all_preds_flat, all_targets_flat, num_classes=3)

    detection_rate = detected_fg / max(1, total_fg)
    metrics["detection_rate"] = detection_rate

    return metrics


def _run_single_patch_aear(
    image_norm, coarse_binary, model, device,
    coarse_size=256, patch_size=128, fine_size=384,
):
    H, W = image_norm.shape[:2]
    prediction = np.zeros((H, W), dtype=np.int64)

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

    with torch.no_grad():
        logits = model(fine_tensor)
        fine_pred = logits.argmax(dim=1)[0].cpu().numpy()

    fine_pred_resized = _resize_mask(fine_pred, (crop_h, crop_w))
    aear_mask = (fine_pred_resized == 2)
    prediction[rmin:rmax, cmin:cmax][aear_mask] = 2

    return prediction



def main():
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

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

    exp28_dir = find_latest_result_dir("exp28_multiseed_aeal")
    exp29_dir = find_latest_result_dir("exp29_multiseed_aear")
    exp30_dir = find_latest_result_dir("exp30_multiseed_coarse")

    missing = []
    if exp28_dir is None:
        missing.append("exp28_multiseed_aeal")
    if exp29_dir is None:
        missing.append("exp29_multiseed_aear")
    if exp30_dir is None:
        missing.append("exp30_multiseed_coarse")

    if missing:
        print(f"ERROR: Missing experiment results: {missing}")
        print("Run exp28, exp29, exp30 first.")
        return

    print(f"exp28 dir: {exp28_dir}")
    print(f"exp29 dir: {exp29_dir}")
    print(f"exp30 dir: {exp30_dir}")

    with open(os.path.join(exp28_dir, "all_checkpoints.json")) as f:
        aeal_checkpoints = json.load(f)
    with open(os.path.join(exp29_dir, "all_checkpoints.json")) as f:
        aear_checkpoints = json.load(f)
    with open(os.path.join(exp30_dir, "all_checkpoints.json")) as f:
        coarse_checkpoints = json.load(f)

    aeal_ckpt_list = [ckpt for seed_ckpts in aeal_checkpoints.values() for ckpt in seed_ckpts]
    aear_ckpt_list = [ckpt for seed_ckpts in aear_checkpoints.values() for ckpt in seed_ckpts]
    coarse_ckpt_list = [ckpt for seed_ckpts in coarse_checkpoints.values() for ckpt in seed_ckpts]

    print(f"Available checkpoints: AEAL={len(aeal_ckpt_list)}, "
          f"AEAR={len(aear_ckpt_list)}, coarse={len(coarse_ckpt_list)}")
    logger.info(f"Available: AEAL={len(aeal_ckpt_list)}, "
                f"AEAR={len(aear_ckpt_list)}, coarse={len(coarse_ckpt_list)}")

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

    device = torch.device(cfg.DEVICE)

    exp30_results_path = os.path.join(exp30_dir, "results.json")
    with open(exp30_results_path) as f:
        exp30_results = json.load(f)
    coarse_threshold = exp30_results.get("best_threshold", 0.30)
    print(f"Using coarse threshold from exp30: {coarse_threshold:.2f}")

    print("\n" + "=" * 70)
    print("LOADING COARSE ENSEMBLE")
    print("=" * 70)

    coarse_models = []
    for seed in [42, 43, 44, 45, 46]:
        best_path = os.path.join(exp30_dir, f"seed_{seed}", "best_model.pth")
        if os.path.exists(best_path):
            model = load_coarse_model_from_checkpoint(best_path, device)
            coarse_models.append(model)
            print(f"  Loaded coarse seed {seed}")
    print(f"Coarse ensemble: {len(coarse_models)} models")
    logger.info(f"Coarse ensemble: {len(coarse_models)} models")

    print("\n" + "=" * 70)
    print("GREEDY SELECTION: AEAL MODELS")
    print("=" * 70)

    def eval_aeal_ensemble(models):
        all_preds = []
        all_targets = []
        for vol, seg in zip(val_volumes, val_segs):
            for sl_idx in range(vol.shape[2]):
                gt = seg[:, :, sl_idx].copy()
                if gt.max() == 0:
                    continue
                image = vol[:, :, sl_idx].copy()
                image_norm = _normalize(image.astype(np.float32))

                coarse_binary, _, _ = run_coarse_ensemble(
                    image_norm, coarse_models, device,
                    coarse_size=256, coarse_threshold=coarse_threshold,
                )
                if coarse_binary is None:
                    pred = np.zeros_like(gt, dtype=np.int64)
                else:
                    pred = run_aeal_ensemble(
                        image_norm, coarse_binary, models, device,
                        coarse_size=256, fine_size=384, bbox_padding=50,
                    )
                all_preds.append(pred.ravel())
                all_targets.append(gt.ravel())

        preds_flat = np.concatenate(all_preds)
        targets_flat = np.concatenate(all_targets)
        metrics = compute_all_metrics(preds_flat, targets_flat, num_classes=3)
        return metrics["dice_per_class"].get(1, 0.0)

    selected_aeal_paths = greedy_select_models(
        aeal_ckpt_list,
        load_fn=load_fine_model_from_checkpoint,
        eval_fn=eval_aeal_ensemble,
        device=device,
        max_models=cfg.MAX_MODELS_PER_STRUCTURE,
        logger=logger,
    )

    print("\n" + "=" * 70)
    print("GREEDY SELECTION: AEAR MODELS")
    print("=" * 70)

    def eval_aear_ensemble(models):
        all_preds = []
        all_targets = []
        for vol, seg in zip(val_volumes, val_segs):
            for sl_idx in range(vol.shape[2]):
                gt = seg[:, :, sl_idx].copy()
                if gt.max() == 0:
                    continue
                image = vol[:, :, sl_idx].copy()
                image_norm = _normalize(image.astype(np.float32))

                coarse_binary, _, _ = run_coarse_ensemble(
                    image_norm, coarse_models, device,
                    coarse_size=256, coarse_threshold=coarse_threshold,
                )
                if coarse_binary is None:
                    pred = np.zeros_like(gt, dtype=np.int64)
                else:
                    pred = run_aear_sliding_window_ensemble(
                        image_norm, coarse_binary, models, device,
                        coarse_size=256, patch_size=128, fine_size=384,
                        slide_offsets=cfg.SLIDE_OFFSETS,
                    )
                all_preds.append(pred.ravel())
                all_targets.append(gt.ravel())

        preds_flat = np.concatenate(all_preds)
        targets_flat = np.concatenate(all_targets)
        metrics = compute_all_metrics(preds_flat, targets_flat, num_classes=3)
        return metrics["dice_per_class"].get(2, 0.0)

    selected_aear_paths = greedy_select_models(
        aear_ckpt_list,
        load_fn=load_fine_model_from_checkpoint,
        eval_fn=eval_aear_ensemble,
        device=device,
        max_models=cfg.MAX_MODELS_PER_STRUCTURE,
        logger=logger,
    )

    print("\n" + "=" * 70)
    print("FINAL ENSEMBLE EVALUATION")
    print("=" * 70)

    aeal_models = [load_fine_model_from_checkpoint(p, device) for p in selected_aeal_paths]
    aear_models = [load_fine_model_from_checkpoint(p, device) for p in selected_aear_paths]

    print(f"Final ensemble: {len(coarse_models)} coarse, "
          f"{len(aeal_models)} AEAL, {len(aear_models)} AEAR")
    logger.info(f"Final: {len(coarse_models)} coarse, "
                f"{len(aeal_models)} AEAL, {len(aear_models)} AEAR")

    all_preds = []
    all_targets = []
    total_fg_slices = 0
    detected_fg_slices = 0
    total_bg_slices = 0
    false_positive_bg_slices = 0

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0
    max_vis = 20

    for patient_idx, (vol, seg) in enumerate(
        tqdm(list(zip(val_volumes, val_segs)), desc="Grand ensemble evaluation")
    ):
        for slice_idx in range(vol.shape[2]):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = grand_ensemble_predict_slice(
                image, coarse_models, aeal_models, aear_models, device,
                coarse_size=256, fine_size=cfg.IMG_SIZE,
                patch_size=cfg.PATCH_SIZE,
                coarse_threshold=coarse_threshold,
                bbox_padding=50,
                slide_offsets=cfg.SLIDE_OFFSETS,
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
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                image_norm = _normalize(image.astype(np.float32))
                axes[0].imshow(image_norm, cmap="gray")
                axes[0].set_title("CT Slice")
                axes[0].axis("off")
                axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=2)
                axes[2].set_title("Grand Ensemble")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(
                    str(vis_dir / f"patient{patient_idx}_slice{slice_idx}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
                vis_count += 1

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_targets_flat = np.concatenate([t.ravel() for t in all_targets])
    exp31_metrics = compute_all_metrics(all_preds_flat, all_targets_flat, num_classes=3)

    detection_rate = detected_fg_slices / max(1, total_fg_slices)
    fp_rate = false_positive_bg_slices / max(1, total_bg_slices)

    print("\n" + "=" * 70)
    print("RUNNING EXP23 BASELINE FOR COMPARISON")
    print("=" * 70)

    del aeal_models, aear_models, coarse_models
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    exp23_metrics = run_exp23_baseline(val_volumes, val_segs, device, cfg)

    print("\n" + "=" * 70)
    print("COMPARISON: exp23 vs exp31 (Grand Ensemble)")
    print("=" * 70)

    def _ser(m):
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

    header = f"{'Metric':<20s} {'exp23':>10s} {'exp31':>10s} {'Delta':>10s}"
    print(header)
    print("-" * len(header))

    rows = []
    if exp23_metrics:
        comparisons = [
            ("Mean FG Dice", exp23_metrics["mean_fg_dice"], exp31_metrics["mean_fg_dice"]),
            ("AEAL Dice", exp23_metrics["dice_per_class"].get(1, 0), exp31_metrics["dice_per_class"].get(1, 0)),
            ("AEAR Dice", exp23_metrics["dice_per_class"].get(2, 0), exp31_metrics["dice_per_class"].get(2, 0)),
            ("Mean FG Recall", exp23_metrics["mean_fg_recall"], exp31_metrics["mean_fg_recall"]),
            ("Mean FG Precision", exp23_metrics["mean_fg_precision"], exp31_metrics["mean_fg_precision"]),
        ]
        if "detection_rate" in exp23_metrics:
            comparisons.append(
                ("Coarse Det Rate", exp23_metrics["detection_rate"], detection_rate)
            )

        for name, v23, v31 in comparisons:
            delta = v31 - v23
            print(f"{name:<20s} {v23:>10.4f} {v31:>10.4f} {delta:>+10.4f}")
            rows.append({"metric": name, "exp23": v23, "exp31": v31, "delta": delta})
            logger.info(f"{name}: exp23={v23:.4f}, exp31={v31:.4f}, delta={delta:+.4f}")
    else:
        print("exp23 baseline not available for comparison")
        logger.info("exp23 baseline not available")

    print("-" * len(header))

    print(f"\nexp31 per-class breakdown:")
    for c in sorted(exp31_metrics['dice_per_class'].keys()):
        name = ["BG", "AEAL", "AEAR"][c] if c < 3 else f"Class{c}"
        print(
            f"  {name}: "
            f"Dice={exp31_metrics['dice_per_class'][c]:.4f}  "
            f"Recall={exp31_metrics['recall_per_class'][c]:.4f}  "
            f"Precision={exp31_metrics['precision_per_class'][c]:.4f}"
        )

    print(f"\nDetection: {detected_fg_slices}/{total_fg_slices} "
          f"({100*detection_rate:.1f}%), "
          f"FP: {false_positive_bg_slices}/{total_bg_slices} ({100*fp_rate:.1f}%)")

    results = {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "description": cfg.DESCRIPTION,
        "output_dir": str(output_dir),

        "exp28_dir": exp28_dir,
        "exp29_dir": exp29_dir,
        "exp30_dir": exp30_dir,

        "selected_aeal_paths": selected_aeal_paths,
        "selected_aear_paths": selected_aear_paths,
        "num_coarse_models": len([42, 43, 44, 45, 46]),
        "num_aeal_models": len(selected_aeal_paths),
        "num_aear_models": len(selected_aear_paths),
        "coarse_threshold": coarse_threshold,

        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
            "false_positive_rate": fp_rate,
        },

        "exp31_metrics": _ser(exp31_metrics),

        "exp23_metrics": _ser(exp23_metrics) if exp23_metrics else None,

        "comparison": rows if rows else None,

        "sliding_window": {
            "offsets": cfg.SLIDE_OFFSETS,
            "grid_size": "3x3",
        },

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
    print(f"  Ensemble composition:")
    print(f"    Coarse: 5 models (exp30)")
    print(f"    AEAL: {len(selected_aeal_paths)} models (exp28, greedy-selected)")
    print(f"    AEAR: {len(selected_aear_paths)} models (exp29, greedy-selected)")
    print(f"    AEAR sliding window: 3x3 grid, offsets={cfg.SLIDE_OFFSETS}")
    print(f"    Coarse threshold: {coarse_threshold:.2f}")
    print(f"")
    print(f"  Results:")
    print(f"    Mean FG Dice: {exp31_metrics['mean_fg_dice']:.4f}")
    print(f"    AEAL Dice: {exp31_metrics['dice_per_class'].get(1, 0):.4f}")
    print(f"    AEAR Dice: {exp31_metrics['dice_per_class'].get(2, 0):.4f}")
    print(f"    Detection rate: {100*detection_rate:.1f}%")
    if exp23_metrics:
        delta = exp31_metrics['mean_fg_dice'] - exp23_metrics['mean_fg_dice']
        print(f"    vs exp23: {delta:+.4f}")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
