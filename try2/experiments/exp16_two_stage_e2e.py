
import sys
import os
import logging
import json
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.models import create_model, create_coarse_model
from shared.metrics import compute_all_metrics, compute_dice_score, compute_recall, compute_precision
from shared.two_stage_inference import two_stage_predict_slice
from shared.dataset import _normalize
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices



class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp16_two_stage_e2e"
    DESCRIPTION = "Two-stage pipeline end-to-end evaluation on full-resolution images"

    COARSE_MODEL_DIR: str = ""
    FINE_MODEL_DIR: str = ""

    COARSE_SIZE = 256
    FINE_SIZE = 384
    COARSE_THRESHOLD = 0.3
    BBOX_PADDING = 30

    USE_TTA = True
    USE_CC_FILTER = True
    CC_MIN_SIZE = 3
    CC_MAX_SIZE = 1000



def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                labeled = get_labeled_slice_indices(seg)
                if len(labeled) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations


def find_latest_model_dir(results_base: str, experiment_name: str) -> Optional[str]:
    exp_dir = os.path.join(results_base, experiment_name)
    if not os.path.isdir(exp_dir):
        return None
    subDirs = sorted(
        [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))],
        reverse=True,
    )
    for sd in subDirs:
        mp = os.path.join(exp_dir, sd, "best_model.pth")
        if os.path.exists(mp):
            return os.path.join(exp_dir, sd)
    return None


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


def load_fine_model(model_dir: str, device: torch.device) -> nn.Module:
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

    print(f"Loaded fine model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val Dice: {checkpoint.get('val_dice', '?'):.4f}")
    return model


def plot_two_stage_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    prediction: np.ndarray,
    info: dict,
    save_path: str,
    title: str = "",
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Full CT Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    if info.get("bbox") is not None:
        rmin, rmax, cmin, cmax = info["bbox"]
        rect = patches.Rectangle(
            (cmin, rmin), cmax - cmin, rmax - rmin,
            linewidth=2, edgecolor='lime', facecolor='none',
        )
        axes[2].add_patch(rect)
        status = "detected"
    elif info.get("detected"):
        status = "fallback (full)"
    else:
        status = "nothing detected"
    axes[2].set_title(f"Stage 1: {status}")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="tab10", vmin=0, vmax=2)
    axes[3].set_title("Final Prediction")
    axes[3].axis("off")

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

    device = torch.device(cfg.DEVICE)

    results_base = cfg.OUTPUT_BASE

    coarse_dir = cfg.COARSE_MODEL_DIR
    if not coarse_dir:
        coarse_dir = find_latest_model_dir(results_base, "exp14_two_stage_coarse")
    if coarse_dir is None:
        print("ERROR: No trained coarse model found. Run exp14 first.")
        print(f"  Searched in: {results_base}/exp14_two_stage_coarse/")
        return
    print(f"\nCoarse model dir: {coarse_dir}")

    fine_dir = cfg.FINE_MODEL_DIR
    if not fine_dir:
        fine_dir = find_latest_model_dir(results_base, "exp15_two_stage_fine")
    if fine_dir is None:
        print("ERROR: No trained fine model found. Run exp15 first.")
        print(f"  Searched in: {results_base}/exp15_two_stage_fine/")
        return
    print(f"Fine model dir: {fine_dir}")

    coarse_model = load_coarse_model(coarse_dir, device)
    fine_model = load_fine_model(fine_dir, device)

    print("\nLoading data...")
    volumes, segmentations = load_data(cfg.DATA_DIR)
    print(f"Loaded {len(volumes)} patients")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Exiting.")
        return

    idxs = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        idxs, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]
    print(f"Validation patients: {len(val_volumes)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"evaluation_{timestamp}.log"

    logger = logging.getLogger(f"eval.{cfg.EXPERIMENT_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(formatter)

    logger.info(cfg.summary())
    logger.info(f"Coarse model: {coarse_dir}")
    logger.info(f"Fine model  : {fine_dir}")
    logger.info(f"Coarse threshold: {cfg.COARSE_THRESHOLD}")
    logger.info(f"Bbox padding    : {cfg.BBOX_PADDING}")
    logger.info(f"Use TTA         : {cfg.USE_TTA}")
    logger.info(f"Use CC filter   : {cfg.USE_CC_FILTER}")

    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION ON FULL-RESOLUTION IMAGES")
    print("=" * 70)

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    total_fg_slices = 0
    detected_fg_slices = 0
    total_bg_slices = 0
    false_positive_bg_slices = 0
    fallback_count = 0

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0
    max_vis = 20

    for patient_idx, (vol, seg) in enumerate(
        tqdm(list(zip(val_volumes, val_segs)), desc="Evaluating patients")
    ):
        n_slices = vol.shape[2]

        for slice_idx in range(n_slices):
            image = vol[:, :, slice_idx].copy()
            gt_mask = seg[:, :, slice_idx].copy()
            has_fg = gt_mask.max() > 0

            pred, info = two_stage_predict_slice(
                image,
                coarse_model,
                fine_model,
                device,
                coarse_size=cfg.COARSE_SIZE,
                fine_size=cfg.FINE_SIZE,
                coarse_threshold=cfg.COARSE_THRESHOLD,
                bbox_padding=cfg.BBOX_PADDING,
                use_tta=cfg.USE_TTA,
                use_cc_filter=cfg.USE_CC_FILTER,
                cc_min_size=cfg.CC_MIN_SIZE,
                cc_max_size=cfg.CC_MAX_SIZE,
            )

            all_preds.append(pred)
            all_targets.append(gt_mask)

            if has_fg:
                total_fg_slices += 1
                if info["detected"]:
                    detected_fg_slices += 1
                if info["fallback_full"]:
                    fallback_count += 1
            else:
                total_bg_slices += 1
                if info["detected"]:
                    false_positive_bg_slices += 1

            if has_fg and vis_count < max_vis:
                image_norm = _normalize(image.astype(np.float32))
                plot_two_stage_visualization(
                    image_norm, gt_mask, pred, info,
                    save_path=str(vis_dir / f"patient{patient_idx}_slice{slice_idx}.png"),
                    title=f"Patient {patient_idx}, Slice {slice_idx}",
                )
                vis_count += 1

    print("\n" + "=" * 70)
    print("RESULTS")
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
    false_positive_rate = (false_positive_bg_slices / max(1, total_bg_slices))

    # print out the stage 1 stats
    print(f"\nStage 1 Detection Statistics:")
    print(f"  Foreground slices: {total_fg_slices}")
    print(f"  Detected         : {detected_fg_slices} ({100*detection_rate:.1f}%)")
    print(f"  Missed           : {total_fg_slices - detected_fg_slices}")
    print(f"  Fallback (full)  : {fallback_count}")
    print(f"  BG slices        : {total_bg_slices}")
    print(f"  False positives  : {false_positive_bg_slices} ({100*false_positive_rate:.1f}%)")

    print(f"\nEnd-to-End Metrics (ALL slices, full resolution):")
    print(f"  Dice      : {all_metrics['mean_fg_dice']:.4f}")
    print(f"  Recall    : {all_metrics['mean_fg_recall']:.4f}")
    print(f"  Precision : {all_metrics['mean_fg_precision']:.4f}")
    print(f"  F2        : {all_metrics['mean_fg_f2']:.4f}")

    print(f"\nEnd-to-End Metrics (FG slices only, full resolution):")
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
    logger.info(f"E2E all-slices Dice={all_metrics['mean_fg_dice']:.4f} "
                f"Recall={all_metrics['mean_fg_recall']:.4f} "
                f"F2={all_metrics['mean_fg_f2']:.4f}")
    logger.info(f"E2E fg-only Dice={fg_metrics['mean_fg_dice']:.4f} "
                f"Recall={fg_metrics['mean_fg_recall']:.4f} "
                f"F2={fg_metrics['mean_fg_f2']:.4f}")
    logger.info(f"Saved {vis_count} visualizations to {vis_dir}")

    comparison_experiments = ["exp13_full_pipeline", "exp03_aggressive_tversky"]
    prior_results = {}

    for exp_name in comparison_experiments:
        exp_result_dir = find_latest_model_dir(results_base, exp_name)
        if exp_result_dir:
            results_json = os.path.join(exp_result_dir, "results.json")
            if os.path.exists(results_json):
                with open(results_json) as f:
                    prior_results[exp_name] = json.load(f)

    if prior_results:
        print(f"\n{'=' * 70}")
        print("COMPARISON WITH PRIOR EXPERIMENTS (ROI-cropped evaluation)")
        print(f"{'=' * 70}")
        print(f"{'Experiment':<35s} {'Dice':>8s} {'Recall':>8s} {'F2':>8s}  {'Note':s}")
        print("-" * 80)

        for exp_name, res in prior_results.items():
            dice = res.get("best_val_dice", res.get("pipeline_metrics", {}).get("mean_fg_dice", 0))
            recall = res.get("best_val_recall", res.get("pipeline_metrics", {}).get("mean_fg_recall", 0))
            f2 = res.get("pipeline_metrics", {}).get("mean_fg_f2", 0)
            print(f"{exp_name:<35s} {dice:>8.4f} {recall:>8.4f} {f2:>8.4f}  ROI-cropped")

        print(f"{'exp16 (E2E, all slices)':<35s} "
              f"{all_metrics['mean_fg_dice']:>8.4f} "
              f"{all_metrics['mean_fg_recall']:>8.4f} "
              f"{all_metrics['mean_fg_f2']:>8.4f}  Full resolution")
        print(f"{'exp16 (E2E, FG slices only)':<35s} "
              f"{fg_metrics['mean_fg_dice']:>8.4f} "
              f"{fg_metrics['mean_fg_recall']:>8.4f} "
              f"{fg_metrics['mean_fg_f2']:>8.4f}  Full resolution")
        print("-" * 80)

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

        "coarse_model_dir": coarse_dir,
        "fine_model_dir": fine_dir,

        "coarse_threshold": cfg.COARSE_THRESHOLD,
        "bbox_padding": cfg.BBOX_PADDING,
        "use_tta": cfg.USE_TTA,
        "use_cc_filter": cfg.USE_CC_FILTER,

        "stage1_detection": {
            "total_fg_slices": total_fg_slices,
            "detected_fg_slices": detected_fg_slices,
            "detection_rate": detection_rate,
            "total_bg_slices": total_bg_slices,
            "false_positive_bg_slices": false_positive_bg_slices,
            "false_positive_rate": false_positive_rate,
            "fallback_count": fallback_count,
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
    print(f"  Stage 1 detection rate: {100*detection_rate:.1f}%")
    print(f"  E2E Dice (all)  : {all_metrics['mean_fg_dice']:.4f}")
    print(f"  E2E Recall (all): {all_metrics['mean_fg_recall']:.4f}")
    print(f"  E2E F2 (all)    : {all_metrics['mean_fg_f2']:.4f}")
    print(f"  E2E Dice (FG)   : {fg_metrics['mean_fg_dice']:.4f}")
    print(f"  E2E Recall (FG) : {fg_metrics['mean_fg_recall']:.4f}")
    print(f"  E2E F2 (FG)     : {fg_metrics['mean_fg_f2']:.4f}")
    print(f"")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
