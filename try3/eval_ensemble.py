
import sys
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import get_labeled_slice_indices

from shared import config
from shared.data import load_all_patients, patient_split
from shared.models import create_model, create_coarse_model
from shared.metrics import compute_all_metrics
from shared.inference import ensemble_predict_slice
from shared.visualization import create_overlay, save_montage, save_comparison_figure


def find_latest_model(stage_name):
    base = Path(config.OUTPUT_BASE) / stage_name
    if not base.exists():
        raise FileNotFoundError(f"No results dir for {stage_name}: {base}")
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")
    latest = runs[-1]
    model_path = latest / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No best_model.pth in {latest}")
    return model_path


def main(coarse_path=None, aeal_path=None, aear_path=None):
    print("=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)

    device = torch.device(config.DEVICE)

    if coarse_path is None:
        coarse_path = find_latest_model("coarse")
    if aeal_path is None:
        aeal_path = find_latest_model("aeal")
    if aear_path is None:
        aear_path = find_latest_model("aear")

    print(f"Coarse model: {coarse_path}")
    print(f"AEAL model:   {aeal_path}")
    print(f"AEAR model:   {aear_path}")

    # load the coarse localizer
    coarse_model = create_coarse_model(in_channels=1, num_classes=2).to(device)
    coarseCkpt = torch.load(coarse_path, map_location=device, weights_only=False)
    coarse_model.load_state_dict(coarseCkpt["model_state_dict"])
    coarse_model.eval()

    aeal_model = create_model(in_channels=1, num_classes=3).to(device)
    aealCkpt = torch.load(aeal_path, map_location=device, weights_only=False)
    aeal_model.load_state_dict(aealCkpt["model_state_dict"])
    aeal_model.eval()

    aear_model = create_model(in_channels=1, num_classes=3).to(device)
    aearCkpt = torch.load(aear_path, map_location=device, weights_only=False)
    aear_model.load_state_dict(aearCkpt["model_state_dict"])
    aear_model.eval()

    volumes, segmentations, infos = load_all_patients()
    n_patients = len(volumes)

    train_idx, val_idx = patient_split(n_patients)
    print(f"\nEvaluating on {len(val_idx)} validation patients")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_BASE) / "ensemble" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    all_preds = []
    all_gts = []
    per_patient_results = []
    overlay_images = []
    overlay_titles = []

    for pi in val_idx:
        vol = volumes[pi]
        seg = segmentations[pi]
        patient_id = infos[pi]["patient_id"]

        labeled_slices = get_labeled_slice_indices(seg)
        all_slice_indices = list(range(vol.shape[2]))

        patient_preds = []
        patient_gts = []

        for si in all_slice_indices:
            image = vol[:, :, si].copy()
            gt = seg[:, :, si].copy()

            pred, info = ensemble_predict_slice(
                image, coarse_model, aeal_model, aear_model, device,
                coarse_size=256, fine_size=384, patch_size=128,
                coarse_threshold=0.3, bbox_padding=30,
            )

            patient_preds.append(pred)
            patient_gts.append(gt)

            if si in labeled_slices and gt.max() > 0:
                overlay_images.append(create_overlay(image, pred))
                overlay_titles.append(f"{patient_id} s{si}")

                if len(overlay_images) <= 20:
                    save_comparison_figure(
                        image, gt, pred,
                        vis_dir / f"{patient_id}_slice{si:03d}.png",
                        title=f"{patient_id} slice {si}",
                    )

        patientPredsArr = np.stack(patient_preds)
        patientGtsArr = np.stack(patient_gts)

        metrics = compute_all_metrics(patientPredsArr, patientGtsArr, num_classes=3)
        per_patient_results.append({
            "patient_id": patient_id,
            **{k: float(v) if isinstance(v, (float, int, np.floating)) else
               {str(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
               for k, v in metrics.items()},
        })

        all_preds.append(patientPredsArr)
        all_gts.append(patientGtsArr)

        print(f"  {patient_id}: Dice={metrics['mean_fg_dice']:.3f} "
              f"Recall={metrics['mean_fg_recall']:.3f} "
              f"AEAL={metrics['dice_per_class'].get(1, float('nan')):.3f} "
              f"AEAR={metrics['dice_per_class'].get(2, float('nan')):.3f}")

    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_gts_arr = np.concatenate(all_gts, axis=0)
    overall = compute_all_metrics(all_preds_arr, all_gts_arr, num_classes=3)

    fg_mask = all_gts_arr.reshape(all_gts_arr.shape[0], -1).max(axis=1) > 0
    if fg_mask.any():
        fg_preds = all_preds_arr[fg_mask]
        fg_gts = all_gts_arr[fg_mask]
        fg_metrics = compute_all_metrics(fg_preds, fg_gts, num_classes=3)
    else:
        fg_metrics = overall

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS (ALL SLICES)")
    print("=" * 70)
    print(f"Mean FG Dice:      {overall['mean_fg_dice']:.4f}")
    print(f"Mean FG Recall:    {overall['mean_fg_recall']:.4f}")
    print(f"Mean FG Precision: {overall['mean_fg_precision']:.4f}")
    print(f"Mean FG F2:        {overall['mean_fg_f2']:.4f}")
    print(f"AEAL Dice:         {overall['dice_per_class'].get(1, float('nan')):.4f}")
    print(f"AEAR Dice:         {overall['dice_per_class'].get(2, float('nan')):.4f}")

    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS (FG SLICES ONLY)")
    print("=" * 70)
    print(f"Mean FG Dice:      {fg_metrics['mean_fg_dice']:.4f}")
    print(f"Mean FG Recall:    {fg_metrics['mean_fg_recall']:.4f}")
    print(f"Mean FG Precision: {fg_metrics['mean_fg_precision']:.4f}")
    print(f"Mean FG F2:        {fg_metrics['mean_fg_f2']:.4f}")
    print(f"AEAL Dice:         {fg_metrics['dice_per_class'].get(1, float('nan')):.4f}")
    print(f"AEAR Dice:         {fg_metrics['dice_per_class'].get(2, float('nan')):.4f}")

    if overlay_images:
        save_montage(overlay_images[:20], output_dir / "overlay_montage.png",
                     ncols=5, titles=overlay_titles[:20])

    results = {
        "overall_all_slices": {k: float(v) if isinstance(v, (float, int, np.floating)) else
                               {str(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                               for k, v in overall.items()},
        "overall_fg_slices": {k: float(v) if isinstance(v, (float, int, np.floating)) else
                              {str(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                              for k, v in fg_metrics.items()},
        "per_patient": per_patient_results,
        "model_paths": {
            "coarse": str(coarse_path),
            "aeal": str(aeal_path),
            "aear": str(aear_path),
        },
        "n_val_patients": len(val_idx),
        "n_all_slices": int(all_preds_arr.shape[0]),
        "n_fg_slices": int(fg_mask.sum()) if fg_mask is not None else 0,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
