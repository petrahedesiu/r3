import os, sys
import gc
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "try2"))
sys.path.insert(0, str(project_root))

import torch
from shared.config import ExperimentConfig as Config
from shared.metrics import compute_all_metrics
from shared.models import create_model, create_coarse_model
from shared.two_stage_inference import two_stage_predict_slice

EXP18_MODEL_DIR = project_root / "try2" / "results" / "exp18_tight_crops" / "20260224_164831"
EXP14_BASE = project_root / "try2" / "results" / "exp14_two_stage_coarse"

cfg = Config()
cfg.EXPERIMENT_NAME = "exp18_e2e_eval"
COARSE_SIZE = 256
FINE_SIZE = 384
E2E_BBOX_PADDING = 10
COARSE_THRESHOLD = 0.3

def find_latest_model_dir(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    for r in reversed(runs):
        if (r / "best_model.pth").exists():
            return r
    return None

def main():
    device = torch.device(cfg.DEVICE)

    coarse_dir = find_latest_model_dir(EXP14_BASE)
    if coarse_dir is None:
        print("ERROR: No exp14 coarse model found!")
        return
    print(f"Coarse model: {coarse_dir}")

    coarse_ckpt = torch.load(coarse_dir / "best_model.pth", map_location=device, weights_only=False)
    coarse_model = create_coarse_model(
        in_channels=cfg.IN_CHANNELS, num_classes=2
    ).to(device)
    coarse_model.load_state_dict(coarse_ckpt["model_state_dict"])
    coarse_model.eval()

    print(f"Fine model: {EXP18_MODEL_DIR}")
    fine_ckpt = torch.load(EXP18_MODEL_DIR / "best_model.pth", map_location=device, weights_only=False)
    fine_model = create_model(
        in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME, attention_type=cfg.ATTENTION_TYPE,
    ).to(device)
    fine_model.load_state_dict(fine_ckpt["model_state_dict"])
    fine_model.eval()

    print(f"Fine model from epoch {fine_ckpt.get('epoch', '?')}, "
          f"val_dice={fine_ckpt.get('val_dice', '?')}")

    from data_utils import discover_patients, load_patient_data
    patients = discover_patients(cfg.DATA_DIR)
    all_volumes, all_segs = [], []
    for p in tqdm(patients, desc="Loading patients"):
        vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
        all_volumes.append(vol)
        all_segs.append(seg)

    np.random.seed(42)
    n = len(all_volumes)
    perm = np.random.permutation(n)
    valStart = int(0.8 * n)
    val_indices = perm[valStart:]
    val_volumes = [all_volumes[i] for i in val_indices]
    val_segs = [all_segs[i] for i in val_indices]
    print(f"\nVal patients: {len(val_volumes)}")

    all_preds = []
    all_targets = []
    total_fg = 0
    detected_fg = 0

    for vol, seg in tqdm(list(zip(val_volumes, val_segs)), desc="E2E eval"):
        nSlices = vol.shape[2]
        for si in range(nSlices):
            image = vol[:, :, si].copy()
            gt_mask = seg[:, :, si].copy()
            hasFg = gt_mask.max() > 0
            pred, info = two_stage_predict_slice(
                image, coarse_model, fine_model, device,
                coarse_size=COARSE_SIZE, fine_size=FINE_SIZE,
                coarse_threshold=COARSE_THRESHOLD,
                bbox_padding=E2E_BBOX_PADDING,
                use_tta=False, use_cc_filter=False,
            )
            if hasFg:
                total_fg += 1
                if info.get("detected", False):
                    detected_fg += 1

            all_preds.append(pred.ravel())
            all_targets.append(gt_mask.ravel())

        if cfg.DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)

    metrics = compute_all_metrics(all_preds_flat, all_targets_flat, num_classes=3)

    fg_mask = all_targets_flat > 0
    if fg_mask.any():
        fg_preds = all_preds_flat[fg_mask]
        fg_targets = all_targets_flat[fg_mask]
        fg_metrics = compute_all_metrics(fg_preds, fg_targets, num_classes=3)
    else:
        fg_metrics = {}

    detection_rate = detected_fg / max(total_fg, 1) * 100

    print("\n" + "=" * 70)
    print("EXP18 END-TO-END RESULTS")
    print("=" * 70)
    print(f"Detection rate: {detected_fg}/{total_fg} = {detection_rate:.1f}%")
    print(f"\nAll-pixel metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"\nForeground-only metrics:")
    for k, v in sorted(fg_metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    results = {
        "experiment": "exp18_tight_crops_e2e",
        "fine_model_dir": str(EXP18_MODEL_DIR),
        "fine_model_epoch": fine_ckpt.get("epoch"),
        "fine_model_val_dice": fine_ckpt.get("val_dice"),
        "detection_rate": detection_rate,
        "detected_fg_slices": detected_fg,
        "total_fg_slices": total_fg,
        "all_pixel_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in metrics.items()},
        "fg_only_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in fg_metrics.items()},
    }

    results_path = EXP18_MODEL_DIR / "e2e_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results -> {results_path}")

if __name__ == "__main__":
    main()
