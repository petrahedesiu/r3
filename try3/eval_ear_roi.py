
import sys
import os
import json
import glob

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared import config
from shared.data import load_all_patients, patient_split
from shared.models import create_model
from shared.atlas import crop_roi, place_roi, roi_bounds, in_depth_band
from shared.windowing import bone_window

IMG_SIZE = 256


def _find_latest_model():
    cands = sorted(glob.glob(os.path.join(
        config.OUTPUT_BASE, "ear_roi", "*", "best_model.pth")))
    if not cands:
        sys.exit("No ear_roi model found. Train first with train_ear_roi.py")
    return cands[-1]


@torch.no_grad()
def predict_volume(model, volume, device):
    H, W, D = volume.shape
    pred = np.zeros((H, W, D), dtype=np.int64)
    for si in range(D):
        if not in_depth_band(si, D):
            continue
        sl = volume[:, :, si].astype(np.float32)
        for cid, side in ((1, "L"), (2, "R")):
            crop = np.ascontiguousarray(crop_roi(sl, side))
            ch, cw = crop.shape
            inp = bone_window(crop)
            inp = cv2.resize(inp, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)
            logits = model(t)
            pm = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pm = cv2.resize(pm, (cw, ch), interpolation=cv2.INTER_NEAREST)
            full = place_roi(pm.astype(np.int64), H, W, side)
            pred[full > 0] = cid
    return pred


def main():
    modelPath = sys.argv[1] if len(sys.argv) > 1 else _find_latest_model()
    print(f"Model: {modelPath}")

    volumes, segmentations, infos = load_all_patients()
    train_idx, val_idx = patient_split(len(volumes))
    print(f"Evaluating on {len(val_idx)} held-out patients")

    device = torch.device(config.DEVICE)
    model = create_model(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(modelPath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tp = {1: 0.0, 2: 0.0}
    fp = {1: 0.0, 2: 0.0}
    fn = {1: 0.0, 2: 0.0}
    per_patient = {1: [], 2: []}

    for i in val_idx:
        vol, seg = volumes[i], segmentations[i]
        pred = predict_volume(model, vol, device)
        for cid in (1, 2):
            p = (pred == cid)
            g = (seg == cid)
            t = float(np.logical_and(p, g).sum())
            f_p = float(np.logical_and(p, ~g).sum())
            f_n = float(np.logical_and(~p, g).sum())
            tp[cid] += t
            fp[cid] += f_p
            fn[cid] += f_n
            if g.sum() > 0:
                d = (2 * t) / (2 * t + f_p + f_n + 1e-7)
                per_patient[cid].append(d)
        d1 = per_patient[1][-1] if (seg == 1).sum() > 0 else float("nan")
        d2 = per_patient[2][-1] if (seg == 2).sum() > 0 else float("nan")
        print(f"  {infos[i]['patient_id'][:28]:<30} "
              f"AEAL dice={d1:.3f}  AEAR dice={d2:.3f}")

    print("\n" + "=" * 70)
    print("HELD-OUT RESULTS  (dataset-level, then per-patient mean)")
    print("=" * 70)
    rows = []
    for cid, name in ((1, "AEAL"), (2, "AEAR")):
        dsDice = (2 * tp[cid]) / (2 * tp[cid] + fp[cid] + fn[cid] + 1e-7)
        dsRec = tp[cid] / (tp[cid] + fn[cid] + 1e-7)
        dsPrec = tp[cid] / (tp[cid] + fp[cid] + 1e-7)
        pp = per_patient[cid]
        ppDice = float(np.mean(pp)) if pp else 0.0
        rows.append((name, dsDice, dsRec, dsPrec, ppDice))
        print(f"{name}: dataset Dice={dsDice:.4f}  Recall={dsRec:.4f}  "
              f"Precision={dsPrec:.4f}  | per-patient mean Dice={ppDice:.4f}")

    mean_fg = np.mean([r[1] for r in rows])
    print(f"\nMean FG Dice (dataset-level): {mean_fg:.4f}")
    print(f"Baseline (try3 AEAL specialist): 0.0000  -> delta {mean_fg:+.4f}")

    out = {
        "model_path": modelPath,
        "n_val_patients": len(val_idx),
        "per_class": {r[0]: {"dataset_dice": r[1], "recall": r[2],
                             "precision": r[3], "patient_mean_dice": r[4]}
                      for r in rows},
        "mean_fg_dice": float(mean_fg),
    }
    out_path = os.path.join(os.path.dirname(modelPath), "eval_ear_roi.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
