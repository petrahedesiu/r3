import os
import sys
import glob
import json
import numpy as np


def dice_3d(pred, gt, num_classes=3, smooth=1e-7):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    out = {}
    for c in range(1, num_classes):
        p = pred == c
        g = gt == c
        inter = float(np.logical_and(p, g).sum())
        denom = float(p.sum() + g.sum())
        out[c] = (2 * inter + smooth) / (denom + smooth) if g.sum() > 0 else float("nan")
    return out


def _read(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def summarize(per_case, class_names=("AEAL", "AEAR")):
    cls = sorted({c for v in per_case.values() for c in v})
    means = {}
    for c in cls:
        vals = [v[c] for v in per_case.values() if c in v and not np.isnan(v[c])]
        means[c] = float(np.mean(vals)) if vals else float("nan")
    fg = [means[c] for c in cls if not np.isnan(means[c])]
    meanFg = float(np.mean(fg)) if fg else float("nan")
    print(f"\n{'case':<16} " + " ".join(f"{class_names[c-1]:>8}" for c in cls))
    for cid in sorted(per_case):
        row = per_case[cid]
        print(f"{cid:<16} " + " ".join(f"{row.get(c, float('nan')):>8.4f}" for c in cls))
    print("-" * 40)
    for c in cls:
        print(f"  {class_names[c-1]} mean 3D Dice: {means[c]:.4f}")
    print(f"  MEAN FG 3D DICE : {meanFg:.4f}   (n={len(per_case)} patients)")
    return {"per_case": {k: {str(kk): vv for kk, vv in v.items()} for k, v in per_case.items()},
            "per_class_mean": {str(k): v for k, v in means.items()},
            "mean_fg_dice": meanFg, "n_patients": len(per_case)}


def eval_dir(pred_dir, gt_dir, num_classes=3):
    per_case = {}
    for pp in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
        cid = os.path.basename(pp)[:-7]
        gp = os.path.join(gt_dir, cid + ".nii.gz")
        if not os.path.exists(gp):
            continue
        per_case[cid] = dice_3d(_read(pp), _read(gp), num_classes)
    return per_case


def eval_nnunet(model_dir, num_classes=3):
    raw_labels = None
    for cand in [os.environ.get("nnUNet_raw", ""), "/content/nnunet/nnUNet_raw"]:
        d = os.path.join(cand, "Dataset001_EarCT", "labelsTr")
        if os.path.isdir(d):
            raw_labels = d
            break
    if raw_labels is None:
        raise RuntimeError("could not locate nnUNet_raw labelsTr")

    per_case = {}
    for fold_dir in sorted(glob.glob(os.path.join(model_dir, "fold_*"))):
        vdir = os.path.join(fold_dir, "validation")
        if not os.path.isdir(vdir):
            print(f"  (skip {os.path.basename(fold_dir)}: no validation dir)")
            continue
        foldCases = eval_dir(vdir, raw_labels, num_classes)
        per_case.update(foldCases)
        print(f"  {os.path.basename(fold_dir)}: {len(foldCases)} cases")
    return per_case


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "nnunet":
        pc = eval_nnunet(sys.argv[2])
    elif mode == "dir":
        pc = eval_dir(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
    summary = summarize(pc)
    out = sys.argv[-1] if sys.argv[-1].endswith(".json") else "/content/eval_3d_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nsaved {out}")
