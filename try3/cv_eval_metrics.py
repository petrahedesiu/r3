import numpy as np
import sys, os, glob, json
import cv2
import torch
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import config
from train_25d import Tight25DDataset, build_model

CACHE = "/content/cache_data.pkl"
IMG, CROP, BATCH, N_FOLDS = 256, 64, 16, 5
THRESH = 0.45


def dice(p, g):
    p, g = p > 0, g > 0
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(p, g).sum() / s)


def tolerant_dice(pred, gt, tol=1):
    if pred.sum()==0 and gt.sum()==0:
        return 1.0
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(gt.astype(np.uint8), kern)
    pd = cv2.dilate(pred.astype(np.uint8), kern)
    inter = (np.logical_and(pred > 0, gd > 0).sum()
             + np.logical_and(gt > 0, pd > 0).sum())
    return float(inter / s)


def _surface(m):
    m = m.astype(bool)
    if m.sum() == 0:
        return m
    return m & ~binary_erosion(m)


def surface_dists(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    sp, sg = _surface(pred), _surface(gt)
    dt_to_g = distance_transform_edt(~sg)
    dt_to_p = distance_transform_edt(~sp)
    d_p2g = dt_to_g[sp]
    d_g2p = dt_to_p[sg]
    return d_p2g, d_g2p


def predict_prob(model, x):
    return torch.softmax(model(x), dim=1)[:, 1]


def main():
    cvdir = (sys.argv[1] if len(sys.argv) > 1
             else sorted(glob.glob(os.path.join(config.OUTPUT_BASE, "cv", "*")))[-1])
    print(f"Eval models in: {cvdir}")
    data = __import__("pickle").load(open(CACHE, "rb"))
    vols, segs = data["vols"], data["segs"]
    n = len(vols)
    device = torch.device(config.DEVICE)

    import albumentations as A
    val_tf = A.Compose([A.Resize(IMG, IMG)])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(range(n)))

    tp = fp = fn = 0.0
    td1, td2, nsd1, nsd2, assd, hd95 = [], [], [], [], [], []
    det = []
    raw_dice = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        models = []
        for arch in ("unet", "unetpp"):
            path = os.path.join(cvdir, f"f{fi}_{arch}", "best_model.pth")
            m = build_model(arch, "efficientnet-b0").to(device)
            sd = torch.load(path, map_location=device)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            m.load_state_dict(sd)
            m.eval()
            models.append(m)

        val_ds = Tight25DDataset(vols, segs, list(va_idx), transform=val_tf,
                                 is_train=False, crop=CROP, seed=fi + 100)
        vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

        with torch.no_grad():
            for images, masks in vl:
                images = images.to(device).float()
                prob = sum(predict_prob(m, images) for m in models) / len(models)
                prob = prob.cpu().numpy()
                tgt = masks.numpy()
                for b in range(prob.shape[0]):
                    p64 = cv2.resize(prob[b], (CROP, CROP),
                                     interpolation=cv2.INTER_AREA)
                    pred = (p64 >= THRESH).astype(np.uint8)
                    gt = cv2.resize(tgt[b].astype(np.uint8), (CROP, CROP),
                                    interpolation=cv2.INTER_NEAREST)
                    gt = (gt == 1).astype(np.uint8)

                    inter = float(np.logical_and(pred, gt).sum())
                    tp += inter
                    fp += float(pred.sum()) - inter
                    fn += float(gt.sum()) - inter

                    if gt.sum() == 0:
                        continue
                    raw_dice.append(dice(pred, gt))
                    td1.append(tolerant_dice(pred, gt, 1))
                    td2.append(tolerant_dice(pred, gt, 2))
                    det.append(1.0 if inter > 0 else 0.0)
                    sd = surface_dists(pred, gt)
                    if sd is not None:
                        d_p2g, d_g2p = sd
                        alld = np.concatenate([d_p2g, d_g2p])
                        assd.append(float(alld.mean()))
                        hd95.append(float(max(np.percentile(d_p2g, 95),
                                              np.percentile(d_g2p, 95))))
                        nsd1.append(float((np.concatenate(
                            [d_p2g <= 1, d_g2p <= 1])).mean()))
                        nsd2.append(float((np.concatenate(
                            [d_p2g <= 2, d_g2p <= 2])).mean()))
        print(f"  fold {fi+1}/{N_FOLDS} done ({len(va_idx)} patients)")

    ds_dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    m = lambda x: float(np.mean(x)) if x else 0.0

    out = {
        "n_fg_crops": len(raw_dice),
        "standard_dice_dataset": round(ds_dice, 4),
        "standard_dice_percrop": round(m(raw_dice), 4),
        "tolerant_dice_1px": round(m(td1), 4),
        "tolerant_dice_2px": round(m(td2), 4),
        "nsd_1px": round(m(nsd1), 4),
        "nsd_2px": round(m(nsd2), 4),
        "assd_px": round(m(assd), 3),
        "hd95_px": round(m(hd95), 3),
        "detection_rate": round(m(det), 4),
    }
    print("\n" + "=" * 60)
    print("FULL METRIC SUITE  (out-of-fold, native 64 px grid)")
    print("=" * 60)
    print(f"  foreground crops scored      : {out['n_fg_crops']}")
    print(f"  standard Dice (dataset)      : {out['standard_dice_dataset']}")
    print(f"  standard Dice (per-crop mean): {out['standard_dice_percrop']}")
    print(f"  tolerant Dice  (1 px slop)   : {out['tolerant_dice_1px']}")
    print(f"  tolerant Dice  (2 px slop)   : {out['tolerant_dice_2px']}")
    print(f"  NSD  tau=1 px                : {out['nsd_1px']}")
    print(f"  NSD  tau=2 px                : {out['nsd_2px']}")
    print(f"  ASSD                         : {out['assd_px']} px")
    print(f"  HD95                         : {out['hd95_px']} px")
    print(f"  detection rate               : {out['detection_rate']}")
    print("=" * 60)
    json.dump(out, open("/content/cv_metrics_full.json", "w"), indent=2)
    print("Saved -> /content/cv_metrics_full.json")


if __name__ == "__main__":
    main()
