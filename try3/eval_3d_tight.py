import sys, os, glob, pickle

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import config
from shared.data import patient_split
from shared.model_3d import create_3d_model
from train_3d_tight import Tight3DDataset, tolerant_dice, surface_dists, CACHE, THRESH, BATCH


@torch.no_grad()
def evaluate_all(model, loader, device):
    model.eval()
    tp = fp = fn = 0.0
    per_slice_dice = []
    per_struct_dice = []
    td2, nsd2, assd, hd95, det = [], [], [], [], []

    for images, masks in loader:
        images = images.to(device).float()
        prob = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
        tgt = masks.numpy()
        for b in range(prob.shape[0]):
            predBlk = (prob[b] >= THRESH).astype(np.uint8)
            gtBlk = (tgt[b] == 1).astype(np.uint8)

            for k in range(predBlk.shape[0]):
                pk, gk = predBlk[k], gtBlk[k]
                inter = float(np.logical_and(pk, gk).sum())
                tp += inter
                fp += float(pk.sum()) - inter
                fn += float(gk.sum()) - inter
                if gk.sum()==0:
                    continue
                s = pk.sum() + gk.sum()
                per_slice_dice.append(2.0 * inter / s if s else 1.0)
                td2.append(tolerant_dice(pk, gk, 2))
                det.append(1.0 if inter > 0 else 0.0)
                sd = surface_dists(pk, gk)
                if sd is not None:
                    d_p2g, d_g2p = sd
                    a = np.concatenate([d_p2g, d_g2p])
                    assd.append(float(a.mean()))
                    hd95.append(float(max(np.percentile(d_p2g, 95),
                                          np.percentile(d_g2p, 95))))
                    nsd2.append(float(np.concatenate(
                        [d_p2g <= 2, d_g2p <= 2]).mean()))

            if gtBlk.sum() == 0:
                continue
            inter3d = float(np.logical_and(predBlk, gtBlk).sum())
            denom3d = float(predBlk.sum() + gtBlk.sum())
            per_struct_dice.append(2.0 * inter3d / denom3d if denom3d else 1.0)

    m = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "micro_perslice_dice": (2 * tp) / (2 * tp + fp + fn + 1e-7),
        "macro_perslice_dice": m(per_slice_dice),
        "macro_3d_dice": m(per_struct_dice),
        "tol_dice_2px": m(td2),
        "nsd_2px": m(nsd2),
        "assd_px": m(assd),
        "hd95_px": m(hd95),
        "detection_rate": m(det),
        "n_fg_structures": len(per_struct_dice),
        "n_fg_slices": len(per_slice_dice),
    }


def main():
    ck = (sys.argv[1] if len(sys.argv) > 1
          else sorted(glob.glob(os.path.join(config.OUTPUT_BASE,
                                             "tight_3d", "*", "best_model.pth")))[-1])
    print(f"Checkpoint: {ck}")
    device = torch.device(config.DEVICE)

    d = pickle.load(open(CACHE, "rb"))
    vols, segs = d["vols"], d["segs"]
    _, val_idx = patient_split(len(vols))
    valDs = Tight3DDataset(vols, segs, val_idx, is_train=False, seed=2)
    val_loader = DataLoader(valDs, batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Val: {len(val_idx)} patients, "
          f"{valDs.n_fg} fg structures, {valDs.n_neg} neg crops")

    model = create_3d_model(in_channels=1, num_classes=2, base_filters=16,
                            deep_supervision=False).to(device)
    sd = torch.load(ck, map_location=device, weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)

    r = evaluate_all(model, val_loader, device)
    print("\n" + "=" * 64)
    print("3D U-NET  --  all Dice definitions, same checkpoint, same val set")
    print("=" * 64)
    print(f"  micro per-slice Dice   : {r['micro_perslice_dice']:.4f}   "
          f"<- the misleading '0.74' (not comparable)")
    print(f"  macro per-slice Dice   : {r['macro_perslice_dice']:.4f}")
    print(f"  macro 3D-volume Dice   : {r['macro_3d_dice']:.4f}   "
          f"<- USE THIS as the SCORES.md 3D row")
    print("-" * 64)
    print(f"  2px-tolerant Dice      : {r['tol_dice_2px']:.4f}")
    print(f"  NSD @ 2px              : {r['nsd_2px']:.4f}")
    print(f"  ASSD                   : {r['assd_px']:.3f} px")
    print(f"  HD95                   : {r['hd95_px']:.3f} px")
    print(f"  detection rate         : {r['detection_rate']:.4f}")
    print(f"  n structures / slices  : {r['n_fg_structures']} / {r['n_fg_slices']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
