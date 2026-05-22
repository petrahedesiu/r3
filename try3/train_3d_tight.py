import sys, os, pickle, random
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import config
from shared.data import patient_split
from shared.atlas import roi_bounds
from shared.windowing import bone_window
from shared.model_3d import create_3d_model

CACHE = "/content/cache_data.pkl"
CROP = 64
PD = 16
NUM_EPOCHS = 45
BATCH = 8
OVERSAMPLE = 8
NEG_RATIO = 1.0
JITTER = 8
JITTER_D = 2
THRESH = 0.45


def _window(H, W, cy, cx, S):
    half = S // 2
    cy = int(np.clip(cy, half, H - (S - half) - 1))
    cx = int(np.clip(cx, half, W - (S - half) - 1))
    return cy - half, cx - half


class Tight3DDataset(Dataset):

    def __init__(self, vols, segs, idxs, is_train=True, seed=0):
        self.vols, self.segs = vols, segs
        self.is_train = is_train
        self.jit = JITTER if is_train else 0
        self.jit_d = JITTER_D if is_train else 0
        rng = random.Random(seed)

        fg, neg = [], []
        for pi in idxs:
            seg = segs[pi]
            H, W, D = seg.shape
            for cid, side in ((1, "L"), (2, "R")):
                sl = []
                for si in range(D):
                    m = seg[:, :, si] == cid
                    if side == "R":
                        m = m[:, ::-1]
                    if m.any():
                        sl.append(si)
                if not sl:
                    continue
                ys, xs, zs = [], [], []
                for si in sl:
                    m = seg[:, :, si] == cid
                    if side == "R":
                        m = m[:, ::-1]
                    yy, xx = np.where(m)
                    ys.append(yy); xs.append(xx); zs += [si] * len(yy)
                cy = int(np.concatenate(ys).mean())
                cx = int(np.concatenate(xs).mean())
                cz = int(np.mean(zs))
                fg.append((pi, side, cid, cy, cx, cz, False))

                r0, r1, c0, c1 = roi_bounds(H, W)
                half = CROP // 2
                for _ in range(20):
                    ncy = rng.randint(r0 + half, max(r0 + half, r1 - half))
                    ncx = rng.randint(c0 + half, max(c0 + half, c1 - half))
                    ncz = cz + rng.randint(-PD, PD)
                    wr, wc = _window(H, W, ncy, ncx, CROP)
                    hit = False
                    for si in range(max(0, ncz - PD // 2),
                                    min(D, ncz + PD // 2)):
                        m = seg[:, :, si] == cid
                        if side == "R":
                            m = m[:, ::-1]
                        if m[wr:wr + CROP, wc:wc + CROP].sum() > 0:
                            hit = True
                            break
                    if not hit:
                        neg.append((pi, side, cid, ncy, ncx, ncz, True))
                        break

        reps = OVERSAMPLE if is_train else 1
        self.samples = fg * reps + neg * (reps if is_train else 1)
        self.n_fg, self.n_neg = len(fg), len(neg)

    def __len__(self):
        return len(self.samples)

    def _crop_block(self, pi, side, cid, cy, cx, cz):
        vol, seg = self.vols[pi], self.segs[pi]
        H, W, D = vol.shape
        if self.jit:
            cy += random.randint(-self.jit, self.jit)
            cx += random.randint(-self.jit, self.jit)
        if self.jit_d:
            cz += random.randint(-self.jit_d, self.jit_d)
        wr, wc = _window(H, W, cy, cx, CROP)
        z0 = int(np.clip(cz - PD // 2, 0, max(0, D - PD)))
        img = np.zeros((PD, CROP, CROP), np.float32)
        msk = np.zeros((PD, CROP, CROP), np.int64)
        for k in range(PD):
            sj = z0 + k
            if sj >= D:
                break
            sl = vol[:, :, sj].astype(np.float32)
            ms = (seg[:, :, sj] == cid).astype(np.int64)
            if side == "R":
                sl = sl[:, ::-1]
                ms = ms[:, ::-1]
            img[k] = sl[wr:wr + CROP, wc:wc + CROP]
            msk[k] = ms[wr:wr + CROP, wc:wc + CROP]
        return img, msk

    def __getitem__(self, idx):
        pi, side, cid, cy, cx, cz, is_neg = self.samples[idx]
        img, msk = self._crop_block(pi, side, cid, cy, cx, cz)
        img = bone_window(img)
        if self.is_train and random.random() < 0.4:
            img = np.clip(img * random.uniform(0.9, 1.1)
                          + random.uniform(-0.05, 0.05), 0, 1)
        image = torch.from_numpy(np.ascontiguousarray(img[None])).float()
        mask = torch.from_numpy(np.ascontiguousarray(msk)).long()
        return image, mask


class Loss3D(nn.Module):

    def __init__(self, class_weights, alpha=0.2, beta=0.8, gamma=2.0,
                 focal_w=0.5, tversky_w=0.5, smooth=1e-6):
        super().__init__()
        self.register_buffer("cw", class_weights)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.fw, self.tw, self.s = focal_w, tversky_w, smooth

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.cw, reduction="none")
        pt = torch.exp(-ce)
        focal = (0.25 * (1.0 - pt) ** self.gamma * ce).mean()
        prob = F.softmax(logits, dim=1)[:, 1]
        g = (target == 1).float()
        tp = (prob * g).sum()
        fp = (prob * (1.0 - g)).sum()
        fn = ((1.0 - prob) * g).sum()
        tversky = 1.0 - (tp + self.s) / (tp + self.alpha * fp
                                         + self.beta * fn + self.s)
        return self.fw * focal + self.tw * tversky


def tolerant_dice(pred, gt, tol=1):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(gt.astype(np.uint8), k)
    pd = cv2.dilate(pred.astype(np.uint8), k)
    inter = (np.logical_and(pred > 0, gd > 0).sum()
             + np.logical_and(gt > 0, pd > 0).sum())
    return float(inter / s)


def _surf(m):
    m = m.astype(bool)
    return m & ~binary_erosion(m) if m.sum() else m


def surface_dists(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    sp, sg = _surf(pred), _surf(gt)
    d_p2g = distance_transform_edt(~sg)[sp]
    d_g2p = distance_transform_edt(~sp)[sg]
    return d_p2g, d_g2p


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0.0
    td1, td2, nsd2, assd, hd95, det = [], [], [], [], [], []
    for images, masks in loader:
        images = images.to(device).float()
        prob = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
        tgt = masks.numpy()
        for b in range(prob.shape[0]):
            for k in range(prob.shape[1]):
                pred = (prob[b, k] >= THRESH).astype(np.uint8)
                gt = (tgt[b, k] == 1).astype(np.uint8)
                inter = float(np.logical_and(pred, gt).sum())
                tp += inter
                fp += float(pred.sum()) - inter
                fn += float(gt.sum()) - inter
                if gt.sum()==0:
                    continue
                td1.append(tolerant_dice(pred, gt, 1))
                td2.append(tolerant_dice(pred, gt, 2))
                det.append(1.0 if inter > 0 else 0.0)
                sd = surface_dists(pred, gt)
                if sd is not None:
                    d_p2g, d_g2p = sd
                    a = np.concatenate([d_p2g, d_g2p])
                    assd.append(float(a.mean()))
                    hd95.append(float(max(np.percentile(d_p2g, 95),
                                          np.percentile(d_g2p, 95))))
                    nsd2.append(float(np.concatenate(
                        [d_p2g <= 2, d_g2p <= 2]).mean()))
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-7),
        "tol_dice_1px": m(td1), "tol_dice_2px": m(td2),
        "nsd_2px": m(nsd2), "assd": m(assd), "hd95": m(hd95),
        "detection": m(det), "n_fg_slices": len(td1),
    }


def main():
    print("=" * 70)
    print(f"3D TIGHT-CROP  ({CROP}x{CROP}x{PD} native, oracle centring)")
    print("=" * 70)
    d = pickle.load(open(CACHE, "rb"))
    vols, segs = d["vols"], d["segs"]
    n = len(vols)
    train_idx, val_idx = patient_split(n)
    print(f"Train {len(train_idx)} / Val {len(val_idx)} patients")

    trainDs = Tight3DDataset(vols, segs, train_idx, is_train=True, seed=1)
    valDs = Tight3DDataset(vols, segs, val_idx, is_train=False, seed=2)
    print(f"Train {len(trainDs)} (fg={trainDs.n_fg} neg={trainDs.n_neg})  "
          f"Val {len(valDs)} (fg={valDs.n_fg} neg={valDs.n_neg})")

    train_loader = DataLoader(trainDs, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(valDs, batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device(config.DEVICE)
    model = create_3d_model(in_channels=1, num_classes=2, base_filters=16,
                            deep_supervision=False).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"3D UNet params: {nparams/1e6:.2f}M")

    cw = torch.tensor([1.0, 3.0], dtype=torch.float32, device=device)
    criterion = Loss3D(cw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(config.OUTPUT_BASE) / "tight_3d" / ts
    out.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0.0
        for images, masks in train_loader:
            images = images.to(device).float()
            masks = masks.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        scheduler.step()
        vm = evaluate(model, val_loader, device)
        flag = ""
        if vm["dice"] > best:
            best = vm["dice"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "metrics": vm}, out / "best_model.pth")
            flag = "  *BEST*"
        print(f"E{epoch:02d}  loss {tl/len(train_loader):.4f}  "
              f"val Dice {vm['dice']:.4f}  tol2 {vm['tol_dice_2px']:.3f}  "
              f"NSD2 {vm['nsd_2px']:.3f}  det {vm['detection']:.3f}{flag}")

    print("\n" + "=" * 70)
    best_ck = torch.load(out / "best_model.pth", map_location=device,
                         weights_only=False)
    bm = best_ck["metrics"]
    print(f"BEST 3D (epoch {best_ck['epoch']}):")
    for k in ("dice", "tol_dice_1px", "tol_dice_2px", "nsd_2px",
              "assd", "hd95", "detection"):
        print(f"  {k:14s}: {bm[k]:.4f}")
    print(f"\n2D ref (same split, train_tight.py): val Dice 0.459")
    print(f"2D ref (CV out-of-fold suite): Dice 0.488  tol2 0.710  "
          f"NSD2 0.803  det 0.735")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
