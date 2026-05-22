
import random
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset


def _normalize(image: np.ndarray) -> np.ndarray:
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32)


def _roi_crop(image, mask, padding=50):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return image, mask
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding)
    rmax = min(image.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(image.shape[1], cmax + padding + 1)
    return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]


def _to_tensors(image, mask, transform=None):
    if transform is not None:
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented['mask']
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
    else:
        mask = mask.long()
    return image, mask


class CoarseDataset(Dataset):

    def __init__(self, volumes, segmentations, transform=None, oversample=5):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform

        self.indices = []
        for pi, seg in enumerate(segmentations):
            for si in range(seg.shape[2]):
                has_fg = seg[:, :, si].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((pi, si))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pi, si = self.indices[idx]
        image = _normalize(self.volumes[pi][:, :, si].copy())
        mask = (self.segmentations[pi][:, :, si] > 0).astype(np.int64)
        return _to_tensors(image, mask, self.transform)


class FGCenteredDataset(Dataset):

    def __init__(self, volumes, segmentations, transform=None,
                 fg_ratio=0.5, patch_size=384, oversample=3):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.fg_ratio = fg_ratio
        self.patch_size = patch_size

        self.indices = []
        for pi, seg in enumerate(segmentations):
            for si in range(seg.shape[2]):
                has_fg = seg[:, :, si].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((pi, si))

        self.fg_pixels: Dict[Tuple[int, int], np.ndarray] = {}
        for pi, seg in enumerate(segmentations):
            for si in range(seg.shape[2]):
                s = seg[:, :, si]
                if s.max() > 0:
                    ys, xs = np.where(s > 0)
                    self.fg_pixels[(pi, si)] = np.stack([ys, xs], axis=1)

    def __len__(self):
        return len(self.indices)

    def _crop_patch_around(self, image, mask, cy, cx):
        ps = self.patch_size
        jitter = ps // 8
        cy += random.randint(-jitter, jitter)
        cx += random.randint(-jitter, jitter)
        H, W = image.shape[:2]
        y0, x0 = cy - ps // 2, cx - ps // 2
        y1, x1 = y0 + ps, x0 + ps
        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - H)
        pad_right = max(0, x1 - W)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(H, y1), min(W, x1)
        img_crop = image[y0:y1, x0:x1]
        msk_crop = mask[y0:y1, x0:x1]
        if pad_top or pad_bottom or pad_left or pad_right:
            img_crop = np.pad(img_crop, ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode="constant", constant_values=0)
            msk_crop = np.pad(msk_crop, ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode="constant", constant_values=0)
        return img_crop, msk_crop

    def __getitem__(self, idx):
        pi, si = self.indices[idx]
        image = _normalize(self.volumes[pi][:, :, si].copy())
        mask = self.segmentations[pi][:, :, si].copy()
        key = (pi, si)
        if random.random() < self.fg_ratio and key in self.fg_pixels:
            fgLocs = self.fg_pixels[key]
            chosen = fgLocs[random.randint(0, len(fgLocs) - 1)]
            image, mask = self._crop_patch_around(image, mask, int(chosen[0]), int(chosen[1]))
        else:
            if mask.max() > 0:
                image, mask = _roi_crop(image, mask, padding=50)
        return _to_tensors(image, mask, self.transform)


class FinePatchDataset(Dataset):

    def __init__(self, volumes, segmentations, transform=None,
                 patch_size=128, jitter=10, oversample=3):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.patch_size = patch_size
        self.jitter = jitter

        self.indices = []
        for pi, seg in enumerate(segmentations):
            for si in range(seg.shape[2]):
                if seg[:, :, si].max() > 0:
                    for _ in range(oversample):
                        self.indices.append((pi, si))

    def __len__(self):
        return len(self.indices)

    def _patch_crop(self, image, mask):
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return image, mask
        cr, cc = int(rows.mean()), int(cols.mean())
        if self.jitter > 0:
            cr += random.randint(-self.jitter, self.jitter)
            cc += random.randint(-self.jitter, self.jitter)
        H, W = image.shape[:2]
        half = self.patch_size // 2
        rmin = max(0, cr - half)
        rmax = min(H, cr + half)
        cmin = max(0, cc - half)
        cmax = min(W, cc + half)
        if rmax - rmin < self.patch_size:
            if rmin == 0:
                rmax = min(H, self.patch_size)
            else:
                rmin = max(0, rmax - self.patch_size)
        if cmax - cmin < self.patch_size:
            if cmin == 0:
                cmax = min(W, self.patch_size)
            else:
                cmin = max(0, cmax - self.patch_size)
        return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]

    def __getitem__(self, idx):
        pi, si = self.indices[idx]
        image = _normalize(self.volumes[pi][:, :, si].copy())
        mask = self.segmentations[pi][:, :, si].copy()
        image, mask = self._patch_crop(image, mask)
        return _to_tensors(image, mask, self.transform)


class StandardDataset(Dataset):

    def __init__(self, volumes, segmentations, transform=None, oversample=3):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform

        self.indices = []
        for pi, seg in enumerate(segmentations):
            for si in range(seg.shape[2]):
                has_fg = seg[:, :, si].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((pi, si))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pi, si = self.indices[idx]
        image = _normalize(self.volumes[pi][:, :, si].copy())
        mask = self.segmentations[pi][:, :, si].copy()
        if mask.max() > 0:
            image, mask = _roi_crop(image, mask, padding=50)
        return _to_tensors(image, mask, self.transform)
