
import sys
import os
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import load_patient_data, get_labeled_slice_indices, discover_patients


def _normalize(image: np.ndarray) -> np.ndarray:
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32)


def _roi_crop(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _to_tensors(
    image: np.ndarray,
    mask: np.ndarray,
    transform=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if transform is not None:
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)

    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
    else:
        mask = mask.long()

    return image, mask


def _to_tensors_multichannel(
    image: np.ndarray,
    mask: np.ndarray,
    transform=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_hwc = np.transpose(image, (1, 2, 0))

    if transform is not None:
        augmented = transform(image=image_hwc, mask=mask)
        image_hwc = augmented["image"]
        mask = augmented["mask"]

    if not isinstance(image_hwc, torch.Tensor):
        image_t = torch.from_numpy(np.ascontiguousarray(image_hwc)).float()
        if image_t.dim() == 3:
            image_t = image_t.permute(2, 0, 1)
        elif image_t.dim() == 2:
            image_t = image_t.unsqueeze(0)
    else:
        image_t = image_hwc
        if image_t.dim() == 3 and image_t.shape[0] != image.shape[0]:
            pass

    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
    else:
        mask = mask.long()

    return image_t, mask


class StandardDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                has_fg = seg[:, :, slice_idx].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((patient_idx, slice_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)

        if mask.max() > 0:
            image, mask = _roi_crop(image, mask, padding=50)

        return _to_tensors(image, mask, self.transform)


class FGCenteredDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        fg_ratio: float = 0.5,
        patch_size: int = 384,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.fg_ratio = fg_ratio
        self.patch_size = patch_size

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                has_fg = seg[:, :, slice_idx].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((patient_idx, slice_idx))

        self.fg_pixels: Dict[Tuple[int, int], np.ndarray] = {}
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                s = seg[:, :, slice_idx]
                if s.max() > 0:
                    ys, xs = np.where(s > 0)
                    self.fg_pixels[(patient_idx, slice_idx)] = np.stack(
                        [ys, xs], axis=1
                    )

    def __len__(self) -> int:
        return len(self.indices)

    def _crop_patch_around(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center_y: int,
        center_x: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps = self.patch_size
        jitter = ps // 8

        cy = center_y + random.randint(-jitter, jitter)
        cx = center_x + random.randint(-jitter, jitter)

        H, W = image.shape[:2]

        y0 = cy - ps // 2
        x0 = cx - ps // 2
        y1 = y0 + ps
        x1 = x0 + ps

        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - H)
        pad_right = max(0, x1 - W)

        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(H, y1)
        x1 = min(W, x1)

        img_crop = image[y0:y1, x0:x1]
        msk_crop = mask[y0:y1, x0:x1]

        if pad_top or pad_bottom or pad_left or pad_right:
            img_crop = np.pad(
                img_crop,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            msk_crop = np.pad(
                msk_crop,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        return img_crop, msk_crop

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)

        key = (patient_idx, slice_idx)
        use_fg_center = (
            random.random() < self.fg_ratio and key in self.fg_pixels
        )

        if use_fg_center:
            fg_locs = self.fg_pixels[key]
            chosen = fg_locs[random.randint(0, len(fg_locs) - 1)]
            image, mask = self._crop_patch_around(
                image, mask, int(chosen[0]), int(chosen[1])
            )
        else:
            if mask.max() > 0:
                image, mask = _roi_crop(image, mask, padding=50)

        return _to_tensors(image, mask, self.transform)


class TwoPointFiveDDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        n_adjacent: int = 2,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.n_adjacent = n_adjacent
        self.n_channels = 2 * n_adjacent + 1

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                has_fg = seg[:, :, slice_idx].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((patient_idx, slice_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        vol = self.volumes[patient_idx]
        seg = self.segmentations[patient_idx]
        H, W, D = vol.shape

        channels: List[np.ndarray] = []
        # loop over adjacent slices
        for offset in range(-self.n_adjacent, self.n_adjacent + 1):
            s = slice_idx + offset
            if 0 <= s < D:
                sl = vol[:, :, s].copy().astype(np.float32)
            else:
                sl = np.zeros((H, W), dtype=np.float32)
            channels.append(sl)

        image = np.stack(channels, axis=0)

        for c in range(image.shape[0]):
            image[c] = _normalize(image[c])

        mask = seg[:, :, slice_idx].copy()

        if mask.max() > 0:
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                padding = 50
                rmin = max(0, rmin - padding)
                rmax = min(H, rmax + padding + 1)
                cmin = max(0, cmin - padding)
                cmax = min(W, cmax + padding + 1)
                image = image[:, rmin:rmax, cmin:cmax]
                mask = mask[rmin:rmax, cmin:cmax]

        return _to_tensors_multichannel(image, mask, self.transform)


class CopyPasteDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        copy_paste_prob: float = 0.3,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.copy_paste_prob = copy_paste_prob

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                has_fg = seg[:, :, slice_idx].max() > 0
                repeats = oversample if has_fg else 1
                for _ in range(repeats):
                    self.indices.append((patient_idx, slice_idx))

        self.donors: List[Tuple[int, int, int, int, int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                s = seg[:, :, slice_idx]
                if s.max() > 0:
                    rows = np.any(s > 0, axis=1)
                    cols = np.any(s > 0, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        self.donors.append(
                            (patient_idx, slice_idx, rmin, rmax + 1, cmin, cmax + 1)
                        )

    def __len__(self) -> int:
        return len(self.indices)

    def _apply_copy_paste(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.donors:
            return image, mask

        donor = self.donors[random.randint(0, len(self.donors) - 1)]
        d_pidx, d_sidx, d_rmin, d_rmax, d_cmin, d_cmax = donor

        donor_img = self.volumes[d_pidx][:, :, d_sidx]
        donor_msk = self.segmentations[d_pidx][:, :, d_sidx]

        patch_img = donor_img[d_rmin:d_rmax, d_cmin:d_cmax].copy().astype(np.float32)
        patch_msk = donor_msk[d_rmin:d_rmax, d_cmin:d_cmax].copy()

        patch_img = _normalize(patch_img)

        ph, pw = patch_img.shape[:2]
        H, W = image.shape[:2]

        if ph > H or pw > W:
            return image, mask

        max_offset = max(ph, pw)
        paste_r = random.randint(
            max(0, d_rmin - max_offset),
            min(H - ph, d_rmin + max_offset),
        ) if d_rmin + max_offset < H and d_rmin - max_offset >= 0 else random.randint(0, H - ph)
        paste_c = random.randint(
            max(0, d_cmin - max_offset),
            min(W - pw, d_cmin + max_offset),
        ) if d_cmin + max_offset < W and d_cmin - max_offset >= 0 else random.randint(0, W - pw)

        fg_mask = patch_msk > 0
        region_img = image[paste_r : paste_r + ph, paste_c : paste_c + pw]
        region_msk = mask[paste_r : paste_r + ph, paste_c : paste_c + pw]

        region_img[fg_mask] = patch_img[fg_mask]
        region_msk[fg_mask] = patch_msk[fg_mask]

        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)

        if random.random() < self.copy_paste_prob:
            image, mask = self._apply_copy_paste(image, mask)

        if mask.max() > 0:
            image, mask = _roi_crop(image, mask, padding=50)

        return _to_tensors(image, mask, self.transform)
