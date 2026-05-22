
import sys, os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import load_patient_data, get_labeled_slice_indices, discover_patients

from shared.dataset import _normalize, _to_tensors


class FinePatchDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        patch_size: int = 128,
        jitter: int = 10,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.patch_size = patch_size
        self.jitter = jitter

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            n_slices = seg.shape[2]
            for slice_idx in range(n_slices):
                if seg[:, :, slice_idx].max() > 0:
                    for _ in range(oversample):
                        self.indices.append((patient_idx, slice_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def _patch_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)
        image, mask = self._patch_crop(image, mask)
        return _to_tensors(image, mask, self.transform)
