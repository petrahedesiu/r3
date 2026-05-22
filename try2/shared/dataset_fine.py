
import sys
import os
import random
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import load_patient_data, get_labeled_slice_indices, discover_patients

from shared.dataset import _normalize, _to_tensors


class FineBBoxCropDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        padding: int = 50,
        jitter: int = 15,
        oversample: int = 3,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform
        self.padding = padding
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

    def _bbox_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any() or not cols.any():
            return image, mask

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        H, W = image.shape[:2]

        if self.jitter > 0:
            pad_top = self.padding + random.randint(-self.jitter, self.jitter)
            pad_bot = self.padding + random.randint(-self.jitter, self.jitter)
            padLeft = self.padding + random.randint(-self.jitter, self.jitter)
            pad_right = self.padding + random.randint(-self.jitter, self.jitter)
        else:
            pad_top = pad_bot = padLeft = pad_right = self.padding

        pad_top = max(0, pad_top)
        pad_bot = max(0, pad_bot)
        padLeft = max(0, padLeft)
        pad_right = max(0, pad_right)

        rmin = max(0, rmin - pad_top)
        rmax = min(H, rmax + pad_bot + 1)
        cmin = max(0, cmin - padLeft)
        cmax = min(W, cmax + pad_right + 1)

        return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.indices[idx]

        image = self.volumes[patient_idx][:, :, slice_idx].copy()
        mask = self.segmentations[patient_idx][:, :, slice_idx].copy()

        image = _normalize(image)
        image, mask = self._bbox_crop(image, mask)

        return _to_tensors(image, mask, self.transform)
