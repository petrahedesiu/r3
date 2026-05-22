
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import load_patient_data, get_labeled_slice_indices, discover_patients

from shared.dataset import _normalize, _to_tensors


class CoarseFullSliceDataset(Dataset):

    def __init__(
        self,
        volumes: List[np.ndarray],
        segmentations: List[np.ndarray],
        transform=None,
        oversample: int = 5,
    ):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform

        self.indices: List[Tuple[int, int]] = []
        for patient_idx, seg in enumerate(segmentations):
            nSlices = seg.shape[2]
            for slice_idx in range(nSlices):
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
        # binarize mask
        mask = (mask > 0).astype(np.int64)

        return _to_tensors(image, mask, self.transform)
