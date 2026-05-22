
import numpy as np
from torch.utils.data import Dataset

from .datasets import _to_tensors
from .atlas import crop_roi
from .windowing import bone_window


class EarROIDataset(Dataset):

    def __init__(self, volumes, segmentations, transform=None,
                 oversample=4, neg_band=20, is_train=True):
        super().__init__()
        self.volumes = volumes
        self.segmentations = segmentations
        self.transform = transform

        self.indices = []
        for pi, seg in enumerate(segmentations):
            D = seg.shape[2]
            fg_slices = {1: set(), 2: set()}
            for si in range(D):
                s = seg[:, :, si]
                if (s == 1).any():
                    fg_slices[1].add(si)
                if (s == 2).any():
                    fg_slices[2].add(si)

            for cid, side in ((1, "L"), (2, 'R')):
                fg = fg_slices[cid]
                neg = set()
                for f in fg:
                    lo = max(0, f - neg_band)
                    hi = min(D, f + neg_band + 1)
                    for d in range(lo, hi):
                        if d not in fg:
                            neg.add(d)
                reps = oversample if is_train else 1
                for si in sorted(fg):
                    for _ in range(reps):
                        self.indices.append((pi, si, side, cid))
                for si in sorted(neg):
                    self.indices.append((pi, si, side, cid))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pi, si, side, cid = self.indices[idx]
        image = self.volumes[pi][:, :, si].astype(np.float32)
        mask = (self.segmentations[pi][:, :, si] == cid).astype(np.int64)
        image = np.ascontiguousarray(crop_roi(image, side))
        mask = np.ascontiguousarray(crop_roi(mask, side))
        image = bone_window(image)
        return _to_tensors(image, mask, self.transform)
