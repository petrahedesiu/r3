
import numpy as np
import torch
from torch.utils.data import Dataset

from .windowing import multi_window


class VolumetricPatchDataset(Dataset):

    def __init__(
        self,
        volumes,
        segmentations,
        patch_size=(96, 96, 32),
        jitter_hw=15,
        jitter_d=5,
        oversample=8,
        augment=True,
    ):
        self.volumes = volumes
        self.segmentations = segmentations
        self.patch_size = patch_size
        self.jitter_hw = jitter_hw
        self.jitter_d = jitter_d
        self.oversample = oversample
        self.augment = augment

        self.centroids = []
        for seg in segmentations:
            fgMask = seg > 0
            if fgMask.any():
                coords = np.argwhere(fgMask)
                centroid = coords.mean(axis=0).astype(int)
                self.centroids.append(centroid)
            else:
                self.centroids.append(
                    np.array([seg.shape[0] // 2, seg.shape[1] // 2, seg.shape[2] // 2])
                )

    def __len__(self):
        return len(self.volumes) * self.oversample

    def _extract_patch(self, volume, seg, centroid):
        H, W, D = volume.shape
        pH, pW, pD = self.patch_size

        ch = centroid[0] + np.random.randint(-self.jitter_hw, self.jitter_hw + 1)
        cw = centroid[1] + np.random.randint(-self.jitter_hw, self.jitter_hw+1)
        cd = centroid[2] + np.random.randint(-self.jitter_d, self.jitter_d + 1)

        h_start = ch - pH // 2
        w_start = cw - pW // 2
        d_start = cd - pD // 2

        h_end = h_start + pH
        w_end = w_start + pW
        d_end = d_start + pD

        src_h0 = max(0, h_start)
        src_w0 = max(0, w_start)
        src_d0 = max(0, d_start)
        src_h1 = min(H, h_end)
        src_w1 = min(W, w_end)
        src_d1 = min(D, d_end)

        dst_h0 = src_h0 - h_start
        dst_w0 = src_w0 - w_start
        dst_d0 = src_d0 - d_start
        dst_h1 = dst_h0 + (src_h1 - src_h0)
        dst_w1 = dst_w0 + (src_w1 - src_w0)
        dst_d1 = dst_d0 + (src_d1 - src_d0)

        vol_patch = np.zeros((pH, pW, pD), dtype=volume.dtype)
        seg_patch = np.zeros((pH, pW, pD), dtype=seg.dtype)

        vol_patch[dst_h0:dst_h1, dst_w0:dst_w1, dst_d0:dst_d1] = \
            volume[src_h0:src_h1, src_w0:src_w1, src_d0:src_d1]
        seg_patch[dst_h0:dst_h1, dst_w0:dst_w1, dst_d0:dst_d1] = \
            seg[src_h0:src_h1, src_w0:src_w1, src_d0:src_d1]

        return vol_patch, seg_patch

    def _augment_3d(self, volume, seg):
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=0).copy()
            seg = np.flip(seg, axis=0).copy()
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()
            seg = np.flip(seg, axis=1).copy()
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()
            seg = np.flip(seg, axis=2).copy()

        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k=k, axes=(0, 1)).copy()
            seg = np.rot90(seg, k=k, axes=(0, 1)).copy()
        return volume, seg

    def __getitem__(self, idx):
        patient_idx = idx % len(self.volumes)
        volume = self.volumes[patient_idx]
        seg = self.segmentations[patient_idx]
        centroid = self.centroids[patient_idx]

        vol_patch, seg_patch = self._extract_patch(volume, seg, centroid)
        vol_windowed = multi_window(vol_patch)

        if self.augment:
            augVol = np.empty_like(vol_windowed)
            flip0 = np.random.random() > 0.5
            flip1 = np.random.random() > 0.5
            flip2 = np.random.random() > 0.5
            rot_k = np.random.randint(0, 4)

            for c in range(3):
                ch = vol_windowed[c]
                if flip0:
                    ch = np.flip(ch, axis=0)
                if flip1:
                    ch = np.flip(ch, axis=1)
                if flip2:
                    ch = np.flip(ch, axis=2)
                if rot_k > 0:
                    ch = np.rot90(ch, k=rot_k, axes=(0, 1))
                augVol[c] = ch.copy()

            seg_aug = seg_patch
            if flip0:
                seg_aug = np.flip(seg_aug, axis=0)
            if flip1:
                seg_aug = np.flip(seg_aug, axis=1)
            if flip2:
                seg_aug = np.flip(seg_aug, axis=2)
            if rot_k > 0:
                seg_aug = np.rot90(seg_aug, k=rot_k, axes=(0, 1))

            vol_windowed = augVol
            seg_patch = seg_aug.copy()

        vol_out = np.transpose(vol_windowed, (0, 3, 1, 2))
        seg_out = np.transpose(seg_patch, (2, 0, 1))

        image = torch.from_numpy(vol_out.copy()).float()
        mask = torch.from_numpy(seg_out.copy()).long()
        return image, mask
