
import numpy as np


def apply_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width/2.0
    hi = center + width / 2.0
    clipped = np.clip(volume, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


BONE_WL = 700.0
BONE_WW = 4000.0


def bone_window(image: np.ndarray) -> np.ndarray:
    return apply_window(image, center=BONE_WL, width=BONE_WW)


def multi_window(volume: np.ndarray) -> np.ndarray:
    # three different windows stacked into channels
    bone = apply_window(volume, center=700, width=4000)
    soft = apply_window(volume, center=50, width=400)
    narrow = apply_window(volume, center=40, width=80)
    return np.stack([bone, soft, narrow], axis=0)
