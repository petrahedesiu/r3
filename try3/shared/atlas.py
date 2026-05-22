
import numpy as np

ROI_ROW = (0.30, 0.72)
ROI_COL = (0.13, 0.52)
ROI_DEPTH = (0.45, 0.97)


def roi_bounds(h, w):
    r0 = int(round(ROI_ROW[0] * h))
    r1 = int(round(ROI_ROW[1]*h))
    c0 = int(round(ROI_COL[0] * w))
    c1 = int(round(ROI_COL[1] * w))
    return r0, r1, c0, c1


def crop_roi(slice2d, side):
    if side == 'R':
        slice2d = slice2d[:, ::-1]
    h, w = slice2d.shape[:2]
    r0, r1, c0, c1 = roi_bounds(h, w)
    return slice2d[r0:r1, c0:c1]


def place_roi(pred_crop, h, w, side):
    full = np.zeros((h, w), dtype=pred_crop.dtype)
    r0, r1, c0, c1 = roi_bounds(h, w)
    full[r0:r1, c0:c1] = pred_crop
    if side == "R":
        full = full[:, ::-1]
    return np.ascontiguousarray(full)


def in_depth_band(si, depth):
    # compute the fractional depth position
    frac = si / max(depth - 1, 1)
    return ROI_DEPTH[0] <= frac <= ROI_DEPTH[1]
