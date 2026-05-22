
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 100, 255],
], dtype=np.uint8)


def create_overlay(ct_slice, mask, alpha=0.4):
    ct = ct_slice.astype(np.float32)
    mn, mx = ct.min(), ct.max()
    if mx - mn > 1e-8:
        ct = (ct - mn) / (mx - mn)
    ct_u8 = (ct * 255).astype(np.uint8)

    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1)

    for cls in range(1, len(CLASS_COLORS)):
        fg = (mask == cls)
        if fg.any():
            color = CLASS_COLORS[cls]
            rgb[fg] = (
                (1 - alpha) * rgb[fg].astype(np.float32)
                + alpha * color.astype(np.float32)
            ).astype(np.uint8)
    return rgb


def save_comparison_figure(ct_slice, gt_mask, pred_mask, save_path, title=""):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(create_overlay(ct_slice, gt_mask))
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(create_overlay(ct_slice, pred_mask))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2)
    axes[3].set_title("GT Mask")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_montage(images, save_path, ncols=5, titles=None, figsize_per=3):
    n = len(images)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows))
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        if i < n:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
