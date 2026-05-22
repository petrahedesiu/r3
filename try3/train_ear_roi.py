
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import config
from shared.data import load_all_patients, patient_split
from shared.dataset_roi import EarROIDataset
from shared.atlas import crop_roi
from shared.models import create_model
from shared.losses import CompoundLoss
from shared.training import run_training

IMG_SIZE = 256
NUM_EPOCHS = 35
BATCH_SIZE = 8
OVERSAMPLE = 4


def _roi_coverage(volumes, segmentations):
    inside = {1: 0, 2: 0}
    total = {1: 0, 2: 0}
    for vol, seg in zip(volumes, segmentations):
        D = seg.shape[2]
        for cid, side in ((1, "L"), (2, "R")):
            for si in range(D):
                m = (seg[:, :, si] == cid)
                n = int(m.sum())
                if n == 0:
                    continue
                total[cid] += n
                inside[cid] += int(crop_roi(m, side).sum())
    return {c: (inside[c] / total[c] if total[c] else 0.0) for c in (1, 2)}


def main():
    print("=" * 70)
    print("PILOT: UNIFIED EAR ROI SPECIALIST (binary, ROI-cropped, mirror-aug)")
    print("=" * 70)

    volumes, segmentations, infos = load_all_patients()
    n_patients = len(volumes)
    print(f"Loaded {n_patients} patients")

    cov = _roi_coverage(volumes, segmentations)
    print(f"ROI coverage of GT foreground -- AEAL: {cov[1]*100:.2f}%, "
          f"AEAR: {cov[2]*100:.2f}%  (recall ceiling)")
    if min(cov.values()) < 0.97:
        print("WARNING: ROI box clips some foreground -- consider widening atlas box.")

    train_idx, val_idx = patient_split(n_patients)
    print(f"Train: {len(train_idx)} patients, Val: {len(val_idx)} patients")

    train_vols = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    valVols = [volumes[i] for i in val_idx]
    valSegs = [segmentations[i] for i in val_idx]

    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.05, rotate_limit=8,
                           border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.RandomGamma(p=0.2),
    ])
    val_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

    train_ds = EarROIDataset(train_vols, train_segs, transform=train_transform,
                             oversample=OVERSAMPLE, is_train=True)
    val_ds = EarROIDataset(valVols, valSegs, transform=val_transform,
                           oversample=1, is_train=False)
    print(f"Train samples: {len(train_ds)}  (vs ~150 before -- 2x mirror + ROI)")
    print(f"Val samples  : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    device = torch.device(config.DEVICE)
    model = create_model(in_channels=1, num_classes=2).to(device)

    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    criterion = CompoundLoss(
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        boundary_weight=0.0,
        class_weights=class_weights.to(device),
        tversky_alpha=0.2,
        tversky_beta=0.8,
        use_boundary=False,
        num_classes=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_BASE) / "ear_roi" / timestamp

    results = run_training({
        "experiment_name": "ear_roi_specialist",
        "output_dir": str(output_dir),
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "num_epochs": NUM_EPOCHS,
        "device": device,
        "num_classes": 2,
    })

    print(f"\nPilot training complete. Best binary ROI dice: "
          f"{results['best_val_dice']:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pth")
    print(f"Run eval with:  ./venv/bin/python3 try3/eval_ear_roi.py "
          f"{output_dir}/best_model.pth")
    return results


if __name__ == "__main__":
    main()
