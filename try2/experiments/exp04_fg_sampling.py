
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset import FGCenteredDataset
from shared.losses import CompoundLoss
from shared.models import create_model
from shared.training import run_training, compute_class_weights
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp04_fg_sampling"
    DESCRIPTION = "Foreground-centered patch sampling (50% patches centered on FG pixels)"


def get_transforms(train=True, img_size=384):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(img_size, img_size), ToTensorV2()])


def load_data(data_dir):
    patients = discover_patients(data_dir)
    volumes, segmentations = [], []
    for p in tqdm(patients, desc="Loading patients"):
        try:
            vol, seg, meta = load_patient_data(p['dicom_dir'], p['nrrd_path'], verbose=False)
            if meta['alignment_success']:
                labeled = get_labeled_slice_indices(seg)
                if len(labeled) >= 2:
                    volumes.append(vol)
                    segmentations.append(seg)
        except Exception:
            pass
    return volumes, segmentations


def main():
    print(Config.summary())

    output_dir = Config.make_output_dir()
    print(f"Output directory: {output_dir}")

    print("\nLoading data...")
    volumes, segmentations = load_data(Config.DATA_DIR)
    print(f"Loaded {len(volumes)} patients")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Check DATA_DIR.")
        return

    indices = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        indices, test_size=Config.VAL_SPLIT, random_state=Config.RANDOM_SEED
    )

    train_volumes = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    print(f"Train: {len(train_volumes)} patients, Val: {len(val_volumes)} patients")

    classWeights = compute_class_weights(train_segs, num_classes=Config.NUM_CLASSES)
    print(f"Class weights: {classWeights}")

    train_dataset = FGCenteredDataset(
        train_volumes, train_segs,
        transform=get_transforms(train=True, img_size=Config.IMG_SIZE),
        fg_ratio=0.5,
        patch_size=Config.IMG_SIZE,
        oversample=Config.OVERSAMPLE_FACTOR,
    )

    from shared.dataset import StandardDataset
    val_dataset = StandardDataset(
        val_volumes, val_segs,
        transform=get_transforms(train=False, img_size=Config.IMG_SIZE),
        oversample=1,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device(Config.DEVICE)
    model = create_model(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        encoder_name=Config.ENCODER_NAME,
        attention_type=Config.ATTENTION_TYPE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = CompoundLoss(
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        class_weights=classWeights.to(device),
        tversky_alpha=Config.TVERSKY_ALPHA,
        tversky_beta=Config.TVERSKY_BETA,
        use_boundary=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.SCHEDULER_T0,
        T_mult=Config.SCHEDULER_TMULT,
        eta_min=Config.SCHEDULER_ETA_MIN,
    )

    config_dict = {
        "experiment_name": Config.EXPERIMENT_NAME,
        "output_dir": str(output_dir),
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "num_epochs": Config.NUM_EPOCHS,
        "device": device,
    }

    results = run_training(config_dict)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {Config.EXPERIMENT_NAME}")
    print(f"  {Config.DESCRIPTION}")
    print(f"  FG sampling ratio : 0.5 (50% patches centered on foreground)")
    print(f"  Patch size        : {Config.IMG_SIZE}")
    print(f"  Best val Dice     : {results['best_val_dice']:.4f}  (epoch {results['best_epoch']})")
    print(f"  Best val Recall   : {results['best_val_recall']:.4f}")
    print(f"  Best val Prec     : {results['best_val_precision']:.4f}")
    print(f"  Output            : {results['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
