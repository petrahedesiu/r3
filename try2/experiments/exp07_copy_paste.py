
import sys
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset import StandardDataset, TwoPointFiveDDataset, CopyPasteDataset
from shared.losses import CompoundLoss
from shared.models import create_model
from shared.training import run_training, compute_class_weights
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp07_copy_paste"
    DESCRIPTION = "Copy-paste augmentation: paste FG regions from donor samples"

    COPY_PASTE_PROB = 0.3


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
    cfg = Config
    print(cfg.summary())

    output_dir = cfg.make_output_dir()
    print(f"Output directory: {output_dir}")

    print("\nLoading data...")
    volumes, segmentations = load_data(cfg.DATA_DIR)
    print(f"Loaded {len(volumes)} patients")

    if len(volumes) == 0:
        print("ERROR: No valid patients found. Exiting.")
        return

    indices = list(range(len(volumes)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )

    train_volumes = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    print(f"Train: {len(train_volumes)} patients, Val: {len(val_volumes)} patients")

    class_weights = compute_class_weights(train_segs, num_classes=cfg.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    train_transform = get_transforms(train=True, img_size=cfg.IMG_SIZE)
    val_transform = get_transforms(train=False, img_size=cfg.IMG_SIZE)

    # build the copy-paste augmented training set
    train_dataset = CopyPasteDataset(
        train_volumes, train_segs,
        transform=train_transform,
        copy_paste_prob=cfg.COPY_PASTE_PROB,
        oversample=cfg.OVERSAMPLE_FACTOR,
    )

    val_dataset = StandardDataset(
        val_volumes, val_segs,
        transform=val_transform,
        oversample=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Copy-paste probability: {cfg.COPY_PASTE_PROB}")
    print(f"Donor pool size: {len(train_dataset.donors)} slices with FG")

    device = torch.device(cfg.DEVICE)
    model = create_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=cfg.ENCODER_NAME,
        attention_type=cfg.ATTENTION_TYPE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = CompoundLoss(
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        class_weights=class_weights.to(device),
        tversky_alpha=cfg.TVERSKY_ALPHA,
        tversky_beta=cfg.TVERSKY_BETA,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        use_boundary=False,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.SCHEDULER_T0,
        T_mult=cfg.SCHEDULER_TMULT,
        eta_min=cfg.SCHEDULER_ETA_MIN,
    )

    config_dict = {
        "experiment_name": cfg.EXPERIMENT_NAME,
        "output_dir": str(output_dir),
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "num_epochs": cfg.NUM_EPOCHS,
        "device": device,
    }

    results = run_training(config_dict)

    results["description"] = cfg.DESCRIPTION
    results["copy_paste_prob"] = cfg.COPY_PASTE_PROB
    results["config"] = {
        k: v for k, v in vars(cfg).items()
        if not k.startswith("_") and not callable(v)
    }

    results_json_path = output_dir / "results.json"
    import json
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nExperiment complete. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
