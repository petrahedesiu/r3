
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset import StandardDataset
from shared.losses import CompoundLoss
from shared.models import create_model
from shared.training import run_training, compute_class_weights
from shared.postprocessing import test_time_augmentation, connected_component_filter, optimize_threshold, apply_threshold
from shared.metrics import compute_all_metrics
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp10_5fold_cv"
    DESCRIPTION = "3-fold cross-validation for reliable metrics with small dataset"
    NUM_EPOCHS = 10

N_SPLITS = 3


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


def train_fold(fold_idx, train_idx, val_idx, volumes, segmentations, output_dir):
    fold_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"FOLD {fold_idx + 1}/{N_SPLITS}")
    print(f"  Train patients: {len(train_idx)}  Val patients: {len(val_idx)}")
    print(f"  Output: {fold_dir}")
    print(f"{'=' * 70}")

    train_volumes = [volumes[i] for i in train_idx]
    train_segs = [segmentations[i] for i in train_idx]
    val_volumes = [volumes[i] for i in val_idx]
    val_segs = [segmentations[i] for i in val_idx]

    class_weights = compute_class_weights(train_segs, num_classes=Config.NUM_CLASSES)
    print(f"  Class weights: {class_weights}")

    train_dataset = StandardDataset(
        train_volumes, train_segs,
        transform=get_transforms(train=True, img_size=Config.IMG_SIZE),
        oversample=Config.OVERSAMPLE_FACTOR,
    )
    val_dataset = StandardDataset(
        val_volumes, val_segs,
        transform=get_transforms(train=False, img_size=Config.IMG_SIZE),
        oversample=1,
    )

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

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

    criterion = CompoundLoss(
        focal_weight=0.35,
        tversky_weight=0.35,
        lovasz_weight=0.30,
        class_weights=class_weights.to(device),
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
        "experiment_name": f"{Config.EXPERIMENT_NAME}_fold{fold_idx}",
        "output_dir": str(fold_dir),
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "num_epochs": Config.NUM_EPOCHS,
        "device": device,
    }

    fold_results = run_training(config_dict)

    fold_results["fold"] = fold_idx
    fold_results["train_patient_indices"] = train_idx.tolist()
    fold_results["val_patient_indices"] = val_idx.tolist()
    fold_results["num_train_patients"] = len(train_idx)
    fold_results["num_val_patients"] = len(val_idx)

    return fold_results


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

    if len(volumes) < N_SPLITS:
        print(f"WARNING: Only {len(volumes)} patients for {N_SPLITS}-fold CV. "
              f"Some folds may have very few validation samples.")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED)
    patient_indices = np.arange(len(volumes))

    all_fold_results = []
    fold_dices = []
    fold_recalls = []
    fold_precisions = []
    fold_f2s = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_indices)):
        fold_results = train_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            volumes=volumes,
            segmentations=segmentations,
            output_dir=output_dir,
        )

        all_fold_results.append(fold_results)

        fold_dices.append(fold_results["best_val_dice"])
        fold_recalls.append(fold_results["best_val_recall"])
        fold_precisions.append(fold_results["best_val_precision"])

        p = fold_results["best_val_precision"]
        r = fold_results["best_val_recall"]
        beta_sq = 4.0
        if (beta_sq * p + r) > 1e-7:
            f2 = (1.0 + beta_sq) * p * r / (beta_sq * p + r)
        else:
            f2 = 0.0
        fold_f2s.append(f2)

        print(f"\n>>> Fold {fold_idx + 1}/{N_SPLITS} complete: "
              f"Dice={fold_results['best_val_dice']:.4f}  "
              f"Recall={fold_results['best_val_recall']:.4f}  "
              f"Precision={fold_results['best_val_precision']:.4f}  "
              f"F2={f2:.4f}")

    fold_dices = np.array(fold_dices)
    fold_recalls = np.array(fold_recalls)
    fold_precisions = np.array(fold_precisions)
    fold_f2s = np.array(fold_f2s)

    print("\n" + "=" * 70)
    print(f"{N_SPLITS}-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    header = f"{'Fold':>6s} {'Dice':>8s} {'Recall':>8s} {'Precision':>10s} {'F2':>8s}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i in range(N_SPLITS):
        print(
            f"{i + 1:>6d} "
            f"{fold_dices[i]:>8.4f} "
            f"{fold_recalls[i]:>8.4f} "
            f"{fold_precisions[i]:>10.4f} "
            f"{fold_f2s[i]:>8.4f}"
        )

    print(sep)
    print(
        f"{'Mean':>6s} "
        f"{fold_dices.mean():>8.4f} "
        f"{fold_recalls.mean():>8.4f} "
        f"{fold_precisions.mean():>10.4f} "
        f"{fold_f2s.mean():>8.4f}"
    )
    print(
        f"{'Std':>6s} "
        f"{fold_dices.std():>8.4f} "
        f"{fold_recalls.std():>8.4f} "
        f"{fold_precisions.std():>10.4f} "
        f"{fold_f2s.std():>8.4f}"
    )
    print("=" * 70)

    summary = {
        "experiment_name": Config.EXPERIMENT_NAME,
        "description": Config.DESCRIPTION,
        "n_splits": N_SPLITS,
        "num_patients": len(volumes),
        "epochs_per_fold": Config.NUM_EPOCHS,
        "output_dir": str(output_dir),
        "aggregate": {
            "dice_mean": float(fold_dices.mean()),
            "dice_std": float(fold_dices.std()),
            "recall_mean": float(fold_recalls.mean()),
            "recall_std": float(fold_recalls.std()),
            "precision_mean": float(fold_precisions.mean()),
            "precision_std": float(fold_precisions.std()),
            "f2_mean": float(fold_f2s.mean()),
            "f2_std": float(fold_f2s.std()),
        },
        "per_fold": [],
    }

    for i, fr in enumerate(all_fold_results):
        fold_entry = {
            "fold": i,
            "best_epoch": fr["best_epoch"],
            "best_val_dice": fr["best_val_dice"],
            "best_val_recall": fr["best_val_recall"],
            "best_val_precision": fr["best_val_precision"],
            "f2": float(fold_f2s[i]),
            "best_class_dices": fr.get("best_class_dices", {}),
            "best_class_recalls": fr.get("best_class_recalls", {}),
            "train_patient_indices": fr.get("train_patient_indices", []),
            "val_patient_indices": fr.get("val_patient_indices", []),
            "fold_output_dir": fr["output_dir"],
        }
        summary["per_fold"].append(fold_entry)

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved aggregate results -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {Config.EXPERIMENT_NAME}")
    print(f"  {Config.DESCRIPTION}")
    print(f"  Patients         : {len(volumes)}")
    print(f"  Folds            : {N_SPLITS}")
    print(f"  Epochs per fold  : {Config.NUM_EPOCHS}")
    print(f"  Dice             : {fold_dices.mean():.4f} +/- {fold_dices.std():.4f}")
    print(f"  Recall           : {fold_recalls.mean():.4f} +/- {fold_recalls.std():.4f}")
    print(f"  Precision        : {fold_precisions.mean():.4f} +/- {fold_precisions.std():.4f}")
    print(f"  F2               : {fold_f2s.mean():.4f} +/- {fold_f2s.std():.4f}")
    print(f"  Output           : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
