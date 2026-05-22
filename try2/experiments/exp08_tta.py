
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp08_tta"
    DESCRIPTION = "Test-time augmentation with max merge mode to boost recall"


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


def evaluate_with_tta(model, val_loader, device, num_classes=3):
    model.eval()

    all_preds_standard = []
    all_preds_tta_max = []
    all_preds_tta_mean = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="TTA evaluation"):
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            bs = images.shape[0]

            logits = model(images)
            preds_standard = logits.argmax(dim=1).cpu().numpy()

            for i in range(bs):
                single_image = images[i:i+1]

                pred_max, _ = test_time_augmentation(
                    model, single_image, device, merge_mode="max"
                )
                all_preds_tta_max.append(pred_max.numpy())

                pred_mean, _ = test_time_augmentation(
                    model, single_image, device, merge_mode="mean"
                )
                all_preds_tta_mean.append(pred_mean.numpy())

            all_preds_standard.append(preds_standard)
            all_targets.append(masks.numpy())

    all_preds_standard = np.concatenate(all_preds_standard, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    all_preds_tta_max = np.stack(all_preds_tta_max, axis=0)
    all_preds_tta_mean = np.stack(all_preds_tta_mean, axis=0)

    metrics_standard = compute_all_metrics(all_preds_standard, all_targets, num_classes)
    metrics_tta_max = compute_all_metrics(all_preds_tta_max, all_targets, num_classes)
    metrics_tta_mean = compute_all_metrics(all_preds_tta_mean, all_targets, num_classes)

    return {
        "standard": metrics_standard,
        "tta_max": metrics_tta_max,
        "tta_mean": metrics_tta_mean,
    }


def print_comparison_table(tta_results):
    modes = ["standard", "tta_max", "tta_mean"]
    labels = ["Standard", "TTA (max)", "TTA (mean)"]

    header = f"{'Mode':<15s} {'Dice':>8s} {'Recall':>8s} {'Precision':>10s} {'F2':>8s}"
    sep = "-" * len(header)

    print("\n" + "=" * 60)
    print("TTA COMPARISON -- Aggregate Validation Metrics")
    print("=" * 60)
    print(header)
    print(sep)

    for mode, label in zip(modes, labels):
        m = tta_results[mode]
        print(
            f"{label:<15s} "
            f"{m['mean_fg_dice']:>8.4f} "
            f"{m['mean_fg_recall']:>8.4f} "
            f"{m['mean_fg_precision']:>10.4f} "
            f"{m['mean_fg_f2']:>8.4f}"
        )

    print(sep)

    for mode, label in zip(modes, labels):
        m = tta_results[mode]
        print(f"\n  {label} per-class:")
        for c in sorted(m['dice_per_class'].keys()):
            print(
                f"    Class {c}: "
                f"Dice={m['dice_per_class'][c]:.4f}  "
                f"Recall={m['recall_per_class'][c]:.4f}  "
                f"Precision={m['precision_per_class'][c]:.4f}  "
                f"F2={m['f2_per_class'][c]:.4f}"
            )

    print("=" * 60)


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

    class_weights = compute_class_weights(train_segs, num_classes=Config.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

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
    print("POST-TRAINING: Test-Time Augmentation Evaluation")
    print("=" * 70)

    best_model_path = output_dir / "best_model.pth"
    print(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tta_results = evaluate_with_tta(model, val_loader, device, num_classes=Config.NUM_CLASSES)

    print_comparison_table(tta_results)

    def _serialize_metrics(m):
        return {
            "mean_fg_dice": m["mean_fg_dice"],
            "mean_fg_recall": m["mean_fg_recall"],
            "mean_fg_precision": m["mean_fg_precision"],
            "mean_fg_f2": m["mean_fg_f2"],
            "dice_per_class": {str(k): v for k, v in m["dice_per_class"].items()},
            "recall_per_class": {str(k): v for k, v in m["recall_per_class"].items()},
            "precision_per_class": {str(k): v for k, v in m["precision_per_class"].items()},
            "f2_per_class": {str(k): v for k, v in m["f2_per_class"].items()},
        }

    results["tta_comparison"] = {
        "standard": _serialize_metrics(tta_results["standard"]),
        "tta_max": _serialize_metrics(tta_results["tta_max"]),
        "tta_mean": _serialize_metrics(tta_results["tta_mean"]),
    }

    best_mode = max(
        tta_results.keys(),
        key=lambda mode: tta_results[mode]["mean_fg_f2"],
    )
    results["best_tta_mode"] = best_mode
    results["best_tta_f2"] = tta_results[best_mode]["mean_fg_f2"]

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results with TTA metrics -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {Config.EXPERIMENT_NAME}")
    print(f"  {Config.DESCRIPTION}")
    print(f"  Best val Dice   : {results['best_val_dice']:.4f}  (epoch {results['best_epoch']})")
    print(f"  Best val Recall : {results['best_val_recall']:.4f}")
    print(f"  Best val Prec   : {results['best_val_precision']:.4f}")
    print(f"  Best TTA mode   : {best_mode} (F2 = {results['best_tta_f2']:.4f})")
    print(f"  Output          : {results['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
