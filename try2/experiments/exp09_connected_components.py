
import sys
import os
import json
from itertools import product

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

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class Config(ExperimentConfig):
    EXPERIMENT_NAME = "exp09_connected_components"
    DESCRIPTION = "Post-processing: connected component filtering with size priors"


MIN_SIZES = [1, 3, 5, 10]
MAX_SIZES = [100, 500, 1000, 5000]


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


def collect_predictions(model, val_loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Collecting predictions"):
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()
            masks = masks.long()

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_preds, all_targets


def evaluate_cc_grid(all_preds, all_targets, num_classes=3):
    baseline_metrics = compute_all_metrics(all_preds, all_targets, num_classes)

    grid_results = []
    combos = list(product(MIN_SIZES, MAX_SIZES))

    for min_size, max_size in tqdm(combos, desc="CC grid search"):
        filteredPreds = np.stack([
            connected_component_filter(all_preds[i], min_size=min_size, max_size=max_size)
            for i in range(all_preds.shape[0])
        ], axis=0)

        metrics = compute_all_metrics(filteredPreds, all_targets, num_classes)

        entry = {
            "min_size": min_size,
            "max_size": max_size,
            "mean_fg_dice": metrics["mean_fg_dice"],
            "mean_fg_recall": metrics["mean_fg_recall"],
            "mean_fg_precision": metrics["mean_fg_precision"],
            "mean_fg_f2": metrics["mean_fg_f2"],
            "dice_per_class": {str(k): v for k, v in metrics["dice_per_class"].items()},
            "recall_per_class": {str(k): v for k, v in metrics["recall_per_class"].items()},
            "precision_per_class": {str(k): v for k, v in metrics["precision_per_class"].items()},
            "f2_per_class": {str(k): v for k, v in metrics["f2_per_class"].items()},
        }
        grid_results.append(entry)

    return grid_results, baseline_metrics


def print_cc_comparison_table(grid_results, baseline_metrics):
    print("\n" + "=" * 80)
    print("CONNECTED COMPONENT FILTERING -- Grid Search Results")
    print("=" * 80)

    header = (
        f"{'min_size':>8s} {'max_size':>8s} | "
        f"{'Dice':>8s} {'Recall':>8s} {'Precision':>10s} {'F2':>8s}"
    )
    sep = "-" * len(header)

    print(f"\n{'Baseline (no CC filter)':>20s}   | "
          f"{baseline_metrics['mean_fg_dice']:>8.4f} "
          f"{baseline_metrics['mean_fg_recall']:>8.4f} "
          f"{baseline_metrics['mean_fg_precision']:>10.4f} "
          f"{baseline_metrics['mean_fg_f2']:>8.4f}")
    print(sep)
    print(header)
    print(sep)

    sortedResults = sorted(grid_results, key=lambda r: r["mean_fg_f2"], reverse=True)

    for entry in sortedResults:
        print(
            f"{entry['min_size']:>8d} {entry['max_size']:>8d} | "
            f"{entry['mean_fg_dice']:>8.4f} "
            f"{entry['mean_fg_recall']:>8.4f} "
            f"{entry['mean_fg_precision']:>10.4f} "
            f"{entry['mean_fg_f2']:>8.4f}"
        )

    print(sep)

    best = sortedResults[0]
    print(f"\n>>> Best CC params: min_size={best['min_size']}, max_size={best['max_size']}")
    print(f"    F2={best['mean_fg_f2']:.4f}  Dice={best['mean_fg_dice']:.4f}  "
          f"Recall={best['mean_fg_recall']:.4f}  Precision={best['mean_fg_precision']:.4f}")

    delta_f2 = best["mean_fg_f2"] - baseline_metrics["mean_fg_f2"]
    delta_dice = best["mean_fg_dice"] - baseline_metrics["mean_fg_dice"]
    delta_recall = best["mean_fg_recall"] - baseline_metrics["mean_fg_recall"]
    delta_prec = best["mean_fg_precision"] - baseline_metrics["mean_fg_precision"]
    print(f"    Delta vs baseline: "
          f"F2={delta_f2:+.4f}  Dice={delta_dice:+.4f}  "
          f"Recall={delta_recall:+.4f}  Precision={delta_prec:+.4f}")
    print("=" * 80)


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
    print("POST-TRAINING: Connected Component Filtering Grid Search")
    print("=" * 70)

    best_model_path = output_dir / "best_model.pth"
    print(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_targets = collect_predictions(model, val_loader, device)
    print(f"Collected predictions: {all_preds.shape[0]} samples")

    grid_results, baseline_metrics = evaluate_cc_grid(
        all_preds, all_targets, num_classes=Config.NUM_CLASSES
    )

    print_cc_comparison_table(grid_results, baseline_metrics)

    best_entry = max(grid_results, key=lambda r: r["mean_fg_f2"])

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

    results["cc_grid_search"] = {
        "all_results": grid_results,
        "baseline_metrics": _serialize_metrics(baseline_metrics),
        "best_params": {
            "min_size": best_entry["min_size"],
            "max_size": best_entry["max_size"],
        },
        "best_metrics": {
            "mean_fg_dice": best_entry["mean_fg_dice"],
            "mean_fg_recall": best_entry["mean_fg_recall"],
            "mean_fg_precision": best_entry["mean_fg_precision"],
            "mean_fg_f2": best_entry["mean_fg_f2"],
        },
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results with CC grid search -> {results_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {Config.EXPERIMENT_NAME}")
    print(f"  {Config.DESCRIPTION}")
    print(f"  Best val Dice   : {results['best_val_dice']:.4f}  (epoch {results['best_epoch']})")
    print(f"  Best val Recall : {results['best_val_recall']:.4f}")
    print(f"  Best val Prec   : {results['best_val_precision']:.4f}")
    print(f"  Best CC params  : min_size={best_entry['min_size']}, max_size={best_entry['max_size']}")
    print(f"  Best CC F2      : {best_entry['mean_fg_f2']:.4f}")
    print(f"  Output          : {results['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
