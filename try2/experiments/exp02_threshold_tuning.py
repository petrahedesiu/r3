
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import ExperimentConfig
from shared.dataset import StandardDataset
from shared.losses import CompoundLoss
from shared.models import create_model
from shared.training import run_training, compute_class_weights
from shared.postprocessing import optimize_threshold
from shared.metrics import compute_dice_score, compute_recall, compute_precision, compute_f2_score
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
    EXPERIMENT_NAME = "exp02_threshold_tuning"
    DESCRIPTION = "Baseline model + optimized decision threshold (F2-based)"


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


def evaluate_with_threshold(model, dataloader, device, threshold, num_classes=3):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Eval @ thresh={threshold:.2f}"):
            images = images.to(device)
            if images.dtype != torch.float32:
                images = images.float()

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            fg_probs = probs[:, 1:, :, :]
            max_fg_prob, max_fg_class = fg_probs.max(dim=1)

            predLabels = torch.zeros_like(masks)
            fgMask = max_fg_prob > threshold
            predLabels[fgMask] = (max_fg_class[fgMask] + 1).long().cpu()

            all_preds.append(predLabels.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    dice_mean, dice_cls = compute_dice_score(all_preds, all_targets, num_classes)
    recall_mean, recall_cls = compute_recall(all_preds, all_targets, num_classes)
    prec_mean, prec_cls = compute_precision(all_preds, all_targets, num_classes)
    f2_mean, f2_cls = compute_f2_score(all_preds, all_targets, num_classes)

    return {
        "dice": dice_mean,
        "recall": recall_mean,
        "precision": prec_mean,
        "f2": f2_mean,
        "dice_per_class": dice_cls,
        "recall_per_class": recall_cls,
        "precision_per_class": prec_cls,
        "f2_per_class": f2_cls,
    }


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
    print("POST-TRAINING: Optimising decision threshold on validation set")
    print("=" * 70)

    best_ckpt_path = os.path.join(str(output_dir), "best_model.pth")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    best_threshold = optimize_threshold(
        model, val_loader, device, num_classes=Config.NUM_CLASSES
    )
    print(f"\nOptimal threshold: {best_threshold:.2f}")

    print("\n--- Metrics with DEFAULT threshold (argmax) ---")
    default_metrics = evaluate_with_threshold(
        model, val_loader, device, threshold=0.5, num_classes=Config.NUM_CLASSES
    )
    print(f"  Dice      : {default_metrics['dice']:.4f}")
    print(f"  Recall    : {default_metrics['recall']:.4f}")
    print(f"  Precision : {default_metrics['precision']:.4f}")
    print(f"  F2        : {default_metrics['f2']:.4f}")
    for c in range(Config.NUM_CLASSES):
        print(f"  Class {c}: dice={default_metrics['dice_per_class'][c]:.4f} "
              f"recall={default_metrics['recall_per_class'][c]:.4f} "
              f"prec={default_metrics['precision_per_class'][c]:.4f} "
              f"f2={default_metrics['f2_per_class'][c]:.4f}")

    print(f"\n--- Metrics with OPTIMISED threshold ({best_threshold:.2f}) ---")
    opt_metrics = evaluate_with_threshold(
        model, val_loader, device, threshold=best_threshold, num_classes=Config.NUM_CLASSES
    )
    print(f"  Dice      : {opt_metrics['dice']:.4f}")
    print(f"  Recall    : {opt_metrics['recall']:.4f}")
    print(f"  Precision : {opt_metrics['precision']:.4f}")
    print(f"  F2        : {opt_metrics['f2']:.4f}")
    for c in range(Config.NUM_CLASSES):
        print(f"  Class {c}: dice={opt_metrics['dice_per_class'][c]:.4f} "
              f"recall={opt_metrics['recall_per_class'][c]:.4f} "
              f"prec={opt_metrics['precision_per_class'][c]:.4f} "
              f"f2={opt_metrics['f2_per_class'][c]:.4f}")

    dice_delta = opt_metrics['dice'] - default_metrics['dice']
    recall_delta = opt_metrics['recall'] - default_metrics['recall']
    f2_delta = opt_metrics['f2'] - default_metrics['f2']

    print(f"\n--- Improvement from threshold tuning ---")
    print(f"  Dice   : {dice_delta:+.4f}")
    print(f"  Recall : {recall_delta:+.4f}")
    print(f"  F2     : {f2_delta:+.4f}")

    import json
    threshold_results = {
        "optimal_threshold": best_threshold,
        "default_metrics": {
            k: v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}
            for k, v in default_metrics.items()
        },
        "optimised_metrics": {
            k: v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}
            for k, v in opt_metrics.items()
        },
        "improvement": {
            "dice": dice_delta,
            "recall": recall_delta,
            "f2": f2_delta,
        },
    }
    threshold_json_path = os.path.join(str(output_dir), "threshold_results.json")
    with open(threshold_json_path, "w") as f:
        json.dump(threshold_results, f, indent=2)
    print(f"\nSaved threshold results -> {threshold_json_path}")

    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE: {Config.EXPERIMENT_NAME}")
    print(f"  {Config.DESCRIPTION}")
    print(f"  Best val Dice   : {results['best_val_dice']:.4f}  (epoch {results['best_epoch']})")
    print(f"  Best val Recall : {results['best_val_recall']:.4f}")
    print(f"  Best val Prec   : {results['best_val_precision']:.4f}")
    print(f"  Optimal thresh  : {best_threshold:.2f}")
    print(f"  Post-tune Dice  : {opt_metrics['dice']:.4f}")
    print(f"  Post-tune Recall: {opt_metrics['recall']:.4f}")
    print(f"  Post-tune F2    : {opt_metrics['f2']:.4f}")
    print(f"  Output          : {results['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
