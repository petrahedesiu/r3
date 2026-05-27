"""
Train and evaluate AEA classification from segmentation masks.

Usage:
  # Train + LOO-CV on CROP1, then optionally validate on CROP EXT:
  python exp_classification.py [--data-dir DATA_DIR] [--xlsx PATH] [--output-dir DIR]
                               [--ext-dir EXT_DIR]

CROP1 folder format : "016. VA016 VINTELER ANA-MARIA"  → code extracted as "VA016"
CROP EXT structure  :
  EXT_DIR/
    01/
      EXT01/       ← DICOM directory
      EXT01.nrrd   ← segmentation mask
    02/
      EXT02/
      EXT02.nrrd
    ...

EXT patients are matched to the Excel by their EXTxx code.  If the Excel has no
EXT codes, predictions are still written to output_dir/ext_predictions.csv.
"""

import sys
import os
import argparse
import re
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, balanced_accuracy_score
)

from data_utils import load_patient_data, discover_patients, get_labeled_slice_indices
from shared.classification import (
    load_classification_labels,
    build_feature_matrix,
    train_classifiers,
    save_classifiers,
    predict_classifications,
    TASK_TO_FEATURES,
    _make_clf,
)
from shared.config import ExperimentConfig


XLSX_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'AEA_classification_sinuzite.xlsx'
)


# ---------------------------------------------------------------------------
# Patient code extraction
# ---------------------------------------------------------------------------

def _extract_patient_code(dicom_dir: str) -> str:
    """Folder format: "016. VA016 VINTELER ANA-MARIA" — search anywhere in name."""
    basename = os.path.basename(os.path.normpath(dicom_dir))
    m = re.search(r'([A-Z]{2}\d{3})', basename.upper())
    if m:
        return m.group(1)
    return basename.upper().strip()


# ---------------------------------------------------------------------------
# CROP1 data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str):
    patients_meta = discover_patients(data_dir)
    volumes, masks, codes = [], [], []
    for p in tqdm(patients_meta, desc="Loading CROP1"):
        try:
            vol, seg, meta = load_patient_data(
                p['dicom_dir'], p['nrrd_path'], verbose=False
            )
        except Exception as e:
            print(f"  [skip] {p.get('dicom_dir','?')}: {e}")
            continue
        if not meta.get('alignment_success', False):
            continue
        if len(get_labeled_slice_indices(seg)) < 2:
            continue
        volumes.append(vol)
        masks.append(seg)
        codes.append(_extract_patient_code(p['dicom_dir']))
    return volumes, masks, codes


# ---------------------------------------------------------------------------
# CROP EXT data loading
# ---------------------------------------------------------------------------

def discover_ext_patients(ext_dir: str) -> list:
    """
    Walk CROP EXT structure:
      ext_dir/01/EXT01/   (DICOM)  +  ext_dir/01/EXT01.nrrd
      ext_dir/02/EXT02/            +  ext_dir/02/EXT02.nrrd
      ...
    Returns list of dicts: {code, dicom_dir, nrrd_path}
    """
    patients = []
    for index_folder in sorted(os.listdir(ext_dir)):
        index_path = os.path.join(ext_dir, index_folder)
        if not os.path.isdir(index_path):
            continue

        # Find the EXTxx subfolder and matching nrrd
        ext_code = None
        dicom_dir = None
        nrrd_path = None

        for entry in os.listdir(index_path):
            entry_path = os.path.join(index_path, entry)
            if os.path.isdir(entry_path) and re.match(r'^EXT\d+$', entry.upper()):
                dicom_dir = entry_path
                ext_code = entry.upper()
            elif entry.lower().endswith('.nrrd') and re.match(r'^EXT\d+', entry.upper()):
                nrrd_path = entry_path

        if dicom_dir and nrrd_path and ext_code:
            patients.append({
                'code': ext_code,
                'dicom_dir': dicom_dir,
                'nrrd_path': nrrd_path,
            })
        else:
            print(f"  [skip EXT] {index_path}: could not find EXTxx dir + nrrd")

    return patients


def load_ext_data(ext_dir: str):
    patients_meta = discover_ext_patients(ext_dir)
    volumes, masks, codes = [], [], []
    for p in tqdm(patients_meta, desc="Loading CROP EXT"):
        try:
            vol, seg, meta = load_patient_data(
                p['dicom_dir'], p['nrrd_path'], verbose=False
            )
        except Exception as e:
            print(f"  [skip] {p['code']}: {e}")
            continue
        if not meta.get('alignment_success', False):
            continue
        volumes.append(vol)
        masks.append(seg)
        codes.append(p['code'])
    return volumes, masks, codes


# ---------------------------------------------------------------------------
# LOO cross-validation
# ---------------------------------------------------------------------------

def loo_evaluate(X: np.ndarray, y: np.ndarray, task: str) -> dict:
    if len(X) < 4:
        print(f"  [skip] {task}: only {len(X)} samples with matched labels")
        return {}

    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in loo.split(X):
        clf = _make_clf(task)
        clf.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx[0]])
        y_pred.append(clf.predict(X[test_idx])[0])
        y_prob.append(clf.predict_proba(X[test_idx])[0, 1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": auc,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------

def run_ext_validation(
    ext_dir: str,
    clfs: dict,
    labels: dict,
    output_dir: str,
) -> None:
    print(f"\n{'=' * 60}")
    print("External Validation (CROP EXT)")
    print(f"{'=' * 60}")

    volumes, masks, codes = load_ext_data(ext_dir)
    print(f"Loaded {len(volumes)} EXT patients")

    if len(volumes) == 0:
        print("No EXT patients loaded — check ext_dir structure.")
        return

    # Run predictions
    rows = []
    for code, vol, msk in zip(codes, volumes, masks):
        preds = predict_classifications(vol, msk, clfs)
        row = {"patient_code": code}
        row.update(preds)
        # Attach ground-truth labels if available in Excel
        if code in labels:
            for task in TASK_TO_FEATURES:
                row[f"{task}_gt"] = labels[code][task]
        rows.append(row)

    # Save predictions CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ext_predictions.csv")
    fieldnames = ["patient_code"] + list(TASK_TO_FEATURES.keys()) + \
                 [f"{t}_gt" for t in TASK_TO_FEATURES]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions saved → {csv_path}")

    # Compute metrics for patients that have Excel labels
    labeled_rows = [r for r in rows if f"{list(TASK_TO_FEATURES)[0]}_gt" in r]
    if not labeled_rows:
        print("No EXT patients found in Excel — predictions saved, no metrics computed.")
        return

    print(f"\nMetrics on {len(labeled_rows)} EXT patients with Excel labels:")
    for task in TASK_TO_FEATURES:
        y_true = np.array([r[f"{task}_gt"] for r in labeled_rows])
        y_pred = np.array([r[task] for r in labeled_rows])
        if len(np.unique(y_true)) < 2:
            print(f"\n--- {task}: only one class in EXT labels, skipping AUC ---")
            continue
        print(f"\n--- {task} ---")
        print(f"  Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
        print(f"  {classification_report(y_true, y_pred, zero_division=0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=ExperimentConfig.DATA_DIR)
    parser.add_argument("--xlsx",       default=XLSX_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ext-dir",    default=None,
                        help="Path to CROP EXT folder for external validation")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'outputs', 'classification'
    )

    print("=== AEA Classification Experiment ===\n")
    print(f"Data dir : {args.data_dir}")
    print(f"XLSX     : {args.xlsx}")
    print(f"Output   : {output_dir}")
    if args.ext_dir:
        print(f"EXT dir  : {args.ext_dir}")
    print()

    labels = load_classification_labels(args.xlsx)
    print(f"Loaded {len(labels)} patient labels from XLSX")

    print("\nLoading imaging data...")
    volumes, masks, codes = load_data(args.data_dir)
    print(f"Loaded {len(volumes)} valid CROP1 patients")

    matched = [c for c in codes if c in labels]
    print(f"Matched to XLSX labels: {len(matched)}/{len(codes)}")
    unmatched = [c for c in codes if c not in labels]
    if unmatched:
        print(f"  Unmatched codes: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")
        print("  → Adjust _extract_patient_code() if codes look wrong")

    if len(matched) == 0:
        print("\nERROR: No patients matched. Check patient code extraction.")
        return

    X_dict, y_dict = build_feature_matrix(codes, volumes, masks, labels)
    n = len(y_dict["aeal_roof_contact"])
    print(f"\nFeature matrix built: {n} patients with matched labels")

    # LOO cross-validation
    print(f"\n{'=' * 60}")
    print("Leave-One-Out Cross-Validation (CROP1)")
    print(f"{'=' * 60}")

    for task, feat_key in TASK_TO_FEATURES.items():
        X = X_dict[feat_key]
        y = y_dict[task]
        print(f"\n--- {task} ---")
        print(f"  Features : {feat_key}  shape={X.shape}")
        print(f"  Positive : {int(y.sum())}/{len(y)} ({100*y.mean():.1f}%)")
        res = loo_evaluate(X, y, task)
        if res:
            print(f"  Accuracy          : {res['accuracy']:.3f}")
            print(f"  Balanced accuracy : {res['balanced_accuracy']:.3f}")
            print(f"  AUC               : {res['auc']:.3f}")
            print(f"  {res['report']}")

    # Train final classifiers on all CROP1 data
    print(f"\n{'=' * 60}")
    print("Training final classifiers on all CROP1 data...")
    clfs = train_classifiers(X_dict, y_dict)
    save_classifiers(clfs, output_dir)
    print(f"Saved {len(clfs)} classifiers to {output_dir}/")
    for task in clfs:
        print(f"  → {task}.joblib")

    # External validation
    if args.ext_dir:
        run_ext_validation(args.ext_dir, clfs, labels, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
