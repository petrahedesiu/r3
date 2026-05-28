"""
Post-hoc classification of AEA segmentation masks.

Given a 3D volume (H×W×D) and a 3-class segmentation mask (BG=0, AEAL=1, AEAR=2),
predicts 2 binary labels per AEA instance (side-agnostic):

  roof_contact   : AEA in contact with skull roof (1) vs distant (0)
  ethmoid_filled : ethmoid sinus adjacent to AEA is filled/sinusitis (1) vs clear (0)

Each patient contributes 2 samples (left + right), pooled into a single classifier
per task. LOO-CV is patient-level: both sides of a patient are held out together.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_classification_labels(xlsx_path: str) -> Dict[str, Dict[str, int]]:
    """Return dict keyed by patient_code.

    Each entry has per-side labels (still stored separately so we can pool
    them into side-agnostic samples in build_feature_matrix):
      aeal_roof_contact, aear_roof_contact,
      left_ethmoid_filled, right_ethmoid_filled
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    labels: Dict[str, Dict[str, int]] = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        code = row[0]
        if code is None:
            continue
        code = str(code).strip()
        labels[code] = {
            "aeal_roof_contact":   1 if row[4] == "X" else 0,
            "aear_roof_contact":   1 if row[6] == "X" else 0,
            "left_ethmoid_filled": 1 if row[9] == "X" else 0,
            "right_ethmoid_filled": 1 if row[11] == "X" else 0,
        }
    return labels

def extract_roof_features(volume: np.ndarray, mask: np.ndarray, class_id: int) -> np.ndarray:
    """
    Features for skull-roof contact detection.

    Slices are sorted ascending by ImagePositionPatient[2] (standard DICOM), so
    high z-index = superior.  "Roof contact" means the anatomy abuts the skull
    base / roof, detected by sampling CT intensities just *above* (z+1..z+3) the
    topmost mask voxels — bone HU >> soft tissue.

    Returns zeros if class_id is absent.
    """
    voxels = np.argwhere(mask == class_id)  # (N, 3): rows are (h, w, d)
    D = mask.shape[2]

    if len(voxels) == 0:
        return np.zeros(6, dtype=np.float32)

    z_coords = voxels[:, 2].astype(np.float32)
    z_max = int(z_coords.max())
    top_z_norm      = z_max / (D - 1)
    centroid_z_norm = float(z_coords.mean()) / (D - 1)
    z_extent_norm   = (z_coords.max() - z_coords.min()) / (D - 1)
    log_count       = float(np.log1p(len(voxels)))

    # Sample HU values in a shell of slices just above the top of the mask.
    # bone ~400–1900 HU; air ≈ -1000 HU; soft tissue -100..+100 HU.
    shell_hu: List[float] = []
    top_voxels = voxels[voxels[:, 2] == z_max]  # (M, 3)
    for dz in (1, 2, 3):
        z_above = z_max + dz
        if z_above >= D:
            break
        hu_vals = volume[top_voxels[:, 0], top_voxels[:, 1], z_above].astype(np.float32)
        shell_hu.extend(hu_vals.tolist())

    if shell_hu:
        shell_arr = np.array(shell_hu, dtype=np.float32)
        bone_contact_ratio = float(np.mean(shell_arr > 400))
        mean_above_hu      = float(np.mean(shell_arr))
    else:
        bone_contact_ratio = 0.0
        mean_above_hu      = 0.0

    return np.array(
        [top_z_norm, centroid_z_norm, z_extent_norm, log_count,
         bone_contact_ratio, mean_above_hu],
        dtype=np.float32,
    )


def extract_ethmoid_features(
    volume: np.ndarray, mask: np.ndarray, class_id: int
) -> np.ndarray:
    """
    Ethmoid filling features derived from the middle sagittal slab.

    Best config from sweep: slab ±1 voxel around the middle of the W range
    (not the centroid), threshold -200 HU for filled_ratio.

    volume shape: (H, W, D) — W = columns = left-right axis.
    Sagittal plane = H×D at fixed W.
    Returns zeros if class_id is absent.
    """
    voxels = np.argwhere(mask == class_id)  # (N, 3): h, w, d

    if len(voxels) == 0:
        return np.zeros(6, dtype=np.float32)

    # Middle of the W extent of the mask (not centroid — midpoint of range)
    w_min, w_max = int(voxels[:, 1].min()), int(voxels[:, 1].max())
    w_mid = (w_min + w_max) // 2
    w_lo = max(0, w_mid - 1)
    w_hi = min(volume.shape[1] - 1, w_mid + 1)

    slab_mask = (voxels[:, 1] >= w_lo) & (voxels[:, 1] <= w_hi)
    slab_voxels = voxels[slab_mask]

    if len(slab_voxels) == 0:
        return np.zeros(6, dtype=np.float32)

    intensities = volume[
        slab_voxels[:, 0], slab_voxels[:, 1], slab_voxels[:, 2]
    ].astype(np.float32)

    mean_hu      = float(np.mean(intensities))
    median_hu    = float(np.median(intensities))
    std_hu       = float(np.std(intensities))
    p10_hu       = float(np.percentile(intensities, 10))
    p90_hu       = float(np.percentile(intensities, 90))
    # -200 HU threshold: better air/soft-tissue boundary than -500
    filled_ratio = float(np.mean(intensities > -200))

    return np.array(
        [mean_hu, median_hu, std_hu, p10_hu, p90_hu, filled_ratio],
        dtype=np.float32,
    )


def extract_features(
    volume: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return feature vectors for both sides, for both tasks."""
    return {
        "roof_L":    extract_roof_features(volume, mask, class_id=1),
        "roof_R":    extract_roof_features(volume, mask, class_id=2),
        "ethmoid_L": extract_ethmoid_features(volume, mask, class_id=1),
        "ethmoid_R": extract_ethmoid_features(volume, mask, class_id=2),
    }


def build_feature_matrix(
    patient_codes: List[str],
    volumes: List[np.ndarray],
    masks: List[np.ndarray],
    labels: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Pool left and right sides into side-agnostic feature matrices.

    Each patient contributes 2 rows (left side first, then right side).
    Returns:
      X_dict       — {"roof": (2N, 6), "ethmoid": (2N, 6)}
      y_dict       — {"roof_contact": (2N,), "ethmoid_filled": (2N,)}
      patient_idx  — (2N,) integer array mapping each row to its patient index.
                     Used for patient-level LOO-CV (hold out both sides together).
    """
    feat_roof, feat_eth = [], []
    y_roof, y_eth       = [], []
    patient_idx         = []

    for p_idx, (code, vol, msk) in enumerate(zip(patient_codes, volumes, masks)):
        if code not in labels:
            continue
        feats = extract_features(vol, msk)
        lbl   = labels[code]
        # Left side (class_id=1)
        feat_roof.append(feats["roof_L"])
        feat_eth.append(feats["ethmoid_L"])
        y_roof.append(lbl["aeal_roof_contact"])
        y_eth.append(lbl["left_ethmoid_filled"])
        patient_idx.append(p_idx)
        # Right side (class_id=2)
        feat_roof.append(feats["roof_R"])
        feat_eth.append(feats["ethmoid_R"])
        y_roof.append(lbl["aear_roof_contact"])
        y_eth.append(lbl["right_ethmoid_filled"])
        patient_idx.append(p_idx)

    X_dict = {
        "roof":    np.stack(feat_roof) if feat_roof else np.empty((0, 6)),
        "ethmoid": np.stack(feat_eth)  if feat_eth  else np.empty((0, 6)),
    }
    y_dict = {
        "roof_contact":   np.array(y_roof, dtype=np.int32),
        "ethmoid_filled": np.array(y_eth,  dtype=np.int32),
    }
    return X_dict, y_dict, np.array(patient_idx, dtype=np.int32)


def _make_clf(task: str) -> Pipeline:
    """Return the best classifier pipeline for the given task.

    All tasks use LogisticRegression — stable under LOO-CV with small
    positive class sizes (23–26 positives out of 133). RandomForest showed
    higher variance and collapsed recall in LOO-CV despite appearing better
    in a fixed-split sweep.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
    ])


TASK_TO_FEATURES = {
    "roof_contact":   "roof",
    "ethmoid_filled": "ethmoid",
}


def train_classifiers(
    X_dict: Dict[str, np.ndarray],
    y_dict: Dict[str, np.ndarray],
) -> Dict[str, Pipeline]:
    """Train one pipeline per task. Returns dict keyed by task name."""
    clfs: Dict[str, Pipeline] = {}
    for task, feat_key in TASK_TO_FEATURES.items():
        X = X_dict[feat_key]
        y = y_dict[task]
        if len(X) == 0:
            continue
        clf = _make_clf(task)
        clf.fit(X, y)
        clfs[task] = clf
    return clfs


def save_classifiers(clfs: Dict[str, Pipeline], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for task, clf in clfs.items():
        joblib.dump(clf, os.path.join(output_dir, f"{task}.joblib"))


def load_classifiers(model_dir: str) -> Dict[str, Pipeline]:
    clfs: Dict[str, Pipeline] = {}
    for task in TASK_TO_FEATURES:
        path = os.path.join(model_dir, f"{task}.joblib")
        if os.path.exists(path):
            clfs[task] = joblib.load(path)
    return clfs


def predict_classifications(
    volume: np.ndarray,
    mask: np.ndarray,
    clfs: Dict[str, Pipeline],
) -> Dict[str, Dict[str, int]]:
    """
    Given a single patient's volume and mask, return side-agnostic predictions.

    Returns: {"left": {"roof_contact": 0/1, "ethmoid_filled": 0/1},
              "right": {"roof_contact": 0/1, "ethmoid_filled": 0/1}}
    """
    feats = extract_features(volume, mask)
    side_feat = {"left": ("roof_L", "ethmoid_L"), "right": ("roof_R", "ethmoid_R")}
    results: Dict[str, Dict[str, int]] = {}
    for side, (rf_key, eth_key) in side_feat.items():
        results[side] = {}
        if "roof_contact" in clfs:
            results[side]["roof_contact"] = int(
                clfs["roof_contact"].predict(feats[rf_key].reshape(1, -1))[0]
            )
        if "ethmoid_filled" in clfs:
            results[side]["ethmoid_filled"] = int(
                clfs["ethmoid_filled"].predict(feats[eth_key].reshape(1, -1))[0]
            )
    return results


def predict_classifications_proba(
    volume: np.ndarray,
    mask: np.ndarray,
    clfs: Dict[str, Pipeline],
) -> Dict[str, Dict[str, float]]:
    """Same as predict_classifications but returns probability of positive class."""
    feats = extract_features(volume, mask)
    side_feat = {"left": ("roof_L", "ethmoid_L"), "right": ("roof_R", "ethmoid_R")}
    results: Dict[str, Dict[str, float]] = {}
    for side, (rf_key, eth_key) in side_feat.items():
        results[side] = {}
        if "roof_contact" in clfs:
            results[side]["roof_contact"] = float(
                clfs["roof_contact"].predict_proba(feats[rf_key].reshape(1, -1))[0, 1]
            )
        if "ethmoid_filled" in clfs:
            results[side]["ethmoid_filled"] = float(
                clfs["ethmoid_filled"].predict_proba(feats[eth_key].reshape(1, -1))[0, 1]
            )
    return results
