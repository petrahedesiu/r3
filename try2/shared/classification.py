"""
Post-hoc classification of AEA segmentation masks.

Given a 3D volume (H×W×D) and a 3-class segmentation mask (BG=0, AEAL=1, AEAR=2),
predicts 4 binary labels per patient:

  Roof contact (geometric — mask z-position only):
    aeal_roof_contact  : AEAL in contact with skull roof (1) vs distant (0)
    aear_roof_contact  : AEAR in contact with skull roof (1) vs distant (0)

  Ethmoid sinus status (radiological — CT intensity in mask region):
    left_ethmoid_filled  : left ethmoid filled/sinusitis (1) vs clear (0)
    right_ethmoid_filled : right ethmoid filled/sinusitis (1) vs clear (0)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_classification_labels(xlsx_path: str) -> Dict[str, Dict[str, int]]:
    """Return dict keyed by patient_code with 4 binary label values."""
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
    6 CT intensity features sampled from voxels belonging to class_id.
    volume shape: (H, W, D), dtype typically int16 (HU values).
    Returns zeros if class_id is absent.
    """
    voxels = np.argwhere(mask == class_id)

    if len(voxels) == 0:
        return np.zeros(6, dtype=np.float32)

    intensities = volume[voxels[:, 0], voxels[:, 1], voxels[:, 2]].astype(np.float32)

    mean_hu   = float(np.mean(intensities))
    median_hu = float(np.median(intensities))
    std_hu    = float(np.std(intensities))
    p10_hu    = float(np.percentile(intensities, 10))
    p90_hu    = float(np.percentile(intensities, 90))
    # fraction of voxels above air threshold — fluid/mucus >> -500 HU
    filled_ratio = float(np.mean(intensities > -500))

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
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      X_dict — {"roof_L": (N,4), "roof_R": (N,4), "ethmoid_L": (N,6), "ethmoid_R": (N,6)}
      y_dict — {"aeal_roof_contact": (N,), "aear_roof_contact": (N,),
                "left_ethmoid_filled": (N,), "right_ethmoid_filled": (N,)}
    Only patients whose code appears in `labels` are included; returns their
    filtered indices as well.
    """
    feat_roof_L, feat_roof_R = [], []
    feat_eth_L, feat_eth_R   = [], []
    y_aeal_roof, y_aear_roof = [], []
    y_left_eth,  y_right_eth = [], []

    for code, vol, msk in zip(patient_codes, volumes, masks):
        if code not in labels:
            continue
        feats = extract_features(vol, msk)
        feat_roof_L.append(feats["roof_L"])
        feat_roof_R.append(feats["roof_R"])
        feat_eth_L.append(feats["ethmoid_L"])
        feat_eth_R.append(feats["ethmoid_R"])
        lbl = labels[code]
        y_aeal_roof.append(lbl["aeal_roof_contact"])
        y_aear_roof.append(lbl["aear_roof_contact"])
        y_left_eth.append(lbl["left_ethmoid_filled"])
        y_right_eth.append(lbl["right_ethmoid_filled"])

    X_dict = {
        "roof_L":    np.stack(feat_roof_L) if feat_roof_L else np.empty((0, 6)),
        "roof_R":    np.stack(feat_roof_R) if feat_roof_R else np.empty((0, 6)),
        "ethmoid_L": np.stack(feat_eth_L)  if feat_eth_L  else np.empty((0, 6)),
        "ethmoid_R": np.stack(feat_eth_R)  if feat_eth_R  else np.empty((0, 6)),
    }
    y_dict = {
        "aeal_roof_contact":   np.array(y_aeal_roof,  dtype=np.int32),
        "aear_roof_contact":   np.array(y_aear_roof,  dtype=np.int32),
        "left_ethmoid_filled": np.array(y_left_eth,   dtype=np.int32),
        "right_ethmoid_filled": np.array(y_right_eth, dtype=np.int32),
    }
    return X_dict, y_dict


def _make_clf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
    ])


TASK_TO_FEATURES = {
    "aeal_roof_contact":    "roof_L",
    "aear_roof_contact":    "roof_R",
    "left_ethmoid_filled":  "ethmoid_L",
    "right_ethmoid_filled": "ethmoid_R",
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
        clf = _make_clf()
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
) -> Dict[str, int]:
    """
    Given a single patient's volume and predicted mask, return classification
    predictions as a dict of {task_name: 0_or_1}.
    """
    feats = extract_features(volume, mask)
    results: Dict[str, int] = {}
    for task, feat_key in TASK_TO_FEATURES.items():
        if task not in clfs:
            continue
        x = feats[feat_key].reshape(1, -1)
        results[task] = int(clfs[task].predict(x)[0])
    return results


def predict_classifications_proba(
    volume: np.ndarray,
    mask: np.ndarray,
    clfs: Dict[str, Pipeline],
) -> Dict[str, float]:
    """Same as predict_classifications but returns probability of positive class."""
    feats = extract_features(volume, mask)
    results: Dict[str, float] = {}
    for task, feat_key in TASK_TO_FEATURES.items():
        if task not in clfs:
            continue
        x = feats[feat_key].reshape(1, -1)
        results[task] = float(clfs[task].predict_proba(x)[0, 1])
    return results
