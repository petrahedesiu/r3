
import sys
import os
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import discover_patients, load_patient_data, get_labeled_slice_indices

from . import config


def load_all_patients(
    data_dirs: List[str] = None,
    min_labeled_slices: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    if data_dirs is None:
        data_dirs = config.DATA_DIRS

    volumes = []
    segmentations = []
    patient_infos = []

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"WARNING: data dir not found: {data_dir}")
            continue

        patients = discover_patients(data_dir)
        print(f"Found {len(patients)} patients in {os.path.basename(data_dir)}")

        for p in patients:
            try:
                vol, seg, meta = load_patient_data(p["dicom_dir"], p["nrrd_path"])
            except Exception as e:
                print(f"  SKIP {p['patient_id']}: {e}")
                continue

            if not meta.get("alignment_success", False):
                print(f"  SKIP {p['patient_id']}: alignment failed")
                continue

            labeled = get_labeled_slice_indices(seg)
            if len(labeled) < min_labeled_slices:
                continue

            volumes.append(vol)
            segmentations.append(seg)
            patient_infos.append({
                "patient_id": p["patient_id"],
                "source_dir": data_dir,
                "n_slices": vol.shape[2],
                "n_labeled": len(labeled),
                "dicom_dir": p["dicom_dir"],
                "nrrd_path": p["nrrd_path"],
            })

    print(f"\nLoaded {len(volumes)} patients total, "
          f"{sum(p['n_labeled'] for p in patient_infos)} labeled slices")
    return volumes, segmentations, patient_infos


def patient_split(
    n_patients: int,
    val_split: float = None,
    seed: int = None,
) -> Tuple[List[int], List[int]]:
    if val_split is None:
        val_split = config.VAL_SPLIT
    if seed is None:
        seed = config.RANDOM_SEED

    indices = list(range(n_patients))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed)
    return sorted(train_idx), sorted(val_idx)
