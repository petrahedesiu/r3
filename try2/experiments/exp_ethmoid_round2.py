"""
Round 2 ethmoid sinus filling prediction experiments.

Base config: middle-W slab ±1, threshold -200 HU, LogisticRegression (balanced).

Experiments:
  G. I-S stratification   — superior vs inferior half of D axis
  H. Texture / distribution shape — kurtosis, skew, fluid/mucus fractions, bimodality
  I. HU histogram as features — 10-bin histogram + PCA(3) before LogReg
  J. Sinus air volume proxy — fractions below -800, -500, -200 HU
  K. Left-right symmetry  — joint classifier with cross-side symmetry feature
  L. Combination          — best features from G+H+J together
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from exp_classification import load_data
from shared.classification import load_classification_labels
from shared.config import ExperimentConfig

XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'AEA_classification_sinuzite.xlsx')
DATA_DIR = ExperimentConfig.DATA_DIR

# Base config constants
BASE_HALF_WIDTH = 1
BASE_THRESHOLD = -200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_slab_voxels_and_intensities(volume, mask, class_id):
    """Return (slab_voxels, intensities) using base config: middle-W slab ±1."""
    voxels = np.argwhere(mask == class_id)
    if len(voxels) == 0:
        return None, None
    w_min, w_max = int(voxels[:, 1].min()), int(voxels[:, 1].max())
    w_mid = (w_min + w_max) // 2
    w_lo = max(0, w_mid - BASE_HALF_WIDTH)
    w_hi = min(volume.shape[1] - 1, w_mid + BASE_HALF_WIDTH)
    sel = (voxels[:, 1] >= w_lo) & (voxels[:, 1] <= w_hi)
    sv = voxels[sel]
    if len(sv) == 0:
        return None, None
    intensities = volume[sv[:, 0], sv[:, 1], sv[:, 2]].astype(np.float32)
    return sv, intensities


def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)),
    ])


def loo_cv(X, y, clf_factory=make_lr):
    loo = LeaveOneOut()
    y_true, y_prob = [], []
    for train_idx, test_idx in loo.split(X):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx[0]])
        y_prob.append(clf.predict_proba(X[test_idx])[0, 1])
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return auc, bal_acc


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def features_G(volume, mask, class_id):
    """G. I-S stratification: superior vs inferior half of D axis + gradient."""
    sv, intensities = get_slab_voxels_and_intensities(volume, mask, class_id)
    if sv is None:
        return np.zeros(5, dtype=np.float32)
    d_coords = sv[:, 2]
    d_min, d_max = int(d_coords.min()), int(d_coords.max())
    d_mid = (d_min + d_max) // 2
    inf_mask = d_coords <= d_mid
    sup_mask = d_coords > d_mid
    inf_ints = intensities[inf_mask]
    sup_ints = intensities[sup_mask]
    filled_ratio_inf = float(np.mean(inf_ints > BASE_THRESHOLD)) if len(inf_ints) > 0 else 0.0
    filled_ratio_sup = float(np.mean(sup_ints > BASE_THRESHOLD)) if len(sup_ints) > 0 else 0.0
    filled_ratio_all = float(np.mean(intensities > BASE_THRESHOLD))
    gradient = filled_ratio_inf - filled_ratio_sup  # positive = more filling inferiorly
    # fraction of voxels in inferior half
    inf_frac = float(len(inf_ints)) / max(1, len(intensities))
    return np.array([filled_ratio_inf, filled_ratio_sup, filled_ratio_all, gradient, inf_frac],
                    dtype=np.float32)


def features_H(volume, mask, class_id):
    """H. Texture / distribution shape features."""
    _, intensities = get_slab_voxels_and_intensities(volume, mask, class_id)
    if intensities is None or len(intensities) < 3:
        return np.zeros(8, dtype=np.float32)
    mean_hu = float(np.mean(intensities))
    std_hu = float(np.std(intensities))
    filled_ratio = float(np.mean(intensities > BASE_THRESHOLD))
    skew = float(scipy_stats.skew(intensities))
    kurt = float(scipy_stats.kurtosis(intensities, fisher=True))
    fluid_frac = float(np.mean((intensities >= 0) & (intensities <= 100)))
    mucus_frac = float(np.mean((intensities >= -100) & (intensities < 0)))
    # bimodality coefficient
    n = len(intensities)
    if kurt > -1 and n > 3:
        bimodality = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3) + 1e-9))
    else:
        bimodality = 0.0
    return np.array([mean_hu, std_hu, filled_ratio, skew, kurt, fluid_frac, mucus_frac, bimodality],
                    dtype=np.float32)


def features_I_raw(volume, mask, class_id):
    """I. 10-bin HU histogram features (raw, before PCA)."""
    _, intensities = get_slab_voxels_and_intensities(volume, mask, class_id)
    if intensities is None or len(intensities) == 0:
        return np.zeros(10, dtype=np.float32)
    counts, _ = np.histogram(intensities, bins=10, range=(-1000, 500))
    total = max(1, counts.sum())
    return (counts / total).astype(np.float32)


def features_J(volume, mask, class_id):
    """J. Sinus air volume proxy: fractions below -800, -500, -200 HU."""
    _, intensities = get_slab_voxels_and_intensities(volume, mask, class_id)
    if intensities is None or len(intensities) == 0:
        return np.zeros(3, dtype=np.float32)
    frac_air   = float(np.mean(intensities < -800))
    frac_below500 = float(np.mean(intensities < -500))
    frac_below200 = float(np.mean(intensities < -200))
    return np.array([frac_air, frac_below500, frac_below200], dtype=np.float32)


def features_GHJ(volume, mask, class_id):
    """L. Combined G + H + J features."""
    g = features_G(volume, mask, class_id)
    h = features_H(volume, mask, class_id)
    j = features_J(volume, mask, class_id)
    return np.concatenate([g, h, j])


# ---------------------------------------------------------------------------
# Build feature matrices
# ---------------------------------------------------------------------------

def build_X_y(volumes, masks, codes, labels_all, class_id, label_key, feat_fn):
    X_list, y_list = [], []
    for vol, msk, code in zip(volumes, masks, codes):
        if code not in labels_all:
            continue
        feats = feat_fn(vol, msk, class_id)
        X_list.append(feats)
        y_list.append(labels_all[code][label_key])
    if not X_list:
        return np.empty((0, 1)), np.empty(0, dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=int)


def build_X_y_joint(volumes, masks, codes, labels_all):
    """K. Build joint feature matrix with both sides + symmetry features."""
    X_list, y_L_list, y_R_list = [], [], []
    for vol, msk, code in zip(volumes, masks, codes):
        if code not in labels_all:
            continue
        # Base features for both sides
        _, ints_L = get_slab_voxels_and_intensities(vol, msk, class_id=1)
        _, ints_R = get_slab_voxels_and_intensities(vol, msk, class_id=2)

        fr_L = float(np.mean(ints_L > BASE_THRESHOLD)) if ints_L is not None and len(ints_L) > 0 else 0.0
        fr_R = float(np.mean(ints_R > BASE_THRESHOLD)) if ints_R is not None and len(ints_R) > 0 else 0.0

        mean_L = float(np.mean(ints_L)) if ints_L is not None and len(ints_L) > 0 else 0.0
        mean_R = float(np.mean(ints_R)) if ints_R is not None and len(ints_R) > 0 else 0.0

        # Symmetry features
        ratio = fr_L / (fr_R + 1e-6)
        diff = fr_L - fr_R
        mean_diff = mean_L - mean_R

        feats = np.array([fr_L, fr_R, mean_L, mean_R, ratio, diff, mean_diff], dtype=np.float32)
        X_list.append(feats)
        y_L_list.append(labels_all[code]['left_ethmoid_filled'])
        y_R_list.append(labels_all[code]['right_ethmoid_filled'])

    if not X_list:
        return np.empty((0, 7)), np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.stack(X_list), np.array(y_L_list, dtype=int), np.array(y_R_list, dtype=int)


# ---------------------------------------------------------------------------
# PCA-LR factory for experiment I
# ---------------------------------------------------------------------------

def make_pca_lr(n_components=3):
    def factory():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)),
        ])
    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Ethmoid Sinus Filling — Round 2 Experiments ===\n")
    print(f"Base config: middle-W slab ±{BASE_HALF_WIDTH}, threshold {BASE_THRESHOLD} HU, LogReg balanced\n")
    print(f"Data dir : {DATA_DIR}")
    print(f"XLSX     : {XLSX_PATH}\n")

    labels_all = load_classification_labels(XLSX_PATH)

    print("Loading imaging data...")
    volumes, masks, codes = load_data(DATA_DIR)
    print(f"Loaded {len(volumes)} patients\n")

    results = []

    def report(name, side, auc, bal_acc, n_pos, n_total):
        results.append({'experiment': name, 'side': side, 'auc': auc,
                        'bal_acc': bal_acc, 'n_pos': n_pos, 'n_total': n_total})
        tag = f"  {name} | {side:5s} | AUC={auc:.3f} | BalAcc={bal_acc:.3f} | pos={n_pos}/{n_total}"
        print(tag)

    def run_both_sides(name, feat_fn, clf_factory=make_lr):
        for side_name, class_id, label_key in [
            ('Left',  1, 'left_ethmoid_filled'),
            ('Right', 2, 'right_ethmoid_filled'),
        ]:
            X, y = build_X_y(volumes, masks, codes, labels_all, class_id, label_key, feat_fn)
            if len(X) < 4:
                print(f"  {name} | {side_name} | SKIPPED (n={len(X)})")
                continue
            n_pos = int(y.sum())
            n_total = len(y)
            auc, bal_acc = loo_cv(X, y, clf_factory)
            report(name, side_name, auc, bal_acc, n_pos, n_total)

    # -----------------------------------------------------------------------
    # G. I-S stratification
    print("\n--- G. I-S Stratification (superior vs inferior half of D axis) ---")
    run_both_sides("G_IS_stratification", features_G)

    # -----------------------------------------------------------------------
    # H. Texture / distribution shape
    print("\n--- H. Texture / Distribution Shape Features ---")
    run_both_sides("H_texture_dist", features_H)

    # -----------------------------------------------------------------------
    # I. HU histogram + PCA
    print("\n--- I. HU Histogram (10 bins) + PCA(3) + LogReg ---")
    # Check n_components doesn't exceed n_features or n_samples-1
    # We have 10 features; use PCA(3)
    run_both_sides("I_histogram_PCA3", features_I_raw, clf_factory=make_pca_lr(n_components=3))

    # -----------------------------------------------------------------------
    # J. Sinus air volume proxy
    print("\n--- J. Sinus Air Volume Proxy (fractions <-800, <-500, <-200 HU) ---")
    run_both_sides("J_air_proxy", features_J)

    # -----------------------------------------------------------------------
    # K. Left-right symmetry (joint classifier)
    print("\n--- K. Left-Right Symmetry (joint features, separate LOO-CV per side) ---")
    X_joint, y_L, y_R = build_X_y_joint(volumes, masks, codes, labels_all)
    if len(X_joint) >= 4:
        auc_L, bal_L = loo_cv(X_joint, y_L)
        report("K_LR_symmetry_joint", 'Left',  auc_L, bal_L, int(y_L.sum()), len(y_L))
        auc_R, bal_R = loo_cv(X_joint, y_R)
        report("K_LR_symmetry_joint", 'Right', auc_R, bal_R, int(y_R.sum()), len(y_R))
    else:
        print(f"  K_LR_symmetry_joint | SKIPPED (n={len(X_joint)})")

    # -----------------------------------------------------------------------
    # L. Combination G + H + J
    print("\n--- L. Combination: G + H + J features ---")
    run_both_sides("L_GHJ_combined", features_GHJ)

    # -----------------------------------------------------------------------
    # Summary
    print("\n" + "=" * 110)
    print("ROUND 2 SUMMARY — Ranked by Mean AUC (Left+Right average)")
    print("=" * 110)

    exp_summary = {}
    for r in results:
        exp_summary.setdefault(r['experiment'], []).append(r)

    rows = []
    for exp_name, exp_results in exp_summary.items():
        aucs = [r['auc'] for r in exp_results]
        bals = [r['bal_acc'] for r in exp_results]
        mean_auc = float(np.nanmean(aucs))
        mean_bal = float(np.nanmean(bals))
        sides_info = {r['side']: r for r in exp_results}
        rows.append((mean_auc, mean_bal, exp_name, sides_info))

    rows.sort(key=lambda x: x[0], reverse=True)

    ROUND1_BASELINE = {'Left': 0.508, 'Right': 0.688, 'Mean': 0.598}

    header = (f"{'Rank':>4}  {'Experiment':<30}  {'MeanAUC':>7}  {'BalAcc':>7}"
              f"  {'Left_AUC':>8}  {'Right_AUC':>9}  {'Left_vs_R1':>10}  {'n':>6}")
    print(header)
    print("-" * len(header))

    for rank, (mean_auc, mean_bal, exp_name, sides_info) in enumerate(rows, 1):
        left_r  = sides_info.get('Left', {})
        right_r = sides_info.get('Right', {})
        left_auc  = left_r.get('auc', float('nan'))
        right_auc = right_r.get('auc', float('nan'))
        left_diff = left_auc - ROUND1_BASELINE['Left']
        diff_tag = f"{left_diff:+.3f}"
        n_pos   = left_r.get('n_pos', '?')
        n_total = left_r.get('n_total', '?')
        print(f"{rank:>4}  {exp_name:<30}  {mean_auc:>7.3f}  {mean_bal:>7.3f}"
              f"  {left_auc:>8.3f}  {right_auc:>9.3f}  {diff_tag:>10}  {n_pos}/{n_total}")

    print()
    print("Round 1 best: Mean=0.598, Left=0.508, Right=0.688  (RF, slab±1, thr=-200, middle-W)")
    print()

    # Highlight experiments that improved Left AUC
    improved_left = [(mean_auc, exp_name, sides_info['Left'].get('auc', float('nan')))
                     for mean_auc, _, exp_name, sides_info in rows
                     if 'Left' in sides_info and sides_info['Left'].get('auc', 0) > ROUND1_BASELINE['Left']]
    if improved_left:
        print("Experiments that IMPROVED Left AUC vs Round 1 (baseline=0.508):")
        for mean_auc, exp_name, left_auc in sorted(improved_left, key=lambda x: -x[2]):
            print(f"  {exp_name:<30}  Left AUC={left_auc:.3f}  (+{left_auc - ROUND1_BASELINE['Left']:.3f})")
    else:
        print("No experiment exceeded the Round 1 Left AUC baseline of 0.508.")

    print("\nDone.")


if __name__ == '__main__':
    main()
