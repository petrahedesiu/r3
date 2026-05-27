"""
Systematic analysis of ethmoid sinus filling prediction.
Tests HU thresholds, slab widths, distribution features, sagittal position,
classifiers, and combinations via LOO-CV.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from exp_classification import load_data
from shared.classification import load_classification_labels
from shared.config import ExperimentConfig

XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'AEA_classification_sinuzite.xlsx')
DATA_DIR = ExperimentConfig.DATA_DIR


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def get_mask_voxels(mask, class_id):
    return np.argwhere(mask == class_id)


def get_slab_intensities(volume, mask, class_id, w_lo, w_hi):
    voxels = get_mask_voxels(mask, class_id)
    if len(voxels) == 0:
        return None
    sel = (voxels[:, 1] >= w_lo) & (voxels[:, 1] <= w_hi)
    sv = voxels[sel]
    if len(sv) == 0:
        return None
    return volume[sv[:, 0], sv[:, 1], sv[:, 2]].astype(np.float32)


def get_slab_bounds(voxels, volume_shape, half_width, position='center'):
    """Return (w_lo, w_hi) for a slab.
    position: 'center', 'anterior', 'middle', 'posterior', or 'full'
    half_width: ±half_width voxels (ignored if position='full')
    """
    W = volume_shape[1]
    if position == 'full':
        return 0, W - 1

    w_vals = voxels[:, 1]
    w_min, w_max = int(w_vals.min()), int(w_vals.max())
    w_range = w_max - w_min
    third = max(1, w_range // 3)

    if position == 'anterior':
        # anterior = low W index (remember H=A-P, W=L-R ... actually W is sagittal/L-R)
        # "anterior" in the A-P sense is H, not W — but task says "anterior third of W range"
        # so we interpret as the first third of the mask's W extent
        anchor = w_min + third // 2
    elif position == 'middle':
        anchor = w_min + third + third // 2
    elif position == 'posterior':
        anchor = w_min + 2 * third + third // 2
    else:  # center = W centroid
        anchor = int(round(float(w_vals.mean())))

    if half_width == 0:
        return anchor, anchor
    return max(0, anchor - half_width), min(W - 1, anchor + half_width)


def base_features(intensities, threshold=-500):
    """6 features matching original: mean, median, std, p10, p90, filled_ratio."""
    if intensities is None or len(intensities) == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        float(np.mean(intensities)),
        float(np.median(intensities)),
        float(np.std(intensities)),
        float(np.percentile(intensities, 10)),
        float(np.percentile(intensities, 90)),
        float(np.mean(intensities > threshold)),
    ], dtype=np.float32)


def distribution_features(intensities):
    """Extra distribution shape features: kurtosis, fluid_frac, soft_frac, entropy."""
    if intensities is None or len(intensities) == 0:
        return np.zeros(4, dtype=np.float32)
    kurt = float(scipy_kurtosis(intensities, fisher=True))
    fluid_frac = float(np.mean((intensities >= 0) & (intensities <= 100)))
    soft_frac = float(np.mean((intensities >= -100) & (intensities <= 100)))
    # entropy of 20-bin histogram from -1000 to 500 HU
    counts, _ = np.histogram(intensities, bins=20, range=(-1000, 500))
    counts = counts.astype(np.float64)
    counts += 1e-10  # avoid log(0)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    return np.array([kurt, fluid_frac, soft_frac, entropy], dtype=np.float32)


# ---------------------------------------------------------------------------
# LOO-CV runner
# ---------------------------------------------------------------------------

def make_lr():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)),
    ])


def make_rf():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
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
# Build feature matrices for all experiments
# ---------------------------------------------------------------------------

def build_features_for_experiment(volumes, masks, codes, labels, class_id,
                                   half_width, position, threshold,
                                   use_dist_features=False):
    X_list, y_list = [], []
    for vol, msk, code in zip(volumes, masks, codes):
        if code not in labels:
            continue
        voxels = get_mask_voxels(msk, class_id)
        if len(voxels) == 0:
            # still need to append a zero row to keep alignment - but skip patient
            continue
        w_lo, w_hi = get_slab_bounds(voxels, vol.shape, half_width, position)
        intensities = get_slab_intensities(vol, msk, class_id, w_lo, w_hi)
        feats = base_features(intensities, threshold)
        if use_dist_features:
            feats = np.concatenate([feats, distribution_features(intensities)])
        X_list.append(feats)
        y_list.append(labels[code])
    if not X_list:
        return np.empty((0, 6)), np.empty(0, dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Ethmoid Sinus Filling — Systematic Analysis ===\n")
    print(f"Data dir : {DATA_DIR}")
    print(f"XLSX     : {XLSX_PATH}\n")

    # Load data
    labels_all = load_classification_labels(XLSX_PATH)

    print("Loading imaging data (this may take a while)...")
    volumes, masks, codes = load_data(DATA_DIR)
    print(f"Loaded {len(volumes)} patients\n")

    results = []  # (experiment_name, side, auc, bal_acc, n_pos, n_total)

    sides = [
        ('Left',  1, 'left_ethmoid_filled'),
        ('Right', 2, 'right_ethmoid_filled'),
    ]

    def run_exp(name, half_width, position, threshold, use_dist, clf_factory=make_lr):
        for side_name, class_id, label_key in sides:
            # build label dict for this task
            task_labels = {code: labels_all[code][label_key]
                           for code in codes if code in labels_all}
            X, y = build_features_for_experiment(
                volumes, masks, codes, task_labels, class_id,
                half_width, position, threshold, use_dist
            )
            if len(X) < 4:
                continue
            n_pos = int(y.sum())
            n_total = len(y)
            auc, bal_acc = loo_cv(X, y, clf_factory)
            results.append({
                'experiment': name,
                'side': side_name,
                'auc': auc,
                'bal_acc': bal_acc,
                'n_pos': n_pos,
                'n_total': n_total,
            })
            print(f"  {name} | {side_name:5s} | AUC={auc:.3f} | BalAcc={bal_acc:.3f} | pos={n_pos}/{n_total}")

    # -----------------------------------------------------------------------
    # BASELINE (reproduce original)
    print("\n--- BASELINE (slab=±2, threshold=-500, LR) ---")
    run_exp("Baseline(slab±2,thr=-500,LR)", half_width=2, position='center',
            threshold=-500, use_dist=False, clf_factory=make_lr)

    # -----------------------------------------------------------------------
    # A. HU threshold sweep
    print("\n--- A. HU Threshold Sweep (slab=±2, center) ---")
    for thr in [-800, -500, -200, 0, 100]:
        run_exp(f"A_thr={thr}", half_width=2, position='center',
                threshold=thr, use_dist=False, clf_factory=make_lr)

    # -----------------------------------------------------------------------
    # B. Sagittal slab width sweep
    print("\n--- B. Slab Width Sweep (threshold=-500, center) ---")
    slab_configs = [
        ('±0(single)', 0),
        ('±1',         1),
        ('±2',         2),
        ('±5',         5),
        ('full3D',    999),  # we'll handle full3D via position='full'
    ]
    for label, hw in slab_configs:
        pos = 'full' if hw == 999 else 'center'
        hw_eff = 0 if hw == 999 else hw
        run_exp(f"B_slab={label}", half_width=hw_eff, position=pos,
                threshold=-500, use_dist=False, clf_factory=make_lr)

    # -----------------------------------------------------------------------
    # C. Distribution shape features (slab=±2, threshold=-500)
    print("\n--- C. Distribution Shape Features (slab=±2, thr=-500) ---")
    run_exp("C_base_only", half_width=2, position='center',
            threshold=-500, use_dist=False, clf_factory=make_lr)
    run_exp("C_base+dist", half_width=2, position='center',
            threshold=-500, use_dist=True, clf_factory=make_lr)

    # -----------------------------------------------------------------------
    # D. Sagittal position (slab=±2, threshold=-500)
    # Also find which position gives highest filled_ratio variance
    print("\n--- D. Sagittal Position (slab=±2, thr=-500) ---")

    # Compute variance of filled_ratio across patients for each position to pick best
    positions_to_test = ['anterior', 'middle', 'posterior', 'center']
    best_pos = {}
    for side_name, class_id, label_key in sides:
        task_labels = {code: labels_all[code][label_key]
                       for code in codes if code in labels_all}
        best_var = -1
        best_p = 'center'
        for pos in positions_to_test:
            ratios = []
            for vol, msk, code in zip(volumes, masks, codes):
                if code not in task_labels:
                    continue
                voxels = get_mask_voxels(msk, class_id)
                if len(voxels) == 0:
                    continue
                w_lo, w_hi = get_slab_bounds(voxels, vol.shape, 2, pos)
                intensities = get_slab_intensities(vol, msk, class_id, w_lo, w_hi)
                if intensities is not None and len(intensities) > 0:
                    ratios.append(float(np.mean(intensities > -500)))
            if ratios:
                v = float(np.var(ratios))
                if v > best_var:
                    best_var = v
                    best_p = pos
        best_pos[side_name] = best_p
        print(f"  Best position for {side_name}: {best_p} (filled_ratio variance={best_var:.4f})")

    for pos in positions_to_test:
        run_exp(f"D_pos={pos}", half_width=2, position=pos,
                threshold=-500, use_dist=False, clf_factory=make_lr)

    # -----------------------------------------------------------------------
    # E. Classifier comparison (slab=±2, threshold=-500, center, no dist)
    print("\n--- E. Classifier Comparison (slab=±2, thr=-500, center, no dist) ---")
    run_exp("E_LR", half_width=2, position='center',
            threshold=-500, use_dist=False, clf_factory=make_lr)
    run_exp("E_RF", half_width=2, position='center',
            threshold=-500, use_dist=False, clf_factory=make_rf)

    # -----------------------------------------------------------------------
    # F. Best combination
    # From the results so far, pick best slab width, threshold, position
    # Then test both classifiers with dist features
    print("\n--- F. Best Combination ---")

    # Gather AUCs from experiments B (slab width) and A (threshold) for each side
    # to determine best settings programmatically
    def best_setting(prefix, param_key):
        subset = [r for r in results if r['experiment'].startswith(prefix)]
        if not subset:
            return None
        best = max(subset, key=lambda r: r['auc'] if not np.isnan(r['auc']) else -1)
        return best['experiment']

    # Pick best slab from B experiments (average over sides)
    b_exps = {}
    for r in results:
        if r['experiment'].startswith('B_'):
            key = r['experiment']
            b_exps.setdefault(key, []).append(r['auc'])
    b_avg = {k: np.nanmean(v) for k, v in b_exps.items()}
    best_slab_exp = max(b_avg, key=lambda k: b_avg[k]) if b_avg else 'B_slab=±2'
    # extract half_width from best_slab_exp name
    slab_map = {'±0(single)': (0, 'center'), '±1': (1, 'center'), '±2': (2, 'center'),
                '±5': (5, 'center'), 'full3D': (0, 'full')}
    best_slab_label = best_slab_exp.replace('B_slab=', '')
    best_hw, best_pos_f = slab_map.get(best_slab_label, (2, 'center'))
    print(f"  Best slab from B: {best_slab_exp} (hw={best_hw}, pos={best_pos_f})")

    # Pick best threshold from A experiments
    a_exps = {}
    for r in results:
        if r['experiment'].startswith('A_'):
            key = r['experiment']
            a_exps.setdefault(key, []).append(r['auc'])
    a_avg = {k: np.nanmean(v) for k, v in a_exps.items()}
    best_thr_exp = max(a_avg, key=lambda k: a_avg[k]) if a_avg else 'A_thr=-500'
    best_thr = int(best_thr_exp.replace('A_thr=', ''))
    print(f"  Best threshold from A: {best_thr_exp} (thr={best_thr})")

    # Best position from D
    d_exps = {}
    for r in results:
        if r['experiment'].startswith('D_'):
            key = r['experiment']
            d_exps.setdefault(key, []).append(r['auc'])
    d_avg = {k: np.nanmean(v) for k, v in d_exps.items()}
    best_pos_exp = max(d_avg, key=lambda k: d_avg[k]) if d_avg else 'D_pos=center'
    best_pos_d = best_pos_exp.replace('D_pos=', '')
    print(f"  Best position from D: {best_pos_exp} (pos={best_pos_d})")

    # Combination experiments
    for use_dist in [False, True]:
        for clf_name, clf_f in [('LR', make_lr), ('RF', make_rf)]:
            dist_tag = '+dist' if use_dist else ''
            name = f"F_slab={best_slab_label},thr={best_thr},pos={best_pos_d}{dist_tag},{clf_name}"
            run_exp(name, half_width=best_hw, position=best_pos_d,
                    threshold=best_thr, use_dist=use_dist, clf_factory=clf_f)

    # -----------------------------------------------------------------------
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE — All Experiments Ranked by Mean AUC (averaged over Left+Right)")
    print("=" * 100)

    # Aggregate per experiment (mean over sides)
    exp_summary = {}
    for r in results:
        key = r['experiment']
        exp_summary.setdefault(key, []).append(r)

    rows_summary = []
    for exp_name, exp_results in exp_summary.items():
        aucs = [r['auc'] for r in exp_results]
        bals = [r['bal_acc'] for r in exp_results]
        mean_auc = float(np.nanmean(aucs))
        mean_bal = float(np.nanmean(bals))
        # individual side info
        sides_info = {r['side']: r for r in exp_results}
        rows_summary.append((mean_auc, mean_bal, exp_name, sides_info))

    rows_summary.sort(key=lambda x: x[0], reverse=True)

    header = f"{'Rank':>4}  {'Experiment':<55}  {'MeanAUC':>7}  {'MeanBalAcc':>10}  {'Left_AUC':>8}  {'Right_AUC':>9}  {'pos/n':>7}"
    print(header)
    print("-" * len(header))
    for rank, (mean_auc, mean_bal, exp_name, sides_info) in enumerate(rows_summary, 1):
        left_r = sides_info.get('Left', {})
        right_r = sides_info.get('Right', {})
        left_auc = f"{left_r.get('auc', float('nan')):.3f}" if left_r else '  N/A '
        right_auc = f"{right_r.get('auc', float('nan')):.3f}" if right_r else '  N/A '
        # use left side for pos/n (representative)
        n_pos = left_r.get('n_pos', '?')
        n_total = left_r.get('n_total', '?')
        pos_n = f"{n_pos}/{n_total}"
        print(f"{rank:>4}  {exp_name:<55}  {mean_auc:>7.3f}  {mean_bal:>10.3f}  {left_auc:>8}  {right_auc:>9}  {pos_n:>7}")

    print("\nDone.")


if __name__ == '__main__':
    main()
