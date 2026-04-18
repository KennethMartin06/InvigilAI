"""
smote_enn.py -- Combined over/undersampling via SMOTE + Edited Nearest Neighbors.

SMOTE-ENN generates synthetic minority examples then cleans noisy/borderline
majority samples via ENN. Produces a cleaner, more balanced training set.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import RANDOM_SEED


def apply_smote_enn(X, y, sampling_strategy="auto", k_neighbors=5,
                    n_neighbors_enn=3, verbose=True):
    """Apply SMOTE oversampling followed by ENN undersampling cleaning."""
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
    except ImportError:
        if verbose:
            print("[SMOTE-ENN] imbalanced-learn not available, using fallback")
        return _fallback_smote_enn(X, y, k_neighbors, verbose)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_SEED,
    )
    enn = EditedNearestNeighbours(n_neighbors=n_neighbors_enn)
    sampler = SMOTEENN(smote=smote, enn=enn, random_state=RANDOM_SEED)

    X_res, y_res = sampler.fit_resample(X, y)

    if verbose:
        print(f"[SMOTE-ENN] {len(y)} -> {len(y_res)} samples")
        unique, counts = np.unique(y_res, return_counts=True)
        for c, cnt in zip(unique, counts):
            print(f"  Class {c}: {cnt}")
    return X_res, y_res


def _fallback_smote_enn(X, y, k, verbose):
    """Pure-numpy SMOTE+ENN fallback."""
    rng = np.random.default_rng(RANDOM_SEED)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_aug, y_aug = [X], [y]

    for cls in unique:
        mask = y == cls
        X_cls = X[mask]
        n_gen = max_count - mask.sum()
        if n_gen <= 0 or len(X_cls) < 2:
            continue

        k_eff = min(k, len(X_cls) - 1)
        synthetic = np.zeros((n_gen, X.shape[1]))
        for i in range(n_gen):
            idx = rng.integers(0, len(X_cls))
            # find k nearest among X_cls
            diffs = X_cls - X_cls[idx]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            nn_idx = np.argsort(dists)[1:k_eff + 1]
            neighbor = X_cls[rng.choice(nn_idx)]
            alpha = rng.random()
            synthetic[i] = X_cls[idx] + alpha * (neighbor - X_cls[idx])

        X_aug.append(synthetic)
        y_aug.append(np.full(n_gen, cls))

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)

    # Simple ENN: remove samples whose majority of neighbors disagree
    keep_mask = np.ones(len(X_out), dtype=bool)
    for i in range(len(X_out)):
        diffs = X_out - X_out[i]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        nn = np.argsort(dists)[1:k + 1]
        majority = np.bincount(y_out[nn].astype(int)).argmax()
        if majority != y_out[i]:
            keep_mask[i] = False

    X_out = X_out[keep_mask]
    y_out = y_out[keep_mask]

    if verbose:
        print(f"[SMOTE-ENN fallback] {len(y)} -> {len(y_out)} samples")
    return X_out, y_out
