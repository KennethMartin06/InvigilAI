"""
preprocess.py -- Data cleaning, derived features, SMOTE-ENN, augmentation, scaling.

v5: 16 derived features (16 -> 32 dim), Borderline-SMOTE+ENN, data augmentation.
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    SCALER_PATH,
    MODELS_DIR,
    USE_SMOTE,
    USE_BORDERLINE_SMOTE,
    USE_SMOTE_ENN,
    USE_DATA_AUGMENTATION,
    USE_KNN_IMPUTATION,
    KNN_IMPUTE_NEIGHBORS,
    USE_ROBUST_SCALER,
    USE_FEATURE_SELECTION,
    FEATURE_SELECTION_K,
    N_BASE_FEATURES,
)


def replace_invalid_values(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with KNN imputation or per-column median."""
    X = X.copy().astype(np.float64)

    has_bad = False
    for col in range(X.shape[1]):
        if not np.all(np.isfinite(X[:, col])):
            has_bad = True
            break

    if not has_bad:
        return X

    if USE_KNN_IMPUTATION:
        try:
            from sklearn.impute import KNNImputer
            X[~np.isfinite(X)] = np.nan
            imputer = KNNImputer(n_neighbors=KNN_IMPUTE_NEIGHBORS)
            X = imputer.fit_transform(X)
            return X
        except ImportError:
            pass

    # Fallback: median imputation
    for col in range(X.shape[1]):
        col_data = X[:, col]
        bad_mask = ~np.isfinite(col_data)
        if bad_mask.any():
            finite_vals = col_data[np.isfinite(col_data)]
            median = np.median(finite_vals) if len(finite_vals) > 0 else 0.0
            col_data[bad_mask] = median
            X[:, col] = col_data
    return X


def _safe_entropy(values):
    """Compute entropy of a vector (treating it as a discrete distribution)."""
    eps = 1e-10
    abs_vals = np.abs(values) + eps
    total = abs_vals.sum()
    if total < eps:
        return 0.0
    p = abs_vals / total
    return -np.sum(p * np.log(p + eps))


def add_derived_features(X: np.ndarray) -> np.ndarray:
    """Add 16 derived features from the base 16 features.

    Input:  (n, 16) -- [visual(8) | behavioral(8)]
    Output: (n, 32) -- [visual(8) | behavioral(8) | derived(16)]
    """
    if X.shape[1] > N_BASE_FEATURES:
        return X  # already has derived features

    eps = 1e-6

    # Base features
    gaze_yaw    = X[:, 0]
    gaze_pitch  = X[:, 1]
    head_yaw    = X[:, 2]
    head_pitch  = X[:, 3]
    head_roll   = X[:, 4]
    gaze_dev_ratio       = X[:, 6]
    keystroke_rate       = X[:, 8]
    mean_dwell_time      = X[:, 9]
    mean_flight_time     = X[:, 10]
    burst_coefficient    = X[:, 11]
    cursor_velocity      = X[:, 12]
    click_frequency      = X[:, 13]
    idle_ratio           = X[:, 14]
    trajectory_linearity = X[:, 15]

    # Original 4 derived features
    gaze_speed = np.sqrt(gaze_yaw**2 + gaze_pitch**2)
    head_mag   = np.sqrt(head_yaw**2 + head_pitch**2 + head_roll**2)
    ks_irreg   = burst_coefficient * keystroke_rate
    act_imbal  = cursor_velocity / (keystroke_rate + eps)

    # v4 batch (6)
    typing_rhythm         = mean_dwell_time / (mean_flight_time + eps)
    interaction_intensity = keystroke_rate * click_frequency
    gaze_head_coupling    = gaze_speed * head_mag
    suspicious_idle       = idle_ratio * burst_coefficient
    trajectory_dev        = (1.0 - trajectory_linearity) * cursor_velocity
    engagement            = (1.0 - idle_ratio) * keystroke_rate

    # v5 batch (6)
    gaze_fixation_score   = 1.0 / (gaze_speed + 1.0)
    head_gaze_divergence  = np.abs(head_mag - gaze_speed)
    keystroke_variability = np.abs(mean_dwell_time - mean_flight_time) / (mean_dwell_time + mean_flight_time + eps)
    movement_complexity   = trajectory_dev * head_mag
    focus_score           = (1.0 - gaze_dev_ratio) * (1.0 - idle_ratio)
    # Behavioral entropy: entropy over the 8 behavioral features per sample
    behavioral_entropy    = np.array([_safe_entropy(X[i, 8:16]) for i in range(len(X))])

    derived = np.column_stack([
        gaze_speed, head_mag, ks_irreg, act_imbal,
        typing_rhythm, interaction_intensity, gaze_head_coupling,
        suspicious_idle, trajectory_dev, engagement,
        gaze_fixation_score, head_gaze_divergence, keystroke_variability,
        movement_complexity, focus_score, behavioral_entropy,
    ])
    return np.hstack([X, derived])


def split_dataset(X, y):
    """Stratified train / validation / test split (70 / 15 / 15)."""
    test_size = VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO)
    val_relative = VAL_RATIO / test_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_relative), stratify=y_temp, random_state=RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, verbose=True):
    """Apply Borderline-SMOTE+ENN (or fallbacks) to balance classes."""
    try:
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    except ImportError:
        if verbose:
            print("[SMOTE] imbalanced-learn not installed, skipping.")
        return X_train, y_train

    unique, counts = np.unique(y_train, return_counts=True)
    if verbose:
        print("[SMOTE] Before:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt}")

    min_count = counts.min()
    k = min(5, min_count - 1)
    if k < 1:
        if verbose:
            print("[SMOTE] Smallest class too small, skipping.")
        return X_train, y_train

    # Step 1: Oversample with Borderline-SMOTE
    if USE_BORDERLINE_SMOTE:
        try:
            smote = BorderlineSMOTE(
                random_state=RANDOM_SEED, k_neighbors=k, kind="borderline-1",
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            if verbose:
                print("[SMOTE] Using Borderline-SMOTE (borderline-1)")
        except Exception:
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            if verbose:
                print("[SMOTE] Borderline failed, using basic SMOTE")
    else:
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    # Step 2: Clean with ENN (Edited Nearest Neighbours)
    if USE_SMOTE_ENN:
        try:
            from imblearn.under_sampling import EditedNearestNeighbours
            enn = EditedNearestNeighbours(n_neighbors=3)
            X_res, y_res = enn.fit_resample(X_res, y_res)
            if verbose:
                print("[SMOTE] Applied ENN cleaning")
        except Exception as e:
            if verbose:
                print(f"[SMOTE] ENN cleaning skipped ({e})")

    if verbose:
        unique2, counts2 = np.unique(y_res, return_counts=True)
        print("[SMOTE] After:")
        for cls, cnt in zip(unique2, counts2):
            print(f"  Class {cls}: {cnt}")
        print(f"[SMOTE] {len(X_train)} -> {len(X_res)} samples")

    return X_res, y_res


def apply_feature_selection(X_train, y_train, X_val, X_test, verbose=True):
    """Select top K features using mutual information."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    k = min(FEATURE_SELECTION_K, X_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    if verbose:
        mask = selector.get_support()
        from cheating_detection.config import ALL_FEATURE_NAMES
        selected = [ALL_FEATURE_NAMES[i] for i in range(len(mask)) if i < len(ALL_FEATURE_NAMES) and mask[i]]
        print(f"[FeatureSelect] Selected {k}/{len(mask)} features: {selected}")

    return X_train, X_val, X_test, selector


def fit_scaler(X_train):
    """Fit RobustScaler (or StandardScaler fallback) on training features."""
    if USE_ROBUST_SCALER:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, path=SCALER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path=SCALER_PATH):
    return joblib.load(path)


def preprocess(X, y, verbose=True):
    """Full pipeline: clean -> derived -> split -> augment -> SMOTE-ENN -> scale."""

    # 1. Clean
    X = replace_invalid_values(X)

    # 2. Add derived features (16 -> 32)
    X = add_derived_features(X)
    if verbose:
        print(f"[Preprocess] Features: {X.shape[1]} ({N_BASE_FEATURES} base + {X.shape[1] - N_BASE_FEATURES} derived)")

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    if verbose:
        print(f"[Preprocess] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # 4. Data augmentation (training only, before SMOTE)
    if USE_DATA_AUGMENTATION:
        try:
            from cheating_detection.preprocessing.augment import augment_minority_classes
            X_train, y_train = augment_minority_classes(X_train, y_train, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[Augment] Skipped ({e})")

    # 5. SMOTE-ENN (training only)
    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train, verbose=verbose)

    # 6. Fit scaler on train, transform all
    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 7. Optional feature selection
    selector = None
    if USE_FEATURE_SELECTION:
        X_train_s, X_val_s, X_test_s, selector = apply_feature_selection(
            X_train_s, y_train, X_val_s, X_test_s, verbose=verbose,
        )

    # 8. Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_scaler(scaler)
    if verbose:
        smote_type = "Borderline-SMOTE"
        if USE_SMOTE_ENN:
            smote_type += " + ENN"
        print(f"[Preprocess] Final train: {X_train_s.shape[0]} x {X_train_s.shape[1]} features")
        print(f"[Preprocess] Scaler: {'RobustScaler' if USE_ROBUST_SCALER else 'StandardScaler'}")
        print(f"[Preprocess] Oversampling: {smote_type}")
        if USE_DATA_AUGMENTATION:
            print(f"[Preprocess] Data augmentation: enabled")
        print(f"[Preprocess] Saved -> {SCALER_PATH}")

    return {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "selector": selector,
    }
