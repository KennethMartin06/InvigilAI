"""
preprocess.py -- Data cleaning, normalisation, SMOTE oversampling, and splitting.

All preprocessing decisions (scaler fitting, imputation) are derived from
the training set only and then applied to val and test to prevent data leakage.
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    SCALER_PATH,
    MODELS_DIR,
    USE_SMOTE,
)


def replace_invalid_values(X: np.ndarray) -> np.ndarray:
    """Replace NaN and infinite values with the per-column median."""
    X = X.copy().astype(np.float64)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        bad_mask = ~np.isfinite(col_data)
        if bad_mask.any():
            finite_vals = col_data[np.isfinite(col_data)]
            median = np.median(finite_vals) if len(finite_vals) > 0 else 0.0
            col_data[bad_mask] = median
            X[:, col] = col_data
    return X


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """Stratified train / validation / test split (70 / 15 / 15)."""
    test_size = VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO)   # = 0.30
    val_relative = VAL_RATIO / test_size                      # 0.15 / 0.30 = 0.50

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_relative),   # 0.50 -> equal val / test
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> tuple:
    """Apply SMOTE to oversample minority classes in training data only."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        if verbose:
            print("[SMOTE] imbalanced-learn not installed -- pip install imbalanced-learn")
            print("[SMOTE] Skipping SMOTE, using original training data.")
        return X_train, y_train

    unique, counts = np.unique(y_train, return_counts=True)
    if verbose:
        print("[SMOTE] Before oversampling:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt}")

    # Set k_neighbors based on smallest class (must be < count)
    min_count = counts.min()
    k = min(5, min_count - 1)
    if k < 1:
        if verbose:
            print("[SMOTE] Smallest class too small for SMOTE, skipping.")
        return X_train, y_train

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if verbose:
        unique2, counts2 = np.unique(y_resampled, return_counts=True)
        print("[SMOTE] After oversampling:")
        for cls, cnt in zip(unique2, counts2):
            print(f"  Class {cls}: {cnt}")
        print(f"[SMOTE] {len(X_train)} -> {len(X_resampled)} samples")

    return X_resampled, y_resampled


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the training features."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    """Persist the fitted scaler to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """Load a previously saved StandardScaler."""
    return joblib.load(path)


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Full preprocessing pipeline: clean -> split -> SMOTE -> normalise -> save scaler.

    SMOTE is applied AFTER splitting but BEFORE scaling, on training data only.
    """
    # 1. Clean
    X = replace_invalid_values(X)

    # 2. Split (stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    if verbose:
        print(f"[Preprocess] Train : {X_train.shape[0]} windows")
        print(f"[Preprocess] Val   : {X_val.shape[0]} windows")
        print(f"[Preprocess] Test  : {X_test.shape[0]} windows")

    # 3. SMOTE oversampling (training data only)
    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train, verbose=verbose)

    # 4. Fit scaler on train only (after SMOTE so synthetic samples are also normalized)
    scaler = fit_scaler(X_train)

    # 5. Transform all splits
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 6. Save scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_scaler(scaler)

    if verbose:
        print(f"[Preprocess] Final train size (after SMOTE): {X_train_s.shape[0]}")
        print(f"[Preprocess] Scaler saved -> {SCALER_PATH}")

    return {
        "X_train": X_train_s,
        "X_val":   X_val_s,
        "X_test":  X_test_s,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "scaler":  scaler,
    }
