"""
preprocess.py — Data cleaning, normalisation, and train/val/test splitting.

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
)


def replace_invalid_values(X: np.ndarray) -> np.ndarray:
    """
    Replace NaN and infinite values with the per-column median.

    Computes medians ignoring NaN/Inf so that even heavily corrupted columns
    can be salvaged.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray — cleaned array of the same shape.
    """
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / validation / test split.

    Ratios are taken from config (70 / 15 / 15).  Stratification preserves
    class proportions in every split.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
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
        test_size=(1 - val_relative),   # 0.50 → equal val / test
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on the training features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)

    Returns
    -------
    sklearn.preprocessing.StandardScaler — fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    """
    Persist the fitted scaler to disk.

    Parameters
    ----------
    scaler : StandardScaler
    path : str — destination file path (.joblib).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """
    Load a previously saved StandardScaler.

    Parameters
    ----------
    path : str

    Returns
    -------
    StandardScaler
    """
    return joblib.load(path)


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Full preprocessing pipeline: clean → split → normalise → save scaler.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix from the data generator.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    verbose : bool
        Print split statistics when True.

    Returns
    -------
    dict with keys:
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
        "scaler"
    """
    # 1. Clean
    X = replace_invalid_values(X)

    # 2. Split (stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # 3. Fit scaler on train only
    scaler = fit_scaler(X_train)

    # 4. Transform all splits
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 5. Save scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_scaler(scaler)

    if verbose:
        print(f"[Preprocess] Train : {X_train_s.shape[0]} windows")
        print(f"[Preprocess] Val   : {X_val_s.shape[0]} windows")
        print(f"[Preprocess] Test  : {X_test_s.shape[0]} windows")
        print(f"[Preprocess] Scaler saved → {SCALER_PATH}")

    return {
        "X_train": X_train_s,
        "X_val":   X_val_s,
        "X_test":  X_test_s,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "scaler":  scaler,
    }
