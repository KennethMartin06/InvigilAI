"""
model_utils.py — Shared helpers for saving, loading, and metric computation.
"""

import os
import json
import numpy as np
import joblib
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import CLASS_NAMES, RESULTS_JSON, OUTPUTS_DIR


# ── Saving / loading sklearn models ──────────────────────────────────────────

def save_sklearn_model(model, path: str) -> None:
    """
    Persist a scikit-learn estimator to disk using joblib.

    Parameters
    ----------
    model : fitted sklearn estimator
    path : str — destination path (e.g. ``models/svm_model.joblib``).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_model(path: str):
    """
    Load a joblib-serialised sklearn estimator.

    Parameters
    ----------
    path : str

    Returns
    -------
    Fitted sklearn estimator.
    """
    return joblib.load(path)


# ── Saving / loading PyTorch models ──────────────────────────────────────────

def save_pytorch_model(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model's state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
    path : str — destination .pth file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_pytorch_model(model_class, path: str, **model_kwargs) -> torch.nn.Module:
    """
    Instantiate a PyTorch model and load saved weights.

    Parameters
    ----------
    model_class : type — the nn.Module subclass (e.g. ``MLP``).
    path : str — .pth file produced by ``save_pytorch_model``.
    **model_kwargs — constructor arguments (e.g. input_dim, hidden_layers).

    Returns
    -------
    torch.nn.Module — model in eval mode.
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute macro-averaged classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1.
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> None:
    """
    Print a formatted per-class classification report.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    model_name : str — printed as a header.
    """
    header = f"Classification Report — {model_name}" if model_name else "Classification Report"
    print(f"\n{'='*60}")
    print(header)
    print('='*60)
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))


# ── Binary cheating probability ───────────────────────────────────────────────

def cheating_probability(proba: np.ndarray) -> np.ndarray:
    """
    Convert multi-class probabilities to a binary cheating score.

    Cheating probability = 1 − P(class=0 | Normal).

    Parameters
    ----------
    proba : np.ndarray, shape (n_samples, n_classes)

    Returns
    -------
    np.ndarray, shape (n_samples,) — cheating probability in [0, 1].
    """
    return 1.0 - proba[:, 0]


# ── Results JSON ──────────────────────────────────────────────────────────────

def save_results(results: dict, path: str = RESULTS_JSON) -> None:
    """
    Append / overwrite a results dictionary to a JSON file.

    Parameters
    ----------
    results : dict
    path : str — destination JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing results if present
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    existing.update(results)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ── Confidence calibration / uncertainty helpers ────────────────────────────

def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to raw logits and return softmax probabilities."""
    scaled = logits / max(temperature, 1e-6)
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Per-sample predictive entropy of class probabilities."""
    eps = 1e-12
    return -np.sum(proba * np.log(proba + eps), axis=1)


def confidence_intervals(y_true, y_pred, confidence=0.95):
    """Wilson score interval for overall accuracy."""
    from scipy.stats import norm
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    correct = int((y_true == y_pred).sum())
    acc = correct / n
    z = norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n
    center = (acc + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(acc * (1 - acc) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(acc), float(max(0.0, center - half)), float(min(1.0, center + half))


# ── OOD / cascade helpers ───────────────────────────────────────────────────

def mahalanobis_scores(X: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = X - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


def cascade_predict(primary, secondary, X, threshold: float = 0.85):
    """Route low-confidence samples from primary to secondary model."""
    p1 = primary.predict_proba(X)
    conf = p1.max(axis=1)
    preds = p1.argmax(axis=1)
    low = conf < threshold
    if low.any() and secondary is not None:
        preds[low] = secondary.predict(X[low])
    return preds, conf, low


def batch_predict(model, X, batch_size: int = 512) -> np.ndarray:
    """Memory-safe batched prediction for any model exposing .predict()."""
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model.predict(X[i:i + batch_size]))
    return np.concatenate(out) if out else np.array([])


# ── Formatted summary table ───────────────────────────────────────────────────

def print_summary_table(results: dict) -> None:
    """
    Print a formatted comparison table for all evaluated classifiers.

    Parameters
    ----------
    results : dict
        Keys are model names; values are dicts with accuracy/precision/recall/f1.
    """
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    sep = "-" * len(header)
    print(f"\n{'='*60}")
    print("  CLASSIFIER COMPARISON (Test Set — Macro-averaged)")
    print('='*60)
    print(header)
    print(sep)
    for name, m in results.items():
        print(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>8.4f} {m['f1']:>8.4f}"
        )
    print(sep)
