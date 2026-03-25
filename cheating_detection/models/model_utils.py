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
