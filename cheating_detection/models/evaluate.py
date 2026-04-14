"""
evaluate.py — Comprehensive evaluation, ablation, threshold, and curve analysis.

All evaluations operate on the held-out test set.  Results are saved both as
PNG plots and into outputs/results.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    CLASS_NAMES,
    N_VISUAL_FEATURES,
    N_BEHAVIORAL_FEATURES,
    N_TOTAL_FEATURES,
    THRESHOLD_RANGE_START,
    THRESHOLD_RANGE_END,
    THRESHOLD_STEP,
    CONFUSION_MATRIX_PNG,
    ABLATION_PNG,
    THRESHOLD_PNG,
    ROC_PNG,
    PR_CURVE_PNG,
    RESULTS_JSON,
    OUTPUTS_DIR,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    RANDOM_SEED,
)
from cheating_detection.models.model_utils import (
    compute_metrics,
    print_classification_report,
    cheating_probability,
    save_results,
    print_summary_table,
)
from cheating_detection.models.train import train_mlp, MLP, EnsembleModel


os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Helper: get probabilities ─────────────────────────────────────────────────

def _get_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Return class probabilities regardless of model type (sklearn or PyTorch).

    Parameters
    ----------
    model : fitted sklearn estimator or MLP instance
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
    """
    if isinstance(model, EnsembleModel):
        return model.predict_proba(X)
    if isinstance(model, MLP):
        x_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        return model.predict_proba(x_t)
    return model.predict_proba(X)


def _get_preds(model, X: np.ndarray) -> np.ndarray:
    """
    Return class predictions regardless of model type.

    Parameters
    ----------
    model : fitted estimator
    X : np.ndarray

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    if isinstance(model, EnsembleModel):
        return model.predict(X)
    if isinstance(model, MLP):
        proba = _get_proba(model, X)
        return np.argmax(proba, axis=1)
    return model.predict(X)


# ── 5.1 Classification Metrics ────────────────────────────────────────────────

def evaluate_classifiers(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Evaluate all three classifiers on the test set and return metric dicts.

    Parameters
    ----------
    models : dict — {"SVM": svm_model, "Random Forest": rf_model, "MLP": mlp_model}
    X_test : np.ndarray, shape (n_test, 16)
    y_test : np.ndarray, shape (n_test,)
    verbose : bool

    Returns
    -------
    dict — {model_name: {"accuracy", "precision", "recall", "f1"}}
    """
    all_metrics = {}
    for name, model in models.items():
        y_pred = _get_preds(model, X_test)
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        if verbose:
            print_classification_report(y_test, y_pred, model_name=name)
    return all_metrics


# ── 5.2 Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "MLP",
    save_path: str = CONFUSION_MATRIX_PNG,
) -> None:
    """
    Generate and save a confusion matrix heatmap for the given model.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : test data
    model_name : str — used in the plot title.
    save_path : str — destination PNG.
    """
    y_pred = _get_preds(model, X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ── 5.3 Ablation Study ────────────────────────────────────────────────────────

def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = ABLATION_PNG,
    verbose: bool = True,
) -> dict:
    """
    Re-train the MLP on three feature subsets and compare performance.

    Subsets:
      • Visual Only   — first 8 columns
      • Behavioral Only — last 8 columns
      • Multi-Modal   — all 16 columns

    Parameters
    ----------
    X_train, y_train : training data (full 16 features, scaled)
    X_val, y_val : validation data
    X_test, y_test : test data
    save_path : str — destination PNG for bar chart.
    verbose : bool

    Returns
    -------
    dict — {subset_name: {accuracy, precision, recall, f1}}
    """
    subsets = {
        "Visual Only":     (slice(None, N_VISUAL_FEATURES), N_VISUAL_FEATURES),
        "Behavioral Only": (slice(N_VISUAL_FEATURES, None), N_BEHAVIORAL_FEATURES),
        "Multi-Modal":     (slice(None),                    N_TOTAL_FEATURES),
    }
    ablation_results = {}

    for label, (cols, in_dim) in subsets.items():
        if verbose:
            print(f"\n[Ablation] Training MLP — {label} (input_dim={in_dim})")
        model, _ = train_mlp(
            X_train[:, cols], y_train,
            X_val[:, cols],   y_val,
            input_dim=in_dim,
            verbose=False,
        )
        y_pred = _get_preds(model, X_test[:, cols])
        metrics = compute_metrics(y_test, y_pred)
        ablation_results[label] = metrics
        if verbose:
            print(
                f"  Acc={metrics['accuracy']:.4f}  "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )

    # Bar chart
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, mvals) in enumerate(ablation_results.items()):
        bars = [mvals[k] for k in metric_keys]
        ax.bar(x + i * width, bars, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Feature Modality Comparison (MLP)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation results saved → {save_path}")

    return ablation_results


# ── 5.4 Threshold Analysis ────────────────────────────────────────────────────

def threshold_analysis(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = THRESHOLD_PNG,
    verbose: bool = True,
) -> dict:
    """
    Sweep binary cheating threshold and record precision/recall at each point.

    Parameters
    ----------
    model : fitted estimator with predict_proba
    X_test : np.ndarray
    y_test : np.ndarray
    save_path : str
    verbose : bool

    Returns
    -------
    dict — {"threshold": [...], "precision": [...], "recall": [...], "optimal_theta": float}
    """
    proba = _get_proba(model, X_test)
    cheat_prob = cheating_probability(proba)
    y_binary = (y_test > 0).astype(int)

    thresholds = np.arange(THRESHOLD_RANGE_START, THRESHOLD_RANGE_END + 1e-9, THRESHOLD_STEP)
    precisions = []
    recalls = []

    for theta in thresholds:
        y_pred_binary = (cheat_prob >= theta).astype(int)
        p = precision_score(y_binary, y_pred_binary, zero_division=0)
        r = recall_score(y_binary, y_pred_binary, zero_division=0)
        precisions.append(float(p))
        recalls.append(float(r))

    # Optimal theta: recall >= 0.88 AND precision >= 0.85
    optimal_theta = None
    for theta, p, r in zip(thresholds, precisions, recalls):
        if r >= 0.88 and p >= 0.85:
            optimal_theta = float(theta)
            break

    if verbose:
        print(f"\n[Threshold] Optimal θ (P≥0.85 & R≥0.88): {optimal_theta}")
        print(f"  {'θ':>6} | {'Precision':>10} | {'Recall':>8}")
        print("  " + "-" * 32)
        for theta, p, r in zip(thresholds, precisions, recalls):
            mark = " ←" if optimal_theta is not None and abs(theta - optimal_theta) < 1e-6 else ""
            print(f"  {theta:>6.2f} | {p:>10.4f} | {r:>8.4f}{mark}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, "b-o", label="Precision", linewidth=2, markersize=6)
    ax.plot(thresholds, recalls, "r-s", label="Recall", linewidth=2, markersize=6)
    ax.axhline(0.88, color="red", linestyle=":", alpha=0.6, label="Recall target (0.88)")
    ax.axhline(0.85, color="blue", linestyle=":", alpha=0.6, label="Precision target (0.85)")
    if optimal_theta is not None:
        ax.axvline(optimal_theta, color="green", linestyle="--", linewidth=2,
                   label=f"Optimal θ = {optimal_theta:.2f}")
    ax.set_xlabel("Threshold θ")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs. Cheating Threshold (MLP)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Threshold analysis saved → {save_path}")

    return {
        "threshold": thresholds.tolist(),
        "precision": precisions,
        "recall": recalls,
        "optimal_theta": optimal_theta,
    }


# ── 5.5 ROC and Precision-Recall Curves ──────────────────────────────────────

def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "MLP",
    save_path: str = ROC_PNG,
) -> float:
    """
    Plot and save the ROC curve (binary: cheating vs normal).

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : test data
    model_name : str
    save_path : str

    Returns
    -------
    float — AUC score.
    """
    proba = _get_proba(model, X_test)
    cheat_prob = cheating_probability(proba)
    y_binary = (y_test > 0).astype(int)

    fpr, tpr, _ = roc_curve(y_binary, cheat_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})", color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name} (Binary: Cheating vs Normal)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curve saved → {save_path}  (AUC={roc_auc:.4f})")
    return float(roc_auc)


def plot_precision_recall_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "MLP",
    save_path: str = PR_CURVE_PNG,
) -> float:
    """
    Plot and save the Precision-Recall curve.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : test data
    model_name : str
    save_path : str

    Returns
    -------
    float — Average Precision score.
    """
    proba = _get_proba(model, X_test)
    cheat_prob = cheating_probability(proba)
    y_binary = (y_test > 0).astype(int)

    precision, recall, _ = precision_recall_curve(y_binary, cheat_prob)
    ap = average_precision_score(y_binary, cheat_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2,
            label=f"{model_name} (AP = {ap:.4f})", color="steelblue")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PR curve saved → {save_path}  (AP={ap:.4f})")
    return float(ap)


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_evaluation(
    models: dict,
    splits: dict,
    best_model_name: str = "MLP",
    verbose: bool = True,
) -> dict:
    """
    Orchestrate all evaluation steps and persist results.

    Parameters
    ----------
    models : dict — {"SVM": ..., "Random Forest": ..., "MLP": ...}
    splits : dict — output of preprocess() containing X_*/y_* arrays.
    best_model_name : str — which model to use for confusion matrix / curves.
    verbose : bool

    Returns
    -------
    dict — aggregated evaluation results.
    """
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    # 5.1 Metrics for all models
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.1 — Classification Metrics (Test Set)")
        print("="*60)
    clf_metrics = evaluate_classifiers(models, X_test, y_test, verbose=verbose)

    # 5.2 Confusion matrix for best model
    if verbose:
        print(f"\n[Eval] Plotting confusion matrix for {best_model_name} …")
    best_model = models[best_model_name]
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_model_name)

    # 5.3 Ablation
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.3 — Ablation Study")
        print("="*60)
    ablation = run_ablation(X_train, y_train, X_val, y_val, X_test, y_test, verbose=verbose)

    # 5.4 Threshold analysis
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.4 — Threshold Analysis")
        print("="*60)
    thresh_results = threshold_analysis(best_model, X_test, y_test, verbose=verbose)

    # 5.5 ROC & PR curves
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.5 — ROC and Precision-Recall Curves")
        print("="*60)
    roc_auc = plot_roc_curve(best_model, X_test, y_test, model_name=best_model_name)
    avg_prec = plot_precision_recall_curve(best_model, X_test, y_test, model_name=best_model_name)

    # Print comparison table
    print_summary_table(clf_metrics)

    # Aggregate and save results
    full_results = {
        "classifier_metrics": clf_metrics,
        "ablation": ablation,
        "threshold_analysis": {
            "threshold": thresh_results["threshold"],
            "precision": thresh_results["precision"],
            "recall": thresh_results["recall"],
            "optimal_theta": thresh_results["optimal_theta"],
        },
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
    }
    save_results(full_results)
    print(f"\n[Eval] Results saved → {RESULTS_JSON}")

    return full_results
