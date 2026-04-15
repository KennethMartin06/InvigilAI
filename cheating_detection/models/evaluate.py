"""
evaluate.py -- Comprehensive evaluation, ablation, threshold, curve analysis,
SHAP, calibration, MC Dropout uncertainty, and feature importance comparison.

v4: Added uncertainty estimation via MC Dropout, per-class detailed analysis,
    feature importance comparison across models.
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
    USE_SHAP,
    USE_MC_DROPOUT,
    MC_DROPOUT_SAMPLES,
    SHAP_PNG,
    CALIBRATION_PNG,
    UNCERTAINTY_PNG,
    FEATURE_IMPORTANCE_PNG,
    ALL_FEATURE_NAMES,
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


# -- Helper: get probabilities ------------------------------------------------

def _get_proba(model, X: np.ndarray) -> np.ndarray:
    """Return class probabilities regardless of model type."""
    if isinstance(model, EnsembleModel):
        return model.predict_proba(X)
    if isinstance(model, MLP):
        x_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        return model.predict_proba(x_t)
    return model.predict_proba(X)


def _get_preds(model, X: np.ndarray) -> np.ndarray:
    """Return class predictions regardless of model type."""
    if isinstance(model, EnsembleModel):
        return model.predict(X)
    if isinstance(model, MLP):
        proba = _get_proba(model, X)
        return np.argmax(proba, axis=1)
    return model.predict(X)


# -- 5.1 Classification Metrics -----------------------------------------------

def evaluate_classifiers(models, X_test, y_test, verbose=True):
    """Evaluate all classifiers on the test set and return metric dicts."""
    all_metrics = {}
    for name, model in models.items():
        y_pred = _get_preds(model, X_test)
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        if verbose:
            print_classification_report(y_test, y_pred, model_name=name)
    return all_metrics


# -- 5.2 Confusion Matrix -----------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, model_name="MLP",
                          save_path=CONFUSION_MATRIX_PNG):
    """Generate and save a confusion matrix heatmap."""
    y_pred = _get_preds(model, X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix -- {model_name}", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved -> {save_path}")


# -- 5.3 Ablation Study -------------------------------------------------------

def run_ablation(X_train, y_train, X_val, y_val, X_test, y_test,
                 save_path=ABLATION_PNG, verbose=True):
    """Re-train MLP on feature subsets and compare performance."""
    subsets = {
        "Visual Only":     (slice(None, N_VISUAL_FEATURES), N_VISUAL_FEATURES),
        "Behavioral Only": (slice(N_VISUAL_FEATURES, N_VISUAL_FEATURES + N_BEHAVIORAL_FEATURES),
                           N_BEHAVIORAL_FEATURES),
        "Multi-Modal":     (slice(None), N_TOTAL_FEATURES),
    }
    ablation_results = {}

    for label, (cols, in_dim) in subsets.items():
        if verbose:
            print(f"\n[Ablation] Training MLP -- {label} (input_dim={in_dim})")
        model, _ = train_mlp(
            X_train[:, cols], y_train,
            X_val[:, cols], y_val,
            input_dim=in_dim, verbose=False,
        )
        y_pred = _get_preds(model, X_test[:, cols])
        metrics = compute_metrics(y_test, y_pred)
        ablation_results[label] = metrics
        if verbose:
            print(f"  Acc={metrics['accuracy']:.4f}  "
                  f"P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  "
                  f"F1={metrics['f1']:.4f}")

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
    ax.set_title("Ablation Study -- Feature Modality Comparison (MLP)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation results saved -> {save_path}")

    return ablation_results


# -- 5.4 Threshold Analysis ---------------------------------------------------

def threshold_analysis(model, X_test, y_test, save_path=THRESHOLD_PNG, verbose=True):
    """Sweep binary cheating threshold and record precision/recall."""
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
        print(f"\n[Threshold] Optimal t (P>=0.85 & R>=0.88): {optimal_theta}")
        print(f"  {'t':>6} | {'Precision':>10} | {'Recall':>8}")
        print("  " + "-" * 32)
        for theta, p, r in zip(thresholds, precisions, recalls):
            mark = " <-" if optimal_theta is not None and abs(theta - optimal_theta) < 1e-6 else ""
            print(f"  {theta:>6.2f} | {p:>10.4f} | {r:>8.4f}{mark}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, "b-o", label="Precision", linewidth=2, markersize=6)
    ax.plot(thresholds, recalls, "r-s", label="Recall", linewidth=2, markersize=6)
    ax.axhline(0.88, color="red", linestyle=":", alpha=0.6, label="Recall target (0.88)")
    ax.axhline(0.85, color="blue", linestyle=":", alpha=0.6, label="Precision target (0.85)")
    if optimal_theta is not None:
        ax.axvline(optimal_theta, color="green", linestyle="--", linewidth=2,
                   label=f"Optimal t = {optimal_theta:.2f}")
    ax.set_xlabel("Threshold t")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs. Cheating Threshold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Threshold analysis saved -> {save_path}")

    return {
        "threshold": thresholds.tolist(),
        "precision": precisions,
        "recall": recalls,
        "optimal_theta": optimal_theta,
    }


# -- 5.5 ROC and Precision-Recall Curves --------------------------------------

def plot_roc_curve(model, X_test, y_test, model_name="MLP", save_path=ROC_PNG):
    """Plot and save the ROC curve (binary: cheating vs normal)."""
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
    ax.set_title(f"ROC Curve -- {model_name} (Binary: Cheating vs Normal)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curve saved -> {save_path}  (AUC={roc_auc:.4f})")
    return float(roc_auc)


def plot_precision_recall_curve(model, X_test, y_test, model_name="MLP",
                                save_path=PR_CURVE_PNG):
    """Plot and save the Precision-Recall curve."""
    proba = _get_proba(model, X_test)
    cheat_prob = cheating_probability(proba)
    y_binary = (y_test > 0).astype(int)

    precision, recall, _ = precision_recall_curve(y_binary, cheat_prob)
    ap = average_precision_score(y_binary, cheat_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, label=f"{model_name} (AP = {ap:.4f})", color="steelblue")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve -- {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] PR curve saved -> {save_path}  (AP={ap:.4f})")
    return float(ap)


# -- 5.6 SHAP Feature Importance ----------------------------------------------

def plot_shap_importance(rf_model, X_test, save_path=SHAP_PNG, verbose=True):
    """Plot SHAP feature importance for the Random Forest model."""
    import shap

    if verbose:
        print("[SHAP] Computing SHAP values (TreeExplainer) ...")

    explainer = shap.TreeExplainer(rf_model)
    n_sample = min(1000, len(X_test))
    X_sample = X_test[:n_sample]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=1)

    feature_names = ALL_FEATURE_NAMES[:len(mean_shap)]
    indices = np.argsort(mean_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(mean_shap))
    top_idx = indices[:top_n]
    ax.barh(range(top_n), mean_shap[top_idx][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[SHAP] Top 5 features:")
        for rank, idx in enumerate(indices[:5]):
            print(f"  {rank+1}. {feature_names[idx]}: {mean_shap[idx]:.4f}")
        print(f"[Plot] SHAP importance -> {save_path}")


# -- 5.7 Calibration Curve ----------------------------------------------------

def plot_calibration_curve(model, X_test, y_test, model_name="Ensemble",
                           save_path=CALIBRATION_PNG, verbose=True):
    """Plot reliability diagram (calibration curve) for binary cheating detection."""
    from sklearn.calibration import calibration_curve as sk_calibration_curve

    proba = _get_proba(model, X_test)
    cheat_prob = cheating_probability(proba)
    y_binary = (y_test > 0).astype(int)

    prob_true, prob_pred = sk_calibration_curve(y_binary, cheat_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label=model_name, linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve -- {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[Plot] Calibration curve -> {save_path}")


# -- 5.8 MC Dropout Uncertainty -----------------------------------------------

def plot_uncertainty_analysis(mlp_model, X_test, y_test,
                              save_path=UNCERTAINTY_PNG, verbose=True):
    """Analyze prediction uncertainty using MC Dropout."""
    if not isinstance(mlp_model, MLP):
        if verbose:
            print("[Uncertainty] Skipping -- model is not MLP")
        return None

    if verbose:
        print(f"[Uncertainty] Running MC Dropout ({MC_DROPOUT_SAMPLES} forward passes) ...")

    x_t = torch.tensor(X_test, dtype=torch.float32)
    if next(mlp_model.parameters()).is_cuda:
        x_t = x_t.cuda()

    mean_proba, std_proba = mlp_model.predict_proba_mc(x_t, n_samples=MC_DROPOUT_SAMPLES)

    # Per-sample uncertainty = mean std across classes
    uncertainty = std_proba.mean(axis=1)

    # Predictions from mean probabilities
    y_pred = np.argmax(mean_proba, axis=1)
    correct = (y_pred == y_test)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Uncertainty distribution
    axes[0].hist(uncertainty[correct], bins=30, alpha=0.7, label="Correct", color="green", density=True)
    axes[0].hist(uncertainty[~correct], bins=30, alpha=0.7, label="Incorrect", color="red", density=True)
    axes[0].set_xlabel("Prediction Uncertainty (std)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Uncertainty Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Uncertainty by class
    class_uncertainties = []
    for c in range(len(CLASS_NAMES)):
        mask = y_test == c
        if mask.any():
            class_uncertainties.append(uncertainty[mask])
        else:
            class_uncertainties.append(np.array([0]))

    axes[1].boxplot(class_uncertainties, labels=CLASS_NAMES)
    axes[1].set_ylabel("Uncertainty (std)")
    axes[1].set_title("Uncertainty by True Class")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Accuracy vs confidence threshold
    thresholds = np.linspace(0, uncertainty.max(), 20)
    accs = []
    coverages = []
    for t in thresholds:
        mask = uncertainty <= t
        if mask.sum() > 0:
            accs.append((y_pred[mask] == y_test[mask]).mean())
            coverages.append(mask.mean())
        else:
            accs.append(0)
            coverages.append(0)

    ax3 = axes[2]
    ax3.plot(thresholds, accs, "b-o", label="Accuracy", markersize=4)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(thresholds, coverages, "r-s", label="Coverage", markersize=4)
    ax3.set_xlabel("Max Uncertainty Threshold")
    ax3.set_ylabel("Accuracy", color="blue")
    ax3_twin.set_ylabel("Coverage", color="red")
    ax3.set_title("Accuracy vs Coverage (Selective Prediction)")
    ax3.grid(alpha=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        mean_unc_correct = uncertainty[correct].mean() if correct.any() else 0
        mean_unc_incorrect = uncertainty[~correct].mean() if (~correct).any() else 0
        print(f"[Uncertainty] Mean uncertainty (correct):   {mean_unc_correct:.4f}")
        print(f"[Uncertainty] Mean uncertainty (incorrect): {mean_unc_incorrect:.4f}")
        print(f"[Uncertainty] Uncertainty ratio: {mean_unc_incorrect / (mean_unc_correct + 1e-8):.2f}x")
        print(f"[Plot] Uncertainty analysis -> {save_path}")

    return {
        "mean_uncertainty_correct": float(mean_unc_correct),
        "mean_uncertainty_incorrect": float(mean_unc_incorrect),
    }


# -- 5.9 Feature Importance Comparison ----------------------------------------

def plot_feature_importance_comparison(models, X_test, y_test,
                                       save_path=FEATURE_IMPORTANCE_PNG, verbose=True):
    """Compare feature importance across RF and LightGBM models."""
    importances = {}

    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importances[name] = imp

    if len(importances) < 1:
        if verbose:
            print("[FeatImp] No models with feature_importances_, skipping.")
        return

    n_feats = len(list(importances.values())[0])
    feature_names = ALL_FEATURE_NAMES[:n_feats]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(n_feats)
    width = 0.8 / len(importances)

    for i, (name, imp) in enumerate(importances.items()):
        ax.bar(x + i * width, imp, width, label=name, alpha=0.8)

    ax.set_xticks(x + width * (len(importances) - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature Importance")
    ax.set_title("Feature Importance Comparison Across Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[Plot] Feature importance comparison -> {save_path}")


# -- Full evaluation pipeline -------------------------------------------------

def run_full_evaluation(models, splits, best_model_name="Ensemble", verbose=True):
    """Orchestrate all evaluation steps and persist results."""
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    # 5.1 Metrics for all models
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.1 -- Classification Metrics (Test Set)")
        print("="*60)
    clf_metrics = evaluate_classifiers(models, X_test, y_test, verbose=verbose)

    # 5.2 Confusion matrix for best model
    if verbose:
        print(f"\n[Eval] Plotting confusion matrix for {best_model_name} ...")
    best_model = models[best_model_name]
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_model_name)

    # 5.3 Ablation
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.3 -- Ablation Study")
        print("="*60)
    ablation = run_ablation(X_train, y_train, X_val, y_val, X_test, y_test, verbose=verbose)

    # 5.4 Threshold analysis
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.4 -- Threshold Analysis")
        print("="*60)
    thresh_results = threshold_analysis(best_model, X_test, y_test, verbose=verbose)

    # 5.5 ROC & PR curves
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.5 -- ROC and Precision-Recall Curves")
        print("="*60)
    roc_auc = plot_roc_curve(best_model, X_test, y_test, model_name=best_model_name)
    avg_prec = plot_precision_recall_curve(best_model, X_test, y_test, model_name=best_model_name)

    # 5.6 SHAP feature importance
    shap_done = False
    if USE_SHAP and "Random Forest" in models:
        if verbose:
            print("\n" + "="*60)
            print("  STEP 5.6 -- SHAP Feature Importance")
            print("="*60)
        try:
            plot_shap_importance(models["Random Forest"], X_test, verbose=verbose)
            shap_done = True
        except Exception as e:
            if verbose:
                print(f"  SHAP skipped ({e})")

    # 5.7 Calibration curve
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.7 -- Calibration Curve")
        print("="*60)
    try:
        plot_calibration_curve(best_model, X_test, y_test, model_name=best_model_name)
    except Exception as e:
        if verbose:
            print(f"  Calibration plot skipped ({e})")

    # 5.8 MC Dropout Uncertainty
    uncertainty_results = None
    if USE_MC_DROPOUT and "MLP" in models:
        if verbose:
            print("\n" + "="*60)
            print("  STEP 5.8 -- MC Dropout Uncertainty Analysis")
            print("="*60)
        try:
            uncertainty_results = plot_uncertainty_analysis(
                models["MLP"], X_test, y_test, verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  Uncertainty analysis skipped ({e})")

    # 5.9 Feature importance comparison
    if verbose:
        print("\n" + "="*60)
        print("  STEP 5.9 -- Feature Importance Comparison")
        print("="*60)
    try:
        plot_feature_importance_comparison(models, X_test, y_test, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  Feature importance comparison skipped ({e})")

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
    if uncertainty_results:
        full_results["uncertainty"] = uncertainty_results
    save_results(full_results)
    print(f"\n[Eval] Results saved -> {RESULTS_JSON}")

    return full_results
