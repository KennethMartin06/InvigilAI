"""
evaluate.py -- Comprehensive evaluation, ablation, threshold, curve analysis,
SHAP, calibration, MC Dropout uncertainty, and feature importance comparison.

v5: Added cost-sensitive evaluation, fairness metrics, adversarial robustness
    (FGSM/PGD), OOD detection (Mahalanobis), concept drift (PSI), model
    degradation tracking, per-class uncertainty (Wilson CIs), and conformal
    prediction set sizes.
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
    # Advanced eval
    RUN_ADVANCED_EVAL,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    USE_FAIRNESS_METRICS,
    FAIRNESS_DEMOGRAPHIC_GROUPS,
    DISPARATE_IMPACT_THRESHOLD,
    FAIRNESS_PNG,
    USE_ADVERSARIAL_TEST,
    ADVERSARIAL_EPSILON,
    ADVERSARIAL_PNG,
    USE_OOD_DETECTION,
    OOD_PERCENTILE,
    OOD_PNG,
    USE_DRIFT_DETECTION,
    DRIFT_PSI_THRESHOLD,
    DRIFT_PNG,
    USE_DEGRADATION_TRACKING,
    DEGRADATION_PNG,
    USE_CONFORMAL_PREDICTION,
    CONFORMAL_ALPHA,
    USE_PER_CLASS_UNCERTAINTY,
    UNCERTAINTY_CONFIDENCE_LEVEL,
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


# -- 5.10 Cost-Sensitive Evaluation -------------------------------------------

def evaluate_cost_sensitive(model, X_test, y_test, verbose=True):
    """Compute total misclassification cost using asymmetric FN/FP weights."""
    y_pred = _get_preds(model, X_test)
    y_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)

    fn = int(((y_binary == 1) & (y_pred_binary == 0)).sum())
    fp = int(((y_binary == 0) & (y_pred_binary == 1)).sum())
    total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    normalized = total_cost / max(len(y_test), 1)

    result = {
        "false_negatives": fn,
        "false_positives": fp,
        "total_cost": float(total_cost),
        "cost_per_sample": float(normalized),
        "cost_fn_weight": COST_FALSE_NEGATIVE,
        "cost_fp_weight": COST_FALSE_POSITIVE,
    }
    if verbose:
        print(f"[CostEval] FN={fn} (×{COST_FALSE_NEGATIVE}), FP={fp} (×{COST_FALSE_POSITIVE})")
        print(f"[CostEval] Total cost={total_cost:.1f}, per-sample={normalized:.4f}")
    return result


# -- 5.11 Fairness Metrics ----------------------------------------------------

def evaluate_fairness(model, X_test, y_test, group_labels=None, verbose=True):
    """
    Compute fairness metrics: disparate impact and statistical parity.
    If group_labels is None, splits are simulated by random partitioning.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(y_test)
    if group_labels is None:
        n_groups = len(FAIRNESS_DEMOGRAPHIC_GROUPS)
        group_labels = rng.choice(n_groups, size=n)

    y_pred = _get_preds(model, X_test)
    results = {}

    # Disparate impact: min(P(pos|group)) / max(P(pos|group))
    positive_rates = []
    for g in np.unique(group_labels):
        mask = group_labels == g
        if mask.sum() == 0:
            continue
        rate = float((y_pred[mask] > 0).mean())
        positive_rates.append(rate)
        results[f"group_{g}_positive_rate"] = rate

    if len(positive_rates) >= 2:
        di = min(positive_rates) / max(positive_rates + [1e-9])
        results["disparate_impact"] = float(di)
        results["disparate_impact_threshold"] = DISPARATE_IMPACT_THRESHOLD
        results["disparate_impact_pass"] = bool(di >= DISPARATE_IMPACT_THRESHOLD)
        if verbose:
            status = "PASS" if results["disparate_impact_pass"] else "FAIL"
            print(f"[Fairness] Disparate impact={di:.4f} [{status}] (threshold={DISPARATE_IMPACT_THRESHOLD})")

    # Equal opportunity: TPR per group
    for g in np.unique(group_labels):
        mask = group_labels == g
        pos_mask = mask & (y_test > 0)
        if pos_mask.sum() > 0:
            tpr = float((y_pred[pos_mask] > 0).mean())
            results[f"group_{g}_tpr"] = tpr

    # Plot fairness bar chart
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = [k for k in results if k.endswith("_positive_rate")]
        vals = [results[k] for k in groups]
        ax.bar(groups, vals, color="steelblue")
        ax.axhline(max(vals) * DISPARATE_IMPACT_THRESHOLD, color="red", linestyle="--",
                   label=f"80% threshold ({max(vals)*DISPARATE_IMPACT_THRESHOLD:.3f})")
        ax.set_ylabel("Positive Prediction Rate")
        ax.set_title("Fairness — Positive Rate by Demographic Group")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        plt.savefig(FAIRNESS_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"[Plot] Fairness analysis -> {FAIRNESS_PNG}")
    except Exception as e:
        if verbose:
            print(f"[Fairness] Plot skipped ({e})")

    return results


# -- 5.12 Adversarial Robustness (FGSM) ---------------------------------------

def evaluate_adversarial_robustness(model, X_test, y_test, verbose=True):
    """
    Apply FGSM perturbation to X_test and measure accuracy drop.
    Works for any model exposing predict_proba; gradient sign is approximated
    via finite differences when autograd is unavailable.
    """
    y_clean = _get_preds(model, X_test)
    acc_clean = float((y_clean == y_test).mean())

    eps = ADVERSARIAL_EPSILON
    X_adv = X_test.copy()

    try:
        if isinstance(model, MLP):
            import torch
            x_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
            device = next(model.parameters()).device
            x_t = x_t.to(device)
            logits = model(x_t)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.tensor(y_test, dtype=torch.long).to(device),
            )
            loss.backward()
            with torch.no_grad():
                X_adv = (X_test + eps * x_t.grad.cpu().numpy().sign()).astype(np.float32)
        else:
            # Finite-difference sign approximation
            delta = 1e-4
            proba_base = model.predict_proba(X_test)
            signs = np.zeros_like(X_test)
            for j in range(X_test.shape[1]):
                X_plus = X_test.copy()
                X_plus[:, j] += delta
                proba_plus = model.predict_proba(X_plus)
                grad_j = (proba_plus - proba_base).sum(axis=1)
                signs[:, j] = np.sign(grad_j)
            X_adv = X_test + eps * signs
    except Exception as e:
        if verbose:
            print(f"[Adversarial] FGSM gradient step skipped ({e}), using random noise")
        rng = np.random.RandomState(RANDOM_SEED)
        X_adv = X_test + eps * rng.choice([-1, 1], size=X_test.shape)

    y_adv = _get_preds(model, X_adv)
    acc_adv = float((y_adv == y_test).mean())
    drop = acc_clean - acc_adv

    result = {
        "epsilon": eps,
        "accuracy_clean": acc_clean,
        "accuracy_adversarial": acc_adv,
        "accuracy_drop": float(drop),
        "robustness_ratio": float(acc_adv / max(acc_clean, 1e-9)),
    }

    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Clean", "Adversarial (FGSM)"], [acc_clean, acc_adv],
               color=["steelblue", "tomato"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Adversarial Robustness (ε={eps})")
        for i, v in enumerate([acc_clean, acc_adv]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(ADVERSARIAL_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"[Plot] Adversarial robustness -> {ADVERSARIAL_PNG}")
    except Exception as e:
        if verbose:
            print(f"[Adversarial] Plot skipped ({e})")

    if verbose:
        print(f"[Adversarial] Clean acc={acc_clean:.4f}, Adv acc={acc_adv:.4f}, Drop={drop:.4f}")
    return result


# -- 5.13 OOD Detection (Mahalanobis) -----------------------------------------

def evaluate_ood_detection(X_train, X_test, y_test, model, verbose=True):
    """Detect out-of-distribution samples using Mahalanobis distance."""
    from cheating_detection.models.model_utils import mahalanobis_scores

    try:
        mean = X_train.mean(axis=0)
        cov = np.cov(X_train.T)
        cov_inv = np.linalg.pinv(cov)
        scores_train = mahalanobis_scores(X_train, mean, cov_inv)
        scores_test = mahalanobis_scores(X_test, mean, cov_inv)

        threshold = np.percentile(scores_train, OOD_PERCENTILE)
        ood_mask = scores_test > threshold
        n_ood = int(ood_mask.sum())

        # Accuracy on in-distribution vs OOD samples
        y_pred = _get_preds(model, X_test)
        acc_id = float((y_pred[~ood_mask] == y_test[~ood_mask]).mean()) if (~ood_mask).any() else 0.0
        acc_ood = float((y_pred[ood_mask] == y_test[ood_mask]).mean()) if ood_mask.any() else 0.0

        result = {
            "n_ood": n_ood,
            "ood_rate": float(n_ood / len(X_test)),
            "mahalanobis_threshold": float(threshold),
            "accuracy_in_distribution": acc_id,
            "accuracy_ood": acc_ood,
        }

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(scores_train, bins=40, alpha=0.6, label="Train", color="blue", density=True)
            axes[0].hist(scores_test, bins=40, alpha=0.6, label="Test", color="orange", density=True)
            axes[0].axvline(threshold, color="red", linestyle="--",
                            label=f"OOD threshold (p{OOD_PERCENTILE}={threshold:.2f})")
            axes[0].set_xlabel("Mahalanobis Distance")
            axes[0].set_ylabel("Density")
            axes[0].set_title("OOD Score Distribution")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].bar(["In-Distribution", "OOD"], [acc_id, acc_ood],
                        color=["steelblue", "tomato"])
            axes[1].set_ylabel("Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].set_title(f"Accuracy by OOD Status (n_ood={n_ood})")
            for i, v in enumerate([acc_id, acc_ood]):
                axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center")
            axes[1].grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(OOD_PNG, dpi=150, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"[Plot] OOD detection -> {OOD_PNG}")
        except Exception as e:
            if verbose:
                print(f"[OOD] Plot skipped ({e})")

        if verbose:
            print(f"[OOD] Detected {n_ood}/{len(X_test)} OOD samples ({result['ood_rate']*100:.1f}%)")
            print(f"[OOD] Acc in-dist={acc_id:.4f}, acc OOD={acc_ood:.4f}")
        return result

    except Exception as e:
        if verbose:
            print(f"[OOD] Skipped ({e})")
        return {}


# -- 5.14 Concept Drift (PSI) -------------------------------------------------

def evaluate_concept_drift(X_train, X_test, verbose=True):
    """Compute Population Stability Index (PSI) per feature to detect drift."""
    try:
        from cheating_detection.monitoring.drift_detector import compute_psi, detect_drift
        psi_per_feature = []
        for j in range(X_train.shape[1]):
            psi = compute_psi(X_train[:, j], X_test[:, j])
            psi_per_feature.append(float(psi))

        mean_psi = float(np.mean(psi_per_feature))
        max_psi = float(np.max(psi_per_feature))
        drifted = [i for i, p in enumerate(psi_per_feature) if p > DRIFT_PSI_THRESHOLD]

        result = {
            "mean_psi": mean_psi,
            "max_psi": max_psi,
            "n_drifted_features": len(drifted),
            "drifted_feature_indices": drifted,
            "psi_threshold": DRIFT_PSI_THRESHOLD,
        }

        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = ["tomato" if p > DRIFT_PSI_THRESHOLD else "steelblue"
                      for p in psi_per_feature]
            ax.bar(range(len(psi_per_feature)), psi_per_feature, color=colors)
            ax.axhline(DRIFT_PSI_THRESHOLD, color="red", linestyle="--",
                       label=f"PSI threshold ({DRIFT_PSI_THRESHOLD})")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("PSI")
            ax.set_title(f"Concept Drift — PSI per Feature (mean={mean_psi:.4f})")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(DRIFT_PNG, dpi=150, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"[Plot] Drift analysis -> {DRIFT_PNG}")
        except Exception as e:
            if verbose:
                print(f"[Drift] Plot skipped ({e})")

        if verbose:
            print(f"[Drift] Mean PSI={mean_psi:.4f}, Max PSI={max_psi:.4f}, "
                  f"Drifted features: {len(drifted)}/{X_train.shape[1]}")
        return result

    except Exception as e:
        if verbose:
            print(f"[Drift] Skipped ({e})")
        return {}


# -- 5.15 Model Degradation Tracking ------------------------------------------

def run_degradation_tracking(model, X_test, y_test, verbose=True):
    """Simulate rolling-window degradation tracking over the test set."""
    try:
        from cheating_detection.monitoring.degradation_tracker import (
            ModelDegradationTracker, plot_degradation,
        )
        proba = _get_proba(model, X_test)
        y_pred = _get_preds(model, X_test)
        baseline_acc = float((y_pred == y_test).mean())
        baseline_conf = float(proba.max(axis=1).mean())

        tracker = ModelDegradationTracker()
        tracker.set_baseline(baseline_acc, baseline_conf)

        batch_size = max(10, len(X_test) // 10)
        for i in range(0, len(X_test), batch_size):
            bx = X_test[i:i+batch_size]
            by = y_test[i:i+batch_size]
            bp = proba[i:i+batch_size]
            by_pred = _get_preds(model, bx)
            tracker.record_batch(by, by_pred, y_proba=bp)

        status = tracker.current_status()
        tracker.save_history()
        plot_degradation(tracker, save_path=DEGRADATION_PNG, verbose=verbose)

        if verbose:
            print(f"[Degradation] Status={status['status']}, "
                  f"Rolling acc={status['rolling_accuracy']:.4f}, "
                  f"Alerts={status['n_alerts']}")
        return status

    except Exception as e:
        if verbose:
            print(f"[Degradation] Skipped ({e})")
        return {}


# -- 5.16 Conformal Prediction ------------------------------------------------

def evaluate_conformal_prediction(model, X_val, y_val, X_test, y_test, verbose=True):
    """
    Compute conformal prediction sets using the RAPS (softmax score) method.
    Calibrates on val set and measures coverage + average set size on test set.
    """
    try:
        proba_val = _get_proba(model, X_val)
        proba_test = _get_proba(model, X_test)

        # Non-conformity score: 1 - P(true class)
        scores_val = 1.0 - proba_val[np.arange(len(y_val)), y_val.astype(int)]
        qhat = np.quantile(scores_val, 1 - CONFORMAL_ALPHA)

        # Prediction sets
        prediction_sets = proba_test >= (1.0 - qhat)
        set_sizes = prediction_sets.sum(axis=1)

        # Coverage: fraction of test samples where true class is in set
        coverage = float(prediction_sets[np.arange(len(y_test)), y_test.astype(int)].mean())
        avg_set_size = float(set_sizes.mean())

        result = {
            "conformal_alpha": CONFORMAL_ALPHA,
            "target_coverage": 1 - CONFORMAL_ALPHA,
            "achieved_coverage": coverage,
            "average_set_size": avg_set_size,
            "qhat": float(qhat),
        }
        if verbose:
            print(f"[Conformal] Target coverage={1-CONFORMAL_ALPHA:.2f}, "
                  f"Achieved={coverage:.4f}, Avg set size={avg_set_size:.2f}")
        return result

    except Exception as e:
        if verbose:
            print(f"[Conformal] Skipped ({e})")
        return {}


# -- 5.17 Per-Class Uncertainty (Wilson CIs) ----------------------------------

def evaluate_per_class_uncertainty(model, X_test, y_test, verbose=True):
    """Compute Wilson score confidence intervals for accuracy per class."""
    from cheating_detection.models.model_utils import confidence_intervals
    y_pred = _get_preds(model, X_test)
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = y_test == c
        if not mask.any():
            continue
        acc, lo, hi = confidence_intervals(y_test[mask], y_pred[mask],
                                           confidence=UNCERTAINTY_CONFIDENCE_LEVEL)
        results[name] = {"accuracy": acc, "ci_lower": lo, "ci_upper": hi}
        if verbose:
            print(f"[PerClassCI] {name}: acc={acc:.4f} [{lo:.4f}, {hi:.4f}]")
    return results


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

    # -- Advanced evaluation block (v5) ----------------------------------------
    cost_results = {}
    fairness_results = {}
    adversarial_results = {}
    ood_results = {}
    drift_results = {}
    degradation_results = {}
    conformal_results = {}
    per_class_ci = {}

    if RUN_ADVANCED_EVAL:
        # 5.10 Cost-sensitive
        if verbose:
            print("\n" + "="*60)
            print("  STEP 5.10 -- Cost-Sensitive Evaluation")
            print("="*60)
        try:
            cost_results = evaluate_cost_sensitive(best_model, X_test, y_test, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Cost-sensitive eval skipped ({e})")

        # 5.11 Fairness
        if USE_FAIRNESS_METRICS:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.11 -- Fairness Metrics")
                print("="*60)
            try:
                fairness_results = evaluate_fairness(best_model, X_test, y_test, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  Fairness eval skipped ({e})")

        # 5.12 Adversarial robustness
        if USE_ADVERSARIAL_TEST:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.12 -- Adversarial Robustness (FGSM)")
                print("="*60)
            try:
                adversarial_results = evaluate_adversarial_robustness(
                    best_model, X_test, y_test, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  Adversarial eval skipped ({e})")

        # 5.13 OOD detection
        if USE_OOD_DETECTION:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.13 -- OOD Detection (Mahalanobis)")
                print("="*60)
            try:
                ood_results = evaluate_ood_detection(X_train, X_test, y_test,
                                                     best_model, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  OOD eval skipped ({e})")

        # 5.14 Concept drift
        if USE_DRIFT_DETECTION:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.14 -- Concept Drift (PSI)")
                print("="*60)
            try:
                drift_results = evaluate_concept_drift(X_train, X_test, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  Drift eval skipped ({e})")

        # 5.15 Degradation tracking
        if USE_DEGRADATION_TRACKING:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.15 -- Model Degradation Tracking")
                print("="*60)
            try:
                degradation_results = run_degradation_tracking(
                    best_model, X_test, y_test, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  Degradation tracking skipped ({e})")

        # 5.16 Conformal prediction
        if USE_CONFORMAL_PREDICTION:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.16 -- Conformal Prediction")
                print("="*60)
            try:
                conformal_results = evaluate_conformal_prediction(
                    best_model, X_val, y_val, X_test, y_test, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  Conformal prediction skipped ({e})")

        # 5.17 Per-class uncertainty (Wilson CIs)
        if USE_PER_CLASS_UNCERTAINTY:
            if verbose:
                print("\n" + "="*60)
                print("  STEP 5.17 -- Per-Class Uncertainty (Wilson CIs)")
                print("="*60)
            try:
                per_class_ci = evaluate_per_class_uncertainty(
                    best_model, X_test, y_test, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  Per-class uncertainty skipped ({e})")

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
    if cost_results:
        full_results["cost_sensitive"] = cost_results
    if fairness_results:
        full_results["fairness"] = fairness_results
    if adversarial_results:
        full_results["adversarial_robustness"] = adversarial_results
    if ood_results:
        full_results["ood_detection"] = ood_results
    if drift_results:
        full_results["concept_drift"] = drift_results
    if degradation_results:
        full_results["degradation"] = degradation_results
    if conformal_results:
        full_results["conformal_prediction"] = conformal_results
    if per_class_ci:
        full_results["per_class_uncertainty"] = per_class_ci
    save_results(full_results)
    print(f"\n[Eval] Results saved -> {RESULTS_JSON}")

    return full_results
