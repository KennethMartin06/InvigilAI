"""
advanced_metrics.py -- Advanced evaluation: uncertainty, cost, fairness, adversarial.

Implements:
  - Per-class uncertainty bounds (Wilson / bootstrap)
  - Cost-sensitive evaluation with custom FP/FN costs
  - Fairness metrics (disparate impact, equal opportunity, demographic parity)
  - Adversarial robustness (FGSM / PGD)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    CLASS_NAMES,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    ADVERSARIAL_EPSILON,
    DISPARATE_IMPACT_THRESHOLD,
    UNCERTAINTY_CONFIDENCE_LEVEL,
    FAIRNESS_PNG,
    OUTPUTS_DIR,
)


# -- Per-Class Uncertainty --------------------------------------------------

def wilson_interval(n_correct, n_total, confidence=UNCERTAINTY_CONFIDENCE_LEVEL):
    """Wilson score interval for a binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def per_class_uncertainty(y_true, y_pred, class_names=CLASS_NAMES,
                          confidence=UNCERTAINTY_CONFIDENCE_LEVEL, verbose=True):
    """Per-class accuracy with Wilson confidence intervals."""
    results = {}
    for c, name in enumerate(class_names):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        correct = (y_pred[mask] == c).sum()
        total = int(mask.sum())
        acc = correct / total
        lo, hi = wilson_interval(correct, total, confidence)
        results[name] = {
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n": total,
        }
        if verbose:
            print(f"[Uncertainty] {name}: {acc:.4f} "
                  f"[{lo:.3f}, {hi:.3f}] (n={total})")
    return results


def bootstrap_uncertainty(y_true, y_pred, n_boot=1000, confidence=UNCERTAINTY_CONFIDENCE_LEVEL):
    """Bootstrap CI for overall accuracy."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append((y_pred[idx] == y_true[idx]).mean())
    alpha = (1 - confidence) / 2
    return float(np.quantile(accs, alpha)), float(np.quantile(accs, 1 - alpha))


# -- Cost-Sensitive Evaluation ---------------------------------------------

def cost_sensitive_evaluation(y_true, y_pred, y_proba=None,
                               cost_fn=COST_FALSE_NEGATIVE,
                               cost_fp=COST_FALSE_POSITIVE,
                               cheating_classes=(1, 2, 3, 4),
                               verbose=True):
    """Binary cheating/no-cheating cost evaluation.

    FN = model said Normal but true was cheating
    FP = model said cheating but true was Normal
    """
    y_true_bin = np.isin(y_true, cheating_classes).astype(int)
    y_pred_bin = np.isin(y_pred, cheating_classes).astype(int)

    tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
    tn = ((y_pred_bin == 0) & (y_true_bin == 0)).sum()
    fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()

    total_cost = cost_fn * fn + cost_fp * fp
    cost_per_sample = total_cost / max(len(y_true), 1)

    # Sweep threshold to find min-cost
    best_thresh = 0.5
    best_cost = total_cost
    if y_proba is not None and y_proba.ndim == 2:
        cheat_proba = y_proba[:, list(cheating_classes)].sum(axis=1) if y_proba.shape[1] > 1 else y_proba.ravel()
        for t in np.arange(0.1, 0.95, 0.05):
            yp = (cheat_proba >= t).astype(int)
            f_n = ((yp == 0) & (y_true_bin == 1)).sum()
            f_p = ((yp == 1) & (y_true_bin == 0)).sum()
            c = cost_fn * f_n + cost_fp * f_p
            if c < best_cost:
                best_cost = c
                best_thresh = float(t)

    result = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_cost": float(total_cost),
        "cost_per_sample": float(cost_per_sample),
        "best_threshold": best_thresh,
        "best_cost": float(best_cost),
        "cost_fn": cost_fn, "cost_fp": cost_fp,
    }
    if verbose:
        print(f"[CostEval] TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"[CostEval] Total cost={total_cost:.2f}  per-sample={cost_per_sample:.4f}")
        print(f"[CostEval] Best threshold={best_thresh:.2f}  cost={best_cost:.2f}")
    return result


# -- Fairness Metrics -------------------------------------------------------

def fairness_metrics(y_true, y_pred, group_labels,
                     disparate_impact_threshold=DISPARATE_IMPACT_THRESHOLD,
                     cheating_classes=(1, 2, 3, 4),
                     verbose=True):
    """Compute disparate impact, statistical parity difference, equal opportunity."""
    y_true_bin = np.isin(y_true, cheating_classes).astype(int)
    y_pred_bin = np.isin(y_pred, cheating_classes).astype(int)

    groups = np.unique(group_labels)
    results = {}
    positive_rates = {}
    tpr_per_group = {}
    fpr_per_group = {}

    for g in groups:
        mask = group_labels == g
        if mask.sum() == 0:
            continue
        yp = y_pred_bin[mask]
        yt = y_true_bin[mask]
        pr = yp.mean()
        positive_rates[g] = float(pr)
        tpr_per_group[g] = float((yp[yt == 1]).mean()) if (yt == 1).any() else 0.0
        fpr_per_group[g] = float((yp[yt == 0]).mean()) if (yt == 0).any() else 0.0

    # Disparate impact: min(pr) / max(pr)
    if positive_rates:
        vals = list(positive_rates.values())
        max_pr = max(vals)
        min_pr = min(vals)
        di = min_pr / max_pr if max_pr > 0 else 1.0
        spd = max_pr - min_pr   # statistical parity difference

        # Equal opportunity difference: max_TPR - min_TPR
        tpr_vals = list(tpr_per_group.values())
        eod = max(tpr_vals) - min(tpr_vals) if tpr_vals else 0.0
    else:
        di = 1.0
        spd = 0.0
        eod = 0.0

    results = {
        "disparate_impact": float(di),
        "statistical_parity_difference": float(spd),
        "equal_opportunity_difference": float(eod),
        "positive_rates": {str(k): v for k, v in positive_rates.items()},
        "tpr_per_group": {str(k): v for k, v in tpr_per_group.items()},
        "fpr_per_group": {str(k): v for k, v in fpr_per_group.items()},
        "passes_80_rule": bool(di >= disparate_impact_threshold),
    }

    if verbose:
        print(f"[Fairness] Disparate impact: {di:.3f} (threshold {disparate_impact_threshold})")
        print(f"[Fairness] Statistical parity diff: {spd:.3f}")
        print(f"[Fairness] Equal opportunity diff: {eod:.3f}")
        print(f"[Fairness] 80% rule: {'PASS' if results['passes_80_rule'] else 'FAIL'}")
        for g in groups:
            print(f"  group={g}  PR={positive_rates.get(g, 0):.3f}  "
                  f"TPR={tpr_per_group.get(g, 0):.3f}  FPR={fpr_per_group.get(g, 0):.3f}")
    return results


def plot_fairness(fairness_results, save_path=FAIRNESS_PNG, verbose=True):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    prs = fairness_results["positive_rates"]
    tprs = fairness_results["tpr_per_group"]
    fprs = fairness_results["fpr_per_group"]
    groups = list(prs.keys())
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.25
    ax.bar(x - w, [prs[g] for g in groups], w, label="Positive Rate")
    ax.bar(x, [tprs[g] for g in groups], w, label="TPR")
    ax.bar(x + w, [fprs[g] for g in groups], w, label="FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("Rate")
    ax.set_title(f"Fairness by Group (DI={fairness_results['disparate_impact']:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[Plot] Fairness -> {save_path}")


# -- Adversarial Robustness ------------------------------------------------

def fgsm_attack(model, X, y, epsilon=ADVERSARIAL_EPSILON):
    """Fast Gradient Sign Method on a PyTorch model with .predict/.forward."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return X

    if not hasattr(model, "parameters"):
        # Non-torch model -- craft perturbation from feature variance
        return X + epsilon * np.sign(np.random.randn(*X.shape))

    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    model.eval()
    logits = model(X_t)
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()
    perturb = epsilon * X_t.grad.sign()
    X_adv = (X_t + perturb).detach().cpu().numpy()
    return X_adv


def pgd_attack(model, X, y, epsilon=ADVERSARIAL_EPSILON, alpha=0.01, steps=10):
    """Projected Gradient Descent on a PyTorch model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return X
    if not hasattr(model, "parameters"):
        return X + epsilon * np.sign(np.random.randn(*X.shape))

    device = next(model.parameters()).device
    X_orig = torch.tensor(X, dtype=torch.float32, device=device)
    X_adv = X_orig.clone().detach()
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    for _ in range(steps):
        X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = nn.CrossEntropyLoss()(logits, y_t)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
        X_adv = X_orig + delta
    return X_adv.cpu().numpy()


def adversarial_robustness(model, X, y, epsilons=None, attack="fgsm", verbose=True):
    """Evaluate accuracy under perturbations of varying magnitude."""
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]

    results = {}
    for eps in epsilons:
        if eps == 0.0:
            X_adv = X
        elif attack == "pgd":
            X_adv = pgd_attack(model, X, y, epsilon=eps)
        else:
            X_adv = fgsm_attack(model, X, y, epsilon=eps)

        try:
            y_pred = model.predict(X_adv) if hasattr(model, "predict") else model(X_adv).argmax(dim=1).cpu().numpy()
        except Exception:
            y_pred = y * 0
        acc = (y_pred == y).mean() if len(y_pred) == len(y) else 0.0
        results[f"eps_{eps}"] = float(acc)
        if verbose:
            print(f"[Adversarial] eps={eps}  acc={acc:.4f}")
    return results
