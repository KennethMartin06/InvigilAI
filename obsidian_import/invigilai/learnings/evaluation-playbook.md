---
title: ML Evaluation Playbook — InvigilAI
project: invigilai
tags: [ml, evaluation, monitoring, learnings]
source: cheating_detection/models/evaluate.py
---

# ML Evaluation Playbook

A superset of evaluation techniques you rarely see together. Use as a template for any ML eval file.

## Basic metrics
- `accuracy`, `precision`, `recall`, `f1` (macro-averaged for imbalance)
- Per-class classification report
- Confusion matrix heatmap

## Curve analysis
- **ROC + AUC** — binary cheating vs normal
- **Precision-Recall + AP** — more informative than ROC for imbalanced data
- **Threshold sweep** — find θ that meets (P ≥ 0.85, R ≥ 0.88) jointly

## Explainability
- **SHAP** (TreeExplainer for RF) — feature-level attribution
- **Calibration curve** — reliability diagram with 10 bins
- **MC Dropout** — 30-sample uncertainty per prediction

## Advanced / production-grade

| Technique | What it catches | Formula / Threshold |
|-----------|----------------|---------------------|
| **Cost-sensitive** | FN vs FP asymmetry | cost = FN·5 + FP·1 |
| **Fairness (disparate impact)** | Demographic bias | min/max positive rate ≥ 0.80 |
| **Equal opportunity** | TPR gap across groups | monitor per-group |
| **Adversarial (FGSM)** | Robustness to perturbation | ε=0.1 sign-gradient step |
| **OOD (Mahalanobis)** | Test samples outside train manifold | d > 95th percentile |
| **Concept drift (PSI)** | Feature distribution shift | PSI > 0.2 per feature |
| **Degradation tracking** | Accuracy decay over time | rolling window=100, alert @ 5% drop |
| **Conformal prediction** | Guaranteed coverage | α=0.10 → 90% coverage |
| **Wilson CI** | Per-class accuracy bounds | 95% confidence |

## Pattern: graceful degradation

Every advanced step is wrapped in `try/except`:
```python
try:
    result = some_advanced_eval(...)
except Exception as e:
    if verbose:
        print(f"  Skipped ({e})")
    result = {}
```
This way missing deps (e.g. `scipy`, `shap`) don't kill the pipeline.

## Output artifacts

Each technique produces:
1. A PNG visualization saved to `outputs/`
2. A numeric summary appended to `results.json`

## Reuse checklist

- [ ] Copy `evaluate.py` wholesale into a new project
- [ ] Adjust `CLASS_NAMES`, `COST_FALSE_NEGATIVE`, `OOD_PERCENTILE`
- [ ] Set `RUN_ADVANCED_EVAL=False` during development, `True` for final eval
- [ ] Persist `results.json` → feed into degradation-tracker history for longitudinal comparison
