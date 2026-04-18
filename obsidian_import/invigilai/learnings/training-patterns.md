---
title: Training Patterns — InvigilAI
project: invigilai
tags: [ml, training, learnings]
source: cheating_detection/models/train.py
---

# Training Patterns

Distilled from `train.py` + `tune.py`. Reusable recipes for classical + deep learning tabular pipelines.

## Classical models with Optuna-tuned params

| Model | Key config | Why it works here |
|-------|-----------|-------------------|
| Random Forest | 300 trees, `class_weight="balanced"` | Robust baseline for imbalanced tabular |
| LightGBM | 300 trees, depth=7, lr=0.05 | Best tree-based model on small tabular |
| XGBoost | 300 trees, depth=6 | Industry standard, good calibration |

All three accept a `tuned_params` dict from Optuna and override defaults if `APPLY_OPTUNA_RESULTS=True`.

## MLP training recipe (the stack)

```
Focal Loss (γ=2.0) + Label smoothing (ε=0.1)
+ MixUp (α=0.4)
+ Cosine Annealing with warm restarts (T0=20, Tmult=2)
+ Linear LR warmup (first 5 epochs)
+ Gradient clipping (max_norm=1.0)
+ Stochastic Weight Averaging (starts epoch 75 of 150)
+ MC Dropout at inference (30 samples)
```

Why this stack:
- **Focal Loss** — down-weights easy Normal examples (class 0 is 76% of data)
- **MixUp** — provides smooth decision boundaries between cheating types
- **SWA** — averages the last 75 epochs of weights → ~0.5–1% accuracy gain
- **Warmup** — prevents early divergence when using cosine annealing
- **MC Dropout** — free uncertainty estimate at inference

## Sequence models (Transformer / TCN / GNN)

All three use a `_SequenceModelAdapter` so they expose `.predict()` / `.predict_proba()` on flat `(n, d)` arrays, internally reshaping to sequences of length `PIPELINE_SEQ_LEN=10`.

- **Transformer** — 3 layers, dim=128, 4 heads, CLS-token pooling
- **TCN** — dilated causal conv channels [32, 64, 128], kernel=3
- **GNN** — 3-layer GCN over feature-correlation graph (edge threshold=0.3)

## NoisyStudent self-training

Pattern from `train_noisy_student_rf`:
1. Train teacher on labeled data
2. Pseudo-label unlabeled data; keep only `confidence ≥ 0.80`
3. Train student (same arch) on union + noise
4. Student becomes new teacher → iterate 3×

## Reuse checklist

- [ ] Swap in domain's feature set (edit `N_BASE_FEATURES` + derived names)
- [ ] Adjust class counts → regenerate `CLASS_COUNTS` for weighting
- [ ] Choose MLP arch via `MLP_HIDDEN_LAYERS` — 3-layer pyramid works well
- [ ] Set focal γ higher (2.5–3.0) if imbalance is more severe
- [ ] Only enable SWA if you train ≥ 100 epochs
