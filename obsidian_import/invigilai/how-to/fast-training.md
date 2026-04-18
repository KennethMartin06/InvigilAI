---
title: Fast Training (Presentation Mode) — InvigilAI
project: invigilai
tags: [how-to, training, quickstart]
source: train_demo.py
---

# Fast Training (~20 min on CPU)

## Commands

```bash
cd ~/invigilai
source venv/bin/activate
pip install lightgbm xgboost optuna shap imbalanced-learn onnx onnxruntime
python3 train_demo.py
```

## What runs (fast mode)

| Stage | Time |
|-------|------|
| Preprocessing + SMOTE-ENN | 2–3 min |
| Random Forest (300 trees) | 1–2 min |
| LightGBM (300 trees) | 1–2 min |
| XGBoost (300 trees) | 2–3 min |
| MLP (60 epochs) | 8–12 min |
| Stacking Ensemble + Eval | 3–5 min |
| **Total** | **~20 min** |

## What's skipped (for speed)

- GAN synthesis (would add ~12 min)
- NoisyStudent self-training (~5 min)
- Temporal augmentation (~2 min)
- Transformer / TCN / GNN (~30 min combined on CPU)
- Optuna tuning (~10 min)
- SHAP (~3–5 min on CPU)

Re-enable any of these by setting `USE_*` flags at the top of `train_demo.py` back to `True`.

## Output artifacts

Saved to `cheating_detection/outputs/`:
- `confusion_matrix.png` — main result slide
- `roc_curve.png` + `precision_recall_curve.png`
- `training_curves.png` — MLP learning curves
- `calibration_curve.png`, `uncertainty_analysis.png`
- `fairness_analysis.png`, `ood_detection.png`, `model_degradation.png`
- `ablation_results.png` — multi-modal vs single-modal comparison
- `results.json` — all numeric metrics

## Troubleshooting

- **`ModuleNotFoundError: lightgbm`** → pip install wasn't run
- **Out of memory** → set `MLP_BATCH_SIZE = 32` in config.py
- **CUDA not available** → expected on CPU-only machines; training still works, just slower
