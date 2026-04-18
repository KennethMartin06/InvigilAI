"""
train_demo.py -- Fast presentation-mode training (~20 min CPU).

Runs: RF + LightGBM + XGBoost + MLP + Stacking Ensemble
Skips: GAN, NoisyStudent, Transformer, TCN, GNN, Optuna, SHAP (all slow)
Produces: confusion matrix, ROC, PR curve, calibration, training curves, results.json
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Override slow config knobs BEFORE any module imports them ─────────────────
import cheating_detection.config as _cfg

_cfg.USE_GAN_SYNTHETIC           = False   # skip 200-epoch GAN
_cfg.USE_NOISY_STUDENT           = False   # skip self-training
_cfg.USE_TEMPORAL_AUGMENTATION   = False   # skip time-warp
_cfg.USE_DATA_AUGMENTATION       = True    # keep lightweight augment
_cfg.USE_SMOTE                   = True    # keep SMOTE-ENN
_cfg.OPTUNA_N_TRIALS             = 0       # skip Optuna
_cfg.MLP_MAX_EPOCHS              = 60      # was 150
_cfg.MLP_PATIENCE                = 10      # was 20
_cfg.USE_SHAP                    = False   # SHAP is slow on CPU
_cfg.USE_MC_DROPOUT              = True    # fast, keep
_cfg.USE_ADVERSARIAL_TEST        = False   # skip (slow finite-diff)
_cfg.USE_DRIFT_DETECTION         = False   # skip
_cfg.USE_DEGRADATION_TRACKING    = True    # fast
_cfg.USE_CONFORMAL_PREDICTION    = True    # fast
_cfg.USE_FAIRNESS_METRICS        = True    # fast
_cfg.USE_OOD_DETECTION           = True    # fast
_cfg.USE_PER_CLASS_UNCERTAINTY   = True    # fast
_cfg.RUN_ADVANCED_EVAL           = True
_cfg.USE_ADVANCED_PIPELINE       = False   # skip Transformer/TCN/GNN
_cfg.TRAIN_TRANSFORMER_IN_PIPELINE = False
_cfg.TRAIN_TCN_IN_PIPELINE       = False
_cfg.TRAIN_GNN_IN_PIPELINE       = False
_cfg.USE_DOMAIN_ADAPTATION       = False
_cfg.USE_LSTM_FEATURES           = False
_cfg.USE_CROSS_MODAL_ATTENTION   = False

# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from cheating_detection.data.generate_dataset import generate_dataset
from cheating_detection.preprocessing.preprocess import preprocess
from cheating_detection.models.train import (
    train_random_forest, train_lightgbm, train_xgboost,
    train_mlp, train_ensemble, plot_training_curves,
)
from cheating_detection.models.evaluate import run_full_evaluation
from cheating_detection.config import OUTPUTS_DIR, MODELS_DIR

COMBINED = os.path.join(os.path.dirname(__file__),
                        "cheating_detection", "data", "combined_dataset.npz")


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    t_start = time.time()

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    banner("STEP 1 — Load Dataset")
    if os.path.exists(COMBINED):
        d = np.load(COMBINED)
        X, y = d["X"], d["y"]
        print(f"  Loaded combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        X, y = generate_dataset(save_npz=True, save_csv=True, verbose=True)
    unique, counts = np.unique(y, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  Class {int(c)}: {n} samples")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    banner("STEP 2 — Preprocessing (KNN → 32 features → SMOTE-ENN → RobustScaler)")
    t0 = time.time()
    splits = preprocess(X, y, verbose=True)
    X_train = splits["X_train"]; X_val = splits["X_val"]; X_test = splits["X_test"]
    y_train = splits["y_train"]; y_val  = splits["y_val"]; y_test  = splits["y_test"]
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3a. Random Forest ─────────────────────────────────────────────────────
    banner("STEP 3a — Random Forest (300 trees)")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3b. LightGBM ─────────────────────────────────────────────────────────
    banner("STEP 3b — LightGBM (300 trees)")
    t0 = time.time()
    lgb_model = train_lightgbm(X_train, y_train, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3c. XGBoost ───────────────────────────────────────────────────────────
    banner("STEP 3c — XGBoost (300 trees)")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3d. MLP ───────────────────────────────────────────────────────────────
    banner("STEP 3d — MLP (60 epochs · Focal Loss · MixUp · SWA · MC Dropout)")
    t0 = time.time()
    mlp_model, history = train_mlp(X_train, y_train, X_val, y_val, verbose=True)
    plot_training_curves(history)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3e. Stacking Ensemble ─────────────────────────────────────────────────
    banner("STEP 3e — Stacking Ensemble (RF + LGB + XGB + MLP)")
    t0 = time.time()
    base_models = [("Random Forest", rf_model), ("MLP", mlp_model)]
    if lgb_model: base_models.insert(1, ("LightGBM", lgb_model))
    if xgb_model: base_models.insert(-1, ("XGBoost", xgb_model))
    ensemble = train_ensemble(base_models, X_train, y_train, X_val, y_val, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    banner("STEP 4 — Full Evaluation (confusion matrix, ROC, PR, calibration, fairness, OOD ...)")
    models = {"Random Forest": rf_model, "MLP": mlp_model, "Ensemble": ensemble}
    if lgb_model: models["LightGBM"] = lgb_model
    if xgb_model: models["XGBoost"] = xgb_model
    results = run_full_evaluation(models, splits, best_model_name="Ensemble", verbose=True)

    # ── 5. ONNX Export ────────────────────────────────────────────────────────
    banner("STEP 5 — ONNX Export")
    try:
        from cheating_detection.models.export import export_mlp_to_onnx, verify_onnx_model
        path = export_mlp_to_onnx(mlp_model, verbose=True)
        if path: verify_onnx_model(path, verbose=True)
    except Exception as e:
        print(f"  ONNX skipped ({e})")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"DONE — {elapsed/60:.1f} min total")

    ens_m = results.get("classifier_metrics", {}).get("Ensemble", {})
    print(f"\n  Ensemble accuracy : {ens_m.get('accuracy', 0):.4f}")
    print(f"  Ensemble F1 (macro): {ens_m.get('f1', 0):.4f}")
    print(f"\n  Output files saved to: {OUTPUTS_DIR}")
    for f in sorted(os.listdir(OUTPUTS_DIR)):
        print(f"    {f}")
    print()


if __name__ == "__main__":
    main()
