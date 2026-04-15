"""
main.py -- End-to-end pipeline for Multi-Modal AI Cheating Detection System.

v3: Focal Loss, LightGBM, MixUp, Cosine Annealing, Stacking Ensemble,
    derived features (20-dim), SHAP analysis, calibration curves.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.data.generate_dataset import generate_dataset
from cheating_detection.data.real_data_pipeline import run as build_combined_dataset
from cheating_detection.data.mpiigaze_pipeline import run as build_mpiigaze_dataset
from cheating_detection.data.daisee_pipeline import run as build_daisee_dataset
from cheating_detection.data.hmdb51_pipeline import run_hmdb51_pipeline, merge_with_combined as merge_hmdb51
from cheating_detection.data.custom_video_pipeline import process_video, merge_and_save as merge_custom_video
from cheating_detection.preprocessing.preprocess import preprocess
from cheating_detection.models.train import (
    train_random_forest, train_lightgbm, train_mlp, train_ensemble, plot_training_curves,
)
from cheating_detection.models.audio_classifier import train_audio_classifier
from cheating_detection.models.evaluate import run_full_evaluation
from cheating_detection.config import OUTPUTS_DIR, MODELS_DIR

_DATA_DIR = Path(__file__).parent / "data"
_COMBINED_PATH = _DATA_DIR / "combined_dataset.npz"
_CUSTOM_VIDEO  = Path.home() / "invigilai" / "datasets" / "custom_cheat.mp4"

try:
    from cheating_detection.data.librispeech_pipeline import run as train_librispeech_audio
    HAS_LIBRISPEECH = True
except ImportError:
    HAS_LIBRISPEECH = False


def _reload_combined():
    data = np.load(_COMBINED_PATH)
    return data["X"], data["y"]


def banner(text):
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    t0_total = time.time()

    # -- Step 1a: Synthetic dataset -----------------------------------------
    banner("STEP 1a -- Synthetic Dataset Generation")
    t0 = time.time()
    X, y = generate_dataset(save_npz=True, save_csv=True, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 1b: CMU Keystroke ---------------------------------------------
    banner("STEP 1b -- Augment with CMU Keystroke Data")
    t0 = time.time()
    try:
        X, y = build_combined_dataset()
        print(f"  Dataset: {len(X)} samples | Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Skipping CMU Keystroke ({e})")

    # -- Step 1c: MPIIGaze --------------------------------------------------
    banner("STEP 1c -- Augment with MPIIGaze Gaze Data")
    t0 = time.time()
    try:
        X, y = build_mpiigaze_dataset()
        if X is not None:
            print(f"  Dataset: {len(X)} samples | Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping (not found).")
    except Exception as e:
        print(f"  Skipping MPIIGaze ({e})")

    # -- Step 1d: DAiSEE ----------------------------------------------------
    banner("STEP 1d -- Augment with DAiSEE Engagement Data")
    t0 = time.time()
    try:
        X, y = build_daisee_dataset()
        if X is not None:
            print(f"  Dataset: {len(X)} samples | Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping (not found).")
    except Exception as e:
        print(f"  Skipping DAiSEE ({e})")

    # -- Step 1e: HMDB-51 ---------------------------------------------------
    banner("STEP 1e -- Augment with HMDB-51 Action Videos")
    t0 = time.time()
    try:
        X_hmdb, y_hmdb = run_hmdb51_pipeline()
        if len(X_hmdb) > 0:
            merge_hmdb51(X_hmdb, y_hmdb)
            X, y = _reload_combined()
            print(f"  HMDB-51 added {len(X_hmdb)} -> total {len(X)}")
        else:
            print("  Skipping (no samples).")
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Skipping HMDB-51 ({e})")

    # -- Step 1f: Custom video ----------------------------------------------
    banner("STEP 1f -- Augment with Custom Cheat Video")
    t0 = time.time()
    try:
        if _CUSTOM_VIDEO.exists():
            X_vid, y_vid = process_video(str(_CUSTOM_VIDEO))
            if len(X_vid) > 0:
                merge_custom_video(X_vid, y_vid)
                X, y = _reload_combined()
                print(f"  Custom video added {len(X_vid)} -> total {len(X)}")
            else:
                print("  No faces detected -- skipping.")
        else:
            print(f"  Not found: {_CUSTOM_VIDEO} -- skipping.")
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Skipping custom video ({e})")

    # -- Step 2: Preprocess (KNN impute, derived feats, SMOTE, RobustScaler)
    banner("STEP 2 -- Preprocessing (KNN impute -> derived features -> SMOTE -> RobustScaler)")
    t0 = time.time()
    splits = preprocess(X, y, verbose=True)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3a: Random Forest (balanced, 300 trees) -----------------------
    banner("STEP 3a -- Train Random Forest (300 trees, balanced)")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3b: LightGBM -------------------------------------------------
    banner("STEP 3b -- Train LightGBM (300 trees, balanced)")
    t0 = time.time()
    lgb_model = train_lightgbm(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3c: MLP (Focal Loss + MixUp + Cosine Annealing) ---------------
    banner("STEP 3c -- Train MLP (Focal Loss, MixUp, CosineAnnealing, BatchNorm)")
    t0 = time.time()
    mlp_model, history = train_mlp(X_train, y_train, X_val, y_val, verbose=True)
    plot_training_curves(history)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3d: Stacking Ensemble -----------------------------------------
    banner("STEP 3d -- Train Stacking Ensemble (RF + LGB + MLP)")
    t0 = time.time()
    base_models = [("Random Forest", rf_model), ("MLP", mlp_model)]
    if lgb_model is not None:
        base_models.insert(1, ("LightGBM", lgb_model))
    ensemble_model = train_ensemble(base_models, X_train, y_train, X_val, y_val, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3e: Audio Classifier ------------------------------------------
    banner("STEP 3e -- Train Audio Classifier (ESC-50 + LibriSpeech)")
    t0 = time.time()
    try:
        if HAS_LIBRISPEECH:
            train_librispeech_audio()
        else:
            train_audio_classifier()
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Audio training failed ({e})")
        try:
            train_audio_classifier()
        except Exception as e2:
            print(f"  Audio classifier unavailable ({e2})")

    # -- Steps 4-6: Full evaluation (SHAP, calibration, ablation) -----------
    banner("STEPS 4-6 -- Evaluation, Ablation, SHAP, Calibration")
    models = {"Random Forest": rf_model, "MLP": mlp_model, "Ensemble": ensemble_model}
    if lgb_model is not None:
        models["LightGBM"] = lgb_model
    run_full_evaluation(models, splits, best_model_name="Ensemble", verbose=True)

    # -- Summary ------------------------------------------------------------
    banner("PIPELINE COMPLETE")
    elapsed = time.time() - t0_total
    print(f"  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Improvements applied:")
    print(f"    + KNN imputation (k=5) + RobustScaler")
    print(f"    + 4 derived features (gaze_speed, head_mag, ks_irreg, act_imbal)")
    print(f"    + SMOTE oversampling for class imbalance")
    print(f"    + RF: 300 trees, balanced class weights")
    print(f"    + LightGBM: 300 trees, balanced, depth=7")
    print(f"    + MLP: [256,128,64] + BatchNorm + Focal Loss (gamma=2)")
    print(f"    + MixUp augmentation (alpha=0.4)")
    print(f"    + Cosine Annealing LR with warm restarts")
    print(f"    + Label smoothing (eps=0.1)")
    print(f"    + Stacking Ensemble (RF + LGB + MLP -> LogisticRegression)")
    print(f"    + SHAP feature importance analysis")
    print(f"    + Calibration curve")
    print(f"  Total training samples: {len(X)}")
    print()


if __name__ == "__main__":
    main()
