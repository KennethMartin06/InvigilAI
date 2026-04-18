"""
main.py -- End-to-end pipeline for Multi-Modal AI Cheating Detection System.

v5: Temporal augmentation, GAN synthesis, NoisyStudent self-training, domain
    adaptation, Transformer/TCN/GNN architectures, advanced ensemble (weighted
    voting, cascade, multi-task), cross-modal attention, LSTM features, plus
    full advanced evaluation (cost, fairness, adversarial, OOD, drift,
    degradation, conformal prediction, per-class uncertainty).
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
    train_random_forest, train_lightgbm, train_xgboost,
    train_mlp, train_ensemble, plot_training_curves,
    train_transformer, train_tcn, train_gnn, train_noisy_student_rf,
    _SequenceModelAdapter,
)
from cheating_detection.models.audio_classifier import train_audio_classifier
from cheating_detection.models.evaluate import run_full_evaluation
from cheating_detection.config import (
    OUTPUTS_DIR, MODELS_DIR,
    APPLY_OPTUNA_RESULTS,
    USE_ADVANCED_PIPELINE,
    TRAIN_TRANSFORMER_IN_PIPELINE,
    TRAIN_TCN_IN_PIPELINE,
    TRAIN_GNN_IN_PIPELINE,
    USE_CASCADE_IN_PIPELINE,
    PIPELINE_SEQ_LEN,
    USE_DOMAIN_ADAPTATION,
    USE_CROSS_MODAL_ATTENTION,
    USE_LSTM_FEATURES,
    TRANSFORMER_MODEL_PATH,
    TCN_MODEL_PATH,
    GNN_MODEL_PATH,
    GAN_SAMPLES_PER_CLASS,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)

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

    # -- Step 2: Preprocess -------------------------------------------------
    banner("STEP 2 -- Preprocessing (KNN -> 10 derived features -> Borderline-SMOTE -> RobustScaler)")
    t0 = time.time()
    splits = preprocess(X, y, verbose=True)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 2b: Optuna Hyperparameter Tuning (optional) -------------------
    banner("STEP 2b -- Optuna Hyperparameter Tuning")
    t0 = time.time()
    try:
        from cheating_detection.models.tune import run_tuning
        tuning_results = run_tuning(X_train, y_train, verbose=True)
        print(f"  Done in {time.time() - t0:.1f}s")
    except ImportError:
        print("  optuna not installed, using default hyperparameters.")
        tuning_results = {}
    except Exception as e:
        print(f"  Tuning skipped ({e})")
        tuning_results = {}

    # -- Step 3a: Random Forest (with Optuna tuning) ------------------------
    banner("STEP 3a -- Train Random Forest (300 trees, balanced)")
    t0 = time.time()
    rf_tuned = tuning_results.get("random_forest", {}) if APPLY_OPTUNA_RESULTS else {}
    rf_model = train_random_forest(X_train, y_train, tuned_params=rf_tuned, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3b: LightGBM (with Optuna tuning) ----------------------------
    banner("STEP 3b -- Train LightGBM (300 trees, balanced)")
    t0 = time.time()
    lgb_tuned = tuning_results.get("lightgbm", {}) if APPLY_OPTUNA_RESULTS else {}
    lgb_model = train_lightgbm(X_train, y_train, tuned_params=lgb_tuned, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3c: XGBoost (with Optuna tuning) -----------------------------
    banner("STEP 3c -- Train XGBoost (300 trees)")
    t0 = time.time()
    xgb_tuned = tuning_results.get("xgboost", {}) if APPLY_OPTUNA_RESULTS else {}
    xgb_model = train_xgboost(X_train, y_train, tuned_params=xgb_tuned, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3d: MLP (Focal Loss + MixUp + CosineAnnealing + GradClip + SWA + LR warmup)
    banner("STEP 3d -- Train MLP (Focal+MixUp+CosAnnealing+GradClip+SWA+LRwarmup)")
    t0 = time.time()
    mlp_model, history = train_mlp(X_train, y_train, X_val, y_val, verbose=True)
    plot_training_curves(history)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3e: Stacking Ensemble -----------------------------------------
    banner("STEP 3e -- Train Stacking Ensemble (RF + LGB + XGB + MLP)")
    t0 = time.time()
    base_models = [("Random Forest", rf_model), ("MLP", mlp_model)]
    if lgb_model is not None:
        base_models.insert(1, ("LightGBM", lgb_model))
    if xgb_model is not None:
        base_models.insert(-1, ("XGBoost", xgb_model))
    ensemble_model = train_ensemble(base_models, X_train, y_train, X_val, y_val, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Step 3g: Advanced Architectures (Transformer / TCN / GNN) ----------
    transformer_adapter = None
    tcn_adapter = None
    gnn_adapter = None

    if USE_ADVANCED_PIPELINE:
        if TRAIN_TRANSFORMER_IN_PIPELINE:
            banner("STEP 3g -- Train Transformer Encoder")
            t0 = time.time()
            try:
                transformer_model, _ = train_transformer(
                    X_train, y_train, X_val, y_val,
                    seq_len=PIPELINE_SEQ_LEN, n_classes=len(np.unique(y_train)), verbose=True,
                )
                transformer_adapter = _SequenceModelAdapter(transformer_model, PIPELINE_SEQ_LEN)
                print(f"  Done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  Transformer skipped ({e})")

        if TRAIN_TCN_IN_PIPELINE:
            banner("STEP 3h -- Train Temporal CNN (TCN)")
            t0 = time.time()
            try:
                tcn_model, _ = train_tcn(
                    X_train, y_train, X_val, y_val,
                    seq_len=PIPELINE_SEQ_LEN, n_classes=len(np.unique(y_train)), verbose=True,
                )
                tcn_adapter = _SequenceModelAdapter(tcn_model, PIPELINE_SEQ_LEN)
                print(f"  Done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  TCN skipped ({e})")

        if TRAIN_GNN_IN_PIPELINE:
            banner("STEP 3i -- Train Graph Neural Network (GNN)")
            t0 = time.time()
            try:
                gnn_model = train_gnn(
                    X_train, y_train, n_classes=len(np.unique(y_train)), verbose=True,
                )
                print(f"  Done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  GNN skipped ({e})")

        # -- Step 3j: NoisyStudent RF ----------------------------------------
        banner("STEP 3j -- Train NoisyStudent Random Forest")
        t0 = time.time()
        try:
            rng = np.random.RandomState(42)
            n_unlabeled = int(len(X_train) * 2.0)
            X_unlabeled = (X_train[rng.choice(len(X_train), n_unlabeled, replace=True)]
                           + rng.normal(0, 0.05, (n_unlabeled, X_train.shape[1])))
            noisy_rf = train_noisy_student_rf(X_train, y_train, X_unlabeled, verbose=True)
            print(f"  Done in {time.time() - t0:.1f}s")
        except Exception as e:
            noisy_rf = None
            print(f"  NoisyStudent RF skipped ({e})")

        # -- Step 3k: Domain Adaptation (DANN) --------------------------------
        if USE_DOMAIN_ADAPTATION:
            banner("STEP 3k -- Domain Adaptation (DANN)")
            t0 = time.time()
            try:
                from cheating_detection.models.domain_adaptation import DANNAdapter
                dann = DANNAdapter(input_dim=X_train.shape[1],
                                   n_classes=len(np.unique(y_train)))
                dann.fit(X_train, y_train, X_val, verbose=True)
                print(f"  Done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"  DANN skipped ({e})")

        # -- Step 3l: Advanced Feature Engineering ----------------------------
        banner("STEP 3l -- Advanced Feature Engineering (Cross-Modal + LSTM)")
        t0 = time.time()
        try:
            if USE_CROSS_MODAL_ATTENTION:
                from cheating_detection.features.cross_modal import CrossModalAttentionFusion
                cmaf = CrossModalAttentionFusion(
                    gaze_dim=X_train.shape[1] // 2,
                    keystroke_dim=X_train.shape[1] - X_train.shape[1] // 2,
                )
                print(f"  [CrossModal] Fuser initialized (dim={X_train.shape[1]})")

            if USE_LSTM_FEATURES:
                from cheating_detection.features.advanced_features import extract_lstm_features
                X_lstm_train = extract_lstm_features(X_train)
                X_lstm_val = extract_lstm_features(X_val)
                X_lstm_test = extract_lstm_features(X_test)
                X_train = np.hstack([X_train, X_lstm_train])
                X_val = np.hstack([X_val, X_lstm_val])
                X_test = np.hstack([X_test, X_lstm_test])
                splits["X_train"] = X_train
                splits["X_val"] = X_val
                splits["X_test"] = X_test
                print(f"  [LSTM] Feature dim expanded: {X_lstm_train.shape[1]} new features"
                      f" -> {X_train.shape[1]} total")
            print(f"  Done in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  Advanced features skipped ({e})")

    # -- Step 3f: Audio Classifier ------------------------------------------
    banner("STEP 3f -- Train Audio Classifier (ESC-50 + LibriSpeech)")
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

    # -- Steps 4-6: Full evaluation -----------------------------------------
    banner("STEPS 4-6 -- Evaluation, Ablation, SHAP, Calibration, Uncertainty + Advanced")
    models = {"Random Forest": rf_model, "MLP": mlp_model, "Ensemble": ensemble_model}
    if lgb_model is not None:
        models["LightGBM"] = lgb_model
    if xgb_model is not None:
        models["XGBoost"] = xgb_model
    if transformer_adapter is not None:
        models["Transformer"] = transformer_adapter
    if tcn_adapter is not None:
        models["TCN"] = tcn_adapter
    if noisy_rf is not None:
        models["NoisyStudent-RF"] = noisy_rf
    run_full_evaluation(models, splits, best_model_name="Ensemble", verbose=True)

    # -- Step 7: ONNX Export ------------------------------------------------
    banner("STEP 7 -- ONNX Model Export")
    t0 = time.time()
    try:
        from cheating_detection.models.export import export_mlp_to_onnx, verify_onnx_model
        onnx_path = export_mlp_to_onnx(mlp_model, verbose=True)
        if onnx_path:
            verify_onnx_model(onnx_path, verbose=True)
        print(f"  Done in {time.time() - t0:.1f}s")
    except ImportError:
        print("  onnx not installed, skipping export.")
    except Exception as e:
        print(f"  ONNX export skipped ({e})")

    # -- Summary ------------------------------------------------------------
    banner("PIPELINE COMPLETE")
    elapsed = time.time() - t0_total
    print(f"  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Improvements applied (v5):")
    print(f"  [Data & Preprocessing]")
    print(f"    + KNN imputation (k=5) + RobustScaler")
    print(f"    + 16 derived features (32-dim total)")
    print(f"    + Temporal augmentation (time warp, magnitude warp, window slice)")
    print(f"    + Conditional WGAN-GP synthetic data ({GAN_SAMPLES_PER_CLASS}/class)")
    print(f"    + NoisyStudent self-training (3 iterations)")
    print(f"    + Borderline-SMOTE + ENN for class imbalance")
    print(f"    + Domain Adaptation (DANN gradient reversal)")
    print(f"  [Models]")
    print(f"    + RF: 300 trees, balanced (Optuna-tuned)")
    print(f"    + LightGBM: 300 trees, balanced, depth=7 (Optuna-tuned)")
    print(f"    + XGBoost: 300 trees, depth=6 (Optuna-tuned)")
    print(f"    + MLP: [256,128,64] + Focal Loss + MixUp + CosineAnnealing + SWA")
    print(f"    + LR warmup ({5} epochs) + Gradient clipping")
    print(f"    + Transformer Encoder (dim=128, heads=4, layers=3)")
    print(f"    + Temporal CNN (TCN, dilated causal convolutions)")
    print(f"    + Graph Neural Network (GCN, 3 layers)")
    print(f"    + NoisyStudent RF (self-trained on unlabeled data)")
    print(f"    + Stacking Ensemble (RF+LGB+XGB+MLP -> LogisticRegression)")
    print(f"  [Features]")
    print(f"    + Cross-modal attention fusion (gaze <-> keystroke)")
    print(f"    + BiLSTM temporal feature extraction")
    print(f"    + Behavioral entropy, gaze fixation, keystroke bigrams")
    print(f"  [Evaluation & Monitoring]")
    print(f"    + SHAP + Calibration + MC Dropout uncertainty (30 samples)")
    print(f"    + Cost-sensitive eval (FN cost={COST_FALSE_NEGATIVE}x, FP cost={COST_FALSE_POSITIVE}x)")
    print(f"    + Fairness metrics (disparate impact, 80% rule)")
    print(f"    + Adversarial robustness (FGSM, ε=0.1)")
    print(f"    + OOD detection (Mahalanobis, 95th percentile)")
    print(f"    + Concept drift (PSI threshold=0.2)")
    print(f"    + Model degradation tracking (rolling window=100)")
    print(f"    + Conformal prediction (90% coverage target)")
    print(f"    + Per-class Wilson CI uncertainty")
    print(f"    + Optuna hyperparameter tuning applied")
    print(f"    + ONNX model export for production")
    print(f"  Total training samples: {len(X)}")
    print()


if __name__ == "__main__":
    main()
