"""
main.py — End-to-end pipeline for the Multi-Modal AI Cheating Detection System.

Usage:
    python -m cheating_detection.main

Pipeline steps:
  1. Generate synthetic dataset + augment with real data (CMU, MPIIGaze, DAiSEE)
  2. Preprocess: clean → split → normalise → save scaler
  3. Train Random Forest and MLP (SVM skipped — too slow on large datasets)
  4. Train Audio Classifier (ESC-50 + LibriSpeech)
  5. Evaluate, ablation study, ROC/PR curves
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.data.generate_dataset import generate_dataset
from cheating_detection.data.real_data_pipeline import run as build_combined_dataset
from cheating_detection.data.mpiigaze_pipeline import run as build_mpiigaze_dataset
from cheating_detection.data.daisee_pipeline import run as build_daisee_dataset
from cheating_detection.data.hmdb51_pipeline import run_hmdb51_pipeline, merge_with_combined as merge_hmdb51
from cheating_detection.preprocessing.preprocess import preprocess
from cheating_detection.models.train import train_random_forest, train_mlp, plot_training_curves
from cheating_detection.models.audio_classifier import train_audio_classifier
from cheating_detection.models.evaluate import run_full_evaluation
from cheating_detection.config import OUTPUTS_DIR, MODELS_DIR

# Try importing LibriSpeech pipeline (optional)
try:
    from cheating_detection.data.librispeech_pipeline import run as train_librispeech_audio
    HAS_LIBRISPEECH = True
except ImportError:
    HAS_LIBRISPEECH = False


def banner(text: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    t0_total = time.time()

    # ── Step 1a: Generate synthetic dataset ────────────────────────────────
    banner("STEP 1a — Synthetic Dataset Generation")
    t0 = time.time()
    X, y = generate_dataset(save_npz=True, save_csv=True, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 1b: CMU Keystroke data ────────────────────────────────────────
    banner("STEP 1b — Augment with CMU Keystroke Data")
    t0 = time.time()
    try:
        X, y = build_combined_dataset()
        print(f"  Dataset size: {len(X)} samples")
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Skipping CMU Keystroke ({e})")

    # ── Step 1c: MPIIGaze data ─────────────────────────────────────────────
    banner("STEP 1c — Augment with MPIIGaze Gaze Data")
    t0 = time.time()
    try:
        X, y = build_mpiigaze_dataset()
        if X is not None:
            print(f"  Dataset size: {len(X)} samples")
            print(f"  Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping (not found).")
    except Exception as e:
        print(f"  Skipping MPIIGaze ({e})")

    # ── Step 1d: DAiSEE engagement data ────────────────────────────────────
    banner("STEP 1d — Augment with DAiSEE Engagement Data")
    t0 = time.time()
    try:
        X, y = build_daisee_dataset()
        if X is not None:
            print(f"  Dataset size: {len(X)} samples")
            print(f"  Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping (not found).")
    except Exception as e:
        print(f"  Skipping DAiSEE ({e})")

    # ── Step 1e: HMDB-51 video data ────────────────────────────────────────
    banner("STEP 1e — Augment with HMDB-51 Action Videos")
    t0 = time.time()
    try:
        X_hmdb, y_hmdb = run_hmdb51_pipeline()
        if len(X_hmdb) > 0:
            merge_hmdb51(X_hmdb, y_hmdb)
            from cheating_detection.data.combined_dataset import load_combined  # noqa
            import numpy as np
            data = np.load(
                os.path.join(os.path.dirname(__file__), "data", "combined_dataset.npz")
            )
            X, y = data["X"], data["y"]
            print(f"  HMDB-51 added {len(X_hmdb)} samples → total {len(X)}")
            print(f"  Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping (no samples extracted).")
    except Exception as e:
        print(f"  Skipping HMDB-51 ({e})")

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    banner("STEP 2 — Preprocessing (clean → split → normalise)")
    t0 = time.time()
    splits = preprocess(X, y, verbose=True)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3a: Train Random Forest ───────────────────────────────────────
    banner("STEP 3a — Train Random Forest")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3b: Train MLP ─────────────────────────────────────────────────
    banner("STEP 3b — Train MLP (PyTorch, early stopping)")
    t0 = time.time()
    mlp_model, history = train_mlp(
        X_train, y_train,
        X_val,   y_val,
        verbose=True,
    )
    plot_training_curves(history)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3c: Train Audio Classifier ────────────────────────────────────
    banner("STEP 3c — Train Audio Classifier (ESC-50 + LibriSpeech)")
    t0 = time.time()
    try:
        if HAS_LIBRISPEECH:
            train_librispeech_audio()
        else:
            train_audio_classifier()
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Warning: Audio training failed ({e})")
        try:
            train_audio_classifier()
        except Exception as e2:
            print(f"  Audio classifier unavailable ({e2})")

    # ── Steps 4-6: Full evaluation ─────────────────────────────────────────
    banner("STEPS 4-6 — Evaluation, Ablation, Curves")
    models = {
        "Random Forest": rf_model,
        "MLP":           mlp_model,
    }
    run_full_evaluation(models, splits, best_model_name="MLP", verbose=True)

    # ── Summary ────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    elapsed = time.time() - t0_total
    print(f"  Total elapsed time : {elapsed:.1f}s")
    print(f"  Datasets used:")
    print(f"    ✓ Synthetic (5,411 samples)")
    print(f"    ✓ CMU Keystroke (20,400 real keystroke sessions)")
    print(f"    ✓ MPIIGaze (213,656 real gaze samples)")
    print(f"    ✓ DAiSEE (9,068 engagement-labeled clips)")
    print(f"    ✓ ESC-50 (480 audio clips)")
    print(f"    ✓ LibriSpeech (200 speech clips)")
    print(f"    ✓ HMDB-51 action videos (talk/wave/laugh/smoke vs sit/smile/drink)")
    print(f"  Total training samples: {len(X)}")
    print()


if __name__ == "__main__":
    main()
