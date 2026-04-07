"""
main.py — End-to-end pipeline for the Multi-Modal AI Cheating Detection System.

Usage:
    python -m cheating_detection.main
    # or from the cheating_detection/ directory:
    python main.py

Pipeline steps:
  1. Generate synthetic dataset (240 sessions, 5 categories)
  2. Preprocess: clean → split → normalise → save scaler
  3. Train SVM, Random Forest, and MLP
  4. Evaluate all three models on the held-out test set
  5. Run ablation study (visual-only / behavioral-only / multi-modal)
  6. Threshold analysis and ROC / PR curve generation
  7. Print formatted summary table to console
  8. Persist all outputs to outputs/ and models/
"""

import os
import sys
import time

# Make sure the package root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.data.generate_dataset import generate_dataset
from cheating_detection.data.real_data_pipeline import run as build_combined_dataset
from cheating_detection.data.mpiigaze_pipeline import run as build_mpiigaze_dataset
from cheating_detection.preprocessing.preprocess import preprocess
from cheating_detection.models.train import train_svm, train_random_forest, train_mlp, plot_training_curves
from cheating_detection.models.audio_classifier import train_audio_classifier
from cheating_detection.data.librispeech_pipeline import run as train_librispeech_audio
from cheating_detection.models.evaluate import run_full_evaluation
from cheating_detection.config import OUTPUTS_DIR, MODELS_DIR, DATA_DIR


def banner(text: str) -> None:
    """Print a section banner."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    """Run the full cheating-detection pipeline end-to-end."""

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    t0_total = time.time()

    # ── Step 1: Generate synthetic dataset ──────────────────────────────────
    banner("STEP 1 — Synthetic Dataset Generation")
    t0 = time.time()
    X, y = generate_dataset(save_npz=True, save_csv=True, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 1b: Combine with real CMU Keystroke data ────────────────────────
    banner("STEP 1b — Augment with Real CMU Keystroke Data")
    t0 = time.time()
    try:
        X, y = build_combined_dataset()
        print(f"  Combined dataset size: {len(X)} samples")
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Warning: Could not load real keystroke data ({e})")
        print("  Continuing with synthetic data only.")

    # ── Step 1c: Integrate MPIIGaze real gaze data ──────────────────────────
    banner("STEP 1c — Augment with Real MPIIGaze Gaze Data")
    t0 = time.time()
    try:
        X, y = build_mpiigaze_dataset()
        if X is not None:
            print(f"  MPIIGaze merged dataset size: {len(X)} samples")
            print(f"  Done in {time.time() - t0:.1f}s")
        else:
            print("  Skipping MPIIGaze (dataset not found).")
    except Exception as e:
        print(f"  Warning: Could not process MPIIGaze ({e})")
        print("  Continuing without MPIIGaze data.")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
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

    # ── Step 3a: Train SVM ───────────────────────────────────────────────────
    banner("STEP 3a — Train SVM (RBF + GridSearch)")
    t0 = time.time()
    svm_model = train_svm(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3b: Train Random Forest ─────────────────────────────────────────
    banner("STEP 3b — Train Random Forest")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3c: Train MLP ───────────────────────────────────────────────────
    banner("STEP 3c — Train MLP (PyTorch, early stopping)")
    t0 = time.time()
    mlp_model, history = train_mlp(
        X_train, y_train,
        X_val,   y_val,
        verbose=True,
    )
    plot_training_curves(history)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 3d: Train Audio Classifier (ESC-50 + LibriSpeech) ──────────────
    banner("STEP 3d — Train Audio Classifier (ESC-50 + LibriSpeech)")
    t0 = time.time()
    try:
        train_librispeech_audio()
        print(f"  Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Warning: LibriSpeech training failed ({e}), falling back to ESC-50 only")
        try:
            train_audio_classifier()
        except Exception as e2:
            print(f"  Warning: Audio classifier training failed ({e2})")

    # ── Steps 4–6: Full evaluation ───────────────────────────────────────────
    banner("STEPS 4-6 — Evaluation, Ablation, Curves")
    models = {
        "SVM":           svm_model,
        "Random Forest": rf_model,
        "MLP":           mlp_model,
    }
    run_full_evaluation(models, splits, best_model_name="MLP", verbose=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    elapsed = time.time() - t0_total
    print(f"  Total elapsed time : {elapsed:.1f}s")
    print(f"  Saved artifacts:")
    artifacts = [
        "data/synthetic_dataset.npz",
        "data/synthetic_dataset.csv",
        "models/scaler.joblib",
        "models/svm_model.joblib",
        "models/rf_model.joblib",
        "models/mlp_model.pth",
        "outputs/confusion_matrix.png",
        "outputs/training_curves.png",
        "outputs/ablation_results.png",
        "outputs/threshold_analysis.png",
        "outputs/roc_curve.png",
        "outputs/precision_recall_curve.png",
        "outputs/results.json",
        "models/audio_classifier.joblib",
        "models/audio_scaler.joblib",
        "data/combined_dataset.npz",
    ]
    for a in artifacts:
        print(f"    ✓ cheating_detection/{a}")
    print()


if __name__ == "__main__":
    main()
