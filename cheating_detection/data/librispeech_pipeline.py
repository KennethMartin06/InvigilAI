"""
librispeech_pipeline.py — Extract audio features from LibriSpeech test-clean
and add to audio classifier training data.

LibriSpeech structure:
  LibriSpeech/test-clean/{speaker_id}/{chapter_id}/*.flac

Labels for exam cheating:
  0 = normal_sound  (clean speech = someone talking normally, not whispering)
  1 = suspicious    (we use this as a "speech detected" signal)

Combined with ESC-50, this improves the audio model's ability to
distinguish silence/keyboard from active speech.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import librosa
except ImportError:
    os.system("pip install librosa -q")
    import librosa

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
LIBRISPEECH_DIR = BASE_DIR / "datasets" / "audio" / "LibriSpeech" / "test-clean"
ESC50_CACHE    = BASE_DIR / "cheating_detection" / "data" / "audio_features.npz"
OUTPUT_CACHE   = BASE_DIR / "cheating_detection" / "data" / "audio_features_combined.npz"
MODEL_OUT      = BASE_DIR / "cheating_detection" / "models" / "audio_classifier.joblib"
SCALER_OUT     = BASE_DIR / "cheating_detection" / "models" / "audio_scaler.joblib"

MAX_CLIPS = 200   # number of LibriSpeech clips to use


def extract_features(audio_path: Path, sr: int = 22050) -> np.ndarray:
    """Extract 55 audio features from a file (.wav or .flac)."""
    y, sr = librosa.load(str(audio_path), sr=sr, duration=5.0)

    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta    = librosa.feature.delta(mfcc)
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr           = librosa.feature.zero_crossing_rate(y)
    rms           = librosa.feature.rms(y=y)
    chroma        = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.concatenate([
        mfcc.mean(axis=1),           # 13
        mfcc.std(axis=1),            # 13
        mfcc_delta.mean(axis=1),     # 13
        spectral_cent.mean(axis=1),  # 1
        spectral_bw.mean(axis=1),    # 1
        zcr.mean(axis=1),            # 1
        rms.mean(axis=1),            # 1
        chroma.mean(axis=1),         # 12
    ]).astype(np.float32)            # total: 55


def extract_librispeech_features() -> tuple[np.ndarray, np.ndarray]:
    """Extract features from LibriSpeech .flac files."""
    flac_files = list(LIBRISPEECH_DIR.rglob("*.flac"))[:MAX_CLIPS]
    print(f"  Found {len(flac_files)} LibriSpeech clips (using {MAX_CLIPS})")

    X_list, y_list = [], []
    for i, f in enumerate(flac_files):
        try:
            feats = extract_features(f)
            X_list.append(feats)
            y_list.append(1)   # label 1 = suspicious (speech detected)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(flac_files)} clips...")
        except Exception:
            continue

    print(f"  Extracted {len(X_list)} LibriSpeech features")
    return np.array(X_list, dtype=np.float32), np.ones(len(X_list), dtype=np.int64)


def run():
    print("\n=== LibriSpeech + ESC-50 Audio Classifier Training ===")

    if not LIBRISPEECH_DIR.exists():
        print(f"ERROR: {LIBRISPEECH_DIR} not found.")
        return

    # Load existing ESC-50 features
    print("Loading ESC-50 features...")
    esc_data = np.load(ESC50_CACHE)
    X_esc, y_esc = esc_data["X"], esc_data["y"]
    print(f"  ESC-50: {len(X_esc)} samples")

    # Extract LibriSpeech features
    print("Extracting LibriSpeech features...")
    X_libri, y_libri = extract_librispeech_features()

    # Combine
    X = np.vstack([X_esc, X_libri])
    y = np.concatenate([y_esc, y_libri])

    # Save combined cache
    np.savez(OUTPUT_CACHE, X=X, y=y)
    print(f"  Combined audio samples: {len(X)}")

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Label dist: {label_counts}")

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )

    print("Training audio MLP...")
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["normal", "suspicious", "alert"]
    ))

    joblib.dump(model,  MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"  Saved model  → {MODEL_OUT}")
    print(f"  Saved scaler → {SCALER_OUT}")


if __name__ == "__main__":
    run()
