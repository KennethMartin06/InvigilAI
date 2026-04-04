"""
audio_classifier.py — Train an audio classifier on ESC-50 dataset.

Maps ESC-50 categories to exam-relevant labels:
  0 = normal_sound     (keyboard typing, clock tick, silence)
  1 = suspicious_sound (breathing, coughing, laughing, speaking)
  2 = alert_sound      (door knock, footsteps, clapping)

Extracts MFCC + spectral features from .wav files and trains
an MLP classifier. Saves model to models/audio_classifier.joblib.
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
ESC50_META = BASE_DIR / "datasets" / "audio" / "ESC-50" / "meta" / "esc50.csv"
ESC50_AUDIO = BASE_DIR / "datasets" / "audio" / "ESC-50" / "audio"
MODEL_OUT  = BASE_DIR / "cheating_detection" / "models" / "audio_classifier.joblib"
SCALER_OUT = BASE_DIR / "cheating_detection" / "models" / "audio_scaler.joblib"
FEATURES_CACHE = BASE_DIR / "cheating_detection" / "data" / "audio_features.npz"

# ESC-50 target IDs → exam relevance label
# 0 = normal, 1 = suspicious, 2 = alert
LABEL_MAP = {
    # Normal sounds
    34: 0,   # keyboard_typing
    12: 0,   # clock_tick
    30: 0,   # vacuum_cleaner (background noise)
    28: 0,   # washing_machine
    # Suspicious sounds (whispering / speech-like)
     1: 1,   # breathing
     2: 1,   # coughing
    20: 1,   # laughing
    13: 1,   # sneezing
    # Alert sounds (movement/activity)
     8: 2,   # clapping
    31: 2,   # door_wood_knock
    10: 2,   # footsteps
    11: 2,   # door_wood_creaks
}


def extract_features(wav_path: Path) -> np.ndarray:
    """Extract MFCC + spectral features from a .wav file."""
    y, sr = librosa.load(str(wav_path), sr=22050, duration=5.0)

    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta    = librosa.feature.delta(mfcc)
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr           = librosa.feature.zero_crossing_rate(y)
    rms           = librosa.feature.rms(y=y)
    chroma        = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1),           # 13
        mfcc.std(axis=1),            # 13
        mfcc_delta.mean(axis=1),     # 13
        spectral_cent.mean(axis=1),  # 1
        spectral_bw.mean(axis=1),    # 1
        zcr.mean(axis=1),            # 1
        rms.mean(axis=1),            # 1
        chroma.mean(axis=1),         # 12
    ])  # total: 55 features
    return features.astype(np.float32)


def load_or_extract_features(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load cached features or extract from audio files."""
    if FEATURES_CACHE.exists():
        print("  Loading cached audio features...")
        data = np.load(FEATURES_CACHE)
        return data["X"], data["y"]

    print("  Extracting features from audio files...")
    X_list, y_list = [], []
    skipped = 0

    for _, row in meta.iterrows():
        target = int(row["target"])
        if target not in LABEL_MAP:
            continue

        wav_path = ESC50_AUDIO / row["filename"]
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            feats = extract_features(wav_path)
            X_list.append(feats)
            y_list.append(LABEL_MAP[target])
        except Exception as e:
            skipped += 1

    print(f"  Extracted: {len(X_list)} samples, skipped: {skipped}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.savez(FEATURES_CACHE, X=X, y=y)
    print(f"  Cached to: {FEATURES_CACHE}")
    return X, y


def train_audio_classifier():
    print("\n=== Audio Classifier Training (ESC-50) ===")

    meta = pd.read_csv(ESC50_META)
    print(f"  Total ESC-50 files: {len(meta)}")
    print(f"  Using categories: {list(LABEL_MAP.keys())}")

    X, y = load_or_extract_features(meta)

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Label distribution: {label_counts}")
    print(f"    0=normal: {label_counts.get(0,0)}, "
          f"1=suspicious: {label_counts.get(1,0)}, "
          f"2=alert: {label_counts.get(2,0)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )

    print("  Training MLP on audio features...")
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

    return model, scaler


def predict_audio(wav_path: str) -> dict:
    """
    Predict exam-relevance label for a .wav file.
    Returns dict with label, probability, and class name.
    """
    model  = joblib.load(MODEL_OUT)
    scaler = joblib.load(SCALER_OUT)

    feats = extract_features(Path(wav_path)).reshape(1, -1)
    feats_sc = scaler.transform(feats)

    pred   = model.predict(feats_sc)[0]
    proba  = model.predict_proba(feats_sc)[0]

    labels = ["normal", "suspicious", "alert"]
    return {
        "predicted_class": int(pred),
        "class_name": labels[pred],
        "probabilities": dict(zip(labels, proba.tolist())),
        "suspicious_probability": float(proba[1] + proba[2]),
    }


if __name__ == "__main__":
    train_audio_classifier()
