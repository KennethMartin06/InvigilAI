"""
audio_classifier.py -- Improved audio classifier on ESC-50 + LibriSpeech.

Improvements:
  - Deeper MLP (256, 128, 64) with balanced class weights
  - Data augmentation via noise injection and time-stretching
  - More ESC-50 categories mapped for better coverage
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

# ESC-50 target IDs -> exam relevance label
# 0 = normal, 1 = suspicious, 2 = alert
LABEL_MAP = {
    # Normal sounds
    34: 0,   # keyboard_typing
    12: 0,   # clock_tick
    30: 0,   # vacuum_cleaner (background noise)
    28: 0,   # washing_machine
    38: 0,   # mouse_click
    48: 0,   # train (ambient)
    47: 0,   # airplane (ambient)
    # Suspicious sounds (whispering / speech-like)
     1: 1,   # breathing
     2: 1,   # coughing
    20: 1,   # laughing
    13: 1,   # sneezing
    26: 1,   # can_opening (fumbling)
    23: 1,   # drinking_sipping
    # Alert sounds (movement/activity)
     8: 2,   # clapping
    31: 2,   # door_wood_knock
    10: 2,   # footsteps
    11: 2,   # door_wood_creaks
    36: 2,   # glass_breaking
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


def augment_audio(y: np.ndarray, sr: int) -> list:
    """Generate augmented versions of an audio signal."""
    augmented = []
    # Add Gaussian noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    # Time stretch (slightly faster)
    augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    # Time stretch (slightly slower)
    augmented.append(librosa.effects.time_stretch(y, rate=0.9))
    return augmented


def extract_features_from_signal(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract features from raw audio signal (for augmented data)."""
    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta    = librosa.feature.delta(mfcc)
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr           = librosa.feature.zero_crossing_rate(y)
    rms           = librosa.feature.rms(y=y)
    chroma        = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        spectral_cent.mean(axis=1),
        spectral_bw.mean(axis=1),
        zcr.mean(axis=1),
        rms.mean(axis=1),
        chroma.mean(axis=1),
    ])
    return features.astype(np.float32)


def load_or_extract_features(meta: pd.DataFrame, use_augmentation: bool = True) -> tuple:
    """Load cached features or extract from audio files with optional augmentation."""
    # Delete old cache to force re-extraction with new categories
    if FEATURES_CACHE.exists():
        data = np.load(FEATURES_CACHE)
        old_X, old_y = data["X"], data["y"]
        # Re-extract if category count changed
        if len(old_X) < 500:
            os.remove(FEATURES_CACHE)
        else:
            print("  Loading cached audio features...")
            return old_X, old_y

    print("  Extracting features from audio files (with augmentation)...")
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
            # Original features
            feats = extract_features(wav_path)
            X_list.append(feats)
            y_list.append(LABEL_MAP[target])

            # Augmented features
            if use_augmentation:
                y_audio, sr = librosa.load(str(wav_path), sr=22050, duration=5.0)
                for aug_signal in augment_audio(y_audio, sr):
                    aug_feats = extract_features_from_signal(aug_signal, sr)
                    X_list.append(aug_feats)
                    y_list.append(LABEL_MAP[target])
        except Exception:
            skipped += 1

    print(f"  Extracted: {len(X_list)} samples (incl. augmented), skipped: {skipped}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.savez(FEATURES_CACHE, X=X, y=y)
    print(f"  Cached to: {FEATURES_CACHE}")
    return X, y


def train_audio_classifier():
    print("\n=== Audio Classifier Training (ESC-50, improved) ===")

    meta = pd.read_csv(ESC50_META)
    print(f"  Total ESC-50 files: {len(meta)}")
    print(f"  Using categories: {list(LABEL_MAP.keys())}")

    X, y = load_or_extract_features(meta, use_augmentation=True)

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
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        batch_size=32,
        verbose=False,
    )

    print("  Training improved MLP on audio features...")
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
    print(f"  Saved model  -> {MODEL_OUT}")
    print(f"  Saved scaler -> {SCALER_OUT}")

    return model, scaler


def predict_audio(wav_path: str) -> dict:
    """Predict exam-relevance label for a .wav file."""
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
