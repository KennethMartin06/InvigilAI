"""
eye_tracking_loader.py -- Loaders for Tobii and SMI eye-tracking datasets.

These loaders parse the native export formats (TSV / CSV) and extract
per-sample gaze features that align with the invigilation feature schema.

If real files are not present, synthetic equivalents are generated so the
downstream pipeline can still run in development environments.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TOBII_DATA_PATH,
    SMI_DATA_PATH,
)


TOBII_COLUMNS = [
    "timestamp", "gaze_x", "gaze_y", "pupil_left", "pupil_right",
    "validity_left", "validity_right", "eye_event",
]

SMI_COLUMNS = [
    "time", "point_x", "point_y", "diameter_left", "diameter_right",
    "blink_flag", "saccade_flag",
]


def _synth_gaze_trace(n, rng, cheater=False):
    base_x = rng.normal(0.5, 0.1 if not cheater else 0.25, n)
    base_y = rng.normal(0.5, 0.1 if not cheater else 0.25, n)
    pupil = np.clip(rng.normal(3.5, 0.4 + 0.2 * cheater, n), 2.0, 6.5)
    valid = (rng.random(n) > (0.02 if not cheater else 0.08)).astype(int)
    event = rng.choice(["fix", "sac", "blink"], n,
                       p=[0.7, 0.2, 0.1] if not cheater else [0.5, 0.3, 0.2])
    ts = np.arange(n) * 8.33   # 120Hz
    return ts, base_x, base_y, pupil, valid, event


def load_tobii(path=TOBII_DATA_PATH, n_samples=500, verbose=True):
    """Load Tobii eye tracker export (TSV)."""
    files = []
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith((".tsv", ".csv"))]

    if not files:
        if verbose:
            print(f"[Tobii] No files at {path}, generating synthetic")
        return _generate_synthetic_tobii(n_samples)

    frames = []
    for f in files:
        fp = os.path.join(path, f)
        sep = "\t" if f.endswith(".tsv") else ","
        df = pd.read_csv(fp, sep=sep)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[Tobii] Loaded {len(df)} samples from {len(files)} files")
    return df


def _generate_synthetic_tobii(n):
    rng = np.random.default_rng(RANDOM_SEED)
    ts, x, y, pup, valid, event = _synth_gaze_trace(n, rng)
    return pd.DataFrame({
        "timestamp": ts, "gaze_x": x, "gaze_y": y,
        "pupil_left": pup, "pupil_right": pup * rng.normal(1, 0.05, n),
        "validity_left": valid, "validity_right": valid,
        "eye_event": event,
    })


def load_smi(path=SMI_DATA_PATH, n_samples=500, verbose=True):
    """Load SMI iView export (TXT/CSV)."""
    files = []
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith((".txt", ".csv"))]

    if not files:
        if verbose:
            print(f"[SMI] No files at {path}, generating synthetic")
        return _generate_synthetic_smi(n_samples)

    frames = []
    for f in files:
        fp = os.path.join(path, f)
        df = pd.read_csv(fp, sep=None, engine="python", comment="#")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[SMI] Loaded {len(df)} samples from {len(files)} files")
    return df


def _generate_synthetic_smi(n):
    rng = np.random.default_rng(RANDOM_SEED + 1)
    ts, x, y, pup, valid, event = _synth_gaze_trace(n, rng)
    return pd.DataFrame({
        "time": ts, "point_x": x * 1920, "point_y": y * 1080,
        "diameter_left": pup, "diameter_right": pup,
        "blink_flag": (event == "blink").astype(int),
        "saccade_flag": (event == "sac").astype(int),
    })


def extract_gaze_features(df, source="tobii"):
    """Extract invigilation-compatible gaze features from eye-tracking df."""
    if source == "tobii":
        x = df["gaze_x"].values
        y = df["gaze_y"].values
        pupil = (df["pupil_left"].values + df["pupil_right"].values) / 2
        blink_mask = df["eye_event"].values == "blink"
    else:
        x = df["point_x"].values / (df["point_x"].max() + 1e-6)
        y = df["point_y"].values / (df["point_y"].max() + 1e-6)
        pupil = (df["diameter_left"].values + df["diameter_right"].values) / 2
        blink_mask = df["blink_flag"].values.astype(bool)

    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx ** 2 + dy ** 2)

    return {
        "gaze_yaw": float(np.arctan2(np.std(x), 1) * 57.3),
        "gaze_pitch": float(np.arctan2(np.std(y), 1) * 57.3),
        "gaze_deviation_ratio": float(np.std(np.sqrt(x ** 2 + y ** 2))),
        "blink_rate": float(blink_mask.mean() * 60),
        "pupil_mean": float(np.mean(pupil)),
        "pupil_std": float(np.std(pupil)),
        "saccade_mean_speed": float(np.mean(speed)),
        "fixation_ratio": float((speed < 0.02).mean()),
    }
