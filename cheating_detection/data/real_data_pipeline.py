"""
real_data_pipeline.py — Extract real behavioral features from CMU Keystroke
dataset and combine with synthetic data to improve model training.

CMU Keystroke columns:
  subject, sessionIndex, rep,
  H.*  = hold time (dwell time) per key
  DD.* = down-down time (inter-keystroke interval)
  UD.* = up-down time (flight time)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KEYSTROKE_CSV = BASE_DIR / "datasets" / "keystroke" / "cmu_keystroke.csv"
SYNTHETIC_NPZ = BASE_DIR / "cheating_detection" / "data" / "synthetic_dataset.npz"
OUTPUT_NPZ    = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.npz"
OUTPUT_CSV    = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.csv"


def extract_keystroke_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CMU keystroke columns to our 8 behavioral features.

    CMU has per-key timings for one password (14 keys).
    We average across keys to get session-level features.
    """
    h_cols  = [c for c in df.columns if c.startswith("H.")]
    dd_cols = [c for c in df.columns if c.startswith("DD.")]
    ud_cols = [c for c in df.columns if c.startswith("UD.")]

    feats = pd.DataFrame()

    # mean_dwell_time  — average hold time (seconds → ms)
    feats["mean_dwell_time"] = df[h_cols].mean(axis=1) * 1000

    # mean_flight_time — average up-down (flight) time (ms)
    feats["mean_flight_time"] = df[ud_cols].mean(axis=1) * 1000

    # keystroke_rate   — keys per second (1 / mean DD)
    mean_dd = df[dd_cols].mean(axis=1).replace(0, np.nan)
    feats["keystroke_rate"] = 1.0 / mean_dd.fillna(0.3)

    # burst_coefficient — std / mean of DD times (irregularity)
    dd_std  = df[dd_cols].std(axis=1)
    dd_mean = df[dd_cols].mean(axis=1).replace(0, np.nan)
    feats["burst_coefficient"] = (dd_std / dd_mean.fillna(1)).fillna(0)

    # cursor_velocity, click_frequency, idle_ratio, trajectory_linearity
    # not in keystroke dataset — fill with normal-range defaults
    feats["cursor_velocity"]      = np.random.normal(150, 30, len(df)).clip(50, 400)
    feats["click_frequency"]      = np.random.normal(0.5, 0.1, len(df)).clip(0, 2)
    feats["idle_ratio"]           = np.random.normal(0.1, 0.05, len(df)).clip(0, 0.5)
    feats["trajectory_linearity"] = np.random.normal(0.85, 0.05, len(df)).clip(0.5, 1.0)

    return feats


def build_visual_features_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate normal (focused) visual features for real keystroke sessions."""
    return np.column_stack([
        rng.normal(0,   5,   n),   # gaze_yaw   — small deviations
        rng.normal(0,   3,   n),   # gaze_pitch
        rng.normal(0,   8,   n),   # head_yaw
        rng.normal(0,   5,   n),   # head_pitch
        rng.normal(0,   3,   n),   # head_roll
        np.ones(n),                # face_count = 1
        rng.uniform(0, 0.1, n),    # gaze_deviation_ratio — low
        rng.normal(25, 3,   n),    # face_embedding_norm
    ])


def build_visual_features_abnormal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate suspicious visual features for abnormal keystroke sessions."""
    return np.column_stack([
        rng.normal(0,   8,   n),
        rng.normal(0,   5,   n),
        rng.normal(0,  12,   n),
        rng.normal(0,   8,   n),
        rng.normal(0,   5,   n),
        np.ones(n),
        rng.uniform(0.1, 0.4, n),  # slightly elevated deviation
        rng.normal(25,  4,  n),
    ])


def load_and_process_cmu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CMU keystroke CSV and return (X, y) arrays.

    Labels:
      0 = Normal Behavior        (most sessions — baseline typing)
      4 = Abnormal Keystroke     (sessions with high burst coefficient)
    """
    df = pd.read_csv(path)
    rng = np.random.default_rng(42)

    feats = extract_keystroke_features(df)
    feats = feats.fillna(feats.median())

    # Flag sessions as "abnormal keystroke" if burst_coefficient is high
    threshold = feats["burst_coefficient"].quantile(0.85)
    labels = np.where(feats["burst_coefficient"] > threshold, 4, 0)

    n = len(feats)
    normal_mask   = labels == 0
    abnormal_mask = labels == 4

    vis_normal   = build_visual_features_normal(normal_mask.sum(),   rng)
    vis_abnormal = build_visual_features_abnormal(abnormal_mask.sum(), rng)

    visual = np.zeros((n, 8))
    visual[normal_mask]   = vis_normal
    visual[abnormal_mask] = vis_abnormal

    behav = feats[["gaze_yaw" if "gaze_yaw" in feats.columns else "keystroke_rate",
                   "mean_dwell_time", "mean_flight_time", "burst_coefficient",
                   "cursor_velocity", "click_frequency", "idle_ratio",
                   "trajectory_linearity"]].values

    # Use correct column order
    behav_cols = ["keystroke_rate", "mean_dwell_time", "mean_flight_time",
                  "burst_coefficient", "cursor_velocity", "click_frequency",
                  "idle_ratio", "trajectory_linearity"]
    behav = feats[behav_cols].values

    X = np.hstack([visual, behav]).astype(np.float32)
    return X, labels


def combine_with_synthetic(X_real: np.ndarray, y_real: np.ndarray) -> tuple:
    """Combine real CMU data with existing synthetic dataset."""
    data = np.load(SYNTHETIC_NPZ)
    X_syn, y_syn = data["X"].astype(np.float32), data["y"].astype(np.int64)

    X_combined = np.vstack([X_syn, X_real])
    y_combined = np.concatenate([y_syn, y_real.astype(np.int64)])

    # Shuffle
    idx = np.random.default_rng(42).permutation(len(X_combined))
    return X_combined[idx], y_combined[idx]


def run():
    print("Loading CMU Keystroke dataset...")
    X_real, y_real = load_and_process_cmu(KEYSTROKE_CSV)
    print(f"  Real samples: {len(X_real)}")
    print(f"  Label dist:   {dict(zip(*np.unique(y_real, return_counts=True)))}")

    print("Combining with synthetic dataset...")
    X, y = combine_with_synthetic(X_real, y_real)
    print(f"  Combined samples: {len(X)}")
    print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Save combined dataset
    np.savez(OUTPUT_NPZ, X=X, y=y)
    print(f"  Saved: {OUTPUT_NPZ}")

    # Also save as CSV for inspection
    feature_names = [
        "gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch", "head_roll",
        "face_count", "gaze_deviation_ratio", "face_embedding_norm",
        "keystroke_rate", "mean_dwell_time", "mean_flight_time", "burst_coefficient",
        "cursor_velocity", "click_frequency", "idle_ratio", "trajectory_linearity",
        "label"
    ]
    df_out = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=feature_names)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")

    return X, y


if __name__ == "__main__":
    run()
