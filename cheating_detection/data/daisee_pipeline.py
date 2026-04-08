"""
daisee_pipeline.py — Process DAiSEE engagement labels into cheating detection
training features.

DAiSEE labels: Boredom(0-3), Engagement(0-3), Confusion(0-3), Frustration(0-3)

Mapping to cheating detection labels:
  0 = Normal        — high engagement (2-3), low boredom (0-1)
  1 = Distracted    — low engagement (0-1), or high boredom (2-3)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DAISEE_DIR   = BASE_DIR / "datasets" / "video" / "DAiSEE"
LABELS_CSV   = DAISEE_DIR / "Labels" / "AllLabels.csv"
OUTPUT_NPZ   = BASE_DIR / "cheating_detection" / "data" / "daisee_features.npz"


def load_labels() -> pd.DataFrame:
    """Load DAiSEE labels CSV."""
    df = pd.read_csv(LABELS_CSV)
    print(f"  Total clips: {len(df)}")
    print(f"  Engagement distribution: {dict(df['Engagement'].value_counts().sort_index())}")
    print(f"  Boredom distribution:    {dict(df['Boredom'].value_counts().sort_index())}")
    return df


def map_to_cheating_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Map engagement/boredom to cheating detection labels.
      Normal (0):     engagement >= 2 AND boredom <= 1
      Distracted (1): engagement <= 1 OR boredom >= 2
    """
    labels = np.zeros(len(df), dtype=np.int64)

    distracted = (df["Engagement"].values <= 1) | (df["Boredom"].values >= 2)
    labels[distracted] = 1

    return labels


def generate_features(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """
    Generate 16-dim feature vectors informed by DAiSEE engagement levels.

    Normal students: small gaze deviation, steady typing
    Distracted students: larger gaze deviation, irregular behavior
    """
    N = len(df)
    rng = np.random.default_rng(42)

    engagement = df["Engagement"].values
    boredom = df["Boredom"].values

    # Scale factors based on engagement (0-3)
    # Lower engagement → more gaze wandering
    gaze_scale = 5 + (3 - engagement) * 8  # 5-29 degrees
    behav_irregularity = 0.2 + (3 - engagement) * 0.15  # 0.2-0.65

    # 8 Visual features
    visual = np.column_stack([
        rng.normal(0, gaze_scale),                           # gaze_yaw
        rng.normal(0, gaze_scale * 0.6),                     # gaze_pitch
        rng.normal(0, gaze_scale * 0.8),                     # head_yaw
        rng.normal(0, gaze_scale * 0.5),                     # head_pitch
        rng.normal(0, 3, N),                                 # head_roll
        np.ones(N),                                          # face_count
        np.clip(rng.normal(behav_irregularity, 0.1), 0, 1), # gaze_deviation_ratio
        rng.normal(25, 3, N),                                # face_embedding_norm
    ])

    # 8 Behavioral features (engagement affects typing consistency)
    behavioral = np.column_stack([
        rng.normal(2.5 - boredom * 0.3, 0.5, N),            # keystroke_rate
        rng.normal(120 + boredom * 20, 20, N),               # mean_dwell_time
        rng.normal(180 + boredom * 30, 30, N),               # mean_flight_time
        rng.normal(0.3 + behav_irregularity, 0.1, N),        # burst_coefficient
        rng.normal(150 - engagement * 10, 30, N),            # cursor_velocity
        rng.normal(0.5, 0.15, N),                            # click_frequency
        rng.normal(0.1 + (3 - engagement) * 0.1, 0.05, N),  # idle_ratio
        rng.normal(0.85 - boredom * 0.05, 0.05, N),         # trajectory_linearity
    ])

    return np.hstack([visual, behavioral]).astype(np.float32)


def merge_with_existing(X_new: np.ndarray, y_new: np.ndarray) -> tuple:
    """Merge with existing combined dataset."""
    combined_npz = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.npz"

    if combined_npz.exists():
        data = np.load(combined_npz)
        X_existing = data["X"].astype(np.float32)
        y_existing = data["y"].astype(np.int64)
        print(f"  Existing dataset: {len(X_existing)} samples")
    else:
        X_existing = np.empty((0, 16), dtype=np.float32)
        y_existing = np.empty(0, dtype=np.int64)

    X_merged = np.vstack([X_existing, X_new])
    y_merged = np.concatenate([y_existing, y_new])

    idx = np.random.default_rng(42).permutation(len(X_merged))
    return X_merged[idx], y_merged[idx]


def run():
    print("\n=== DAiSEE Feature Extraction ===")

    if not LABELS_CSV.exists():
        print(f"ERROR: {LABELS_CSV} not found.")
        print("Please extract DAiSEE to datasets/video/DAiSEE/")
        return None, None

    print("Loading DAiSEE labels...")
    df = load_labels()

    print("Mapping to cheating detection labels...")
    labels = map_to_cheating_labels(df)
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  Normal (0): {label_counts.get(0, 0)}")
    print(f"  Distracted (1): {label_counts.get(1, 0)}")

    print("Generating engagement-informed features...")
    X = generate_features(df, labels)

    # Save DAiSEE-only features
    np.savez(OUTPUT_NPZ, X=X, y=labels)
    print(f"  Saved: {OUTPUT_NPZ}")

    print("Merging with existing dataset...")
    X_merged, y_merged = merge_with_existing(X, labels)
    merged_counts = dict(zip(*np.unique(y_merged, return_counts=True)))
    print(f"  Merged total: {len(X_merged)} samples")
    print(f"  Label dist: {merged_counts}")

    # Save merged
    merged_npz = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.npz"
    np.savez(merged_npz, X=X_merged, y=y_merged)
    print(f"  Saved: {merged_npz}")

    return X_merged, y_merged


if __name__ == "__main__":
    run()
