"""
mpiigaze_pipeline.py — Extract gaze features from MPIIGaze dataset
and integrate them into the cheating detection training data.

MPIIGaze structure:
  Data/Normalized/p00/day01.mat  — normalized face images + gaze labels
  Evaluation Subset/annotation for face image/p00.txt — landmarks

Each .mat file contains:
  data.left  / data.right — normalized eye images
  data.label             — gaze direction (pitch, yaw) in radians

Labels we produce:
  0 = Normal     (small gaze deviation, looking at screen)
  1 = Looking Away (large gaze deviation, distracted)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Installing scipy...")
    os.system(f"{sys.executable} -m pip install scipy -q")
    import scipy.io as sio

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
MPIIGAZE_DIR = BASE_DIR / "datasets" / "gaze" / "MPIIGaze"
NORMALIZED_DIR = MPIIGAZE_DIR / "Data" / "Normalized"
OUTPUT_NPZ  = BASE_DIR / "cheating_detection" / "data" / "mpiigaze_features.npz"
OUTPUT_CSV  = BASE_DIR / "cheating_detection" / "data" / "mpiigaze_features.csv"

# Gaze deviation threshold — beyond this angle (radians) = looking away
# ~25 degrees = 0.436 radians
GAZE_THRESHOLD = 0.436


def inspect_mat(mat_path: Path) -> dict:
    """Load and inspect a .mat file to find gaze labels."""
    mat = sio.loadmat(str(mat_path), squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def gaze_vector_to_angles(gaze_vec: np.ndarray) -> np.ndarray:
    """
    Convert 3D unit gaze vectors (N, 3) to [pitch, yaw] angles in radians.
    Convention: x=right, y=up, z=forward(-z = away from screen)
    """
    x = gaze_vec[:, 0]
    y = gaze_vec[:, 1]
    z = gaze_vec[:, 2]
    yaw   = np.arctan2(x, -z)          # horizontal angle
    pitch = np.arctan2(y, np.sqrt(x**2 + z**2))  # vertical angle
    return np.column_stack([pitch, yaw]).astype(np.float32)


def extract_gaze_from_mat(mat_path: Path) -> np.ndarray | None:
    """
    Extract gaze angles from MPIIGaze normalized .mat file.

    Structure: data -> right/left -> gaze (N,3), image (N,36,60), pose (N,3)
    Returns array of shape (N, 2) with [pitch, yaw] in radians.
    """
    try:
        mat = sio.loadmat(str(mat_path), squeeze_me=True)

        if "data" not in mat:
            return None

        data = mat["data"]

        # data is a 0-d struct with fields 'right' and 'left'
        # Try right eye first, then left
        for eye in ["right", "left"]:
            try:
                eye_data = data[eye].item()
                gaze_vec = eye_data["gaze"]

                # gaze_vec may be nested in another item()
                if not isinstance(gaze_vec, np.ndarray):
                    gaze_vec = gaze_vec.item()

                gaze_vec = np.array(gaze_vec)

                if gaze_vec.ndim == 2 and gaze_vec.shape[1] == 3:
                    return gaze_vector_to_angles(gaze_vec)
            except Exception:
                continue

        return None
    except Exception:
        return None


def gaze_to_features(gaze_angles: np.ndarray) -> np.ndarray:
    """
    Convert raw gaze angles to our 16-feature vector format.

    gaze_angles: (N, 2) array of [pitch, yaw] in radians

    Returns: (N, 16) feature array
    """
    N = len(gaze_angles)
    rng = np.random.default_rng(42)

    pitch = gaze_angles[:, 0]
    yaw   = gaze_angles[:, 1]

    # Convert radians → degrees
    pitch_deg = np.degrees(pitch)
    yaw_deg   = np.degrees(yaw)

    # Rolling gaze deviation ratio (per-sample, not per-window)
    gaze_deviation = (np.abs(yaw_deg) > 25).astype(np.float32)

    # Build 8 visual features
    visual = np.column_stack([
        yaw_deg,                                          # gaze_yaw
        pitch_deg,                                        # gaze_pitch
        yaw_deg * 0.8 + rng.normal(0, 2, N),             # head_yaw (correlated)
        pitch_deg * 0.6 + rng.normal(0, 2, N),           # head_pitch
        rng.normal(0, 3, N),                              # head_roll
        np.ones(N),                                       # face_count = 1
        gaze_deviation,                                   # gaze_deviation_ratio
        rng.normal(25, 3, N),                             # face_embedding_norm
    ])

    # Build 8 behavioral features (neutral — gaze dataset has no typing data)
    behavioral = np.column_stack([
        rng.normal(2.5, 0.5, N),    # keystroke_rate
        rng.normal(120, 20, N),     # mean_dwell_time
        rng.normal(180, 30, N),     # mean_flight_time
        rng.normal(0.3, 0.1, N),    # burst_coefficient
        rng.normal(150, 30, N),     # cursor_velocity
        rng.normal(0.5, 0.1, N),    # click_frequency
        rng.normal(0.1, 0.05, N),   # idle_ratio
        rng.normal(0.85, 0.05, N),  # trajectory_linearity
    ])

    return np.hstack([visual, behavioral]).astype(np.float32)


def assign_labels(gaze_angles: np.ndarray) -> np.ndarray:
    """
    Label each sample:
      0 = Normal           (|yaw| < 25°)
      1 = Gaze/Distraction (|yaw| >= 25°)
    """
    yaw_deg = np.degrees(gaze_angles[:, 1])
    return np.where(np.abs(yaw_deg) >= 25, 1, 0).astype(np.int64)


def process_all_participants() -> tuple[np.ndarray, np.ndarray]:
    """Process all participant .mat files and return (X, y)."""
    all_X, all_y = [], []

    participants = sorted(NORMALIZED_DIR.glob("p*"))
    if not participants:
        raise FileNotFoundError(
            f"No participant folders found in {NORMALIZED_DIR}\n"
            f"Make sure MPIIGaze is extracted to: {MPIIGAZE_DIR}"
        )

    print(f"  Found {len(participants)} participants")

    for p_dir in participants:
        mat_files = sorted(p_dir.glob("*.mat"))
        p_samples = 0

        for mat_path in mat_files:
            gaze = extract_gaze_from_mat(mat_path)
            if gaze is None or len(gaze) == 0:
                continue

            X = gaze_to_features(gaze)
            y = assign_labels(gaze)

            all_X.append(X)
            all_y.append(y)
            p_samples += len(X)

        if p_samples > 0:
            print(f"  {p_dir.name}: {p_samples} samples")

    if not all_X:
        raise ValueError("No gaze data could be extracted. Check .mat file format.")

    return np.vstack(all_X), np.concatenate(all_y)


def merge_with_existing(X_new: np.ndarray, y_new: np.ndarray) -> tuple:
    """Merge MPIIGaze features with existing combined dataset."""
    combined_npz = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.npz"

    if combined_npz.exists():
        data = np.load(combined_npz)
        X_existing = data["X"].astype(np.float32)
        y_existing = data["y"].astype(np.int64)
        print(f"  Existing dataset: {len(X_existing)} samples")
    else:
        # Fall back to synthetic
        syn_npz = BASE_DIR / "cheating_detection" / "data" / "synthetic_dataset.npz"
        data = np.load(syn_npz)
        X_existing = data["X"].astype(np.float32)
        y_existing = data["y"].astype(np.int64)
        print(f"  Synthetic dataset: {len(X_existing)} samples")

    X_merged = np.vstack([X_existing, X_new])
    y_merged  = np.concatenate([y_existing, y_new])

    # Shuffle
    idx = np.random.default_rng(42).permutation(len(X_merged))
    return X_merged[idx], y_merged[idx]


def run():
    print("\n=== MPIIGaze Feature Extraction ===")

    if not NORMALIZED_DIR.exists():
        print(f"ERROR: {NORMALIZED_DIR} not found.")
        print("Please extract MPIIGaze to datasets/gaze/MPIIGaze/")
        return None, None

    print("Extracting gaze features from .mat files...")
    X_gaze, y_gaze = process_all_participants()

    label_counts = dict(zip(*np.unique(y_gaze, return_counts=True)))
    print(f"  Total gaze samples : {len(X_gaze)}")
    print(f"  Normal (0)         : {label_counts.get(0, 0)}")
    print(f"  Looking Away (1)   : {label_counts.get(1, 0)}")

    # Save gaze-only features
    np.savez(OUTPUT_NPZ, X=X_gaze, y=y_gaze)
    feature_names = [
        "gaze_yaw","gaze_pitch","head_yaw","head_pitch","head_roll",
        "face_count","gaze_deviation_ratio","face_embedding_norm",
        "keystroke_rate","mean_dwell_time","mean_flight_time","burst_coefficient",
        "cursor_velocity","click_frequency","idle_ratio","trajectory_linearity","label"
    ]
    df = pd.DataFrame(np.hstack([X_gaze, y_gaze.reshape(-1,1)]), columns=feature_names)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_NPZ}")

    print("\nMerging with existing dataset...")
    X_merged, y_merged = merge_with_existing(X_gaze, y_gaze)

    merged_counts = dict(zip(*np.unique(y_merged, return_counts=True)))
    print(f"  Merged total    : {len(X_merged)} samples")
    print(f"  Label dist      : {merged_counts}")

    # Save merged dataset (overwrites combined_dataset.npz)
    merged_npz = BASE_DIR / "cheating_detection" / "data" / "combined_dataset.npz"
    np.savez(merged_npz, X=X_merged, y=y_merged)
    print(f"  Saved: {merged_npz}")

    return X_merged, y_merged


if __name__ == "__main__":
    run()
