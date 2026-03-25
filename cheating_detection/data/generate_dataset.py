"""
generate_dataset.py — Synthetic exam-session data generator.

Produces 240 simulated exam sessions, each broken into time windows of
~2 seconds.  Every window carries 8 visual features and 8 behavioral
features whose statistical properties are class-conditional, replicating
the distributions that would be observed in a real proctoring deployment.

Run standalone:
    python -m cheating_detection.data.generate_dataset
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow `python -m` or direct import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    CLASS_COUNTS,
    CLASS_NAMES,
    WINDOWS_PER_SESSION_MIN,
    WINDOWS_PER_SESSION_MAX,
    ALL_FEATURE_NAMES,
    DATASET_NPZ,
    DATASET_CSV,
    DATA_DIR,
)


# ── Per-class feature distributions ──────────────────────────────────────────
#
# Each tuple describes (mean, std) for a single feature under a particular
# cheating category.  This mirrors the statistical specification in the
# project requirements.

_VISUAL_PARAMS = {
    # label: [(gaze_yaw_mu, gaze_yaw_sigma), (gaze_pitch), (head_yaw),
    #          (head_pitch), (head_roll), (face_count), (gaze_dev_ratio),
    #          (face_emb_norm)]
    0: [(0, 8),    (5, 5),   (0, 10),  (0, 8),  (0, 6),  (1.0, 0.05), (0.04, 0.03), (1.00, 0.05)],
    1: [(28, 15),  (12, 8),  (32, 12), (8, 8),  (5, 6),  (1.0, 0.08), (0.55, 0.18), (1.00, 0.06)],
    2: [(18, 14),  (15, 10), (20, 12), (5, 8),  (4, 6),  (1.0, 0.10), (0.35, 0.15), (1.02, 0.08)],
    3: [(5,  10),  (6, 6),   (5, 10),  (3, 6),  (3, 5),  (2.1, 0.45), (0.12, 0.08), (1.25, 0.20)],
    4: [(4,  9),   (5, 5),   (4, 10),  (2, 6),  (2, 5),  (1.0, 0.06), (0.08, 0.05), (1.01, 0.05)],
}

_BEHAVIORAL_PARAMS = {
    # label: [(kr_mu, kr_sigma), (mdt), (mft), (bc), (cv), (cf), (ir), (tl)]
    0: [(4.0, 0.6),  (100, 12), (150, 25), (1.0, 0.15), (180, 40), (0.5, 0.15), (0.10, 0.04), (0.55, 0.08)],
    1: [(3.8, 0.7),  (105, 14), (155, 28), (1.0, 0.18), (175, 45), (0.5, 0.17), (0.12, 0.05), (0.54, 0.09)],
    2: [(2.5, 1.2),  (115, 20), (170, 40), (2.5, 0.60), (120, 55), (0.3, 0.15), (0.50, 0.12), (0.75, 0.10)],
    3: [(3.5, 0.8),  (102, 15), (152, 30), (1.1, 0.20), (165, 50), (0.5, 0.18), (0.14, 0.06), (0.56, 0.09)],
    4: [(7.5, 2.0),  (55,  18), (70,  25), (3.2, 0.80), (200, 60), (0.7, 0.20), (0.28, 0.10), (0.88, 0.07)],
}

# Gaussian noise scale added to every feature (simulates sensor noise)
_NOISE_SCALE = 0.05


def _clip_features(X: np.ndarray) -> np.ndarray:
    """
    Clip feature values to physically plausible ranges.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, 16)

    Returns
    -------
    np.ndarray — same shape, clipped.
    """
    # gaze_yaw [-90, 90]
    X[:, 0] = np.clip(X[:, 0], -90, 90)
    # gaze_pitch [-90, 90]
    X[:, 1] = np.clip(X[:, 1], -90, 90)
    # head_yaw [-90, 90]
    X[:, 2] = np.clip(X[:, 2], -90, 90)
    # head_pitch [-60, 60]
    X[:, 3] = np.clip(X[:, 3], -60, 60)
    # head_roll [-45, 45]
    X[:, 4] = np.clip(X[:, 4], -45, 45)
    # face_count [1, 4]
    X[:, 5] = np.clip(np.round(X[:, 5]), 1, 4)
    # gaze_deviation_ratio [0, 1]
    X[:, 6] = np.clip(X[:, 6], 0, 1)
    # face_embedding_norm [0.5, 2.0]
    X[:, 7] = np.clip(X[:, 7], 0.5, 2.0)
    # keystroke_rate [0, 20]
    X[:, 8] = np.clip(X[:, 8], 0, 20)
    # mean_dwell_time [10, 500]
    X[:, 9] = np.clip(X[:, 9], 10, 500)
    # mean_flight_time [10, 800]
    X[:, 10] = np.clip(X[:, 10], 10, 800)
    # burst_coefficient [0, 10]
    X[:, 11] = np.clip(X[:, 11], 0, 10)
    # cursor_velocity [0, 800]
    X[:, 12] = np.clip(X[:, 12], 0, 800)
    # click_frequency [0, 5]
    X[:, 13] = np.clip(X[:, 13], 0, 5)
    # idle_ratio [0, 1]
    X[:, 14] = np.clip(X[:, 14], 0, 1)
    # trajectory_linearity [0, 1]
    X[:, 15] = np.clip(X[:, 15], 0, 1)
    return X


def _generate_session(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one exam session as an array of time-window feature vectors.

    Parameters
    ----------
    label : int
        Cheating category (0–4).
    rng : np.random.Generator
        Pre-seeded random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_windows, 16)
        Rows are windows; columns are the 16 fused features.
    """
    n_windows = rng.integers(WINDOWS_PER_SESSION_MIN, WINDOWS_PER_SESSION_MAX + 1)

    v_params = _VISUAL_PARAMS[label]
    b_params = _BEHAVIORAL_PARAMS[label]

    visual = np.column_stack([
        rng.normal(mu, sigma, n_windows)
        for mu, sigma in v_params
    ])  # (n_windows, 8)

    behavioral = np.column_stack([
        rng.normal(mu, sigma, n_windows)
        for mu, sigma in b_params
    ])  # (n_windows, 8)

    X = np.hstack([visual, behavioral])  # (n_windows, 16)

    # Add small Gaussian noise across all features
    noise = rng.normal(0, _NOISE_SCALE, X.shape)
    X += noise

    X = _clip_features(X)
    return X


def generate_dataset(
    save_npz: bool = True,
    save_csv: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the full synthetic dataset.

    Creates 240 exam sessions distributed across 5 cheating categories.
    Each session is split into ~2-second time windows.  Windows are
    labelled with the session-level category.

    Parameters
    ----------
    save_npz : bool
        Whether to write ``data/synthetic_dataset.npz``.
    save_csv : bool
        Whether to write ``data/synthetic_dataset.csv``.
    verbose : bool
        Print progress information.

    Returns
    -------
    X : np.ndarray, shape (n_total_windows, 16)
    y : np.ndarray, shape (n_total_windows,)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    all_X = []
    all_y = []
    session_id_col = []

    session_counter = 0
    for label, n_sessions in CLASS_COUNTS.items():
        for _ in range(n_sessions):
            session_data = _generate_session(label, rng)
            n_windows = session_data.shape[0]
            all_X.append(session_data)
            all_y.extend([label] * n_windows)
            session_id_col.extend([session_counter] * n_windows)
            session_counter += 1

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)
    session_ids = np.array(session_id_col, dtype=np.int64)

    if verbose:
        print(f"[DataGen] Generated {session_counter} sessions → {len(y)} windows total")
        for lbl, name in enumerate(CLASS_NAMES):
            count = np.sum(y == lbl)
            print(f"  Label {lbl} ({name}): {count} windows")

    os.makedirs(DATA_DIR, exist_ok=True)

    if save_npz:
        np.savez_compressed(
            DATASET_NPZ,
            X=X,
            y=y,
            session_ids=session_ids,
            feature_names=np.array(ALL_FEATURE_NAMES),
            class_names=np.array(CLASS_NAMES),
        )
        if verbose:
            print(f"[DataGen] Saved .npz → {DATASET_NPZ}")

    if save_csv:
        df = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
        df.insert(0, "session_id", session_ids)
        df.insert(1, "label", y)
        df.insert(2, "class_name", [CLASS_NAMES[i] for i in y])
        df.to_csv(DATASET_CSV, index=False)
        if verbose:
            print(f"[DataGen] Saved .csv  → {DATASET_CSV}")

    return X, y


if __name__ == "__main__":
    generate_dataset()
