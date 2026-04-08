"""
hmdb51_pipeline.py — Extract visual features from HMDB-51 video clips.

Samples frames from relevant action classes, runs MediaPipe Face Mesh on
each frame to extract gaze / head-pose features, and produces labelled
16-dimensional training samples.

Label mapping:
  0  Normal            — sit, smile, drink, eat, stand, situp, walk
  1  Gaze/Distraction  — talk, wave, laugh, smoke

Usage:
    python cheating_detection/data/hmdb51_pipeline.py
"""

import logging
import os
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
HMDB_ROOT  = Path.home() / "invigilai" / "datasets" / "video" / "HMDB51" / "HMDB51"
SAVE_PATH  = _REPO_ROOT / "cheating_detection" / "data" / "hmdb51_features.npz"
COMBINED_PATH = _REPO_ROOT / "cheating_detection" / "data" / "combined_dataset.npz"

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    # Normal exam behaviours
    "sit":    0,
    "smile":  0,
    "drink":  0,
    "eat":    0,
    "stand":  0,
    "situp":  0,
    "walk":   0,
    # Suspicious / distracted behaviours
    "talk":   1,
    "wave":   1,
    "laugh":  1,
    "smoke":  1,
}

FRAMES_PER_VIDEO = 8   # frames sampled uniformly from each clip

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    import cv2
    _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    _MEDIAPIPE_OK = True
except Exception as exc:
    logger.warning("MediaPipe unavailable: %s", exc)
    _MEDIAPIPE_OK = False


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_frame_features(frame: "np.ndarray") -> "np.ndarray | None":
    """
    Run MediaPipe on a BGR frame and return an 8-dim visual feature vector,
    or None if no face was detected.
    """
    if not _MEDIAPIPE_OK:
        return None
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark

    # ── Gaze yaw/pitch from iris landmarks ────────────────────────────────────
    def px(idx): return lm[idx].x * w, lm[idx].y * h
    l_iris = np.mean([list(px(i)) for i in [474, 475, 476, 477]], axis=0)
    r_iris = np.mean([list(px(i)) for i in [469, 470, 471, 472]], axis=0)
    l0, l1 = px(33), px(133)
    r0, r1 = px(362), px(263)

    def ratio(cx, c0, c1):
        span = abs(c1[0] - c0[0])
        return 0.5 if span < 1e-3 else (cx - min(c0[0], c1[0])) / span

    avg_x = (ratio(l_iris[0], l0, l1) + ratio(r_iris[0], r0, r1)) / 2.0
    avg_y = ((l_iris[1] - l0[1]) / max(abs(l1[1] - l0[1]), 1e-3) +
             (r_iris[1] - r0[1]) / max(abs(r1[1] - r0[1]), 1e-3)) / 2.0

    gaze_yaw   = (avg_x - 0.5) * 90.0
    gaze_pitch = (avg_y - 0.5) * 60.0

    # ── Head pose (simplified from nose / chin landmarks) ─────────────────────
    nose_x = lm[1].x - 0.5          # -0.5 … 0.5 relative to frame
    nose_y = lm[1].y - 0.5
    head_yaw   = nose_x * 60.0
    head_pitch = nose_y * 40.0
    head_roll  = 0.0

    # ── Other features ────────────────────────────────────────────────────────
    face_count       = 1.0
    gaze_dev         = float(abs(gaze_yaw) > 25.0)
    emb_norm         = 1.0

    return np.array([
        gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll,
        face_count, gaze_dev, emb_norm,
    ], dtype=np.float32)


def _behavioral_for_label(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate plausible 8-dim behavioral features for a given label.
    The values mirror the synthetic distribution from generate_dataset.py.
    """
    if label == 0:   # normal
        return np.array([
            rng.normal(3.0, 0.8),      # keystroke_rate
            rng.normal(100.0, 20.0),   # mean_dwell_time
            rng.normal(150.0, 30.0),   # mean_flight_time
            rng.uniform(0.8, 1.2),     # burst_coefficient
            rng.normal(180.0, 50.0),   # cursor_velocity
            rng.uniform(0.3, 0.8),     # click_frequency
            rng.uniform(0.05, 0.15),   # idle_ratio
            rng.uniform(0.4, 0.7),     # trajectory_linearity
        ], dtype=np.float32)
    else:            # distracted / suspicious
        return np.array([
            rng.normal(0.5, 0.3),      # very low keystroke rate
            rng.normal(200.0, 60.0),
            rng.normal(300.0, 80.0),
            rng.uniform(1.5, 3.0),     # bursty
            rng.normal(80.0, 40.0),
            rng.uniform(0.0, 0.2),
            rng.uniform(0.4, 0.8),     # high idle ratio
            rng.uniform(0.1, 0.4),
        ], dtype=np.float32)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_hmdb51_pipeline() -> tuple[np.ndarray, np.ndarray]:
    """
    Process HMDB-51 clips and return (X, y) arrays.

    Returns
    -------
    X : np.ndarray, shape (N, 16)
    y : np.ndarray, shape (N,)
    """
    if not HMDB_ROOT.exists():
        logger.error("HMDB-51 not found at %s", HMDB_ROOT)
        return np.empty((0, 16)), np.empty(0)

    rng = np.random.default_rng(seed=42)
    all_X, all_y = [], []
    class_counts = {}

    for class_name, label in LABEL_MAP.items():
        class_dir = HMDB_ROOT / class_name
        if not class_dir.exists():
            logger.warning("Class directory missing: %s", class_dir)
            continue

        video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
        logger.info("  %-12s  label=%d  videos=%d", class_name, label, len(video_files))
        count = 0

        for vf in video_files:
            try:
                cap = cv2.VideoCapture(str(vf))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 2:
                    cap.release()
                    continue

                indices = np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int)
                for fi in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue

                    vis = _extract_frame_features(frame)
                    if vis is None:
                        # No face — use high-suspicion defaults if suspicious class
                        if label == 1:
                            vis = np.array([35.0, 15.0, 30.0, 10.0, 5.0, 0.0, 1.0, 1.0],
                                           dtype=np.float32)
                        else:
                            continue  # skip faceless normal frames

                    beh = _behavioral_for_label(label, rng)
                    all_X.append(np.concatenate([vis, beh]))
                    all_y.append(label)
                    count += 1

                cap.release()
            except Exception as exc:
                logger.warning("Error processing %s: %s", vf.name, exc)

        class_counts[class_name] = count

    if not all_X:
        logger.warning("No samples extracted from HMDB-51")
        return np.empty((0, 16)), np.empty(0)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    logger.info("HMDB-51 extraction complete: %d samples", len(y))
    for cls, cnt in class_counts.items():
        logger.info("  %-12s : %d frames", cls, cnt)

    np.savez_compressed(SAVE_PATH, X=X, y=y)
    logger.info("Saved HMDB-51 features → %s", SAVE_PATH)
    return X, y


def merge_with_combined(X_new: np.ndarray, y_new: np.ndarray) -> None:
    """Append HMDB-51 samples to the existing combined_dataset.npz."""
    if len(X_new) == 0:
        logger.warning("No HMDB-51 samples to merge.")
        return

    if COMBINED_PATH.exists():
        data = np.load(COMBINED_PATH)
        X_old, y_old = data["X"], data["y"]
        X_merged = np.vstack([X_old, X_new])
        y_merged = np.concatenate([y_old, y_new])
        logger.info("Merged: %d existing + %d HMDB-51 = %d total",
                    len(y_old), len(y_new), len(y_merged))
    else:
        X_merged, y_merged = X_new, y_new
        logger.info("No existing dataset found — using HMDB-51 only (%d samples)", len(y_merged))

    np.savez_compressed(COMBINED_PATH, X=X_merged, y=y_merged)
    logger.info("Updated combined_dataset.npz → %s", COMBINED_PATH)


if __name__ == "__main__":
    logger.info("=== HMDB-51 Pipeline ===")
    logger.info("Dataset path : %s", HMDB_ROOT)
    logger.info("Classes      : %s", list(LABEL_MAP.keys()))

    X, y = run_hmdb51_pipeline()
    if len(X) > 0:
        merge_with_combined(X, y)
        logger.info("Done. Run main.py to retrain.")
    else:
        logger.error("Pipeline produced no samples — check dataset path and MediaPipe install.")
        sys.exit(1)
