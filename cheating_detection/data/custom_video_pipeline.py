"""
custom_video_pipeline.py — Extract training samples from a custom cheat video.

Samples frames uniformly, runs MediaPipe Face Mesh on each, then
auto-labels using extracted gaze / head-pose values:

  Label 0  Normal            — face forward, gaze centred
  Label 1  Gaze/Distraction  — large gaze deviation or head turn
  Label 3  Multi-Person      — more than one face detected

Usage:
    python cheating_detection/data/custom_video_pipeline.py \
        --video ~/invigilai/datasets/custom_cheat.mp4
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parents[2]
COMBINED_PATH = _REPO_ROOT / "cheating_detection" / "data" / "combined_dataset.npz"

# ── Labelling thresholds (degrees) ────────────────────────────────────────────
GAZE_YAW_SUSPICIOUS   = 20.0   # |gaze_yaw|   > this → suspicious
HEAD_YAW_SUSPICIOUS   = 18.0   # |head_yaw|   > this → suspicious
HEAD_PITCH_SUSPICIOUS = 20.0   # |head_pitch| > this → suspicious (looking down at notes)

FRAMES_PER_SECOND = 2          # extract 2 frames per second of video

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("cv2 not found — install opencv-python")

try:
    import mediapipe as mp
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    _MEDIAPIPE_OK = True
except Exception as exc:
    logger.warning("MediaPipe unavailable: %s", exc)
    _MEDIAPIPE_OK = False


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract(frame: np.ndarray) -> "tuple[np.ndarray, int] | None":
    """
    Run MediaPipe on a frame and return (8-dim visual vector, face_count).
    Returns None if no face detected.
    """
    if not _MEDIAPIPE_OK:
        return None
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    face_count = len(res.multi_face_landmarks)
    lm = res.multi_face_landmarks[0].landmark

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
    head_yaw   = (lm[1].x - 0.5) * 60.0
    head_pitch = (lm[1].y - 0.5) * 40.0
    head_roll  = 0.0
    gaze_dev   = float(abs(gaze_yaw) > 25.0)
    emb_norm   = 1.0

    vis = np.array([
        gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll,
        float(face_count), gaze_dev, emb_norm,
    ], dtype=np.float32)
    return vis, face_count


def _auto_label(vis: np.ndarray, face_count: int) -> int:
    """Assign a label based on extracted visual features."""
    if face_count > 1:
        return 3   # Multi-Person
    gaze_yaw   = abs(vis[0])
    head_yaw   = abs(vis[2])
    head_pitch = abs(vis[3])
    if (gaze_yaw > GAZE_YAW_SUSPICIOUS or
            head_yaw > HEAD_YAW_SUSPICIOUS or
            head_pitch > HEAD_PITCH_SUSPICIOUS):
        return 1   # Gaze/Distraction
    return 0       # Normal


def _behavioral(label: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic behavioral features matching the detected label."""
    if label == 0:
        return np.array([
            rng.normal(3.0, 0.8), rng.normal(100.0, 20.0),
            rng.normal(150.0, 30.0), rng.uniform(0.8, 1.2),
            rng.normal(180.0, 50.0), rng.uniform(0.3, 0.8),
            rng.uniform(0.05, 0.15), rng.uniform(0.4, 0.7),
        ], dtype=np.float32)
    else:
        return np.array([
            rng.normal(0.5, 0.3), rng.normal(200.0, 60.0),
            rng.normal(300.0, 80.0), rng.uniform(1.5, 3.0),
            rng.normal(80.0, 40.0), rng.uniform(0.0, 0.2),
            rng.uniform(0.4, 0.8), rng.uniform(0.1, 0.4),
        ], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_video(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Process a video and return (X, y) arrays."""
    vp = Path(video_path).expanduser()
    if not vp.exists():
        logger.error("Video not found: %s", vp)
        return np.empty((0, 16)), np.empty(0)

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    step = max(1, int(fps / FRAMES_PER_SECOND))

    logger.info("Video  : %s", vp.name)
    logger.info("FPS    : %.1f  |  Frames: %d  |  Duration: %.1fs", fps, total_frames, duration_s)
    logger.info("Sampling every %d frames (~%d samples expected)", step, total_frames // step)

    rng = np.random.default_rng(seed=7)
    all_X, all_y = [], []
    label_counts = {0: 0, 1: 0, 3: 0}
    fi = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            result = _extract(frame)
            if result is not None:
                vis, face_count = result
                label = _auto_label(vis, face_count)
                beh = _behavioral(label, rng)
                all_X.append(np.concatenate([vis, beh]))
                all_y.append(label)
                label_counts[label] = label_counts.get(label, 0) + 1
        fi += 1

    cap.release()

    if not all_X:
        logger.warning("No faces detected in video. Check MediaPipe installation.")
        return np.empty((0, 16)), np.empty(0)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    logger.info("Extracted %d samples", len(y))
    logger.info("  Normal (0)           : %d", label_counts.get(0, 0))
    logger.info("  Gaze/Distraction (1) : %d", label_counts.get(1, 0))
    logger.info("  Multi-Person (3)     : %d", label_counts.get(3, 0))
    return X, y


def merge_and_save(X_new: np.ndarray, y_new: np.ndarray) -> None:
    """Append new samples to combined_dataset.npz."""
    if len(X_new) == 0:
        return
    if COMBINED_PATH.exists():
        data = np.load(COMBINED_PATH)
        X = np.vstack([data["X"], X_new])
        y = np.concatenate([data["y"], y_new])
        logger.info("Merged: %d existing + %d new = %d total", len(data["y"]), len(y_new), len(y))
    else:
        X, y = X_new, y_new
    np.savez_compressed(COMBINED_PATH, X=X, y=y)
    logger.info("Saved → %s", COMBINED_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="~/invigilai/datasets/custom_cheat.mp4",
                        help="Path to the cheat video file")
    args = parser.parse_args()

    X, y = process_video(args.video)
    if len(X) > 0:
        merge_and_save(X, y)
        logger.info("Done. Run python3 -m cheating_detection.main to retrain.")
    else:
        sys.exit(1)
