"""
visual_features.py — Visual feature extraction from webcam frame data.

In a real deployment each "session_data" dict would contain per-frame
measurements from:
  • A gaze-estimation network (e.g. MPIIGaze / ETH-XGaze) for gaze angles.
  • A head-pose estimator (e.g. FSA-Net / 6DRepNet) for head Euler angles.
  • A face detector (e.g. RetinaFace / MTCNN) for face count.
  • A face-recognition network (e.g. ResNet-50 ArcFace) for embeddings.

Since we operate on pre-generated synthetic windows the function below
demonstrates the computation logic that would apply to raw frame sequences.
"""

import numpy as np


def compute_gaze_deviation_ratio(
    gaze_yaw_frames: np.ndarray,
    threshold_deg: float = 25.0,
) -> float:
    """
    Fraction of frames within a window where the horizontal gaze angle
    exceeds a ±threshold_deg boundary.

    In a real system ``gaze_yaw_frames`` would be a 1-D array of per-frame
    gaze-yaw estimates (in degrees) produced by the gaze estimator for all
    frames captured during a 2-second window.

    Parameters
    ----------
    gaze_yaw_frames : np.ndarray, shape (n_frames,)
        Per-frame gaze-yaw angle in degrees.
    threshold_deg : float
        Boundary beyond which gaze is considered "off-screen".

    Returns
    -------
    float in [0, 1] — fraction of deviant frames.
    """
    if len(gaze_yaw_frames) == 0:
        return 0.0
    deviant = np.abs(gaze_yaw_frames) > threshold_deg
    return float(np.mean(deviant))


def compute_face_embedding_norm(embedding: np.ndarray) -> float:
    """
    L2 norm of a facial embedding vector produced by a recognition network.

    For a unit-normalised embedding (standard ArcFace output) the norm is
    exactly 1.0.  Variations arise from quantisation, model uncertainty, or
    partial occlusion.  Large deviations from 1.0 can signal a different
    person (impersonation) or image quality issues.

    Parameters
    ----------
    embedding : np.ndarray, shape (d,)
        Raw (not necessarily unit-normalised) embedding vector.

    Returns
    -------
    float — L2 norm of the embedding.
    """
    norm = float(np.linalg.norm(embedding))
    return norm if norm > 0 else 1.0


def extract_visual_features(session_data: dict) -> np.ndarray:
    """
    Extract the 8-dimensional visual feature vector for each time window
    in a session.

    Expected keys in ``session_data``:
        "gaze_yaw"          : list/array of per-window mean gaze-yaw (deg)
        "gaze_pitch"        : per-window mean gaze-pitch (deg)
        "head_yaw"          : per-window mean head-yaw (deg)
        "head_pitch"        : per-window mean head-pitch (deg)
        "head_roll"         : per-window mean head-roll (deg)
        "face_count"        : per-window face count (int)
        "gaze_yaw_frames"   : list of per-frame gaze-yaw arrays
                              (one array per window, each shape (n_frames,))
        "face_embeddings"   : list of per-window embedding arrays
                              (one array per window, each shape (d,))

    When the system uses synthetic pre-computed windows (as in this project)
    the deviation ratio and embedding norm are stored directly as scalars
    under "gaze_deviation_ratio" and "face_embedding_norm".

    Parameters
    ----------
    session_data : dict
        Dictionary with the keys described above.

    Returns
    -------
    np.ndarray, shape (n_windows, 8)
        Columns (in order):
          0  gaze_yaw
          1  gaze_pitch
          2  head_yaw
          3  head_pitch
          4  head_roll
          5  face_count
          6  gaze_deviation_ratio
          7  face_embedding_norm
    """
    n_windows = len(session_data["gaze_yaw"])
    features = np.zeros((n_windows, 8), dtype=np.float64)

    features[:, 0] = np.asarray(session_data["gaze_yaw"])
    features[:, 1] = np.asarray(session_data["gaze_pitch"])
    features[:, 2] = np.asarray(session_data["head_yaw"])
    features[:, 3] = np.asarray(session_data["head_pitch"])
    features[:, 4] = np.asarray(session_data["head_roll"])
    features[:, 5] = np.asarray(session_data["face_count"])

    # Gaze deviation ratio — computed per-window from raw frame data
    if "gaze_yaw_frames" in session_data:
        ratios = [
            compute_gaze_deviation_ratio(frames)
            for frames in session_data["gaze_yaw_frames"]
        ]
        features[:, 6] = np.asarray(ratios)
    else:
        features[:, 6] = np.asarray(session_data.get("gaze_deviation_ratio", 0.0))

    # Face embedding norm
    if "face_embeddings" in session_data:
        norms = [
            compute_face_embedding_norm(emb)
            for emb in session_data["face_embeddings"]
        ]
        features[:, 7] = np.asarray(norms)
    else:
        features[:, 7] = np.asarray(session_data.get("face_embedding_norm", 1.0))

    return features
