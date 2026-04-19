"""
feature_fusion.py — Combine visual and behavioral feature dicts into a
16-dimensional numpy array and invoke the ML model.

Feature order must exactly match the training pipeline:
  [0..7]  visual features
  [8..15] behavioral features
"""

import numpy as np
import logging

from .inference import predict, get_model, get_scaler
from .ensemble_inference import get_ensemble
from .config import settings

logger = logging.getLogger(__name__)

# Feature names in the exact order the model expects
_VISUAL_KEYS = [
    "gaze_yaw",
    "gaze_pitch",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_count",
    "gaze_deviation_ratio",
    "face_embedding_norm",
]

_BEHAVIORAL_KEYS = [
    "keystroke_rate",
    "mean_dwell_time",
    "mean_flight_time",
    "burst_coefficient",
    "cursor_velocity",
    "click_frequency",
    "idle_ratio",
    "trajectory_linearity",
]

# Safe defaults used when a key is missing
_VISUAL_DEFAULTS = {
    "gaze_yaw": 0.0,
    "gaze_pitch": 5.0,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "head_roll": 0.0,
    "face_count": 1.0,
    "gaze_deviation_ratio": 0.05,
    "face_embedding_norm": 1.0,
}

_BEHAVIORAL_DEFAULTS = {
    "keystroke_rate": 3.0,
    "mean_dwell_time": 100.0,
    "mean_flight_time": 150.0,
    "burst_coefficient": 1.0,
    "cursor_velocity": 180.0,
    "click_frequency": 0.5,
    "idle_ratio": 0.10,
    "trajectory_linearity": 0.55,
}


def fuse_features(visual: dict, behavioral: dict) -> np.ndarray:
    """
    Merge visual and behavioral feature dicts into a (1, 16) numpy array.

    Missing keys are filled with domain-neutral defaults.

    Parameters
    ----------
    visual : dict — 8 visual feature values.
    behavioral : dict — 8 behavioral feature values.

    Returns
    -------
    np.ndarray, shape (1, 16) — ready for scaler.transform().
    """
    v_vec = [
        float(visual.get(k, _VISUAL_DEFAULTS[k]))
        for k in _VISUAL_KEYS
    ]
    b_vec = [
        float(behavioral.get(k, _BEHAVIORAL_DEFAULTS[k]))
        for k in _BEHAVIORAL_KEYS
    ]
    return np.array(v_vec + b_vec, dtype=np.float64).reshape(1, -1)


def fuse_and_predict(visual: dict, behavioral: dict) -> dict:
    """
    Fuse features and run the loaded ML model.

    If a phone is detected in the visual dict (phone_detected=True), the
    result is immediately overridden to External Device class regardless of
    the MLP output.

    Parameters
    ----------
    visual : dict
    behavioral : dict

    Returns
    -------
    dict — prediction result from inference.predict(), augmented with
           the raw feature vector and phone detection fields.
    """
    feature_vec = fuse_features(visual, behavioral)

    # ── Phone detection override ──────────────────────────────────────────────
    phone_detected  = bool(visual.get("phone_detected", False))
    phone_confidence = float(visual.get("phone_confidence", 0.0))

    if phone_detected:
        conf = max(phone_confidence, 0.92)
        result = {
            "predicted_class": 2,
            "class_name": "External Device",
            "cheating_probability": round(conf, 4),
            "class_probabilities": {
                "Normal": round(1.0 - conf, 4),
                "Gaze/Distraction": 0.0,
                "External Device": round(conf, 4),
                "Multi-Person": 0.0,
                "Abnormal Keystroke": 0.0,
            },
            "is_cheating": True,
        }
        result["feature_vector"]    = feature_vec.squeeze().tolist()
        result["phone_detected"]    = True
        result["phone_confidence"]  = phone_confidence
        return result

    # ── Ensemble inference (MLP + RF + SVM + LGB + XGB + ONNX) ──────────────
    ensemble = get_ensemble()
    if ensemble is not None and ensemble.get_status()["ensemble_size"] > 0:
        try:
            result = ensemble.predict(
                feature_vec.squeeze(),
                threshold=settings.cheating_threshold,
            )
        except Exception as exc:
            logger.error("Ensemble inference failed: %s — falling back to MLP", exc)
            result = _single_mlp_predict(feature_vec)
    else:
        # Fallback: legacy single-MLP inference
        result = _single_mlp_predict(feature_vec)

    result["feature_vector"]   = feature_vec.squeeze().tolist()
    result["phone_detected"]   = False
    result["phone_confidence"] = 0.0
    return result


def _single_mlp_predict(feature_vec: np.ndarray) -> dict:
    """Legacy single-MLP prediction path (used as fallback)."""
    try:
        model = get_model()
        scaler = get_scaler()
        return predict(
            model, scaler, feature_vec.squeeze(),
            threshold=settings.cheating_threshold,
        )
    except RuntimeError as exc:
        logger.error("MLP inference failed: %s — returning Normal default", exc)
        return {
            "predicted_class": 0,
            "class_name": "Normal",
            "cheating_probability": 0.0,
            "class_probabilities": {
                "Normal": 1.0,
                "Gaze/Distraction": 0.0,
                "External Device": 0.0,
                "Multi-Person": 0.0,
                "Abnormal Keystroke": 0.0,
            },
            "is_cheating": False,
        }
