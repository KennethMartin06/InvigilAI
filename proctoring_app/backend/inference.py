"""
inference.py — Load trained MLP + scaler and run predictions on feature vectors.

The MLP architecture is reconstructed here to exactly match the class defined
in cheating_detection/models/train.py so that torch.load() can deserialize
the state_dict correctly.

Architecture: Linear(16,128) → ReLU → Dropout(0.3)
              → Linear(128,64) → ReLU → Dropout(0.3)
              → Linear(64,5)

PyTorch is an OPTIONAL dependency in production — if it's not installed,
the MLP path is skipped and the ensemble falls back to the ONNX runtime
(which is ~80× smaller to ship).
"""

import logging
from typing import Optional

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Optional torch — keep backend tiny in production (onnxruntime only)
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _HAS_TORCH = False
    logger.info("PyTorch not installed — falling back to ONNX-only inference")

CLASS_NAMES = [
    "Normal",
    "Gaze/Distraction",
    "External Device",
    "Multi-Person",
    "Abnormal Keystroke",
]


# ── MLP — only defined when torch is importable ──────────────────────────────

if _HAS_TORCH:

    class MLP(nn.Module):
        """
        Multi-Layer Perceptron for cheating detection.

        Mirrors the architecture from cheating_detection/models/train.py so that
        the saved state_dict can be loaded without errors.
        """

        def __init__(
            self,
            input_dim: int = 16,
            hidden_layers: list = None,
            dropout: float = 0.3,
            n_classes: int = 5,
        ) -> None:
            super().__init__()
            hidden_layers = hidden_layers or [128, 64]
            layers = []
            prev_dim = input_dim
            for h in hidden_layers:
                layers += [
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
                prev_dim = h
            layers.append(nn.Linear(prev_dim, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            """Forward pass — returns raw logits (no softmax)."""
            return self.net(x)

else:

    class MLP:  # type: ignore
        """Stub when torch is unavailable — never instantiated."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not installed — MLP class unavailable")


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Instantiate the MLP and load saved weights from a .pth file.

    Raises RuntimeError if PyTorch is not installed.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not installed — cannot load .pth model")
    model = MLP(input_dim=16, hidden_layers=[128, 64], dropout=0.3, n_classes=5)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_scaler(scaler_path: str):
    """Load the fitted StandardScaler from a .joblib file."""
    return joblib.load(scaler_path)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    model,
    scaler,
    feature_vector: np.ndarray,
    threshold: float = 0.70,
) -> dict:
    """
    Run inference on a single 16-dimensional feature vector via the MLP.

    Raises RuntimeError if PyTorch is not installed.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not installed — use ensemble_inference instead")

    vec = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec_scaled = scaler.transform(vec).astype(np.float32)

    with torch.no_grad():
        x_tensor = torch.tensor(vec_scaled, dtype=torch.float32)
        logits = model(x_tensor)
        proba = torch.softmax(logits, dim=1).squeeze().numpy()

    predicted_class = int(np.argmax(proba))
    cheating_prob = float(1.0 - proba[0])

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "cheating_probability": round(cheating_prob, 4),
        "class_probabilities": {
            name: round(float(p), 4)
            for name, p in zip(CLASS_NAMES, proba)
        },
        "is_cheating": cheating_prob >= threshold,
    }


# ── Module-level singletons (populated on startup) ───────────────────────────

_model = None
_scaler = None


def init_inference(model_path: str, scaler_path: str) -> None:
    """
    Load model and scaler into module-level singletons.

    Skips MLP load gracefully if PyTorch is not installed.
    """
    global _model, _scaler
    _scaler = load_scaler(scaler_path)
    if _HAS_TORCH:
        _model = load_model(model_path)
    else:
        _model = None


def get_model():
    """Return the loaded MLP singleton."""
    if _model is None:
        raise RuntimeError("MLP not available — use ensemble_inference (ONNX) instead")
    return _model


def get_scaler():
    """Return the loaded scaler singleton."""
    if _scaler is None:
        raise RuntimeError("Scaler not initialised — call init_inference() first")
    return _scaler

