"""
inference.py — Load trained MLP + scaler and run predictions on feature vectors.

The MLP architecture is reconstructed here to exactly match the class defined
in cheating_detection/models/train.py so that torch.load() can deserialize
the state_dict correctly.

Architecture: Linear(16,128) → ReLU → Dropout(0.3)
              → Linear(128,64) → ReLU → Dropout(0.3)
              → Linear(64,5)
"""

import numpy as np
import torch
import torch.nn as nn
import joblib
from typing import Optional

CLASS_NAMES = [
    "Normal",
    "Gaze/Distraction",
    "External Device",
    "Multi-Person",
    "Abnormal Keystroke",
]


# ── MLP — must exactly match cheating_detection/models/train.py ──────────────

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits (no softmax)."""
        return self.net(x)


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_model(model_path: str) -> MLP:
    """
    Instantiate the MLP and load saved weights from a .pth file.

    Parameters
    ----------
    model_path : str — path to mlp_model.pth.

    Returns
    -------
    MLP — model in eval mode on CPU.

    Raises
    ------
    RuntimeError if the file cannot be loaded or architecture mismatch.
    """
    model = MLP(input_dim=16, hidden_layers=[128, 64], dropout=0.3, n_classes=5)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_scaler(scaler_path: str):
    """
    Load the fitted StandardScaler from a .joblib file.

    Parameters
    ----------
    scaler_path : str

    Returns
    -------
    sklearn.preprocessing.StandardScaler
    """
    return joblib.load(scaler_path)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    model: MLP,
    scaler,
    feature_vector: np.ndarray,
    threshold: float = 0.70,
) -> dict:
    """
    Run inference on a single 16-dimensional feature vector.

    Parameters
    ----------
    model : MLP — loaded, in eval mode.
    scaler : StandardScaler — fitted on training data.
    feature_vector : np.ndarray — shape (16,) or (1, 16).
    threshold : float — cheating probability threshold.

    Returns
    -------
    dict with keys:
        predicted_class       : int
        class_name            : str
        cheating_probability  : float  (1 - P(Normal))
        class_probabilities   : dict[str, float]
        is_cheating           : bool
    """
    vec = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

    # Replace any NaN/Inf with 0 before scaling
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

_model: Optional[MLP] = None
_scaler = None


def init_inference(model_path: str, scaler_path: str) -> None:
    """
    Load model and scaler into module-level singletons.

    Called once from app.py lifespan on startup.

    Parameters
    ----------
    model_path : str
    scaler_path : str
    """
    global _model, _scaler
    _model = load_model(model_path)
    _scaler = load_scaler(scaler_path)


def get_model() -> MLP:
    """Return the loaded MLP singleton."""
    if _model is None:
        raise RuntimeError("Model not initialised — call init_inference() first")
    return _model


def get_scaler():
    """Return the loaded scaler singleton."""
    if _scaler is None:
        raise RuntimeError("Scaler not initialised — call init_inference() first")
    return _scaler
