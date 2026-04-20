"""
ensemble_inference.py — Multi-model ensemble inference for production.

Loads every available trained model (MLP, RF, SVM, LightGBM, XGBoost, ONNX)
and runs weighted soft voting across all of them.

If a model file is missing or fails to load, it's skipped gracefully — the
ensemble works with whatever subset is available (minimum: MLP).

Each model's probability vector is averaged (weighted) to produce the final
cheating probability.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

from .inference import CLASS_NAMES, load_model, load_scaler

logger = logging.getLogger(__name__)


# ── Default weights (higher = more trusted) ───────────────────────────────────
# Tuned based on validation accuracy during training.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "mlp":         2.5,   # Primary deep model
    "random_forest": 2.0,
    "xgboost":     2.0,
    "lightgbm":    2.0,
    "svm":         1.0,
    "onnx":        2.5,   # Same as MLP (it IS the MLP, just faster)
}


def _probe_model_input_dim(model, probe_dims=(16, 32)) -> Optional[int]:
    """Probe a model to find which input dim it accepts. Returns the first working dim."""
    for dim in probe_dims:
        try:
            x = np.zeros((1, dim), dtype=np.float64)
            if hasattr(model, "predict_proba"):
                model.predict_proba(x)
                return dim
            elif hasattr(model, "predict"):
                model.predict(x)
                return dim
        except Exception:
            continue
    return None


class EnsembleInferencer:
    """Weighted soft-voting ensemble over available trained models."""

    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.weights: Dict[str, float] = {}
        self.input_dims: Dict[str, int] = {}
        self.scaler = None
        self.onnx_session = None
        self.n_classes: int = 5

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(
        self,
        models_dir: str,
        scaler_filename: str = "scaler.joblib",
    ) -> None:
        """
        Discover and load every supported model file from `models_dir`.

        Expected filenames:
          mlp_model.pth           → MLP
          mlp_model.onnx          → ONNX-optimized MLP (preferred for speed)
          rf_model.joblib         → Random Forest
          svm_model.joblib        → SVM
          lgb_model.joblib        → LightGBM
          xgb_model.joblib        → XGBoost
          scaler.joblib           → StandardScaler (16-dim)
        """
        # --- Scaler (required) ------------------------------------------------
        scaler_path = os.path.join(models_dir, scaler_filename)
        if os.path.exists(scaler_path):
            try:
                self.scaler = load_scaler(scaler_path)
                logger.info("[Ensemble] Loaded scaler from %s", scaler_path)
            except Exception as exc:
                logger.error("[Ensemble] Failed to load scaler: %s", exc)

        # --- MLP (PyTorch) — optional; skipped in slim production image ──────
        mlp_path = os.path.join(models_dir, "mlp_model.pth")
        if _HAS_TORCH and os.path.exists(mlp_path):
            try:
                self.models["mlp"] = load_model(mlp_path)
                self.weights["mlp"] = DEFAULT_WEIGHTS["mlp"]
                self.input_dims["mlp"] = 16
                logger.info("[Ensemble] Loaded MLP (PyTorch)")
            except Exception as exc:
                logger.warning("[Ensemble] MLP skipped: %s", exc)
        elif not _HAS_TORCH and os.path.exists(mlp_path):
            logger.info("[Ensemble] MLP .pth found but PyTorch not installed — using ONNX instead")

        # --- ONNX MLP (optimized for production) ------------------------------
        onnx_path = os.path.join(models_dir, "mlp_model.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"],
                )
                self.weights["onnx"] = DEFAULT_WEIGHTS["onnx"]
                self.input_dims["onnx"] = 16
                logger.info("[Ensemble] Loaded ONNX MLP (optimized)")
            except Exception as exc:
                logger.warning("[Ensemble] ONNX skipped: %s", exc)

        # --- Sklearn-compatible models (RF, SVM, LGB, XGB) --------------------
        sklearn_models = [
            ("random_forest", "rf_model.joblib"),
            ("svm",           "svm_model.joblib"),
            ("lightgbm",      "lgb_model.joblib"),
            ("xgboost",       "xgb_model.joblib"),
        ]
        for name, filename in sklearn_models:
            path = os.path.join(models_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                import joblib
                model = joblib.load(path)
                input_dim = _probe_model_input_dim(model)
                if input_dim is None:
                    logger.warning(
                        "[Ensemble] %s loaded but input dim probe failed — skipping",
                        name,
                    )
                    continue
                self.models[name] = model
                self.weights[name] = DEFAULT_WEIGHTS[name]
                self.input_dims[name] = input_dim
                logger.info(
                    "[Ensemble] Loaded %s (input_dim=%d)", name, input_dim,
                )
            except Exception as exc:
                logger.warning("[Ensemble] %s skipped: %s", name, exc)

        total = len(self.models) + (1 if self.onnx_session else 0)
        logger.info("[Ensemble] Ready with %d model(s) loaded", total)
        if total == 0:
            logger.error("[Ensemble] No models loaded — predictions will return defaults!")

    # ── Per-model probability extraction ──────────────────────────────────────

    def _predict_mlp(self, x_scaled_16: np.ndarray) -> np.ndarray:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed — ONNX path should be used instead")
        model = self.models["mlp"]
        with torch.no_grad():
            x = torch.tensor(x_scaled_16, dtype=torch.float32)
            logits = model(x)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba[0]  # shape (n_classes,)

    def _predict_onnx(self, x_scaled_16: np.ndarray) -> np.ndarray:
        inp_name = self.onnx_session.get_inputs()[0].name
        out = self.onnx_session.run(None, {inp_name: x_scaled_16.astype(np.float32)})[0]
        logits = out[0]
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def _predict_sklearn(self, name: str, x_raw_16: np.ndarray, x_scaled_16: np.ndarray) -> Optional[np.ndarray]:
        """Run a sklearn-compatible model. Handles both 16-dim and 32-dim models."""
        model = self.models[name]
        dim = self.input_dims[name]
        # RF/SVM typically work on raw features; LGB/XGB are flexible.
        # If the model was trained on 32 dims, we can't serve it with 16 — skip.
        if dim != 16:
            logger.debug(
                "[Ensemble] Skipping %s at inference (expected %d dims, have 16)",
                name, dim,
            )
            return None
        x_input = x_raw_16 if name in ("random_forest", "svm") else x_scaled_16
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_input)[0]
            else:
                pred = int(model.predict(x_input)[0])
                proba = np.zeros(self.n_classes)
                proba[pred] = 1.0
            # Pad if model was trained on fewer classes (e.g. binary)
            if len(proba) < self.n_classes:
                padded = np.zeros(self.n_classes)
                padded[:len(proba)] = proba
                proba = padded
            elif len(proba) > self.n_classes:
                proba = proba[:self.n_classes]
            return proba
        except Exception as exc:
            logger.debug("[Ensemble] %s prediction failed: %s", name, exc)
            return None

    # ── Public predict ────────────────────────────────────────────────────────

    def predict(
        self,
        feature_vector: np.ndarray,
        threshold: float = 0.70,
    ) -> dict:
        """
        Run the full ensemble on a single 16-dim feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray — shape (16,) or (1, 16).
        threshold : float — probability above which a sample is flagged.

        Returns
        -------
        dict with the same shape as inference.predict() output, plus:
            - per_model_votes : dict[str, list[float]]  per-model probability vectors
            - ensemble_size   : int  how many models contributed
        """
        vec = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler is None:
            logger.error("[Ensemble] No scaler loaded — returning Normal default")
            return self._default_result(vec)

        x_scaled = self.scaler.transform(vec).astype(np.float32)

        # Collect per-model probability vectors
        votes: Dict[str, np.ndarray] = {}

        if "mlp" in self.models:
            votes["mlp"] = self._predict_mlp(x_scaled)

        if self.onnx_session is not None:
            votes["onnx"] = self._predict_onnx(x_scaled)

        for name in ("random_forest", "svm", "lightgbm", "xgboost"):
            if name in self.models:
                p = self._predict_sklearn(name, vec, x_scaled)
                if p is not None:
                    votes[name] = p

        if not votes:
            logger.error("[Ensemble] No valid votes — returning Normal default")
            return self._default_result(vec)

        # Weighted average
        total_weight = 0.0
        combined = np.zeros(self.n_classes)
        for name, proba in votes.items():
            w = self.weights.get(name, 1.0)
            combined += w * proba
            total_weight += w
        combined /= total_weight

        predicted_class = int(np.argmax(combined))
        cheating_prob = float(1.0 - combined[0])

        return {
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[predicted_class],
            "cheating_probability": round(cheating_prob, 4),
            "class_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(CLASS_NAMES, combined)
            },
            "is_cheating": cheating_prob >= threshold,
            "per_model_votes": {
                name: [round(float(p), 4) for p in proba]
                for name, proba in votes.items()
            },
            "ensemble_size": len(votes),
        }

    def _default_result(self, vec: np.ndarray) -> dict:
        return {
            "predicted_class": 0,
            "class_name": "Normal",
            "cheating_probability": 0.0,
            "class_probabilities": {n: (1.0 if n == "Normal" else 0.0) for n in CLASS_NAMES},
            "is_cheating": False,
            "per_model_votes": {},
            "ensemble_size": 0,
        }

    def get_status(self) -> dict:
        """Return which models are loaded — used by /health."""
        return {
            "scaler_loaded": self.scaler is not None,
            "models_loaded": list(self.models.keys()) + (["onnx"] if self.onnx_session else []),
            "ensemble_size": len(self.models) + (1 if self.onnx_session else 0),
            "weights": self.weights,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_ensemble: Optional[EnsembleInferencer] = None


def init_ensemble(models_dir: str) -> EnsembleInferencer:
    """Load all available models into a process-level singleton."""
    global _ensemble
    _ensemble = EnsembleInferencer()
    _ensemble.load_all(models_dir)
    return _ensemble


def get_ensemble() -> Optional[EnsembleInferencer]:
    """Return the loaded ensemble (or None if not initialised)."""
    return _ensemble
