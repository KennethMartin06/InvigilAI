"""
calibration.py -- Temperature scaling and conformal prediction.

Temperature scaling: learns a single scalar T to calibrate softmax outputs.
Conformal prediction: provides prediction sets with guaranteed coverage.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import CONFORMAL_ALPHA


# -- Temperature Scaling -----------------------------------------------------

class TemperatureScaler:
    """Post-hoc temperature scaling for calibrating neural network outputs.

    Learns a single temperature T such that softmax(logits/T) is well-calibrated.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        """Find optimal temperature on validation set using grid search."""
        best_nll = float("inf")
        best_T = 1.0

        for T in np.arange(0.1, 5.0, 0.05):
            scaled = logits / T
            # softmax
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            proba = exp_s / exp_s.sum(axis=1, keepdims=True)
            # NLL
            nll = -np.log(proba[np.arange(len(y_true)), y_true.astype(int)] + 1e-10).mean()
            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        return self

    def transform(self, logits):
        """Apply temperature scaling to logits."""
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_temperature_scaling(model, X_val, y_val, verbose=True):
    """Fit temperature scaling on validation set.

    Works with MLP model by extracting logits.
    """
    import torch
    from cheating_detection.models.train import MLP

    if not isinstance(model, MLP):
        if verbose:
            print("[TempScale] Skipping -- model is not MLP")
        return None

    model.eval()
    x_t = torch.tensor(X_val, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        x_t = x_t.cuda()

    with torch.no_grad():
        logits = model(x_t).cpu().numpy()

    scaler = TemperatureScaler()
    scaler.fit(logits, y_val)

    if verbose:
        print(f"[TempScale] Optimal temperature: {scaler.temperature:.3f}")

    return scaler


# -- Conformal Prediction ---------------------------------------------------

class ConformalPredictor:
    """Split conformal prediction for classification.

    Provides prediction sets with guaranteed marginal coverage >= 1-alpha.
    """

    def __init__(self, alpha=CONFORMAL_ALPHA):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, proba, y_true):
        """Calibrate using held-out data.

        Computes the conformal quantile from nonconformity scores.
        """
        n = len(y_true)
        # Nonconformity score = 1 - P(true class)
        scores = 1.0 - proba[np.arange(n), y_true.astype(int)]
        # Adjusted quantile for finite-sample coverage
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.qhat = np.quantile(scores, q_level)
        return self

    def predict_sets(self, proba):
        """Return prediction sets for each sample.

        Returns list of lists, where each inner list contains the classes
        included in the prediction set.
        """
        if self.qhat is None:
            raise RuntimeError("Must call calibrate() first")

        prediction_sets = []
        for p in proba:
            # Include class c if P(c) >= 1 - qhat
            included = [c for c in range(len(p)) if p[c] >= 1.0 - self.qhat]
            if not included:
                # Always include at least the argmax
                included = [int(np.argmax(p))]
            prediction_sets.append(included)
        return prediction_sets

    def evaluate(self, proba, y_true, verbose=True):
        """Evaluate conformal prediction coverage and set sizes."""
        sets = self.predict_sets(proba)

        # Coverage: fraction of true labels contained in prediction set
        covered = sum(1 for s, y in zip(sets, y_true) if int(y) in s)
        coverage = covered / len(y_true)

        # Average set size
        avg_size = np.mean([len(s) for s in sets])

        # Singleton fraction (sets with exactly 1 class)
        singleton_frac = sum(1 for s in sets if len(s) == 1) / len(sets)

        if verbose:
            print(f"[Conformal] Alpha: {self.alpha:.2f} | qhat: {self.qhat:.4f}")
            print(f"[Conformal] Coverage: {coverage:.4f} (target: {1 - self.alpha:.2f})")
            print(f"[Conformal] Avg set size: {avg_size:.2f}")
            print(f"[Conformal] Singleton fraction: {singleton_frac:.4f}")

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_size),
            "singleton_fraction": float(singleton_frac),
            "qhat": float(self.qhat),
        }
