"""
ood_detector.py -- Out-of-distribution detection using Mahalanobis distance.

Detects test samples that are significantly different from the training
distribution, flagging them as unreliable predictions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    OOD_PERCENTILE,
    CLASS_NAMES,
    OOD_PNG,
    OUTPUTS_DIR,
)


class MahalanobisOODDetector:
    """Detect OOD samples using Mahalanobis distance from class centroids."""

    def __init__(self, percentile=OOD_PERCENTILE):
        self.percentile = percentile
        self.class_means = {}
        self.cov_inv = None
        self.threshold = None

    def fit(self, X_train, y_train):
        """Fit detector on training data: compute class means and shared covariance."""
        classes = np.unique(y_train)

        # Per-class means
        for c in classes:
            self.class_means[int(c)] = X_train[y_train == c].mean(axis=0)

        # Shared covariance (pooled)
        cov = np.cov(X_train.T)
        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

        # Compute training distances to set threshold
        train_distances = self._compute_distances(X_train)
        self.threshold = np.percentile(train_distances, self.percentile)

        return self

    def _mahalanobis(self, x, mean):
        """Compute Mahalanobis distance for a single sample."""
        diff = x - mean
        return np.sqrt(diff @ self.cov_inv @ diff)

    def _compute_distances(self, X):
        """Compute minimum Mahalanobis distance to any class centroid."""
        distances = np.zeros(len(X))
        for i, x in enumerate(X):
            min_dist = min(
                self._mahalanobis(x, mean)
                for mean in self.class_means.values()
            )
            distances[i] = min_dist
        return distances

    def predict(self, X):
        """Return OOD flags (True = OOD) and distances."""
        distances = self._compute_distances(X)
        is_ood = distances > self.threshold
        return is_ood, distances

    def evaluate(self, X_test, y_test, y_pred, verbose=True):
        """Evaluate OOD detection and its correlation with misclassifications."""
        is_ood, distances = self.predict(X_test)
        is_correct = (y_pred == y_test)

        n_ood = is_ood.sum()
        n_total = len(X_test)
        ood_rate = n_ood / n_total

        # Accuracy on ID vs OOD samples
        id_mask = ~is_ood
        acc_id = is_correct[id_mask].mean() if id_mask.any() else 0
        acc_ood = is_correct[is_ood].mean() if is_ood.any() else 0

        if verbose:
            print(f"[OOD] Threshold (p{self.percentile}): {self.threshold:.4f}")
            print(f"[OOD] OOD samples: {n_ood}/{n_total} ({ood_rate:.2%})")
            print(f"[OOD] Accuracy on in-distribution: {acc_id:.4f}")
            print(f"[OOD] Accuracy on OOD samples: {acc_ood:.4f}")

        return {
            "ood_rate": float(ood_rate),
            "accuracy_id": float(acc_id),
            "accuracy_ood": float(acc_ood),
            "threshold": float(self.threshold),
        }


def plot_ood_analysis(X_test, y_test, y_pred, ood_detector,
                      save_path=OOD_PNG, verbose=True):
    """Plot OOD detection analysis."""
    is_ood, distances = ood_detector.predict(X_test)
    is_correct = (y_pred == y_test)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Distance distribution: correct vs incorrect
    axes[0].hist(distances[is_correct], bins=40, alpha=0.7, label="Correct", color="green", density=True)
    axes[0].hist(distances[~is_correct], bins=40, alpha=0.7, label="Incorrect", color="red", density=True)
    axes[0].axvline(ood_detector.threshold, color="black", linestyle="--", label=f"OOD threshold")
    axes[0].set_xlabel("Mahalanobis Distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distance Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Distance by class
    class_dists = []
    for c in range(len(CLASS_NAMES)):
        mask = y_test == c
        if mask.any():
            class_dists.append(distances[mask])
        else:
            class_dists.append(np.array([0]))
    axes[1].boxplot(class_dists, labels=CLASS_NAMES)
    axes[1].set_ylabel("Mahalanobis Distance")
    axes[1].set_title("Distance by True Class")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Accuracy vs rejection rate
    percentiles = np.linspace(0, 100, 50)
    accs = []
    coverages = []
    for p in percentiles:
        thresh = np.percentile(distances, p)
        mask = distances <= thresh
        if mask.sum() > 0:
            accs.append((y_pred[mask] == y_test[mask]).mean())
            coverages.append(mask.mean())
        else:
            accs.append(0)
            coverages.append(0)

    axes[2].plot(coverages, accs, "b-o", markersize=3)
    axes[2].set_xlabel("Coverage (fraction kept)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy vs Coverage (OOD Rejection)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[Plot] OOD analysis -> {save_path}")
