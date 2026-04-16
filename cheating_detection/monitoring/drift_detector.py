"""
drift_detector.py -- Concept drift detection for production monitoring.

Uses Population Stability Index (PSI) and KL divergence to detect when
the input distribution has shifted from training, indicating model
retraining may be needed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    DRIFT_PSI_THRESHOLD,
    ALL_FEATURE_NAMES,
    DRIFT_PNG,
    OUTPUTS_DIR,
)


def compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.2: moderate shift
    PSI > 0.2: significant shift -- consider retraining
    """
    eps = 1e-8

    # Bin the expected distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_kl_divergence(p, q, bins=50):
    """Compute KL divergence D(P || Q) between two distributions."""
    eps = 1e-8
    all_data = np.concatenate([p, q])
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    p_hist = np.histogram(p, bins=bin_edges, density=True)[0] + eps
    q_hist = np.histogram(q, bins=bin_edges, density=True)[0] + eps

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


class DriftDetector:
    """Monitor for concept drift using PSI and KL divergence."""

    def __init__(self, psi_threshold=DRIFT_PSI_THRESHOLD):
        self.psi_threshold = psi_threshold
        self.reference_data = None

    def fit(self, X_train):
        """Store training data as reference distribution."""
        self.reference_data = X_train.copy()
        return self

    def check_drift(self, X_new, verbose=True):
        """Check for drift between reference and new data.

        Returns per-feature PSI values and overall drift assessment.
        """
        n_features = min(X_new.shape[1], self.reference_data.shape[1])
        feature_names = ALL_FEATURE_NAMES[:n_features]

        psi_values = {}
        kl_values = {}
        drifted_features = []

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
            psi = compute_psi(self.reference_data[:, i], X_new[:, i])
            kl = compute_kl_divergence(self.reference_data[:, i], X_new[:, i])
            psi_values[name] = psi
            kl_values[name] = kl

            if psi > self.psi_threshold:
                drifted_features.append(name)

        overall_drift = len(drifted_features) > 0

        if verbose:
            print(f"[Drift] PSI threshold: {self.psi_threshold}")
            print(f"[Drift] Features with drift: {len(drifted_features)}/{n_features}")
            if drifted_features:
                print(f"[Drift] Drifted features: {drifted_features}")
            for name in sorted(psi_values, key=psi_values.get, reverse=True)[:5]:
                status = "DRIFT" if psi_values[name] > self.psi_threshold else "OK"
                print(f"  {name:<30} PSI={psi_values[name]:.4f} KL={kl_values[name]:.4f} [{status}]")

        return {
            "overall_drift": overall_drift,
            "psi_values": psi_values,
            "kl_values": kl_values,
            "drifted_features": drifted_features,
        }


def plot_drift_analysis(drift_results, save_path=DRIFT_PNG, verbose=True):
    """Plot drift analysis results."""
    psi_values = drift_results["psi_values"]
    threshold = DRIFT_PSI_THRESHOLD

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    names = list(psi_values.keys())
    values = [psi_values[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["red" if v > threshold else "steelblue" for v in values]
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift Analysis (PSI)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"[Plot] Drift analysis -> {save_path}")
