"""
degradation_tracker.py -- Track model performance degradation over time.

Maintains a rolling window of prediction accuracy / confidence statistics
and raises alerts when performance drops below baseline by a configurable
threshold. Persists history to JSON for longitudinal analysis.
"""

import json
import numpy as np
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    DEGRADATION_WINDOW_SIZE,
    DEGRADATION_ALERT_THRESHOLD,
    DEGRADATION_PNG,
    OUTPUTS_DIR,
)


class ModelDegradationTracker:
    """Rolling-window performance monitor for deployed models."""

    def __init__(self, window_size=DEGRADATION_WINDOW_SIZE,
                 alert_threshold=DEGRADATION_ALERT_THRESHOLD,
                 history_path=None):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.baseline_acc = None
        self.baseline_conf = None
        self.window_acc = deque(maxlen=window_size)
        self.window_conf = deque(maxlen=window_size)
        self.alerts = []
        self.history = []
        self.history_path = history_path or os.path.join(OUTPUTS_DIR, "degradation_history.json")

    def set_baseline(self, acc, mean_confidence):
        self.baseline_acc = float(acc)
        self.baseline_conf = float(mean_confidence)

    def record_batch(self, y_true, y_pred, y_proba=None):
        """Record a batch of predictions."""
        acc = float((y_pred == y_true).mean())
        mean_conf = float(y_proba.max(axis=1).mean()) if y_proba is not None else 1.0
        self.window_acc.append(acc)
        self.window_conf.append(mean_conf)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": int(len(y_true)),
            "accuracy": acc,
            "mean_confidence": mean_conf,
            "rolling_accuracy": float(np.mean(self.window_acc)),
            "rolling_confidence": float(np.mean(self.window_conf)),
        }
        self.history.append(entry)

        alert = self._check_alert(entry["rolling_accuracy"])
        if alert:
            self.alerts.append(alert)
        return entry

    def _check_alert(self, rolling_acc):
        if self.baseline_acc is None:
            return None
        drop = self.baseline_acc - rolling_acc
        if drop >= self.alert_threshold:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "accuracy_degradation",
                "baseline_accuracy": self.baseline_acc,
                "rolling_accuracy": float(rolling_acc),
                "drop": float(drop),
                "severity": "high" if drop > 2 * self.alert_threshold else "medium",
            }
            return alert
        return None

    def current_status(self):
        if not self.window_acc:
            return {"status": "no_data"}
        rolling_acc = float(np.mean(self.window_acc))
        rolling_conf = float(np.mean(self.window_conf))
        drop = (self.baseline_acc - rolling_acc) if self.baseline_acc is not None else 0.0
        return {
            "status": "degraded" if drop >= self.alert_threshold else "healthy",
            "rolling_accuracy": rolling_acc,
            "rolling_confidence": rolling_conf,
            "baseline_accuracy": self.baseline_acc,
            "accuracy_drop": float(drop),
            "n_alerts": len(self.alerts),
            "window_size": len(self.window_acc),
        }

    def save_history(self, path=None):
        path = path or self.history_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "baseline_accuracy": self.baseline_acc,
                "baseline_confidence": self.baseline_conf,
                "history": self.history,
                "alerts": self.alerts,
            }, f, indent=2)

    def load_history(self, path=None):
        path = path or self.history_path
        if not os.path.isfile(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.baseline_acc = data.get("baseline_accuracy")
        self.baseline_conf = data.get("baseline_confidence")
        self.history = data.get("history", [])
        self.alerts = data.get("alerts", [])
        for entry in self.history[-self.window_size:]:
            self.window_acc.append(entry["accuracy"])
            self.window_conf.append(entry["mean_confidence"])


def plot_degradation(tracker, save_path=DEGRADATION_PNG, verbose=True):
    """Plot accuracy / confidence over time with alert markers."""
    if not tracker.history:
        if verbose:
            print("[Degradation] No history to plot")
        return
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    accs = [h["accuracy"] for h in tracker.history]
    roll = [h["rolling_accuracy"] for h in tracker.history]
    confs = [h["mean_confidence"] for h in tracker.history]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(accs))

    axes[0].plot(x, accs, alpha=0.4, label="Batch acc")
    axes[0].plot(x, roll, linewidth=2, label="Rolling acc")
    if tracker.baseline_acc:
        axes[0].axhline(tracker.baseline_acc, color="green", linestyle="--", label="Baseline")
        axes[0].axhline(tracker.baseline_acc - tracker.alert_threshold,
                        color="red", linestyle=":", label="Alert threshold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Model Accuracy Over Time")

    axes[1].plot(x, confs, color="purple", label="Mean confidence")
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Batch index")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[Plot] Degradation -> {save_path}")
