"""
ensemble_advanced.py -- Advanced ensembling: weighted voting, cascade, multi-task.

Provides:
  - WeightedVotingEnsemble: soft voting with per-model confidence weights
  - CascadeClassifier: reject low-confidence samples to a heavier secondary model
  - MultiTaskLearner: single backbone with multiple classification heads
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    CASCADE_CONFIDENCE_THRESHOLD,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    MLP_LR,
    MLP_BATCH_SIZE,
    MTL_TASKS,
)


class WeightedVotingEnsemble:
    """Soft-voting ensemble weighted by per-model validation accuracy/confidence."""

    def __init__(self, models, weights=None, confidence_weighted=True):
        self.models = models
        self.weights = weights
        self.confidence_weighted = confidence_weighted

    def fit(self, X_val, y_val, verbose=True):
        """Compute weights from per-model validation accuracy."""
        if self.weights is not None:
            return self
        accs = []
        for name, m in self.models.items():
            preds = m.predict(X_val)
            accs.append((preds == y_val).mean())
        accs = np.array(accs)
        # Softmax over accuracies
        exps = np.exp((accs - accs.max()) * 5)   # temperature
        self.weights = dict(zip(self.models.keys(), exps / exps.sum()))
        if verbose:
            print(f"[WVoting] weights: " + ", ".join(f"{k}={v:.3f}" for k, v in self.weights.items()))
        return self

    def predict_proba(self, X):
        probs = None
        for name, m in self.models.items():
            p = m.predict_proba(X)
            w = self.weights.get(name, 1.0 / len(self.models))
            if self.confidence_weighted:
                # Also scale by per-sample confidence
                per_sample_conf = p.max(axis=1, keepdims=True)
                p = p * per_sample_conf * w
            else:
                p = p * w
            probs = p if probs is None else probs + p
        # Re-normalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class CascadeClassifier:
    """Reject low-confidence samples from stage-1 to a stage-2 heavier model."""

    def __init__(self, primary, secondary, threshold=CASCADE_CONFIDENCE_THRESHOLD):
        self.primary = primary
        self.secondary = secondary
        self.threshold = threshold

    def predict(self, X):
        p1 = self.primary.predict_proba(X)
        conf = p1.max(axis=1)
        preds = p1.argmax(axis=1)

        low_conf = conf < self.threshold
        if low_conf.any() and self.secondary is not None:
            preds[low_conf] = self.secondary.predict(X[low_conf])
        return preds

    def predict_proba(self, X):
        p1 = self.primary.predict_proba(X)
        conf = p1.max(axis=1)
        low_conf = conf < self.threshold
        if low_conf.any() and self.secondary is not None:
            p2 = self.secondary.predict_proba(X[low_conf])
            p1[low_conf] = p2
        return p1

    def evaluate(self, X, y, verbose=True):
        p1 = self.primary.predict_proba(X)
        conf = p1.max(axis=1)
        low_conf = conf < self.threshold
        preds = self.predict(X)
        acc = (preds == y).mean()
        routed = low_conf.mean()
        primary_only_acc = (p1.argmax(axis=1)[~low_conf] == y[~low_conf]).mean() if (~low_conf).any() else 0
        if verbose:
            print(f"[Cascade] routed to secondary: {routed:.2%}")
            print(f"[Cascade] accuracy: {acc:.4f} (primary-only on high-conf: {primary_only_acc:.4f})")
        return {"accuracy": float(acc), "routed_fraction": float(routed),
                "primary_high_conf_acc": float(primary_only_acc)}


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class MultiTaskLearner:
    """MLP backbone + multiple classification heads for related tasks."""

    def __init__(self, n_features, task_n_classes,
                 hidden=MLP_HIDDEN_LAYERS,
                 dropout=MLP_DROPOUT,
                 lr=MLP_LR):
        self.n_features = n_features
        self.task_n_classes = task_n_classes   # dict: task_name -> n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class MTLNet(nn.Module):
            def __init__(self, in_dim, hidden, dropout, task_n_classes):
                super().__init__()
                layers = []
                prev = in_dim
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                    prev = h
                self.backbone = nn.Sequential(*layers)
                self.heads = nn.ModuleDict({
                    task: nn.Linear(prev, n) for task, n in task_n_classes.items()
                })

            def forward(self, x):
                h = self.backbone(x)
                return {task: head(h) for task, head in self.heads.items()}

        self.model = MTLNet(self.n_features, self.hidden, self.dropout, self.task_n_classes)

    def fit(self, X, y_dict, epochs=60, batch_size=MLP_BATCH_SIZE, verbose=True):
        """y_dict: {task_name -> labels array}."""
        if not _torch_ok():
            if verbose:
                print("[MTL] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensors = {t: torch.tensor(y_dict[t], dtype=torch.long, device=device) for t in y_dict}

        for ep in range(epochs):
            perm = torch.randperm(len(X_t))
            losses = []
            self.model.train()
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                logits_dict = self.model(X_t[idx])
                loss = sum(ce(logits_dict[t], y_tensors[t][idx]) for t in y_dict)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if verbose and (ep + 1) % 10 == 0:
                print(f"[MTL] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def predict(self, X, task):
        import torch
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)[task]
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X, task):
        import torch
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)[task]
        return torch.softmax(logits, dim=1).cpu().numpy()


def derive_mtl_labels(y_cheating, X):
    """Derive auxiliary labels for multi-task learning.

    - cheating_type:  primary label (0..K)
    - severity:       quantized into {low, mid, high} by class priors
    - confidence:     binary high/low feature variance proxy
    """
    y_sev = np.where(y_cheating == 0, 0, np.where(y_cheating <= 2, 1, 2))
    variance = X.var(axis=1)
    y_conf = (variance > np.median(variance)).astype(int)
    return {"cheating_type": y_cheating, "severity": y_sev, "confidence": y_conf}
