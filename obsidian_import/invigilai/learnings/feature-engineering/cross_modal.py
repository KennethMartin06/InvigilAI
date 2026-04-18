"""
cross_modal.py -- Cross-modal attention between gaze and keystroke streams.

Uses scaled dot-product attention to learn correlations between visual
(gaze) and behavioral (keystroke) feature subsets. Produces a fused
feature vector that captures cross-modal dependencies.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    N_VISUAL_FEATURES,
    N_BEHAVIORAL_FEATURES,
    MLP_LR,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def gaze_keystroke_correlation(gaze_features, keystroke_features):
    """Compute cross-correlation matrix (NumPy, no training)."""
    if gaze_features.ndim == 1:
        gaze_features = gaze_features[None, :]
        keystroke_features = keystroke_features[None, :]
    # Per-feature correlation
    g_norm = (gaze_features - gaze_features.mean(axis=0)) / (gaze_features.std(axis=0) + 1e-6)
    k_norm = (keystroke_features - keystroke_features.mean(axis=0)) / (keystroke_features.std(axis=0) + 1e-6)
    cross_corr = (g_norm.T @ k_norm) / len(gaze_features)
    return cross_corr


class CrossModalAttention:
    """Small attention module producing fused (gaze | keystroke) embeddings."""

    def __init__(self, visual_dim=N_VISUAL_FEATURES, behavioral_dim=N_BEHAVIORAL_FEATURES,
                 hidden=64, heads=4, lr=MLP_LR):
        self.visual_dim = visual_dim
        self.behavioral_dim = behavioral_dim
        self.hidden = hidden
        self.heads = heads
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class CMA(nn.Module):
            def __init__(self, vd, bd, hidden, heads):
                super().__init__()
                self.v_proj = nn.Linear(vd, hidden)
                self.b_proj = nn.Linear(bd, hidden)
                self.attn_vb = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.attn_bv = nn.MultiheadAttention(hidden, heads, batch_first=True)
                self.fuse = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )

            def forward(self, v, b):
                # Treat each sample as a seq of length 1
                v = self.v_proj(v).unsqueeze(1)
                b = self.b_proj(b).unsqueeze(1)
                v_att, _ = self.attn_vb(v, b, b)
                b_att, _ = self.attn_bv(b, v, v)
                fused = torch.cat([v_att.squeeze(1), b_att.squeeze(1)], dim=1)
                return self.fuse(fused)

        self.model = CMA(self.visual_dim, self.behavioral_dim, self.hidden, self.heads)

    def fit_encode(self, X, epochs=20, lr=None, verbose=True):
        """Unsupervised fit via reconstruction of visual+behavioral parts."""
        if not _torch_ok():
            if verbose:
                print("[CMA] PyTorch unavailable")
            return self.encode_numpy(X)

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        if self.model is None:
            self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        decoder = nn.Linear(self.hidden, self.visual_dim + self.behavioral_dim).to(device)

        opt = AdamW(list(self.model.parameters()) + list(decoder.parameters()),
                    lr=lr or self.lr)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        v = X_t[:, :self.visual_dim]
        b = X_t[:, self.visual_dim:self.visual_dim + self.behavioral_dim]

        for ep in range(epochs):
            fused = self.model(v, b)
            recon = decoder(fused)
            loss = ((recon - X_t[:, :self.visual_dim + self.behavioral_dim]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and (ep + 1) % 5 == 0:
                print(f"[CMA] ep {ep+1}/{epochs} recon={loss.item():.4f}")

        with torch.no_grad():
            fused = self.model(v, b).cpu().numpy()
        return fused

    def encode_numpy(self, X):
        v = X[:, :self.visual_dim]
        b = X[:, self.visual_dim:self.visual_dim + self.behavioral_dim]
        # Fallback: simple pooled cross-correlation features
        cross = gaze_keystroke_correlation(v, b)
        # broadcast correlation diagonal summaries
        per_sample = np.concatenate([v, b, v * b[:, :v.shape[1]]
                                     if b.shape[1] >= v.shape[1] else v], axis=1)
        return per_sample


# -- LSTM Temporal Features -------------------------------------------------

from cheating_detection.config import (
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_BIDIRECTIONAL,
)


class LSTMFeatureExtractor:
    """BiLSTM encoder producing temporal embeddings for sequential data."""

    def __init__(self, n_features, hidden=LSTM_HIDDEN_DIM, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, bidirectional=LSTM_BIDIRECTIONAL,
                 lr=MLP_LR):
        self.n_features = n_features
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, in_dim, hidden, layers, dropout, bidirectional, n_classes):
                super().__init__()
                self.lstm = nn.LSTM(
                    in_dim, hidden, layers,
                    dropout=dropout if layers > 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
                out_dim = hidden * (2 if bidirectional else 1)
                self.head = nn.Linear(out_dim, n_classes)

            def forward(self, x, return_embedding=False):
                out, (h, _) = self.lstm(x)
                pooled = out.mean(dim=1)
                if return_embedding:
                    return pooled
                return self.head(pooled)

        return LSTMNet

    def fit(self, X_seq, y, n_classes, epochs=30, batch_size=32, verbose=True):
        if not _torch_ok():
            if verbose:
                print("[LSTM] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        LSTMNet = self._build()
        self.model = LSTMNet(self.n_features, self.hidden, self.layers,
                             self.dropout, self.bidirectional, n_classes)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        opt = AdamW(self.model.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_seq, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        for ep in range(epochs):
            perm = torch.randperm(len(X_t))
            losses = []
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X_t[idx])
                loss = ce(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if verbose and (ep + 1) % 10 == 0:
                print(f"[LSTM] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def extract_embeddings(self, X_seq):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            emb = self.model(torch.tensor(X_seq, dtype=torch.float32, device=device),
                             return_embedding=True)
        return emb.cpu().numpy()

    def predict(self, X_seq):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32, device=device))
        return logits.argmax(dim=1).cpu().numpy()
