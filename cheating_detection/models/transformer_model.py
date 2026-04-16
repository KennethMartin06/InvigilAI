"""
transformer_model.py -- Transformer encoder for sequential cheating patterns.

Uses multi-head self-attention to learn dependencies across windows of a
session. Input shape: (batch, seq_len, n_features). Suitable when the
underlying data is a time-ordered set of window-level feature vectors.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF_DIM,
    MLP_LR,
    MLP_BATCH_SIZE,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class SequenceTransformer:
    """Transformer classifier for sequential feature windows."""

    def __init__(self, n_features, n_classes, seq_len=30,
                 dim=TRANSFORMER_DIM, heads=TRANSFORMER_HEADS,
                 layers=TRANSFORMER_LAYERS, dropout=TRANSFORMER_DROPOUT,
                 ff_dim=TRANSFORMER_FF_DIM, lr=MLP_LR):
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.ff_dim = ff_dim
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class PositionalEncoding(nn.Module):
            def __init__(self, d, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class Transformer(nn.Module):
            def __init__(self, in_dim, dim, heads, layers, dropout, ff, n_classes):
                super().__init__()
                self.input_proj = nn.Linear(in_dim, dim)
                self.pos = PositionalEncoding(dim)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=ff,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
                self.head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(dim // 2, n_classes),
                )

            def forward(self, x):
                x = self.input_proj(x)
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.pos(x)
                x = self.encoder(x)
                return self.head(x[:, 0])

        self.model = Transformer(self.n_features, self.dim, self.heads,
                                 self.layers, self.dropout, self.ff_dim, self.n_classes)

    def fit(self, X_seq, y, epochs=50, batch_size=MLP_BATCH_SIZE, verbose=True):
        if not _torch_ok():
            if verbose:
                print("[Transformer] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        X = torch.tensor(X_seq, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)
        n = len(X)

        for ep in range(epochs):
            perm = torch.randperm(n)
            losses = []
            self.model.train()
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X[idx])
                loss = ce(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if verbose and (ep + 1) % 10 == 0:
                print(f"[Transformer] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def predict(self, X_seq):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32, device=device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X_seq):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def reshape_to_sequences(X, seq_len):
    """Reshape flat (n, features) into (n//seq_len, seq_len, features)."""
    n, f = X.shape
    usable = (n // seq_len) * seq_len
    return X[:usable].reshape(-1, seq_len, f)
