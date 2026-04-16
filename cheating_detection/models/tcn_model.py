"""
tcn_model.py -- Temporal Convolutional Network for keystroke sequences.

Uses dilated causal 1D convolutions with residual connections, following
Bai et al. (2018). Good for long sequences where RNNs struggle.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    TCN_SEQ_LENGTH,
    MLP_LR,
    MLP_BATCH_SIZE,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TCNModel:
    """Temporal convolutional network for keystroke-like sequences."""

    def __init__(self, n_features, n_classes,
                 channels=TCN_CHANNELS,
                 kernel=TCN_KERNEL_SIZE,
                 dropout=TCN_DROPOUT,
                 lr=MLP_LR):
        self.n_features = n_features
        self.n_classes = n_classes
        self.channels = channels
        self.kernel = kernel
        self.dropout = dropout
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class Chomp1d(nn.Module):
            def __init__(self, chomp):
                super().__init__()
                self.chomp = chomp

            def forward(self, x):
                return x[:, :, :-self.chomp].contiguous()

        class TempBlock(nn.Module):
            def __init__(self, in_c, out_c, k, dilation, dropout):
                super().__init__()
                pad = (k - 1) * dilation
                self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation))
                self.chomp1 = Chomp1d(pad)
                self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_c, out_c, k, padding=pad, dilation=dilation))
                self.chomp2 = Chomp1d(pad)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(dropout)
                self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None

            def forward(self, x):
                h = self.drop(self.relu(self.chomp1(self.conv1(x))))
                h = self.drop(self.relu(self.chomp2(self.conv2(h))))
                res = x if self.down is None else self.down(x)
                return self.relu(h + res)

        class TCN(nn.Module):
            def __init__(self, in_dim, channels, k, dropout, n_classes):
                super().__init__()
                blocks = []
                prev = in_dim
                for i, c in enumerate(channels):
                    blocks.append(TempBlock(prev, c, k, dilation=2 ** i, dropout=dropout))
                    prev = c
                self.blocks = nn.Sequential(*blocks)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(channels[-1], n_classes),
                )

            def forward(self, x):
                # x: (B, seq_len, features) -> (B, features, seq_len)
                x = x.transpose(1, 2)
                x = self.blocks(x)
                return self.head(x)

        self.model = TCN(self.n_features, self.channels, self.kernel,
                         self.dropout, self.n_classes)

    def fit(self, X_seq, y, epochs=40, batch_size=MLP_BATCH_SIZE, verbose=True):
        if not _torch_ok():
            if verbose:
                print("[TCN] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        X = torch.tensor(X_seq, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        for ep in range(epochs):
            perm = torch.randperm(len(X))
            losses = []
            self.model.train()
            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X[idx])
                loss = ce(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if verbose and (ep + 1) % 10 == 0:
                print(f"[TCN] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
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
