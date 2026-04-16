"""
vit_model.py -- Vision Transformer for video-frame cheating detection.

Expects batches of single frames or short clips reshaped as images.
Patches are flattened + linearly projected, then processed by a
Transformer encoder with a learnable CLS token.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    VIT_IMAGE_SIZE,
    VIT_PATCH_SIZE,
    VIT_DIM,
    VIT_DEPTH,
    VIT_HEADS,
    VIT_MLP_DIM,
    VIT_CHANNELS,
    MLP_LR,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class ViT:
    """Minimal Vision Transformer for per-frame classification."""

    def __init__(self, n_classes,
                 image_size=VIT_IMAGE_SIZE,
                 patch_size=VIT_PATCH_SIZE,
                 dim=VIT_DIM,
                 depth=VIT_DEPTH,
                 heads=VIT_HEADS,
                 mlp_dim=VIT_MLP_DIM,
                 channels=VIT_CHANNELS,
                 lr=MLP_LR):
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.n_classes = n_classes
        self.lr = lr
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class PatchEmbedding(nn.Module):
            def __init__(self, channels, patch_size, dim):
                super().__init__()
                self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

            def forward(self, x):
                x = self.proj(x)       # (B, dim, H/p, W/p)
                x = x.flatten(2).transpose(1, 2)
                return x

        class ViTNet(nn.Module):
            def __init__(self, channels, patch_size, n_patches, dim, depth,
                         heads, mlp_dim, n_classes):
                super().__init__()
                self.patch_embed = PatchEmbedding(channels, patch_size, dim)
                self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
                self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
                    dropout=0.1, batch_first=True, activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
                self.head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, n_classes),
                )

            def forward(self, x):
                x = self.patch_embed(x)
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = x + self.pos_embed
                x = self.transformer(x)
                return self.head(x[:, 0])

        self.model = ViTNet(self.channels, self.patch_size, self.n_patches,
                            self.dim, self.depth, self.heads,
                            self.mlp_dim, self.n_classes)

    def fit(self, images, y, epochs=30, batch_size=32, verbose=True):
        """Fit on batch of images with shape (N, C, H, W)."""
        if not _torch_ok():
            if verbose:
                print("[ViT] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        X = torch.tensor(images, dtype=torch.float32, device=device)
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
            if verbose and (ep + 1) % 5 == 0:
                print(f"[ViT] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def predict(self, images):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(images, dtype=torch.float32, device=device))
        return logits.argmax(dim=1).cpu().numpy()


def feature_vector_to_image(X, image_size=VIT_IMAGE_SIZE, channels=VIT_CHANNELS):
    """Reshape tabular features into pseudo-images for ViT input.

    Tiles the feature vector into a (channels, image_size, image_size) grid
    so ViT can be tested on tabular data without raw frames.
    """
    n, f = X.shape
    target = channels * image_size * image_size
    out = np.zeros((n, channels, image_size, image_size), dtype=np.float32)
    for i in range(n):
        vec = np.tile(X[i], target // f + 1)[:target]
        out[i] = vec.reshape(channels, image_size, image_size)
    return out
