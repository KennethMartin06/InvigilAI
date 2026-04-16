"""
gan_synthetic.py -- Conditional GAN for generating realistic tabular cheating samples.

A lightweight conditional Wasserstein-GAN with gradient penalty (WGAN-GP)
for synthesizing features conditioned on the target class. Useful for
augmenting rare cheating classes with diverse samples.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    GAN_LATENT_DIM,
    GAN_HIDDEN_DIM,
    GAN_EPOCHS,
    GAN_BATCH_SIZE,
    GAN_LR,
    GAN_SAMPLES_PER_CLASS,
)


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class ConditionalGAN:
    """Conditional GAN for tabular feature synthesis (WGAN-GP)."""

    def __init__(self, n_features, n_classes,
                 latent_dim=GAN_LATENT_DIM,
                 hidden_dim=GAN_HIDDEN_DIM,
                 lr=GAN_LR):
        self.n_features = n_features
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.generator = None
        self.critic = None

    def _build(self):
        import torch
        import torch.nn as nn

        class Generator(nn.Module):
            def __init__(self, latent_dim, n_classes, hidden, out_dim):
                super().__init__()
                self.embed = nn.Embedding(n_classes, n_classes)
                self.net = nn.Sequential(
                    nn.Linear(latent_dim + n_classes, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, out_dim),
                )

            def forward(self, z, y):
                c = self.embed(y)
                return self.net(torch.cat([z, c], dim=1))

        class Critic(nn.Module):
            def __init__(self, in_dim, n_classes, hidden):
                super().__init__()
                self.embed = nn.Embedding(n_classes, n_classes)
                self.net = nn.Sequential(
                    nn.Linear(in_dim + n_classes, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x, y):
                c = self.embed(y)
                return self.net(torch.cat([x, c], dim=1))

        self.generator = Generator(self.latent_dim, self.n_classes, self.hidden_dim, self.n_features)
        self.critic = Critic(self.n_features, self.n_classes, self.hidden_dim)

    def _gradient_penalty(self, real, fake, y):
        import torch
        alpha = torch.rand(real.size(0), 1, device=real.device)
        interp = alpha * real + (1 - alpha) * fake
        interp.requires_grad_(True)
        d_interp = self.critic(interp, y)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X, y, epochs=GAN_EPOCHS, batch_size=GAN_BATCH_SIZE, verbose=True):
        if not _torch_available():
            if verbose:
                print("[GAN] PyTorch not available, skipping")
            return self

        import torch
        from torch.optim import Adam

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator.to(device)
        self.critic.to(device)

        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        opt_c = Adam(self.critic.parameters(), lr=self.lr, betas=(0.5, 0.9))

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)
        n = len(X_t)

        for ep in range(epochs):
            perm = torch.randperm(n)
            d_losses, g_losses = [], []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                real = X_t[idx]
                y_b = y_t[idx]
                bs = real.size(0)

                # Train critic
                for _ in range(3):
                    z = torch.randn(bs, self.latent_dim, device=device)
                    fake = self.generator(z, y_b).detach()
                    d_real = self.critic(real, y_b).mean()
                    d_fake = self.critic(fake, y_b).mean()
                    gp = self._gradient_penalty(real, fake, y_b)
                    d_loss = d_fake - d_real + 10 * gp
                    opt_c.zero_grad()
                    d_loss.backward()
                    opt_c.step()
                    d_losses.append(d_loss.item())

                # Train generator
                z = torch.randn(bs, self.latent_dim, device=device)
                fake = self.generator(z, y_b)
                g_loss = -self.critic(fake, y_b).mean()
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_losses.append(g_loss.item())

            if verbose and (ep + 1) % 50 == 0:
                print(f"[GAN] epoch {ep+1}/{epochs} D={np.mean(d_losses):.3f} G={np.mean(g_losses):.3f}")
        return self

    def generate(self, n_samples, class_label):
        import torch
        if self.generator is None:
            raise RuntimeError("Call fit() before generate()")
        device = next(self.generator.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
        self.generator.eval()
        with torch.no_grad():
            synthetic = self.generator(z, y).cpu().numpy()
        return synthetic


def augment_with_gan(X_train, y_train, samples_per_class=GAN_SAMPLES_PER_CLASS,
                     verbose=True):
    """Train a conditional GAN and generate synthetic samples for all classes."""
    n_classes = int(y_train.max() + 1)
    gan = ConditionalGAN(X_train.shape[1], n_classes)
    gan.fit(X_train, y_train, verbose=verbose)

    X_aug = [X_train]
    y_aug = [y_train]
    for c in range(n_classes):
        X_syn = gan.generate(samples_per_class, c)
        if X_syn is None or len(X_syn) == 0:
            continue
        X_aug.append(X_syn)
        y_aug.append(np.full(samples_per_class, c))

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)

    if verbose:
        print(f"[GAN] {len(y_train)} -> {len(y_out)} samples ({samples_per_class}/class synthesized)")
    return X_out, y_out
