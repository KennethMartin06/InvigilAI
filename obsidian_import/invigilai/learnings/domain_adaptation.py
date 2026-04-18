"""
domain_adaptation.py -- Transfer learning and domain-adaptive training.

Includes:
  - Domain-Adversarial Neural Network (DANN) with gradient reversal
  - Layer-wise fine-tuning from a pretrained source model
  - Deep CORAL (correlation alignment) domain adaptation
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    DA_LAMBDA,
    DA_PRETRAIN_EPOCHS,
    DA_FINETUNE_EPOCHS,
    TRANSFER_FREEZE_LAYERS,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    MLP_LR,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class GradientReversalFunction:
    """Autograd function that reverses the gradient during backprop."""

    @staticmethod
    def apply_fn():
        import torch
        from torch.autograd import Function

        class _GRL(Function):
            @staticmethod
            def forward(ctx, x, lambda_):
                ctx.lambda_ = lambda_
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.lambda_, None

        return _GRL.apply


class DANN:
    """Domain-Adversarial Neural Network for unsupervised domain adaptation."""

    def __init__(self, n_features, n_classes, hidden_layers=MLP_HIDDEN_LAYERS,
                 lambda_=DA_LAMBDA, lr=MLP_LR):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.lambda_ = lambda_
        self.lr = lr
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        feat_dim = self.hidden_layers[-1]

        class FeatureExtractor(nn.Module):
            def __init__(self, in_dim, hidden):
                super().__init__()
                layers = []
                prev = in_dim
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(MLP_DROPOUT)]
                    prev = h
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        class ClassHead(nn.Module):
            def __init__(self, in_dim, n_classes):
                super().__init__()
                self.fc = nn.Linear(in_dim, n_classes)

            def forward(self, h):
                return self.fc(h)

        class DomainHead(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 2))

            def forward(self, h):
                return self.fc(h)

        self.feat = FeatureExtractor(self.n_features, self.hidden_layers)
        self.cls = ClassHead(feat_dim, self.n_classes)
        self.dom = DomainHead(feat_dim)
        self.feat_dim = feat_dim

    def fit(self, X_source, y_source, X_target, epochs=DA_PRETRAIN_EPOCHS,
            batch_size=64, verbose=True):
        if not _torch_ok():
            if verbose:
                print("[DANN] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import Adam

        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for m in [self.feat, self.cls, self.dom]:
            m.to(device)

        grl = GradientReversalFunction.apply_fn()

        params = list(self.feat.parameters()) + list(self.cls.parameters()) + list(self.dom.parameters())
        opt = Adam(params, lr=self.lr)
        ce = nn.CrossEntropyLoss()

        Xs = torch.tensor(X_source, dtype=torch.float32, device=device)
        ys = torch.tensor(y_source, dtype=torch.long, device=device)
        Xt = torch.tensor(X_target, dtype=torch.float32, device=device)

        for ep in range(epochs):
            perm_s = torch.randperm(len(Xs))
            perm_t = torch.randperm(len(Xt))

            losses = []
            for start in range(0, len(Xs), batch_size):
                xs = Xs[perm_s[start:start + batch_size]]
                ysb = ys[perm_s[start:start + batch_size]]
                t_idx = perm_t[start % len(Xt):(start % len(Xt)) + batch_size]
                if len(t_idx) == 0:
                    continue
                xt = Xt[t_idx]

                p = float(start + ep * len(Xs)) / (epochs * max(len(Xs), 1))
                lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                lam *= self.lambda_

                h_s = self.feat(xs)
                h_t = self.feat(xt)
                cls_logits = self.cls(h_s)
                cls_loss = ce(cls_logits, ysb)

                dom_s_logits = self.dom(grl(h_s, lam))
                dom_t_logits = self.dom(grl(h_t, lam))
                dom_labels_s = torch.zeros(len(h_s), dtype=torch.long, device=device)
                dom_labels_t = torch.ones(len(h_t), dtype=torch.long, device=device)
                dom_loss = ce(dom_s_logits, dom_labels_s) + ce(dom_t_logits, dom_labels_t)

                loss = cls_loss + dom_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            if verbose and (ep + 1) % 10 == 0:
                print(f"[DANN] ep {ep+1}/{epochs} loss={np.mean(losses):.4f} lam={lam:.3f}")
        return self

    def predict(self, X):
        import torch
        device = next(self.feat.parameters()).device
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        self.feat.eval(); self.cls.eval()
        with torch.no_grad():
            logits = self.cls(self.feat(Xt))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        device = next(self.feat.parameters()).device
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        self.feat.eval(); self.cls.eval()
        with torch.no_grad():
            logits = self.cls(self.feat(Xt))
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def deep_coral_loss(source_features, target_features):
    """Compute CORAL loss between source and target feature covariances."""
    import torch
    d = source_features.size(1)
    ns, nt = source_features.size(0), target_features.size(0)

    ms = source_features.mean(dim=0, keepdim=True)
    mt = target_features.mean(dim=0, keepdim=True)
    cs = ((source_features - ms).t() @ (source_features - ms)) / (ns - 1 + 1e-6)
    ct = ((target_features - mt).t() @ (target_features - mt)) / (nt - 1 + 1e-6)

    return ((cs - ct) ** 2).sum() / (4 * d * d)


def transfer_finetune(pretrained_model, X_new, y_new, freeze_layers=TRANSFER_FREEZE_LAYERS,
                      epochs=DA_FINETUNE_EPOCHS, lr=MLP_LR * 0.1, verbose=True):
    """Fine-tune a pretrained MLP on institution-specific data.

    Freezes early layers, trains later layers with low LR.
    """
    if not _torch_ok():
        if verbose:
            print("[Transfer] PyTorch unavailable")
        return pretrained_model

    import torch
    import torch.nn as nn
    from torch.optim import Adam

    model = pretrained_model
    # Freeze first N linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for layer in linear_layers[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = Adam(trainable, lr=lr)
    ce = nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    Xt = torch.tensor(X_new, dtype=torch.float32, device=device)
    yt = torch.tensor(y_new, dtype=torch.long, device=device)

    model.train()
    for ep in range(epochs):
        logits = model(Xt)
        loss = ce(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"[Transfer] ep {ep+1}/{epochs} loss={loss.item():.4f}")
    return model
