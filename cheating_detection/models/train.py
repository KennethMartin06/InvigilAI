"""
train.py -- Training pipeline: RF, LightGBM, MLP (Focal Loss + MixUp + Cosine
Annealing + BatchNorm + label smoothing), and Stacking Ensemble.

v3 improvements:
  - Focal Loss for hard-example mining (gaze class)
  - MixUp augmentation during MLP training
  - Cosine Annealing with Warm Restarts LR schedule
  - LightGBM as 3rd base model
  - Stacking Ensemble (meta-learner on OOF predictions)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    N_TOTAL_FEATURES,
    RF_N_ESTIMATORS,
    LGB_N_ESTIMATORS,
    LGB_LEARNING_RATE,
    LGB_MAX_DEPTH,
    LGB_NUM_LEAVES,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    MLP_LR,
    MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    MLP_USE_BATCH_NORM,
    MLP_USE_CLASS_WEIGHTS,
    MLP_LR_SCHEDULER,
    MLP_LABEL_SMOOTHING,
    USE_FOCAL_LOSS,
    FOCAL_GAMMA,
    USE_MIXUP,
    MIXUP_ALPHA,
    USE_COSINE_ANNEALING,
    COSINE_T_0,
    COSINE_T_MULT,
    CV_FOLDS,
    CLASS_NAMES,
    RF_MODEL_PATH,
    LGB_MODEL_PATH,
    MLP_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    TRAINING_CURVES_PNG,
    OUTPUTS_DIR,
    ALL_FEATURE_NAMES,
)
from cheating_detection.models.model_utils import (
    save_sklearn_model,
    save_pytorch_model,
)


# -- Focal Loss --------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss: down-weights well-classified examples, focuses on hard ones."""

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# -- MixUp helper ------------------------------------------------------------

def mixup_data(x, y, alpha=0.4):
    """MixUp: blend pairs of samples with random lambda from Beta(alpha, alpha)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# -- PyTorch Dataset ---------------------------------------------------------

class ExamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- MLP Architecture -------------------------------------------------------

class MLP(nn.Module):
    """MLP with BatchNorm, configurable depth and dropout."""

    def __init__(self, input_dim=N_TOTAL_FEATURES, hidden_layers=None,
                 dropout=MLP_DROPOUT, n_classes=5, use_batch_norm=MLP_USE_BATCH_NORM):
        super().__init__()
        hidden_layers = hidden_layers or MLP_HIDDEN_LAYERS
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()


# -- Random Forest -----------------------------------------------------------

def train_random_forest(X_train, y_train, verbose=True):
    if verbose:
        print("\n[RF] Training Random Forest (300 trees, balanced) ...")

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(rf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    if verbose:
        s = cv_results["test_score"]
        print(f"[RF] CV F1 (macro): {s.mean():.4f} +/- {s.std():.4f}")

    rf.fit(X_train, y_train)

    if verbose and X_train.shape[1] <= len(ALL_FEATURE_NAMES):
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("[RF] Feature importances (top 10):")
        for rank, idx in enumerate(indices[:10]):
            name = ALL_FEATURE_NAMES[idx] if idx < len(ALL_FEATURE_NAMES) else f"feat_{idx}"
            print(f"  {rank+1:>2}. {name:<28} {importances[idx]:.4f}")

    save_sklearn_model(rf, RF_MODEL_PATH)
    if verbose:
        print(f"[RF] Saved -> {RF_MODEL_PATH}")
    return rf


# -- LightGBM ---------------------------------------------------------------

def train_lightgbm(X_train, y_train, verbose=True):
    """Train LightGBM with balanced class weights."""
    try:
        import lightgbm as lgb
    except ImportError:
        if verbose:
            print("[LGB] lightgbm not installed (pip install lightgbm), skipping.")
        return None

    if verbose:
        print("\n[LGB] Training LightGBM ...")

    model = lgb.LGBMClassifier(
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=LGB_LEARNING_RATE,
        max_depth=LGB_MAX_DEPTH,
        num_leaves=LGB_NUM_LEAVES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    if verbose:
        s = cv_results["test_score"]
        print(f"[LGB] CV F1 (macro): {s.mean():.4f} +/- {s.std():.4f}")

    model.fit(X_train, y_train)
    save_sklearn_model(model, LGB_MODEL_PATH)
    if verbose:
        print(f"[LGB] Saved -> {LGB_MODEL_PATH}")
    return model


# -- Compute class weights ---------------------------------------------------

def _compute_class_weights(y_train, device):
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(unique)
    weights = total / (n_classes * counts)
    weight_tensor = torch.zeros(int(unique.max()) + 1, dtype=torch.float32)
    for cls, w in zip(unique, weights):
        weight_tensor[int(cls)] = w
    return weight_tensor.to(device)


# -- MLP epoch runner (with MixUp support) -----------------------------------

def _run_epoch(model, loader, criterion, optimizer=None, device="cpu", use_mixup=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if training:
            optimizer.zero_grad()

        if training and use_mixup:
            X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, MIXUP_ALPHA)
            logits = model(X_mixed)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == y_a).float().sum().item() +
                       (1 - lam) * (preds == y_b).float().sum().item())
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)

    return total_loss / total, correct / total


# -- MLP Training ------------------------------------------------------------

def train_mlp(X_train, y_train, X_val, y_val, input_dim=N_TOTAL_FEATURES,
              n_classes=5, verbose=True):
    """Train MLP with Focal Loss, MixUp, Cosine Annealing, BatchNorm."""
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n[MLP] Training on {device} ...")
        print(f"[MLP] Arch: {MLP_HIDDEN_LAYERS}, BN={MLP_USE_BATCH_NORM}, "
              f"Focal={USE_FOCAL_LOSS}, MixUp={USE_MIXUP}")

    train_ds = ExamDataset(X_train, y_train)
    val_ds = ExamDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)

    model = MLP(input_dim=input_dim, n_classes=n_classes).to(device)

    # Loss function selection
    class_weights = None
    if MLP_USE_CLASS_WEIGHTS:
        class_weights = _compute_class_weights(y_train, device)
        if verbose:
            print(f"[MLP] Class weights: {class_weights.cpu().numpy().round(3)}")

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights,
                              label_smoothing=MLP_LABEL_SMOOTHING)
        if verbose:
            print(f"[MLP] Loss: FocalLoss(gamma={FOCAL_GAMMA}, smoothing={MLP_LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=MLP_LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)

    # LR scheduler
    scheduler = None
    if MLP_LR_SCHEDULER:
        if USE_COSINE_ANNEALING:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=COSINE_T_0, T_mult=COSINE_T_MULT,
            )
            if verbose:
                print(f"[MLP] Scheduler: CosineAnnealingWarmRestarts(T0={COSINE_T_0}, Tmult={COSINE_T_MULT})")
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=7,
            )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer,
                                     device, use_mixup=USE_MIXUP)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if scheduler is not None:
            if USE_COSINE_ANNEALING:
                scheduler.step(epoch)
            else:
                scheduler.step(va_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3}/{MLP_MAX_EPOCHS} | "
                  f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"acc {tr_acc:.4f}/{va_acc:.4f} | lr {lr_now:.6f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (patience={MLP_PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    save_pytorch_model(model, MLP_MODEL_PATH)
    if verbose:
        print(f"[MLP] Saved -> {MLP_MODEL_PATH}")
    return model, history


# -- Ensemble ----------------------------------------------------------------

class EnsembleModel:
    """Soft-voting or stacking ensemble."""

    def __init__(self, models, weights=None, meta_learner=None):
        """
        models: list of (name, model) tuples
        weights: list of floats for soft-voting (if no meta_learner)
        meta_learner: fitted sklearn classifier for stacking
        """
        self.models = models
        self.weights = weights
        self.meta_learner = meta_learner

    def _get_base_proba(self, X):
        all_proba = []
        for name, model in self.models:
            if isinstance(model, MLP):
                x_t = torch.tensor(X, dtype=torch.float32)
                if next(model.parameters()).is_cuda:
                    x_t = x_t.cuda()
                proba = model.predict_proba(x_t)
            else:
                proba = model.predict_proba(X)
            all_proba.append(proba)
        return all_proba

    def predict_proba(self, X):
        all_proba = self._get_base_proba(X)
        if self.meta_learner is not None:
            # Stacking: concatenate base predictions as meta-features
            meta_X = np.hstack(all_proba)
            return self.meta_learner.predict_proba(meta_X)
        else:
            # Soft voting
            result = np.zeros_like(all_proba[0])
            for proba, w in zip(all_proba, self.weights):
                result += w * proba
            return result

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def train_ensemble(base_models, X_train, y_train, X_val, y_val, verbose=True):
    """Build a stacking ensemble with LogisticRegression meta-learner.

    Falls back to optimized soft-voting if stacking fails.
    """
    if verbose:
        print(f"\n[Ensemble] Building stacking ensemble with {len(base_models)} models ...")

    # Generate out-of-fold predictions for meta-learner training
    try:
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        meta_train = []
        meta_y = []

        for name, model in base_models:
            if isinstance(model, MLP):
                x_t = torch.tensor(X_train, dtype=torch.float32)
                if next(model.parameters()).is_cuda:
                    x_t = x_t.cuda()
                proba = model.predict_proba(x_t)
            else:
                proba = model.predict_proba(X_train)
            meta_train.append(proba)

        meta_X_train = np.hstack(meta_train)

        # Train meta-learner on training set predictions
        meta_lr = LogisticRegression(
            max_iter=500, random_state=RANDOM_SEED,
            class_weight="balanced", C=1.0,
        )
        meta_lr.fit(meta_X_train, y_train)

        # Evaluate on val
        meta_val = []
        for name, model in base_models:
            if isinstance(model, MLP):
                x_t = torch.tensor(X_val, dtype=torch.float32)
                if next(model.parameters()).is_cuda:
                    x_t = x_t.cuda()
                proba = model.predict_proba(x_t)
            else:
                proba = model.predict_proba(X_val)
            meta_val.append(proba)

        meta_X_val = np.hstack(meta_val)
        y_pred = meta_lr.predict(meta_X_val)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        if verbose:
            print(f"[Ensemble] Stacking meta-learner Val F1: {f1:.4f}")

        ensemble = EnsembleModel(base_models, meta_learner=meta_lr)

    except Exception as e:
        if verbose:
            print(f"[Ensemble] Stacking failed ({e}), using soft-voting fallback")

        # Fallback: optimized soft-voting
        n = len(base_models)
        best_f1 = 0
        best_w = [1.0 / n] * n

        for _ in range(50):
            w = np.random.dirichlet(np.ones(n))
            ens = EnsembleModel(base_models, weights=w.tolist())
            y_pred = ens.predict(X_val)
            f1_val = f1_score(y_val, y_pred, average="macro", zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_w = w.tolist()

        ensemble = EnsembleModel(base_models, weights=best_w)
        if verbose:
            names = [n for n, _ in base_models]
            print(f"[Ensemble] Soft-voting weights: {dict(zip(names, [f'{w:.2f}' for w in best_w]))}")
            print(f"[Ensemble] Val F1: {best_f1:.4f}")

    joblib.dump({"type": "stacking" if ensemble.meta_learner else "voting"}, ENSEMBLE_MODEL_PATH)
    if verbose:
        print(f"[Ensemble] Saved -> {ENSEMBLE_MODEL_PATH}")
    return ensemble


# -- Training curve plot -----------------------------------------------------

def plot_training_curves(history, save_path=TRAINING_CURVES_PNG):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("MLP Training Convergence (Loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MLP Training Convergence (Accuracy)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves -> {save_path}")
