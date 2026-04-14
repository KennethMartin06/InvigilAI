"""
train.py -- Training pipeline for Random Forest, MLP, and Ensemble classifiers.

Improvements over v1:
  - MLP: BatchNorm, deeper architecture [256,128,64], class weights, LR scheduler
  - RF: 300 estimators, balanced class weights
  - Ensemble: Soft-voting average of RF + MLP probabilities
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    N_TOTAL_FEATURES,
    RF_N_ESTIMATORS,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    MLP_LR,
    MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    MLP_USE_BATCH_NORM,
    MLP_USE_CLASS_WEIGHTS,
    MLP_LR_SCHEDULER,
    CV_FOLDS,
    CLASS_NAMES,
    RF_MODEL_PATH,
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


# -- PyTorch Dataset ---------------------------------------------------------

class ExamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -- MLP Architecture (improved with BatchNorm) -----------------------------

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional BatchNorm for cheating detection.

    Architecture: Input -> [Dense -> BatchNorm -> ReLU -> Dropout] x N -> Output
    """

    def __init__(
        self,
        input_dim: int = N_TOTAL_FEATURES,
        hidden_layers: list = None,
        dropout: float = MLP_DROPOUT,
        n_classes: int = 5,
        use_batch_norm: bool = MLP_USE_BATCH_NORM,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or MLP_HIDDEN_LAYERS
        self.use_batch_norm = use_batch_norm

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()


# -- Random Forest -----------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> RandomForestClassifier:
    """Train a Random Forest with balanced class weights and 300 estimators."""
    if verbose:
        print("\n[RF] Training Random Forest ...")

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # 5-fold CV
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        rf, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    if verbose:
        scores = cv_results["test_score"]
        print(f"[RF] CV F1 (macro): {scores.mean():.4f} +/- {scores.std():.4f}")

    # Final fit on full training set
    rf.fit(X_train, y_train)

    if verbose:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("[RF] Feature importances (top 10):")
        for rank, idx in enumerate(indices[:10]):
            print(f"  {rank+1:>2}. {ALL_FEATURE_NAMES[idx]:<28} {importances[idx]:.4f}")

    save_sklearn_model(rf, RF_MODEL_PATH)
    if verbose:
        print(f"[RF] Model saved -> {RF_MODEL_PATH}")

    return rf


# -- Compute class weights for CrossEntropyLoss -----------------------------

def _compute_class_weights(y_train: np.ndarray, device: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(unique)
    weights = total / (n_classes * counts)
    weight_tensor = torch.zeros(int(unique.max()) + 1, dtype=torch.float32)
    for cls, w in zip(unique, weights):
        weight_tensor[int(cls)] = w
    return weight_tensor.to(device)


# -- MLP epoch runner --------------------------------------------------------

def _run_epoch(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: str = "cpu",
) -> tuple:
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

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


# -- MLP Training (improved) ------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int = N_TOTAL_FEATURES,
    n_classes: int = 5,
    verbose: bool = True,
) -> tuple:
    """Train MLP with BatchNorm, class weights, and LR scheduling."""
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n[MLP] Training on {device} ...")
        print(f"[MLP] Architecture: {MLP_HIDDEN_LAYERS}, BatchNorm={MLP_USE_BATCH_NORM}")
        print(f"[MLP] Class weights: {MLP_USE_CLASS_WEIGHTS}, LR scheduler: {MLP_LR_SCHEDULER}")

    train_ds = ExamDataset(X_train, y_train)
    val_ds = ExamDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)

    model = MLP(input_dim=input_dim, n_classes=n_classes).to(device)

    # Class-weighted loss for imbalanced data
    if MLP_USE_CLASS_WEIGHTS:
        class_weights = _compute_class_weights(y_train, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if verbose:
            print(f"[MLP] Class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    # LR scheduler: reduce LR when val loss plateaus
    scheduler = None
    if MLP_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, verbose=verbose,
        )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        # Step the LR scheduler
        if scheduler is not None:
            scheduler.step(va_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:>3}/{MLP_MAX_EPOCHS} | "
                f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                f"acc {tr_acc:.4f}/{va_acc:.4f} | "
                f"lr {lr_now:.6f}"
            )

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

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    save_pytorch_model(model, MLP_MODEL_PATH)
    if verbose:
        print(f"[MLP] Model saved -> {MLP_MODEL_PATH}")

    return model, history


# -- Ensemble (soft-voting RF + MLP) ----------------------------------------

class EnsembleModel:
    """Soft-voting ensemble that averages RF and MLP predicted probabilities."""

    def __init__(self, rf_model, mlp_model, rf_weight=0.4, mlp_weight=0.6):
        self.rf = rf_model
        self.mlp = mlp_model
        self.rf_weight = rf_weight
        self.mlp_weight = mlp_weight

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rf_proba = self.rf.predict_proba(X)
        x_t = torch.tensor(X, dtype=torch.float32)
        if next(self.mlp.parameters()).is_cuda:
            x_t = x_t.cuda()
        mlp_proba = self.mlp.predict_proba(x_t)
        return self.rf_weight * rf_proba + self.mlp_weight * mlp_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def train_ensemble(
    rf_model,
    mlp_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> EnsembleModel:
    """Create and evaluate a soft-voting ensemble of RF + MLP."""
    if verbose:
        print("\n[Ensemble] Building soft-voting ensemble (RF + MLP) ...")

    # Try different weight combinations on validation set
    best_f1 = 0
    best_weights = (0.5, 0.5)

    for rf_w in np.arange(0.2, 0.8, 0.1):
        mlp_w = 1.0 - rf_w
        ens = EnsembleModel(rf_model, mlp_model, rf_w, mlp_w)
        y_pred = ens.predict(X_val)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (rf_w, mlp_w)

    ensemble = EnsembleModel(rf_model, mlp_model, best_weights[0], best_weights[1])

    if verbose:
        print(f"[Ensemble] Best weights: RF={best_weights[0]:.1f}, MLP={best_weights[1]:.1f}")
        print(f"[Ensemble] Val F1 (macro): {best_f1:.4f}")

    # Save ensemble metadata
    joblib.dump({"rf_weight": best_weights[0], "mlp_weight": best_weights[1]}, ENSEMBLE_MODEL_PATH)
    if verbose:
        print(f"[Ensemble] Saved -> {ENSEMBLE_MODEL_PATH}")

    return ensemble


# -- Training curve plot -----------------------------------------------------

def plot_training_curves(history: dict, save_path: str = TRAINING_CURVES_PNG) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("MLP -- Training Convergence (Loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MLP -- Training Convergence (Accuracy)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved -> {save_path}")
