"""
train.py — Training pipeline for SVM, Random Forest, and MLP classifiers.

Each model is trained on the 16-dimensional fused feature vector.
5-fold stratified cross-validation is run on the training set before a
final fit on the full training data.  Trained models are saved to disk.
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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    N_TOTAL_FEATURES,
    SVM_PARAM_GRID,
    RF_N_ESTIMATORS,
    MLP_HIDDEN_LAYERS,
    MLP_DROPOUT,
    MLP_LR,
    MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    CV_FOLDS,
    CLASS_NAMES,
    SVM_MODEL_PATH,
    RF_MODEL_PATH,
    MLP_MODEL_PATH,
    TRAINING_CURVES_PNG,
    OUTPUTS_DIR,
    BEHAVIORAL_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)
from cheating_detection.models.model_utils import (
    save_sklearn_model,
    save_pytorch_model,
)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ExamDataset(Dataset):
    """
    Wraps numpy feature / label arrays as a torch Dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── MLP Architecture ──────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for cheating detection.

    Architecture: Input(16) → Dense(128, ReLU) → Dropout(0.3)
                            → Dense(64,  ReLU) → Dropout(0.3)
                            → Output(5,  Softmax)

    Parameters
    ----------
    input_dim : int — number of input features (default 16).
    hidden_layers : list[int] — sizes of hidden layers.
    dropout : float — dropout probability.
    n_classes : int — number of output classes.
    """

    def __init__(
        self,
        input_dim: int = N_TOTAL_FEATURES,
        hidden_layers: list = None,
        dropout: float = MLP_DROPOUT,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or MLP_HIDDEN_LAYERS

        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        # No softmax here — CrossEntropyLoss includes log-softmax internally
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Return class probabilities (after softmax) as a numpy array.

        Parameters
        ----------
        x : torch.Tensor, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()


# ── SVM ───────────────────────────────────────────────────────────────────────

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> SVC:
    """
    Train an RBF-kernel SVM with grid-search over C and gamma.

    5-fold stratified CV is used inside GridSearchCV.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 16)
    y_train : np.ndarray, shape (n_train,)
    verbose : bool

    Returns
    -------
    sklearn.svm.SVC — best estimator from grid search.
    """
    if verbose:
        print("\n[SVM] Starting grid search …")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    base_svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        base_svm,
        SVM_PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    if verbose:
        print(f"[SVM] Best params : {grid.best_params_}")
        print(f"[SVM] CV F1 (macro): {grid.best_score_:.4f}")

    save_sklearn_model(best, SVM_MODEL_PATH)
    if verbose:
        print(f"[SVM] Model saved → {SVM_MODEL_PATH}")

    return best


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Train a Random Forest and report feature importances.

    5-fold stratified CV is computed for logging purposes.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 16)
    y_train : np.ndarray, shape (n_train,)
    verbose : bool

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier — fitted estimator.
    """
    if verbose:
        print("\n[RF] Training Random Forest …")

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
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
        print(f"[RF] CV F1 (macro): {scores.mean():.4f} ± {scores.std():.4f}")

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
        print(f"[RF] Model saved → {RF_MODEL_PATH}")

    return rf


# ── MLP (PyTorch) ─────────────────────────────────────────────────────────────

def _run_epoch(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Run one training or validation epoch.

    Parameters
    ----------
    model : MLP
    loader : DataLoader
    criterion : loss function
    optimizer : if None the model is evaluated (no gradient update).
    device : str

    Returns
    -------
    (mean_loss, accuracy) over the epoch.
    """
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


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int = N_TOTAL_FEATURES,
    n_classes: int = 5,
    verbose: bool = True,
) -> tuple[MLP, dict]:
    """
    Train the MLP with early stopping and return training history.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data for early stopping
    input_dim : int
    n_classes : int
    verbose : bool

    Returns
    -------
    (MLP model, history_dict)
        history_dict keys: train_loss, val_loss, train_acc, val_acc
    """
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n[MLP] Training on {device} …")

    train_ds = ExamDataset(X_train, y_train)
    val_ds = ExamDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)

    model = MLP(input_dim=input_dim, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)

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

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:>3}/{MLP_MAX_EPOCHS} | "
                f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                f"acc {tr_acc:.4f}/{va_acc:.4f}"
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
        print(f"[MLP] Model saved → {MLP_MODEL_PATH}")

    return model, history


# ── Training curve plot ───────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = TRAINING_CURVES_PNG) -> None:
    """
    Plot and save MLP training / validation loss and accuracy curves.

    Parameters
    ----------
    history : dict — output of train_mlp.
    save_path : str — destination PNG path.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("MLP — Training Convergence (Loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MLP — Training Convergence (Accuracy)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")
