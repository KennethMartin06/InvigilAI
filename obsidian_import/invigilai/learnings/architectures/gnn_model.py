"""
gnn_model.py -- Feature-relationship Graph Neural Network.

Builds a graph where nodes are features and edges are correlations above
a threshold. Each sample's feature values become node features. A small
GCN then pools node embeddings into a classification.

Implemented without torch_geometric -- uses a hand-rolled GCN layer.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    GNN_HIDDEN_DIM,
    GNN_LAYERS,
    GNN_DROPOUT,
    GNN_EDGE_THRESHOLD,
    MLP_LR,
    MLP_BATCH_SIZE,
)


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def build_feature_graph(X_train, threshold=GNN_EDGE_THRESHOLD):
    """Build feature-feature adjacency by absolute correlation thresholding."""
    corr = np.corrcoef(X_train.T)
    adj = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    # Symmetric normalization: D^-1/2 A D^-1/2
    d = adj.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-6)
    norm_adj = adj * d_inv_sqrt[None, :] * d_inv_sqrt[:, None]
    return norm_adj.astype(np.float32)


class GNNModel:
    """GCN classifier over a feature-feature correlation graph."""

    def __init__(self, n_features, n_classes,
                 hidden=GNN_HIDDEN_DIM,
                 layers=GNN_LAYERS,
                 dropout=GNN_DROPOUT,
                 lr=MLP_LR):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.adj = None
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn

        class GCNLayer(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.w = nn.Linear(in_dim, out_dim)

            def forward(self, x, adj):
                # x: (B, N_nodes, in_dim), adj: (N_nodes, N_nodes)
                x = self.w(x)
                return adj @ x

        class GCN(nn.Module):
            def __init__(self, node_feat_dim, hidden, n_layers, dropout, n_classes):
                super().__init__()
                self.blocks = nn.ModuleList()
                prev = node_feat_dim
                for _ in range(n_layers):
                    self.blocks.append(GCNLayer(prev, hidden))
                    prev = hidden
                self.drop = nn.Dropout(dropout)
                self.act = nn.ReLU()
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_classes),
                )

            def forward(self, x, adj):
                for b in self.blocks:
                    x = self.drop(self.act(b(x, adj)))
                # Global mean pool over nodes
                pooled = x.mean(dim=1)
                return self.head(pooled)

        # Node feature dim = 1 (scalar feature value); could be richer
        self.model = GCN(node_feat_dim=1, hidden=self.hidden, n_layers=self.layers,
                         dropout=self.dropout, n_classes=self.n_classes)

    def _to_graph_batch(self, X):
        # Each feature becomes a node with a single scalar = feature value
        return X[..., None]   # (B, N_features, 1)

    def fit(self, X, y, epochs=40, batch_size=MLP_BATCH_SIZE, verbose=True):
        if not _torch_ok():
            if verbose:
                print("[GNN] PyTorch unavailable")
            return self

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        self.adj = build_feature_graph(X)
        self._build()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        adj_t = torch.tensor(self.adj, dtype=torch.float32, device=device)

        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        X_t = torch.tensor(self._to_graph_batch(X), dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        for ep in range(epochs):
            perm = torch.randperm(len(X_t))
            losses = []
            self.model.train()
            for i in range(0, len(X_t), batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(X_t[idx], adj_t)
                loss = ce(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if verbose and (ep + 1) % 10 == 0:
                print(f"[GNN] ep {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def predict(self, X):
        import torch
        device = next(self.model.parameters()).device
        adj_t = torch.tensor(self.adj, dtype=torch.float32, device=device)
        X_t = torch.tensor(self._to_graph_batch(X), dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t, adj_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        device = next(self.model.parameters()).device
        adj_t = torch.tensor(self.adj, dtype=torch.float32, device=device)
        X_t = torch.tensor(self._to_graph_batch(X), dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t, adj_t)
        return torch.softmax(logits, dim=1).cpu().numpy()
