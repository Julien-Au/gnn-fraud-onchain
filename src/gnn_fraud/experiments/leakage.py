"""Leakage demonstration: temporal split vs random split, same model.

The literature's near-perfect Elliptic numbers come from RANDOM splits that leak
future time steps. This experiment reproduces the inflation on our own code: train
the *same* GraphSAGE under (a) the honest temporal split and (b) a stratified random
split, and compare illicit-class PR-AUC / F1. The random split should score much
higher - not because the model is better, but because it has seen the future.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import BinaryMetrics, best_f1_threshold, evaluate_binary
from gnn_fraud.models.gnn import build_model
from gnn_fraud.train.gnn_trainer import train_gnn


def random_masks(
    y: torch.Tensor, seed: int, val_frac: float = 0.15, test_frac: float = 0.30
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified random train/val/test masks over labeled (0/1) nodes.

    This is the *leaky* split: labels are assigned regardless of time step.
    """
    y_np = y.cpu().numpy()
    n = y_np.shape[0]
    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for cls in (0, 1):  # stratify within each class
        idx = np.where(y_np == cls)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        n_val = int(len(idx) * val_frac)
        test[idx[:n_test]] = True
        val[idx[n_test : n_test + n_val]] = True
        train[idx[n_test + n_val :]] = True
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)


def _fit_with_masks(
    data: Data,
    model_name: str,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> BinaryMetrics:
    """Transductive GNN training with arbitrary train/val/test masks."""
    torch.manual_seed(seed)
    y = data.y
    labels = y[train_mask]
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    total = n0 + n1
    weight = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model = build_model(
        model_name, in_dim=int(data.num_node_features), hidden_dim=hidden_dim, dropout=dropout
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = y[val_mask].cpu().numpy()
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())
    since = 0

    def prob() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(data.x, data.edge_index).softmax(dim=1)[:, 1]

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        pr = evaluate_binary(y_val, prob()[val_mask].cpu().numpy()).pr_auc
        if pr > best_val:
            best_val, best_state, since = pr, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    p = prob()
    thr = best_f1_threshold(y_val, p[val_mask].cpu().numpy())
    return evaluate_binary(y[test_mask].cpu().numpy(), p[test_mask].cpu().numpy(), thr)


def run_leakage_experiment(
    data: Data, timesteps: NDArray[Any], model_name: str = "sage", seed: int = 42
) -> dict[str, dict[str, float | int]]:
    """Compare the same GNN under the temporal split vs a leaky random split."""
    temporal = train_gnn(data, timesteps, model_name, seed=seed).metrics
    train_m, val_m, test_m = random_masks(data.y, seed=seed)
    random_split = _fit_with_masks(data, model_name, train_m, val_m, test_m, seed=seed)
    return {"temporal": temporal.as_row(), "random": random_split.as_row()}
