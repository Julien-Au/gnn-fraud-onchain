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


def run_hetero_leakage(
    data: Any, target: str = "addr", seed: int = 42
) -> dict[str, dict[str, float | int]]:
    """Leakage on the Elliptic++ heterogeneous task: temporal vs random split of ``target`` nodes."""
    from gnn_fraud.train.hetero_trainer import train_hetero

    temporal = train_hetero(data, target=target, seed=seed).metrics
    tr, va, te = random_masks(data[target].y, seed=seed)
    random_split = train_hetero(data, target=target, seed=seed, override_masks=(tr, va, te)).metrics
    return {"temporal": temporal.as_row(), "random": random_split.as_row()}


def _xgb_with_masks(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int = 42,
) -> BinaryMetrics:
    """XGBoost on node features with arbitrary train/val/test masks (threshold on val)."""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr, va, te = train_mask.cpu().numpy(), val_mask.cpu().numpy(), test_mask.cpu().numpy()
    scaler = StandardScaler().fit(x[tr])
    pos, neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(neg / pos) if pos else 1.0,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=4,
    )
    xgb.fit(scaler.transform(x[tr]), y[tr])
    s_va = xgb.predict_proba(scaler.transform(x[va]))[:, 1]
    s_te = xgb.predict_proba(scaler.transform(x[te]))[:, 1]
    thr = best_f1_threshold(y[va], s_va)
    return evaluate_binary(y[te], s_te, thr)


def run_leakage_multi(
    data: Data,
    timesteps: NDArray[Any],
    models: tuple[str, ...] = ("gcn", "sage", "gat"),
    seed: int = 42,
    val_start: int = 30,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Multi-model leakage table: each model under the honest temporal vs a leaky random split.

    Uses one trainer with different masks so the ONLY difference is the split.
    """
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask
    r_train, r_val, r_test = random_masks(data.y, seed=seed)

    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for m in models:
        temporal = _fit_with_masks(data, m, t_train, t_val, t_test, seed=seed)
        random_split = _fit_with_masks(data, m, r_train, r_val, r_test, seed=seed)
        out[m] = {"temporal": temporal.as_row(), "random": random_split.as_row()}
    try:
        out["xgboost"] = {
            "temporal": _xgb_with_masks(data, t_train, t_val, t_test, seed).as_row(),
            "random": _xgb_with_masks(data, r_train, r_val, r_test, seed).as_row(),
        }
    except ImportError:
        pass
    return out
