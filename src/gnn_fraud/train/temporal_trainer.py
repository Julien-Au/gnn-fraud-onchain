"""Train EvolveGCN-O over Elliptic's per-time-step snapshots.

Each transaction lives at one time step, so we cut the graph into 49 snapshots and
process them as a sequence. The model's weights evolve across the sequence (BPTT).

Leakage discipline: weights are trained by evolving through the *sub-train* steps
only (t < val_start); the loss never touches val/test snapshots. At evaluation the
full sequence is replayed under no_grad so weights evolve all the way to the test
steps - using their structure, never their labels. Val (the last train steps) is
used for early stopping and threshold selection; test is t > 34.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary
from gnn_fraud.models.temporal import EvolveGCN
from gnn_fraud.train.gnn_trainer import TRAIN_MAX_TIMESTEP, TrainOutcome

NUM_TIMESTEPS = 49


class Snapshot:
    """One time-step subgraph with local node indexing."""

    __slots__ = ("t", "x", "edge_index", "y", "labeled")

    def __init__(self, t: int, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> None:
        self.t = t
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.labeled = (y == 0) | (y == 1)


def build_snapshots(data: Data, timesteps: NDArray[Any]) -> list[Snapshot]:
    """Cut the full graph into per-time-step snapshots (relabeled node indices)."""
    ts = torch.as_tensor(np.asarray(timesteps))
    snapshots: list[Snapshot] = []
    for t in range(1, NUM_TIMESTEPS + 1):
        mask = ts == t
        if not bool(mask.any()):
            continue
        edge_index, _ = subgraph(
            mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )
        snapshots.append(Snapshot(t, data.x[mask], edge_index, data.y[mask]))
    return snapshots


def _class_weights(snapshots: list[Snapshot], max_t: int) -> torch.Tensor:
    labels = torch.cat([s.y[s.labeled] for s in snapshots if s.t < max_t])
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    total = n0 + n1
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)


def _collect(
    model: EvolveGCN, snapshots: list[Snapshot], lo: int, hi: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Replay the full sequence (no_grad) and gather (y, prob) for steps in [lo, hi]."""
    model.eval()
    model.reset_state()
    probs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    with torch.no_grad():
        for s in snapshots:
            out = model(s.x, s.edge_index)
            if lo <= s.t <= hi and bool(s.labeled.any()):
                probs.append(out.softmax(dim=1)[s.labeled, 1])
                ys.append(s.y[s.labeled])
    return torch.cat(ys).cpu().numpy(), torch.cat(probs).cpu().numpy()


def fit_evolvegcn(
    data: Data,
    timesteps: NDArray[Any],
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
    grad_clip: float | None = None,
) -> tuple[EvolveGCN, list[Snapshot], TrainOutcome]:
    """Train EvolveGCN-O; return the fitted model, its snapshots, and test metrics.

    Loss is applied on train time steps ``< val_start``; ``[val_start, 34]`` is the
    temporal validation slice. Raising ``val_start`` toward 34 gives the model a
    fuller training budget (the "rolling" setup, closer to the static GNNs' budget).
    """
    torch.manual_seed(seed)
    snapshots = build_snapshots(data, timesteps)

    model = EvolveGCN(in_dim=int(data.num_node_features), hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = _class_weights(snapshots, max_t=val_start)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    best_val = -1.0
    best_epoch = -1
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    since_best = 0

    for epoch in range(epochs):
        model.train()
        model.reset_state()
        optimizer.zero_grad()
        losses: list[torch.Tensor] = []
        for s in snapshots:
            if s.t >= val_start:  # train only on sub-train steps (weights evolve through them)
                break
            out = model(s.x, s.edge_index)
            if bool(s.labeled.any()):
                losses.append(loss_fn(out[s.labeled], s.y[s.labeled]))
        loss = torch.stack(losses).mean()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        y_val, p_val = _collect(model, snapshots, val_start, TRAIN_MAX_TIMESTEP)
        val_pr_auc = evaluate_binary(y_val, p_val).pr_auc
        if val_pr_auc > best_val:
            best_val, best_epoch = val_pr_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    y_val, p_val = _collect(model, snapshots, val_start, TRAIN_MAX_TIMESTEP)
    threshold = best_f1_threshold(y_val, p_val)
    y_test, p_test = _collect(model, snapshots, TRAIN_MAX_TIMESTEP + 1, NUM_TIMESTEPS)
    metrics = evaluate_binary(y_test, p_test, threshold)
    outcome = TrainOutcome(
        metrics=metrics,
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_val),
        threshold=float(threshold),
    )
    return model, snapshots, outcome


def train_evolvegcn(
    data: Data,
    timesteps: NDArray[Any],
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
) -> TrainOutcome:
    """Train EvolveGCN-O and return test metrics (threshold chosen on temporal val)."""
    _, _, outcome = fit_evolvegcn(
        data,
        timesteps,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        seed=seed,
        val_start=val_start,
    )
    return outcome


def per_timestep_prauc(
    model: EvolveGCN, snapshots: list[Snapshot], lo: int, hi: int
) -> list[tuple[int, int, float]]:
    """PR-AUC per time step in [lo, hi] (rolling one-step-ahead backtest).

    Returns (time_step, num_illicit, pr_auc) for each step that has both classes.
    Weights evolve through the whole sequence; each step is scored with weights
    evolved up to it - the model never sees a step's labels.
    """
    model.eval()
    model.reset_state()
    out: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for s in snapshots:
            logits = model(s.x, s.edge_index)
            if lo <= s.t <= hi and bool(s.labeled.any()):
                y = s.y[s.labeled].cpu().numpy()
                p = logits.softmax(dim=1)[s.labeled, 1].cpu().numpy()
                n_illicit = int((y == 1).sum())
                if 0 < n_illicit < len(y):  # PR-AUC needs both classes
                    out.append((s.t, n_illicit, float(evaluate_binary(y, p).pr_auc)))
    return out
