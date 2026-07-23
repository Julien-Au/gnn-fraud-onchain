"""Transductive training of a GNN for node classification on Elliptic.

Full-batch: the whole graph (including unknown nodes) participates in message
passing; the loss is applied only on labeled train nodes. Model selection
(early stopping) uses a *temporal* validation slice carved from the train period
(the latest train time steps), so it never peeks at the test period. The test
metrics use the built-in temporal test split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import BinaryMetrics, best_f1_threshold, evaluate_binary
from gnn_fraud.models.gnn import build_model

TRAIN_MAX_TIMESTEP = 34  # PyG's built-in split: train = time steps 1..34


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    metrics: BinaryMetrics
    best_epoch: int
    best_val_pr_auc: float
    threshold: float


def temporal_val_masks(
    data: Data, timesteps: NDArray[Any], val_start: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the built-in train mask into sub-train (early) and val (latest steps).

    Asserts alignment between ``timesteps`` and PyG's masks before trusting it.
    """
    ts = torch.as_tensor(np.asarray(timesteps))
    y = data.y
    labeled = (y == 0) | (y == 1)
    # Alignment check: labeled nodes at t<=34 must equal the built-in train mask.
    expected_train = int((labeled & (ts <= TRAIN_MAX_TIMESTEP)).sum())
    if expected_train != int(data.train_mask.sum()):
        raise AssertionError(
            "timestep alignment failed: labeled(t<=34)="
            f"{expected_train} != train_mask={int(data.train_mask.sum())}"
        )
    train = data.train_mask
    val = train & (ts >= val_start)
    sub_train = train & (ts < val_start)
    return sub_train, val


def _class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = y[mask]
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    total = n0 + n1
    # Inverse-frequency weights (balanced): rare class gets a larger weight.
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)


def _positive_prob(model: torch.nn.Module, data: Data) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        return logits.softmax(dim=1)[:, 1]


def fit_gnn(
    data: Data,
    timesteps: NDArray[Any],
    model_name: str,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
) -> tuple[torch.nn.Module, TrainOutcome]:
    """Train one GNN; return the fitted model and its test metrics."""
    torch.manual_seed(seed)
    sub_train, val = temporal_val_masks(data, timesteps, val_start=val_start)
    y = data.y

    model = build_model(
        model_name, in_dim=int(data.num_node_features), hidden_dim=hidden_dim, dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = _class_weights(y, sub_train)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    y_val_np = y[val].cpu().numpy()
    best_val = -1.0
    best_epoch = -1
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[sub_train], y[sub_train])
        loss.backward()
        optimizer.step()

        prob = _positive_prob(model, data)
        val_pr_auc = evaluate_binary(y_val_np, prob[val].cpu().numpy()).pr_auc
        if val_pr_auc > best_val:
            best_val = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    prob = _positive_prob(model, data)
    threshold = best_f1_threshold(y_val_np, prob[val].cpu().numpy())
    metrics = evaluate_binary(
        y[data.test_mask].cpu().numpy(),
        prob[data.test_mask].cpu().numpy(),
        threshold,
    )
    outcome = TrainOutcome(
        metrics=metrics,
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_val),
        threshold=float(threshold),
    )
    return model, outcome


def train_gnn(
    data: Data,
    timesteps: NDArray[Any],
    model_name: str,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
) -> TrainOutcome:
    """Train one GNN and return test metrics (threshold chosen on temporal val)."""
    _, outcome = fit_gnn(
        data,
        timesteps,
        model_name,
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
