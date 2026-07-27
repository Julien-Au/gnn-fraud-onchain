"""Train the domain-adversarial supervised GraphSAGE (drift-robust).

Supervised illicit/licit loss on train nodes + a gradient-reversed time-period loss
over all nodes, so the encoder learns time-invariant features. Evaluated on the honest
temporal test split; the baseline to beat is a plain GraphSAGE (PR-AUC 0.488).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary
from gnn_fraud.models.dann import DANNGraphSAGE
from gnn_fraud.train.gnn_trainer import TRAIN_MAX_TIMESTEP, TrainOutcome, temporal_val_masks


def train_dann(
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
    gamma: float = 1.0,
) -> TrainOutcome:
    """Train DANN-GraphSAGE; return test metrics (threshold chosen on temporal val)."""
    torch.manual_seed(seed)
    sub_train, val = temporal_val_masks(data, timesteps, val_start=val_start)
    y = data.y
    x, ei = data.x, data.edge_index
    domain = torch.as_tensor(timesteps > TRAIN_MAX_TIMESTEP).long()

    labels = y[sub_train]
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    total = n0 + n1
    weight = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)

    model = DANNGraphSAGE(int(data.num_node_features), hidden_dim=hidden_dim, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_val = y[val].cpu().numpy()
    best_val, best_epoch, since = -1.0, -1, 0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())

    def prob() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, ei, 0.0)
            return logits.softmax(dim=1)[:, 1]

    for epoch in range(epochs):
        model.train()
        p = epoch / max(1, epochs - 1)
        lambd = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        opt.zero_grad()
        class_logits, domain_logits = model(x, ei, lambd)
        loss_cls = F.cross_entropy(class_logits[sub_train], y[sub_train], weight=weight)
        loss_dom = F.cross_entropy(domain_logits, domain)
        (loss_cls + gamma * loss_dom).backward()
        opt.step()

        val_pr = evaluate_binary(y_val, prob()[val].cpu().numpy()).pr_auc
        if val_pr > best_val:
            best_val, best_epoch, since = val_pr, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    s = prob()
    threshold = best_f1_threshold(y_val, s[val].cpu().numpy())
    metrics = evaluate_binary(
        y[data.test_mask].cpu().numpy(), s[data.test_mask].cpu().numpy(), threshold
    )
    return TrainOutcome(
        metrics=metrics,
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_val),
        threshold=float(threshold),
    )
