"""Train the heterogeneous GNN on Elliptic++ for transaction classification.

Full-batch, transductive. Message passing is made bidirectional (``ToUndirected``)
so tx nodes receive from addresses and vice-versa. Model selection uses a temporal
validation slice (latest train time steps); test is time step > 34. Loss is applied
only on labeled tx nodes; addresses are unsupervised context.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary
from gnn_fraud.models.hetero import HeteroGNN
from gnn_fraud.train.gnn_trainer import TrainOutcome


def _masks(data: HeteroData, val_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = data["tx"].train_mask
    time = data["tx"].time
    sub_train = train & (time < val_start)
    val = train & (time >= val_start)
    return sub_train, val


def _class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = y[mask]
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    total = n0 + n1
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)


def train_hetero(
    data: HeteroData,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
) -> TrainOutcome:
    """Train HeteroGNN and return tx-test metrics (threshold chosen on temporal val)."""
    torch.manual_seed(seed)
    data = T.ToUndirected(merge=False)(data)

    sub_train, val = _masks(data, val_start)
    y = data["tx"].y
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict

    model = HeteroGNN(data.metadata(), hidden_dim=hidden_dim, dropout=dropout)
    with torch.no_grad():  # materialize lazy parameters before the optimizer sees them
        model(x_dict, edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(y, sub_train))

    y_val = y[val].cpu().numpy()
    best_val, best_epoch, since_best = -1.0, -1, 0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())

    def positive_prob() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(x_dict, edge_index_dict).softmax(dim=1)[:, 1]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)
        loss = loss_fn(out[sub_train], y[sub_train])
        loss.backward()
        optimizer.step()

        val_pr_auc = evaluate_binary(y_val, positive_prob()[val].cpu().numpy()).pr_auc
        if val_pr_auc > best_val:
            best_val, best_epoch, since_best = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    prob = positive_prob()
    threshold = best_f1_threshold(y_val, prob[val].cpu().numpy())
    metrics = evaluate_binary(
        y[data["tx"].test_mask].cpu().numpy(),
        prob[data["tx"].test_mask].cpu().numpy(),
        threshold,
    )
    return TrainOutcome(
        metrics=metrics,
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_val),
        threshold=float(threshold),
    )
