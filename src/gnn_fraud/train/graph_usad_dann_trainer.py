"""GraphUSAD v3: domain-adversarial (time-invariant) reconstruction.

v1/v2 failed because the reconstruction error tracked the licit distribution's drift
rather than illicitness. v3 attacks that directly: a domain classifier tries to
predict the *time period* (train vs later) from the latent, and a gradient-reversal
layer trains the encoder to make the latent time-INVARIANT. If the latent no longer
encodes "which period", reconstruction error should reflect abnormality, not drift.

Uses only the time period (known for all nodes transductively), never test labels.
Research prototype - reported as-is.
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
from gnn_fraud.models.graph_usad import DomainHead, GraphUSAD
from gnn_fraud.train.gnn_trainer import TRAIN_MAX_TIMESTEP, TrainOutcome, temporal_val_masks


def _node_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return ((recon - x) ** 2).mean(dim=1)


def train_graph_usad_dann(
    data: Data,
    timesteps: NDArray[Any],
    hidden_dim: int = 64,
    latent_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
    gamma: float = 1.0,
) -> TrainOutcome:
    """Train the domain-adversarial GraphUSAD; evaluate on the temporal test split."""
    torch.manual_seed(seed)
    sub_train, val = temporal_val_masks(data, timesteps, val_start=val_start)
    y = data.y
    normal = sub_train & (y == 0)
    x, ei = data.x, data.edge_index
    # Domain label: 0 = train period (t<=34), 1 = later period. Known for all nodes.
    domain = torch.as_tensor(timesteps > TRAIN_MAX_TIMESTEP).long()

    model = GraphUSAD(int(data.num_node_features), hidden_dim=hidden_dim, latent_dim=latent_dim)
    domain_head = DomainHead(latent_dim)
    params = (
        list(model.enc1.parameters())
        + list(model.enc2.parameters())
        + list(model.dec1.parameters())
        + list(domain_head.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)

    y_val = y[val].cpu().numpy()
    best_val, best_epoch, since = -1.0, -1, 0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())

    def score() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            z = model.encode(x, ei)
            return _node_mse(x, model.dec1(z))

    for epoch in range(epochs):
        model.train()
        domain_head.train()
        # DANN lambda schedule: ramps 0 -> 1 over training.
        p = epoch / max(1, epochs - 1)
        lambd = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        opt.zero_grad()
        z = model.encode(x, ei)
        recon_loss = _node_mse(x[normal], model.dec1(z)[normal]).mean()
        domain_loss = F.cross_entropy(domain_head(z, lambd), domain)
        (recon_loss + gamma * domain_loss).backward()
        opt.step()

        val_pr = evaluate_binary(y_val, score()[val].cpu().numpy()).pr_auc
        if val_pr > best_val:
            best_val, best_epoch, since = val_pr, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    s = score()
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
