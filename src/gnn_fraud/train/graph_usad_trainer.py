"""Train GraphUSAD on Elliptic (unsupervised, on licit/normal nodes only).

Follows USAD's adversarial objective with a clean two-optimizer (GAN-style) split:
- generator (encoder + AE1 decoder) minimizes reconstruction AND fools AE2 into
  reconstructing AE1's output as if it were real;
- discriminator (AE2 decoder) reconstructs real inputs well and AE1's outputs badly.
The adversarial weight follows USAD's 1/n schedule (n = epoch). The anomaly score is
alpha * ||x - AE1|| + beta * ||x - AE2(Encode(AE1))|| per node (higher = anomalous).

Only licit training nodes drive the loss; the encoder still sees the full graph.
This is a research prototype - the result is reported as-is.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary
from gnn_fraud.models.graph_usad import GraphUSAD
from gnn_fraud.train.gnn_trainer import TrainOutcome, temporal_val_masks


def _node_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-node mean squared error over features -> [N]."""
    return ((recon - x) ** 2).mean(dim=1)


def _anomaly_score(
    model: GraphUSAD, data: Data, alpha: float = 0.5, beta: float = 0.5
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        ae1, _, ae2_of_ae1 = model(data.x, data.edge_index)
        return alpha * _node_mse(data.x, ae1) + beta * _node_mse(data.x, ae2_of_ae1)


def train_graph_usad(
    data: Data,
    timesteps: NDArray[Any],
    hidden_dim: int = 64,
    latent_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    val_start: int = 30,
    alpha: float = 0.5,
    beta: float = 0.5,
    normal_recent_steps: int | None = None,
) -> TrainOutcome:
    """Train GraphUSAD unsupervised; evaluate illicit detection on the temporal test.

    ``normal_recent_steps`` (drift-aware v2): if set, the "normal" set uses only the
    latest N train time steps (a rolling window that tracks the drifting normal),
    instead of all early steps. This directly targets v1's failure mode, where the
    reconstruction error tracked the licit distribution's drift rather than illicitness.
    """
    torch.manual_seed(seed)
    sub_train, val = temporal_val_masks(data, timesteps, val_start=val_start)
    y = data.y
    normal = sub_train & (y == 0)  # train on licit (normal) nodes only
    if normal_recent_steps is not None:
        ts = torch.as_tensor(timesteps)
        normal = normal & (ts >= (val_start - normal_recent_steps))

    model = GraphUSAD(int(data.num_node_features), hidden_dim=hidden_dim, latent_dim=latent_dim)
    gen_params = (
        list(model.enc1.parameters())
        + list(model.enc2.parameters())
        + list(model.dec1.parameters())
    )
    opt_g = torch.optim.Adam(gen_params, lr=lr)
    opt_d = torch.optim.Adam(model.dec2.parameters(), lr=lr)

    x, ei = data.x, data.edge_index
    y_val = y[val].cpu().numpy()
    best_val, best_epoch, since_best = -1.0, -1, 0
    best_state: dict[str, Any] = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        n = float(epoch)
        z = model.encode(x, ei)
        ae1 = model.dec1(z)
        ae2_of_ae1 = model.dec2(model.encode(ae1, ei))
        # Generator: reconstruct + fool the discriminator on normal nodes.
        loss_g = (1.0 / n) * _node_mse(x[normal], ae1[normal]).mean() + (1.0 - 1.0 / n) * _node_mse(
            x[normal], ae2_of_ae1[normal]
        ).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # Discriminator: reconstruct real well, AE1's (detached) output badly.
        z_d = model.encode(x, ei).detach()
        ae2_real = model.dec2(z_d)
        ae1_d = model.dec1(model.encode(x, ei)).detach()
        ae2_fake = model.dec2(model.encode(ae1_d, ei).detach())
        loss_d = (1.0 / n) * _node_mse(x[normal], ae2_real[normal]).mean() - (
            1.0 - 1.0 / n
        ) * _node_mse(x[normal], ae2_fake[normal]).mean()
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        score = _anomaly_score(model, data, alpha, beta)
        val_pr_auc = evaluate_binary(y_val, score[val].cpu().numpy()).pr_auc
        if val_pr_auc > best_val:
            best_val, best_epoch, since_best = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    score = _anomaly_score(model, data, alpha, beta)
    threshold = best_f1_threshold(y_val, score[val].cpu().numpy())
    metrics = evaluate_binary(
        y[data.test_mask].cpu().numpy(), score[data.test_mask].cpu().numpy(), threshold
    )
    return TrainOutcome(
        metrics=metrics,
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_val),
        threshold=float(threshold),
    )
