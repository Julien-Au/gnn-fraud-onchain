"""GraphUSAD: a USAD-style adversarial autoencoder on a graph (research prototype).

USAD (Audibert et al., KDD 2020) trains two autoencoders that share an encoder and
play an adversarial game: AE1 learns to reconstruct, AE2 learns to tell real inputs
from AE1's reconstructions, and AE1 learns to fool AE2. The anomaly score combines
both reconstruction errors. This sharpens the boundary around "normal" versus a
plain autoencoder.

Here the encoder is a GNN (GraphSAGE) over node features + graph structure, and the
two decoders (MLPs) reconstruct node features. Trained on licit (normal) nodes only;
the anomaly score is the (adversarially-weighted) node-feature reconstruction error.
This is the graph analogue of USAD and a starting point for research on distribution
shift - reported honestly whatever it scores.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer (DANN): identity forward, negated gradient backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class DomainHead(nn.Module):
    """Predicts the time period (train vs later) from the latent - used with a GRL to
    push the encoder toward a time-INVARIANT representation (v3, drift-robust)."""

    def __init__(self, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = _mlp(latent_dim, hidden, 2)

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(z, lambd))


class GraphUSAD(nn.Module):
    """Shared GNN encoder + two feature-reconstruction decoders (USAD-style)."""

    def __init__(
        self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.enc1 = SAGEConv(in_dim, hidden_dim)
        self.enc2 = SAGEConv(hidden_dim, latent_dim)
        self.dec1 = _mlp(latent_dim, hidden_dim, in_dim)
        self.dec2 = _mlp(latent_dim, hidden_dim, in_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.enc2(h, edge_index)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (AE1, AE2, AE2 of Encoder(AE1)) - the three USAD reconstructions."""
        z = self.encode(x, edge_index)
        ae1 = self.dec1(z)
        ae2 = self.dec2(z)
        z2 = self.encode(ae1, edge_index)
        ae2_of_ae1 = self.dec2(z2)
        return ae1, ae2, ae2_of_ae1
