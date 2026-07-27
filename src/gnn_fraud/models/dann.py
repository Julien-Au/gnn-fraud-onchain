"""Domain-adversarial GraphSAGE (supervised, drift-robust).

The unsupervised reconstruction approach failed under Elliptic's temporal shift
(GraphUSAD, three variants). Here we keep the strong *supervised* signal and add a
domain-adversarial term: a GraphSAGE encoder feeds (a) a supervised illicit/licit
classifier and (b) a time-period discriminator through a gradient-reversal layer, so
the representation is pushed to be time-INVARIANT. The hypothesis: time-invariant
features generalize better from the train period to the post-shift test period, so
this should beat a plain GraphSAGE under the honest temporal split.

This is the supervised analogue of DANN (Ganin & Lempitsky, 2015) on a graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from gnn_fraud.models.graph_usad import DomainHead

NUM_CLASSES = 2


class DANNGraphSAGE(nn.Module):
    """GraphSAGE encoder + supervised classifier + gradient-reversal time-period head."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.domain = DomainHead(hidden_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.relu(self.conv2(h, edge_index))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, lambd: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index)
        return self.classifier(z), self.domain(z, lambd)
