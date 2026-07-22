"""EvolveGCN-O: a temporal GNN whose weights evolve across graph snapshots.

Motivation (from step 3): Elliptic's graph is disconnected per time step and
suffers temporal distribution shift, and each transaction lives at exactly one
time step - so there is no node to "track" over time. EvolveGCN answers this by
evolving the *model parameters* across the sequence of per-time-step snapshots:
a GRU updates each GCN weight matrix from one snapshot to the next, letting the
model adapt as the distribution drifts.

This is a faithful from-scratch implementation of EvolveGCN-O (Pareja et al.,
AAAI 2020): the square weight W_t is the hidden state of a GRU fed with W_{t-1},
and the convolution applies A_hat @ (X @ W_t). No fragile compiled dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

NUM_CLASSES = 2


class EvolveGCNO(nn.Module):
    """One EvolveGCN-O layer: evolve a square weight with a GRU, then GCN-propagate.

    Keeps ``channels`` in == out (the weight is ``channels x channels``), so the
    feature dimension is preserved across the layer, exactly as in the paper.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.initial_weight = nn.Parameter(torch.empty(1, channels, channels))
        nn.init.xavier_uniform_(self.initial_weight)
        self.recurrent = nn.GRU(input_size=channels, hidden_size=channels, num_layers=1)
        self.weight: torch.Tensor | None = None

    def reset_state(self) -> None:
        """Clear the evolved weight; call at the start of each pass over the sequence."""
        self.weight = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        prev = self.initial_weight if self.weight is None else self.weight
        # GRU(input=W_prev, hidden=W_prev) -> new weight as the returned hidden state.
        _, self.weight = self.recurrent(prev, prev)
        weight = self.weight.squeeze(dim=0)  # (channels, channels)

        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), add_self_loops=True)
        x = x @ weight
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, col, norm.unsqueeze(-1) * x[row])
        return out


class EvolveGCN(nn.Module):
    """Project -> two EvolveGCN-O layers -> linear classifier (2 logits/node)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.egcn1 = EvolveGCNO(hidden_dim)
        self.egcn2 = EvolveGCNO(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.dropout = dropout

    def reset_state(self) -> None:
        self.egcn1.reset_state()
        self.egcn2.reset_state()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.relu(self.egcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.egcn2(x, edge_index))
        return self.classifier(x)
