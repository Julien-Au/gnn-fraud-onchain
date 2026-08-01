"""Graph neural network models: GCN, GraphSAGE, GAT.

Ordered by the idea each one adds, so a gain can be attributed to a mechanism:

- **GCN** - the simplest message passing: each node's new representation is a
  normalized *mean* of its neighbors (symmetric degree normalization). One shared
  linear transform, no notion of neighbor importance.
- **GraphSAGE** - separates "self" from "aggregated neighbors" (concatenate then
  transform) and is designed to be *inductive* (generalize to unseen nodes); we
  use mean aggregation here.
- **GAT** - learns *attention* weights over neighbors, so not all neighbors count
  equally. Multi-head for stability.
- **GraphTransformer** - full dot-product attention over neighbors
  (TransformerConv, the UniMP operator), the transformer-style message passing
  used by graph transformers; included so the leakage study covers the
  attention-with-values family, not just GAT-style additive attention.

All models are 2-layer (a common depth that avoids over-smoothing) and output 2
logits per node (binary: licit vs illicit). They run full-batch and transductive:
the whole graph - including unknown nodes - participates in message passing; the
loss is applied only on labeled train nodes (handled by the trainer).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv

NUM_CLASSES = 2


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, NUM_CLASSES)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, NUM_CLASSES)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int = 64, heads: int = 8, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # First layer: `heads` attention heads, concatenated -> hidden_dim*heads.
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # Second layer: average the heads down to the class logits.
        self.conv2 = GATConv(
            hidden_dim * heads, NUM_CLASSES, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphTransformer(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # First layer: multi-head dot-product attention, concatenated heads.
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # Second layer: average the heads down to the class logits.
        self.conv2 = TransformerConv(
            hidden_dim * heads, NUM_CLASSES, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_model(name: str, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> nn.Module:
    """Factory: 'gcn' | 'sage' | 'gat' | 'transformer'."""
    key = name.lower()
    if key == "gcn":
        return GCN(in_dim, hidden_dim, dropout)
    if key == "sage":
        return GraphSAGE(in_dim, hidden_dim, dropout)
    if key == "gat":
        return GAT(in_dim, hidden_dim, dropout=dropout)
    if key == "transformer":
        return GraphTransformer(in_dim, hidden_dim, dropout=dropout)
    raise ValueError(f"unknown model '{name}' (expected: gcn, sage, gat, transformer)")
