"""A heterogeneous GNN for Elliptic++ transaction classification.

Built with ``HeteroConv``: one ``SAGEConv`` per relation (tx<->tx, addr<->tx,
addr<->addr), whose messages are aggregated per destination node type. This is the
"schema-agnostic" pattern the relational-foundation-model framing cares about - the
model is parameterized by the schema's node/edge types, not hard-coded to one graph.
(We use HeteroConv rather than ``to_hetero`` to avoid a torch.fx tracing issue and
keep the per-relation structure explicit.)

Lazy ``SAGEConv((-1, -1), ...)`` lets one definition handle the different feature
dimensions of tx and addr nodes; parameters materialize on the first forward pass
(the trainer runs a warmup before building the optimizer).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

NUM_CLASSES = 2


class HeteroGNN(nn.Module):
    """Two HeteroConv(SAGE) layers + a classifier head on ``tx`` nodes."""

    def __init__(self, metadata: Any, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        edge_types = metadata[1]
        self.conv1 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="sum"
        )
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.dropout = dropout

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[Any, torch.Tensor]
    ) -> torch.Tensor:
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.classifier(F.relu(x_dict["tx"]))
