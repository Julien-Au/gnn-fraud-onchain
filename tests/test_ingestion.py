"""Tests for graph statistics on a tiny synthetic graph (no download).

These exercise the torch/PyG path, so they are skipped when the ``gnn`` extra is
not installed (the fast gate). The ``--full`` gate and the CI ``gnn`` job install
the extra and run them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.ingestion.elliptic import graph_stats  # noqa: E402


def _toy_graph() -> Data:
    # 5 nodes. Node 4 is isolated. Two components: {0,1,2}, {3}, {4}.
    #   0 -> 1 -> 2 -> 0   (a directed triangle)
    #   3 is connected to nobody but itself via no edge -> its own component
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(5, 3)
    # classes: licit(0), illicit(1), unknown(2)
    y = torch.tensor([0, 1, 2, 0, 2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def test_basic_counts() -> None:
    stats = graph_stats(_toy_graph())
    assert stats.num_nodes == 5
    assert stats.num_edges == 3
    assert stats.num_node_features == 3
    assert stats.is_directed is True


def test_class_counts_and_imbalance() -> None:
    stats = graph_stats(_toy_graph())
    assert stats.class_counts == {"licit": 2, "illicit": 1, "unknown": 2}
    assert stats.labeled_nodes == 3
    assert stats.illicit_share_of_labeled == pytest.approx(1 / 3)
    assert stats.illicit_share_of_all == pytest.approx(1 / 5)


def test_isolated_and_components() -> None:
    stats = graph_stats(_toy_graph())
    # nodes 3 and 4 have no edges -> isolated
    assert stats.isolated_nodes == 2
    # components: {0,1,2}, {3}, {4}
    assert stats.num_connected_components == 3
    assert stats.largest_component_fraction == pytest.approx(3 / 5)
