"""Ingestion of the Elliptic Bitcoin dataset via PyTorch Geometric.

The Elliptic graph has three classes: licit (0), illicit (1) and unknown (2).
Only classes 0 and 1 are labeled; the majority of nodes are unknown. PyG ships a
built-in temporal train/test split (early time steps -> train, later -> test),
which we surface and never override with a random split (that would leak the
future; see docs/research-log.md).

``graph_stats`` is deliberately generic - it operates on any PyG ``Data`` - so it
is unit-testable on a tiny synthetic graph without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import degree

# Class encoding used by PyG's EllipticBitcoinDataset.
LICIT = 0
ILLICIT = 1
UNKNOWN = 2
CLASS_NAMES = {LICIT: "licit", ILLICIT: "illicit", UNKNOWN: "unknown"}


def load_elliptic(root: str | Path = "data/raw/elliptic") -> Data:
    """Load the Elliptic Bitcoin graph as a single PyG ``Data`` object.

    Downloads on first use (PyG handles it); subsequent calls read the cache.
    """
    dataset = EllipticBitcoinDataset(root=str(root))
    return dataset[0]


def node_timesteps(root: str | Path = "data/raw/elliptic") -> NDArray[Any]:
    """Return each node's time step (1..49), aligned to PyG node index order.

    PyG assigns node indices in the row order of the features CSV, so the time
    step column (column 1) is already aligned to ``data.x`` row order. Callers
    should assert alignment against the built-in masks (see the trainer).
    """
    feats = pd.read_csv(
        Path(root) / "raw" / "elliptic_txs_features.csv",
        header=None,
        usecols=[1],
        names=["time"],
    )
    return feats["time"].to_numpy()


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Summary statistics for a transaction graph. All values are computed, not
    assumed - this is what goes into the DATA_CARD after ingestion."""

    num_nodes: int
    num_edges: int
    num_node_features: int
    is_directed: bool
    class_counts: dict[str, int]
    labeled_nodes: int
    illicit_share_of_labeled: float
    illicit_share_of_all: float
    mean_degree: float
    max_degree: int
    isolated_nodes: int
    num_connected_components: int
    largest_component_fraction: float


def graph_stats(data: Data) -> GraphStats:
    """Compute summary statistics for any PyG ``Data`` graph."""
    num_nodes = int(data.num_nodes)
    num_edges = int(data.num_edges)

    # Class distribution (named), robust to whichever classes are present.
    y = data.y
    values, counts = torch.unique(y, return_counts=True)
    class_counts = {
        CLASS_NAMES.get(int(v), str(int(v))): int(c)
        for v, c in zip(values.tolist(), counts.tolist(), strict=False)
    }
    n_illicit = class_counts.get("illicit", 0)
    n_licit = class_counts.get("licit", 0)
    labeled = n_illicit + n_licit

    # Degree over the undirected view (in + out), so isolated == truly isolated.
    deg = degree(data.edge_index[0], num_nodes=num_nodes) + degree(
        data.edge_index[1], num_nodes=num_nodes
    )
    isolated = int((deg == 0).sum())

    # Connected components on the undirected adjacency (fast via scipy).
    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()
    ones = np.ones(src.shape[0], dtype=np.int8)
    adj = coo_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes))
    n_components, comp_labels = connected_components(adj, directed=False)
    _, comp_sizes = np.unique(comp_labels, return_counts=True)
    largest_fraction = float(comp_sizes.max()) / num_nodes if num_nodes else 0.0

    return GraphStats(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_node_features=int(data.num_node_features),
        is_directed=bool(data.is_directed()),
        class_counts=class_counts,
        labeled_nodes=labeled,
        illicit_share_of_labeled=(n_illicit / labeled) if labeled else 0.0,
        illicit_share_of_all=(n_illicit / num_nodes) if num_nodes else 0.0,
        mean_degree=float(deg.mean()),
        max_degree=int(deg.max()) if num_nodes else 0,
        isolated_nodes=isolated,
        num_connected_components=int(n_components),
        largest_component_fraction=largest_fraction,
    )
