"""Ingestion of DGraph-Fin: a large, different-domain temporal fraud graph.

DGraph-Fin (Huang et al., NeurIPS 2022) is a financial social graph from Finvolution:
3,700,550 user nodes (17 features), 4,300,999 timestamped directed edges (emergency-
contact relations). Labels: 0 = normal, 1 = fraud (~1.3% of labeled), 2/3 = unlabeled
background. It is a genuinely different domain from Elliptic (Chinese fintech, not
crypto), used here to test whether the leakage inflation generalizes cross-domain.

The dataset is gated (registration + ToS at dgraph.xinye.com); the user downloads
``DGraphFin.zip`` (and optionally the v2 timestamp files) and places them under
``data/raw/dgraphfin/raw/``. We never redistribute it. Per-node time is derived as the
earliest incident edge timestamp (first-seen), enabling a temporal train/test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data


def _clean(feat: NDArray[Any]) -> NDArray[Any]:
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled: NDArray[Any] = ((feat - mean) / std).astype(np.float32)
    return scaled


def load_dgraph(
    root: str | Path = "data/raw/dgraphfin",
    cache: str | Path = "data/processed/dgraph.pt",
) -> Data:
    """Load DGraph-Fin as a PyG ``Data`` with a per-node first-seen ``node_time``."""
    cache_path = Path(cache)
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    npz = np.load(Path(root) / "raw" / "dgraphfin.npz")
    x = _clean(npz["x"].astype(np.float32))
    y = npz["y"].astype(np.int64)
    edge_index = npz["edge_index"].T.copy()  # [E, 2] -> [2, E]
    edge_ts = npz["edge_timestamp"].astype(np.int64)

    # Per-node first-seen time = earliest incident edge timestamp (both endpoints).
    n = y.shape[0]
    node_time = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(node_time, edge_index[0], edge_ts)
    np.minimum.at(node_time, edge_index[1], edge_ts)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(y),
    )
    data.node_time = torch.from_numpy(node_time)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data
