"""Smoke test for GraphUSAD (research prototype; gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.train.graph_usad_trainer import train_graph_usad  # noqa: E402


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 160
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)
    timesteps = np.array([(i % 40) + 1 for i in range(n)], dtype=np.int64)
    labeled = np.arange(0, n, 2)
    for k, i in enumerate(labeled):
        y[i] = 1 if k % 3 == 0 else 0
    x[y == 1] += 2.0  # illicit nodes reconstruct worse -> higher anomaly score
    edge_index = torch.randint(0, n, (2, 300), dtype=torch.long)
    lab = ((y == 0) | (y == 1)).numpy()
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=torch.tensor(lab & (timesteps <= 34)),
        test_mask=torch.tensor(lab & (timesteps > 34)),
    )
    return data, timesteps


def test_train_graph_usad_smoke() -> None:
    data, ts = _synthetic()
    out = train_graph_usad(data, ts, epochs=5, patience=5, seed=0, val_start=30)
    assert 0.0 <= out.metrics.pr_auc <= 1.0
    assert 0.0 <= out.metrics.roc_auc <= 1.0
