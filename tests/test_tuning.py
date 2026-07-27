"""Tests for the GNN tuning sweep (gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.experiments.tuning import run_tuning  # noqa: E402


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 200
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)
    timesteps = np.array([(i % 45) + 1 for i in range(n)], dtype=np.int64)
    for i in range(0, n, 2):
        y[i] = 1 if i % 8 == 0 else 0
    x[y == 1, 0] += 2.5
    lab = ((y == 0) | (y == 1)).numpy()
    data = Data(
        x=x,
        edge_index=torch.randint(0, n, (2, 400)),
        y=y,
        train_mask=torch.tensor(lab & (timesteps <= 34)),
        test_mask=torch.tensor(lab & (timesteps > 34)),
    )
    return data, timesteps


def test_run_tuning_selects_on_val() -> None:
    data, ts = _synthetic()
    tiny_grid = {"hidden_dim": [16, 32], "lr": [0.01], "dropout": [0.3]}
    res = run_tuning(data, ts, model_name="sage", seed=0, grid=tiny_grid)
    assert len(res["trials"]) == 2
    best_val = max(t["val_pr_auc"] for t in res["trials"])
    assert res["selected"]["val_pr_auc"] == best_val  # selection is by val, never test
