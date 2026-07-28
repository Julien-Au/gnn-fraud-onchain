"""Smoke tests for the recipe-replication and windowed-inflation experiments (gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 400
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)
    timesteps = np.array([(i % 49) + 1 for i in range(n)], dtype=np.int64)
    for i in range(0, n, 2):
        y[i] = 1 if i % 8 == 0 else 0
    x[y == 1, 0] += 2.5
    lab = ((y == 0) | (y == 1)).numpy()
    data = Data(
        x=x,
        edge_index=torch.randint(0, n, (2, 800)),
        y=y,
        train_mask=torch.tensor(lab & (timesteps <= 34)),
        test_mask=torch.tensor(lab & (timesteps > 34)),
    )
    return data, timesteps


def test_run_windowed_leakage_smoke() -> None:
    from gnn_fraud.experiments.windowed import run_windowed_leakage

    data, ts = _synthetic()
    res = run_windowed_leakage(data, ts, model_name="sage", seeds=(0,))
    assert "pre_shutdown" in res and "post_shutdown" in res
    assert "post_greater_than_pre" in res["prediction"]
    for w in ("pre_shutdown", "post_shutdown"):
        assert 0.0 <= res[w]["temporal"]["mean"] <= 1.0


def test_run_recipe_smoke() -> None:
    pytest.importorskip("xgboost")
    from gnn_fraud.experiments.recipe import run_recipe

    data, ts = _synthetic()
    res = run_recipe(data, ts, seeds=(0,))
    for arm in ("recipe_random_split", "honest_temporal_split"):
        assert 0.0 <= res[arm]["accuracy"]["mean"] <= 1.0
        assert 0.0 <= res[arm]["illicit_f1"]["mean"] <= 1.0
