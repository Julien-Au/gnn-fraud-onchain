"""Tests for the gap-decomposition experiment (gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.experiments.decompose import match_prevalence, run_decompose  # noqa: E402


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 300
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)
    timesteps = np.array([(i % 45) + 1 for i in range(n)], dtype=np.int64)
    for i in range(0, n, 2):
        y[i] = 1 if i % 8 == 0 else 0
    x[y == 1, 0] += 2.5
    lab = ((y == 0) | (y == 1)).numpy()
    data = Data(
        x=x,
        edge_index=torch.randint(0, n, (2, 600)),
        y=y,
        train_mask=torch.tensor(lab & (timesteps <= 34)),
        test_mask=torch.tensor(lab & (timesteps > 34)),
    )
    return data, timesteps


def test_match_prevalence_hits_target() -> None:
    data, _ = _synthetic()
    y = data.y
    test = data.test_mask
    target = 0.10
    matched = match_prevalence(test, y, target, seed=0)
    # matched is a subset of the original test mask
    assert bool((matched & ~test).sum() == 0)
    prev = float((y[matched] == 1).float().mean())
    assert abs(prev - target) < 0.05  # small-n rounding tolerance


def test_run_decompose_components_sum() -> None:
    data, ts = _synthetic()
    res = run_decompose(data, ts, model_name="sage", seeds=(0,))
    comps = res["components"]
    # base_rate + mp_leak + distribution_access == total_gap (per construction)
    total = comps["total_gap"]["mean"]
    parts = (
        comps["base_rate"]["mean"]
        + comps["message_passing_leak"]["mean"]
        + comps["distribution_access"]["mean"]
    )
    assert abs(total - parts) < 1e-3  # rounding of per-component means
    assert "random_matched_inductive" in res["arms"]
