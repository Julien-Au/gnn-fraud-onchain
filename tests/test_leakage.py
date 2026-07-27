"""Tests for the leakage experiment helpers (gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from gnn_fraud.experiments.leakage import random_masks  # noqa: E402


def test_random_masks_partition_labeled_nodes() -> None:
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4, dtype=torch.long)
    train, val, test = random_masks(y, seed=0)
    # disjoint and covering all labeled nodes
    assert bool((train & val).sum() == 0)
    assert bool((train & test).sum() == 0)
    assert bool((val & test).sum() == 0)
    labeled = ((y == 0) | (y == 1)).sum()
    assert int((train | val | test).sum()) == int(labeled)


def test_random_masks_stratified_and_deterministic() -> None:
    y = torch.tensor([0] * 30 + [1] * 30, dtype=torch.long)
    t1, v1, te1 = random_masks(y, seed=7)
    t2, v2, te2 = random_masks(y, seed=7)
    assert torch.equal(t1, t2) and torch.equal(te1, te2)  # deterministic
    # both classes present in the test split (stratification)
    assert int(y[te1].sum()) > 0 and int((y[te1] == 0).sum()) > 0


def test_run_dgraph_leakage_smoke() -> None:
    from torch_geometric.data import Data

    from gnn_fraud.experiments.leakage import run_dgraph_leakage

    torch.manual_seed(0)
    n = 200
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)  # background default
    idx = torch.arange(n)
    y[idx % 2 == 0] = 0
    y[idx % 10 == 0] = 1  # some fraud
    x[y == 1, 0] += 2.0
    data = Data(x=x, edge_index=torch.randint(0, n, (2, 400)), y=y)
    data.node_time = torch.randint(1, 100, (n,))
    res = run_dgraph_leakage(data, model_name="sage", seed=0)
    assert 0.0 <= res["temporal"]["pr_auc"] <= 1.0
    assert 0.0 <= res["random"]["pr_auc"] <= 1.0
