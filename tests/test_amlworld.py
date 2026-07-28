"""Smoke tests for the AMLworld leakage experiment (gnn tier: needs xgboost)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")

from gnn_fraud.experiments.amlworld_leakage import (  # noqa: E402
    _splits_random,
    _splits_temporal,
    run_amlworld_leakage,
)


def test_temporal_splits_partition_and_order() -> None:
    t = np.arange(1000)
    tr, va, te = _splits_temporal(t)
    assert (tr | va | te).all() and not (tr & va).any() and not (va & te).any()
    assert t[tr].max() < t[va].min() <= t[va].max() < t[te].min()  # strictly ordered


def test_random_splits_stratified() -> None:
    y = np.array([0] * 900 + [1] * 100)
    tr, va, te = _splits_random(y, seed=0)
    assert (tr | va | te).all()
    # both classes present in test
    assert y[te].sum() > 0 and (y[te] == 0).sum() > 0


def test_run_amlworld_leakage_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    t = np.arange(n)
    y = (rng.random(n) < 0.05).astype(np.int64)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    x[y == 1, 0] += 2.0
    res = run_amlworld_leakage(x, y, t, seeds=(0,))
    assert 0.0 <= res["temporal"]["pr_auc"]["mean"] <= 1.0
    assert 0.0 <= res["random"]["pr_auc"]["mean"] <= 1.0
    assert "prevalence" in res
