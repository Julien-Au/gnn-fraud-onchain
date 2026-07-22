"""Smoke test for the baseline pipeline (gnn tier; skipped without torch)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.models.baselines import run_baselines  # noqa: E402


def _synthetic_labeled_graph() -> Data:
    torch.manual_seed(0)
    n = 80
    x = torch.randn(n, 8)
    y = torch.zeros(n, dtype=torch.long)
    y[::5] = 1  # ~20% positives; make them separable on feature 0
    x[y == 1, 0] += 3.0
    idx = torch.arange(n)
    train_mask = idx < 50
    test_mask = idx >= 50
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


def test_run_baselines_returns_metrics() -> None:
    results = run_baselines(_synthetic_labeled_graph(), seed=0)
    # logreg and autoencoder always run; xgboost only if the `boost` extra is present.
    assert "logreg" in results
    assert "autoencoder" in results
    for m in results.values():
        assert 0.0 <= m.pr_auc <= 1.0
        assert 0.0 <= m.roc_auc <= 1.0
        assert 0.0 <= m.f1 <= 1.0
