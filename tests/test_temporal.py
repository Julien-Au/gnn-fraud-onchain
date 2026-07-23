"""Smoke tests for EvolveGCN-O and its temporal trainer (gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.models.temporal import EvolveGCNO  # noqa: E402
from gnn_fraud.train.temporal_trainer import (  # noqa: E402
    build_snapshots,
    fit_evolvegcn,
    per_timestep_prauc,
    train_evolvegcn,
)


def test_evolvegcno_evolves_weight() -> None:
    layer = EvolveGCNO(4)
    x = torch.randn(10, 4)
    ei = torch.randint(0, 10, (2, 20))
    layer.reset_state()
    layer(x, ei)
    w1 = layer.weight.detach().clone()
    layer(x, ei)  # second snapshot: weight should have evolved
    w2 = layer.weight.detach().clone()
    assert not torch.allclose(w1, w2)


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 240
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)
    timesteps = np.array([(i % 40) + 1 for i in range(n)], dtype=np.int64)
    labeled_idx = np.arange(0, n, 2)
    for k, i in enumerate(labeled_idx):
        y[i] = 1 if k % 3 == 0 else 0
    x[y == 1, 0] += 2.5
    edge_index = torch.randint(0, n, (2, 400), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, timesteps


def test_build_snapshots_covers_all_steps() -> None:
    data, ts = _synthetic()
    snaps = build_snapshots(data, ts)
    assert len(snaps) == 40  # steps 1..40 present
    # node counts across snapshots sum to N
    assert sum(int(s.x.shape[0]) for s in snaps) == data.num_nodes


def test_train_evolvegcn_smoke() -> None:
    data, ts = _synthetic()
    out = train_evolvegcn(data, ts, epochs=3, patience=3, seed=0, val_start=30)
    assert 0.0 <= out.metrics.pr_auc <= 1.0
    assert 0.0 <= out.metrics.f1 <= 1.0


def test_per_timestep_backtest() -> None:
    data, ts = _synthetic()
    model, snaps, _ = fit_evolvegcn(data, ts, epochs=3, patience=3, seed=0, val_start=33)
    rows = per_timestep_prauc(model, snaps, 35, 40)
    for t, n_illicit, pr in rows:
        assert 35 <= t <= 40
        assert n_illicit > 0
        assert 0.0 <= pr <= 1.0
