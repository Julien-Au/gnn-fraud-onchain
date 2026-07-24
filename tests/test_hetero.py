"""Smoke test for the heterogeneous GNN pipeline (gnn tier; skipped without torch)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from gnn_fraud.train.hetero_trainer import train_hetero  # noqa: E402


def _synthetic_hetero() -> HeteroData:
    torch.manual_seed(0)
    n_tx, n_addr = 80, 60
    data = HeteroData()
    x_tx = torch.randn(n_tx, 6)
    y = torch.full((n_tx,), -1, dtype=torch.long)
    time = torch.tensor([(i % 40) + 1 for i in range(n_tx)], dtype=torch.long)
    for i in range(n_tx):
        if i % 2 == 0:  # label the even tx
            y[i] = 1 if i % 6 == 0 else 0
    x_tx[y == 1, 0] += 2.5
    data["tx"].x = x_tx
    data["tx"].y = y
    data["tx"].time = time
    labeled = y >= 0
    data["tx"].train_mask = labeled & (time <= 34)
    data["tx"].test_mask = labeled & (time > 34)
    data["addr"].x = torch.randn(n_addr, 5)
    addr_y = torch.full((n_addr,), -1, dtype=torch.long)
    addr_time = torch.tensor([(i % 40) + 1 for i in range(n_addr)], dtype=torch.long)
    for i in range(n_addr):
        if i % 2 == 0:
            addr_y[i] = 1 if i % 6 == 0 else 0
    data["addr"].x[addr_y == 1, 0] += 2.5
    data["addr"].y = addr_y
    data["addr"].time = addr_time
    addr_labeled = addr_y >= 0
    data["addr"].train_mask = addr_labeled & (addr_time <= 34)
    data["addr"].test_mask = addr_labeled & (addr_time > 34)

    data["tx", "to", "tx"].edge_index = torch.randint(0, n_tx, (2, 120))
    data["addr", "to", "tx"].edge_index = torch.stack(
        [torch.randint(0, n_addr, (100,)), torch.randint(0, n_tx, (100,))]
    )
    data["tx", "to", "addr"].edge_index = torch.stack(
        [torch.randint(0, n_tx, (100,)), torch.randint(0, n_addr, (100,))]
    )
    data["addr", "to", "addr"].edge_index = torch.randint(0, n_addr, (2, 80))
    return data


@pytest.mark.parametrize("target", ["tx", "addr"])
def test_train_hetero_smoke(target: str) -> None:
    out = train_hetero(
        _synthetic_hetero(), epochs=3, patience=3, seed=0, val_start=30, target=target
    )
    assert 0.0 <= out.metrics.pr_auc <= 1.0
    assert 0.0 <= out.metrics.f1 <= 1.0


def test_train_hetero_override_masks() -> None:
    from gnn_fraud.experiments.leakage import random_masks

    data = _synthetic_hetero()
    tr, va, te = random_masks(data["addr"].y, seed=0)
    out = train_hetero(
        data, epochs=3, patience=3, seed=0, target="addr", override_masks=(tr, va, te)
    )
    assert 0.0 <= out.metrics.pr_auc <= 1.0
