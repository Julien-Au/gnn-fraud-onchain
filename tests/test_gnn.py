"""Smoke tests for the GNN models and trainer (gnn tier; skipped without torch)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from gnn_fraud.models.gnn import build_model  # noqa: E402
from gnn_fraud.train.gnn_trainer import temporal_val_masks, train_gnn  # noqa: E402


def _synthetic() -> tuple[Data, np.ndarray]:
    torch.manual_seed(0)
    n = 160
    x = torch.randn(n, 6)
    y = torch.full((n,), 2, dtype=torch.long)  # default: unknown
    timesteps = np.array([(i % 40) + 1 for i in range(n)], dtype=np.int64)

    labeled_idx = np.arange(0, n, 2)  # even nodes are labeled
    for k, i in enumerate(labeled_idx):
        y[i] = 1 if k % 3 == 0 else 0  # ~1/3 positives
    x[y == 1, 0] += 2.5  # make positives separable

    labeled = ((y == 0) | (y == 1)).numpy()
    train_mask = torch.tensor(labeled & (timesteps <= 34))
    test_mask = torch.tensor(labeled & (timesteps > 34))
    edge_index = torch.randint(0, n, (2, 300), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data, timesteps


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("nope", in_dim=6)


def test_temporal_val_masks_alignment_and_disjoint() -> None:
    data, ts = _synthetic()
    sub_train, val = temporal_val_masks(data, ts, val_start=30)
    # sub-train and val partition the train mask, and are disjoint.
    assert bool((sub_train & val).sum() == 0)
    assert int((sub_train | val).sum()) == int(data.train_mask.sum())


@pytest.mark.parametrize("model_name", ["gcn", "sage", "gat"])
def test_train_gnn_smoke(model_name: str) -> None:
    data, ts = _synthetic()
    out = train_gnn(data, ts, model_name, epochs=5, patience=5, seed=0, val_start=30)
    assert 0.0 <= out.metrics.pr_auc <= 1.0
    assert 0.0 <= out.metrics.f1 <= 1.0
    assert out.best_epoch >= 0
