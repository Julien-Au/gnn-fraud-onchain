"""Hyperparameter sweep for the GNN baselines under the honest temporal split.

The committee review flagged that our GNNs used near-default hyperparameters while
XGBoost got strong hand-picked ones, and that our temporal GNN scores sit well below
honest-protocol literature values. This sweep selects on TEMPORAL VALIDATION PR-AUC
only (never test), then reports the selected configuration's test metrics - one
honest, tuned number per architecture.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.experiments.leakage import _fit_with_masks

DEFAULT_GRID: dict[str, list[Any]] = {
    "hidden_dim": [64, 128, 256],
    "lr": [0.01, 0.005],
    "dropout": [0.3, 0.5],
}


def run_tuning(
    data: Data,
    timesteps: NDArray[Any],
    model_name: str = "sage",
    seed: int = 42,
    val_start: int = 30,
    grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Sweep the grid on the temporal split; select on val PR-AUC; report test."""
    grid = grid or DEFAULT_GRID
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask

    keys = list(grid.keys())
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for values in product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, values, strict=True))
        metrics, val_pr = _fit_with_masks(data, model_name, t_train, t_val, t_test, seed=seed, **hp)
        trial = {"hp": hp, "val_pr_auc": round(val_pr, 4), "test": metrics.as_row()}
        trials.append(trial)
        if best is None or val_pr > best["val_pr_auc"]:
            best = trial
    assert best is not None
    return {"model": model_name, "seed": seed, "selected": best, "trials": trials}
