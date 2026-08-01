"""Decompose the temporal-vs-random evaluation gap into its components.

The raw temporal-vs-random gap bundles several effects. This module separates them
with two controls, per seed, on threshold-free PR-AUC (so threshold calibration is
out of the picture entirely):

- **Arm T** - temporal split, transductive (the honest protocol).
- **Arm R** - random split, transductive (the leaky protocol).
- **Arm M** - random split with the test set subsampled so illicit prevalence
  matches Arm T's test prevalence. R - M = the **base-rate component** (PR-AUC is
  prevalence-dependent; this much of the gap is arithmetic, not leakage).
- **Arm I** (GNNs only) - like M, but the model is trained **inductively**: all
  test-set nodes are removed from the graph during training, so no test features or
  test-adjacent edges can reach the model. M - I = the **message-passing leakage
  component** (the "double leak" hypothesis, now measured instead of asserted).
- I - T (GNNs) or M - T (tabular) = the **distribution-access component**: training
  examples drawn from the test period teach the model the shifted distribution.
  This is an evaluation artifact too (a deployed model cannot train on the future),
  but it is distinct from classical feature/edge leakage.

All arms share one trainer and hyperparameters; only masks/graph visibility differ.
"""

from __future__ import annotations

import copy
import statistics
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from gnn_fraud.eval.metrics import evaluate_binary
from gnn_fraud.experiments.leakage import _fit_with_masks, _xgb_with_masks, random_masks
from gnn_fraud.models.gnn import build_model


def match_prevalence(
    test_mask: torch.Tensor, y: torch.Tensor, target_prev: float, seed: int
) -> torch.Tensor:
    """Subsample the test mask so the positive prevalence matches ``target_prev``.

    Drops positives (or negatives, if the target is higher) uniformly at random.
    """
    y_np = y.cpu().numpy()
    te = test_mask.cpu().numpy().copy()
    pos = np.where(te & (y_np == 1))[0]
    neg = np.where(te & (y_np == 0))[0]
    rng = np.random.default_rng(seed)
    need_pos = int(round(target_prev * len(neg) / (1.0 - target_prev)))
    if need_pos <= len(pos):
        drop = rng.choice(pos, size=len(pos) - need_pos, replace=False)
    else:
        need_neg = int(round(len(pos) * (1.0 - target_prev) / target_prev))
        drop = rng.choice(neg, size=len(neg) - need_neg, replace=False)
    te[drop] = False
    return torch.from_numpy(te)


def _fit_inductive(
    data: Data,
    model_name: str,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    hidden_dim: int = 64,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> float:
    """Train with ALL test-set nodes removed from the graph; return test PR-AUC.

    Training (and validation scoring) run on the subgraph without test nodes, so no
    test feature or test-adjacent edge is ever visible in training. Final inference
    runs on the full graph (at deployment time the test nodes' own features are
    legitimately available).
    """
    torch.manual_seed(seed)
    keep = ~test_mask
    idx = torch.where(keep)[0]
    sub_ei, _ = subgraph(keep, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    x_sub = data.x[keep]
    y_sub = data.y[keep]
    pos_in_sub = torch.full((data.num_nodes,), -1, dtype=torch.long)
    pos_in_sub[idx] = torch.arange(idx.numel())
    sub_train = pos_in_sub[torch.where(train_mask)[0]]
    sub_val = pos_in_sub[torch.where(val_mask)[0]]

    labels = y_sub[sub_train]
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    total = n0 + n1
    weight = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model = build_model(
        model_name, in_dim=int(data.num_node_features), hidden_dim=hidden_dim, dropout=dropout
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = y_sub[sub_val].cpu().numpy()
    best_val, best_state, since = -1.0, copy.deepcopy(model.state_dict()), 0

    def sub_prob() -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(x_sub, sub_ei).softmax(dim=1)[:, 1]

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x_sub, sub_ei)
        loss = loss_fn(out[sub_train], y_sub[sub_train])
        loss.backward()
        opt.step()
        val_pr = evaluate_binary(y_val, sub_prob()[sub_val].cpu().numpy()).pr_auc
        if val_pr > best_val:
            best_val, best_state, since = val_pr, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():  # full-graph inference: test nodes now legitimately present
        prob = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
    return evaluate_binary(data.y[test_mask].cpu().numpy(), prob[test_mask].cpu().numpy()).pr_auc


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 4),
        "per_seed": [round(v, 4) for v in values],
    }


def run_decompose(
    data: Data,
    timesteps: NDArray[Any],
    model_name: str = "sage",
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    val_start: int = 30,
) -> dict[str, Any]:
    """Run the four-arm decomposition across seeds; return components (PR-AUC)."""
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask
    y = data.y
    t_prev = float((y[t_test] == 1).float().mean())

    is_gnn = model_name != "xgboost"
    arms: dict[str, list[float]] = {"temporal": [], "random": [], "random_matched": []}
    if is_gnn:
        arms["random_matched_inductive"] = []

    for seed in seeds:
        r_train, r_val, r_test = random_masks(y, seed=seed)
        m_test = match_prevalence(r_test, y, t_prev, seed=seed)
        if is_gnn:
            arms["temporal"].append(
                _fit_with_masks(data, model_name, t_train, t_val, t_test, seed=seed)[0].pr_auc
            )
            arms["random"].append(
                _fit_with_masks(data, model_name, r_train, r_val, r_test, seed=seed)[0].pr_auc
            )
            arms["random_matched"].append(
                _fit_with_masks(data, model_name, r_train, r_val, m_test, seed=seed)[0].pr_auc
            )
            arms["random_matched_inductive"].append(
                _fit_inductive(data, model_name, r_train, r_val, m_test, seed=seed)
            )
        else:
            arms["temporal"].append(_xgb_with_masks(data, t_train, t_val, t_test, seed=seed).pr_auc)
            arms["random"].append(_xgb_with_masks(data, r_train, r_val, r_test, seed=seed).pr_auc)
            arms["random_matched"].append(
                _xgb_with_masks(data, r_train, r_val, m_test, seed=seed).pr_auc
            )

    out: dict[str, Any] = {
        "model": model_name,
        "seeds": list(seeds),
        "temporal_test_prevalence": round(t_prev, 4),
        "arms": {k: _mean_std(v) for k, v in arms.items()},
    }
    base_rate = [r - m for r, m in zip(arms["random"], arms["random_matched"], strict=True)]
    components: dict[str, Any] = {"base_rate": _mean_std(base_rate)}
    if is_gnn:
        mp_leak = [
            m - i
            for m, i in zip(arms["random_matched"], arms["random_matched_inductive"], strict=True)
        ]
        dist_access = [
            i - t for i, t in zip(arms["random_matched_inductive"], arms["temporal"], strict=True)
        ]
        components["message_passing_leak"] = _mean_std(mp_leak)
        components["distribution_access"] = _mean_std(dist_access)
    else:
        dist_access = [m - t for m, t in zip(arms["random_matched"], arms["temporal"], strict=True)]
        components["distribution_access"] = _mean_std(dist_access)
    total = [r - t for r, t in zip(arms["random"], arms["temporal"], strict=True)]
    components["total_gap"] = _mean_std(total)
    out["components"] = components
    return out


def run_inductive_temporal(
    data: Data,
    timesteps: NDArray[Any],
    models: tuple[str, ...] = ("gcn", "sage", "gat", "transformer"),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    val_start: int = 30,
) -> dict[str, Any]:
    """Measure graph-visibility under the TEMPORAL protocol (paired, per seed).

    Under a temporal split, test-period nodes are the model's only access to the
    shifted test distribution, so the transductive-vs-inductive contrast measures
    whether graph visibility of the (unlabeled) test period carries usable
    distributional information - the reading suggested by the random-split null
    versus the large transductive-vs-inductive gap reported by concurrent work.

    - **transductive**: the temporal arm as in the main experiments (full graph
      during training; loss on labeled train nodes only).
    - **inductive**: all test-period nodes removed from the graph during training
      (same as Arm I, but with temporal masks); full-graph inference at test time.

    ``visibility_effect`` = transductive - inductive, per seed.
    """
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask

    out: dict[str, Any] = {"seeds": list(seeds), "models": {}}
    for model_name in models:
        trans: list[float] = []
        induc: list[float] = []
        for seed in seeds:
            trans.append(
                _fit_with_masks(data, model_name, t_train, t_val, t_test, seed=seed)[0].pr_auc
            )
            induc.append(_fit_inductive(data, model_name, t_train, t_val, t_test, seed=seed))
        effect = [t - i for t, i in zip(trans, induc, strict=True)]
        out["models"][model_name] = {
            "transductive": _mean_std(trans),
            "inductive": _mean_std(induc),
            "visibility_effect": _mean_std(effect),
        }
    return out
