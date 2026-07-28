"""Replicate the published high-score recipe exactly, then re-score it honestly.

Papers reporting ~0.98 on Elliptic typically combine two choices: (i) a stratified
RANDOM split and (ii) aggregate metrics (accuracy, weighted/micro F1) dominated by
the ~90% licit class. This experiment reproduces that recipe verbatim with a
generic XGBoost and reports the same numbers side by side with the honest protocol
(temporal split, illicit-class metrics). If the recipe lands at ~0.98 while the
honest protocol lands far lower, the published band is reproduced by protocol
alone - no novel model required.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary
from gnn_fraud.experiments.leakage import random_masks


def _fit_xgb_probs(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> tuple[NDArray[Any], NDArray[Any], float]:
    """Fit XGBoost; return (y_test, test probabilities, val-chosen threshold)."""
    from xgboost import XGBClassifier

    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr, va, te = train_mask.cpu().numpy(), val_mask.cpu().numpy(), test_mask.cpu().numpy()
    scaler = StandardScaler().fit(x[tr])
    pos, neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(neg / pos) if pos else 1.0,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=4,
    )
    xgb.fit(scaler.transform(x[tr]), y[tr])
    s_va = xgb.predict_proba(scaler.transform(x[va]))[:, 1]
    s_te = xgb.predict_proba(scaler.transform(x[te]))[:, 1]
    thr = best_f1_threshold(y[va], s_va)
    return y[te], s_te, float(thr)


def _all_metrics(y_true: NDArray[Any], probs: NDArray[Any], thr: float) -> dict[str, float]:
    """Published-style aggregate metrics plus the honest illicit-class ones."""
    pred = (probs >= thr).astype(int)
    honest = evaluate_binary(y_true, probs, thr)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "weighted_f1": round(float(f1_score(y_true, pred, average="weighted")), 4),
        "micro_f1": round(float(f1_score(y_true, pred, average="micro")), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "illicit_f1": round(honest.f1, 4),
        "pr_auc": round(honest.pr_auc, 4),
    }


def run_recipe(
    data: Data,
    timesteps: NDArray[Any],
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    val_start: int = 30,
) -> dict[str, Any]:
    """Run the published recipe (random split, aggregate metrics) vs the honest protocol."""
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask

    per_metric: dict[str, dict[str, list[float]]] = {"recipe": {}, "honest": {}}
    for seed in seeds:
        r_train, r_val, r_test = random_masks(data.y, seed=seed)
        y_r, p_r, thr_r = _fit_xgb_probs(data, r_train, r_val, r_test, seed)
        y_t, p_t, thr_t = _fit_xgb_probs(data, t_train, t_val, t_test, seed)
        for arm, (yy, pp, tt) in (("recipe", (y_r, p_r, thr_r)), ("honest", (y_t, p_t, thr_t))):
            for k, v in _all_metrics(yy, pp, tt).items():
                per_metric[arm].setdefault(k, []).append(v)

    def agg(vals: list[float]) -> dict[str, Any]:
        return {
            "mean": round(statistics.mean(vals), 4),
            "sstd": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "per_seed": vals,
        }

    return {
        "seeds": list(seeds),
        "recipe_random_split": {k: agg(v) for k, v in per_metric["recipe"].items()},
        "honest_temporal_split": {k: agg(v) for k, v in per_metric["honest"].items()},
    }
