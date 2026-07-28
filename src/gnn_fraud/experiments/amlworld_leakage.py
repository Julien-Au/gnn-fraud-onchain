"""Leakage experiment on AMLworld (tabular XGBoost): third dataset, model-agnostic.

Temporal split: earliest 60% of transactions (by timestamp) train, next 10%
validation, final 30% test. Random split: stratified with identical proportions.
The interesting context: AMLworld is synthetic; whether its laundering distribution
drifts over time is measured (per-window prevalence) and reported alongside, since
the distribution-access mechanism predicts inflation only where there is drift.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary


def _splits_temporal(
    t: NDArray[Any], train_q: float = 0.6, val_q: float = 0.7
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    q1, q2 = np.quantile(t, [train_q, val_q])
    return t <= q1, (t > q1) & (t <= q2), t > q2


def _splits_random(
    y: NDArray[Any], seed: int, train_q: float = 0.6, val_q: float = 0.7
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for cls in (0, 1):  # stratified
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(len(idx) * train_q)
        n_va = int(len(idx) * (val_q - train_q))
        train[idx[:n_tr]] = True
        val[idx[n_tr : n_tr + n_va]] = True
        test[idx[n_tr + n_va :]] = True
    return train, val, test


def _fit_eval(
    x: NDArray[Any],
    y: NDArray[Any],
    tr: NDArray[Any],
    va: NDArray[Any],
    te: NDArray[Any],
    seed: int,
) -> dict[str, float]:
    from xgboost import XGBClassifier

    pos, neg = int(y[tr].sum()), int((y[tr] == 0).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(neg / pos) if pos else 1.0,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=8,
    )
    model.fit(x[tr], y[tr])
    s_va = model.predict_proba(x[va])[:, 1]
    s_te = model.predict_proba(x[te])[:, 1]
    thr = best_f1_threshold(y[va], s_va)
    m = evaluate_binary(y[te], s_te, thr)
    return {"pr_auc": round(m.pr_auc, 4), "f1": round(m.f1, 4), "roc_auc": round(m.roc_auc, 4)}


def run_amlworld_leakage(
    x: NDArray[Any],
    y: NDArray[Any],
    t: NDArray[Any],
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Temporal vs random split on AMLworld transactions (XGBoost, multi-seed)."""
    tr_t, va_t, te_t = _splits_temporal(t)
    # Drift context: laundering prevalence per temporal segment.
    prevalence = {
        "train_window": round(float(y[tr_t].mean()), 5),
        "val_window": round(float(y[va_t].mean()), 5),
        "test_window": round(float(y[te_t].mean()), 5),
    }

    arms: dict[str, list[dict[str, float]]] = {"temporal": [], "random": []}
    for seed in seeds:
        tr_r, va_r, te_r = _splits_random(y, seed)
        arms["temporal"].append(_fit_eval(x, y, tr_t, va_t, te_t, seed))
        arms["random"].append(_fit_eval(x, y, tr_r, va_r, te_r, seed))

    def agg(key: str, rows: list[dict[str, float]]) -> dict[str, Any]:
        vals = [r[key] for r in rows]
        return {
            "mean": round(statistics.mean(vals), 4),
            "sstd": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "per_seed": vals,
        }

    out: dict[str, Any] = {"seeds": list(seeds), "prevalence": prevalence}
    for arm, rows in arms.items():
        out[arm] = {k: agg(k, rows) for k in ("pr_auc", "f1", "roc_auc")}
    infl = [
        r["pr_auc"] - t_["pr_auc"] for r, t_ in zip(arms["random"], arms["temporal"], strict=True)
    ]
    out["inflation_pr_auc"] = {
        "mean": round(statistics.mean(infl), 4),
        "sstd": round(statistics.stdev(infl) if len(infl) > 1 else 0.0, 4),
        "per_seed": [round(v, 4) for v in infl],
    }
    return out
