"""Windowed inflation: a falsifiable prediction of the distribution-access mechanism.

If the temporal-vs-random gap is driven by distribution access (random splits train
on the test period's distribution), the inflation should be LARGEST where the
distribution shift is largest: the post-shutdown window (time steps >= 43), where
the fraud base rate collapses. Within each window the temporal and random arms are
evaluated on nodes of the same period (so window base rates are comparable across
arms); the comparison across windows is of the inflation, not the absolute PR-AUC.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import evaluate_binary
from gnn_fraud.experiments.leakage import _fit_with_masks, random_masks


def _xgb_full_probs(data: Data, train_mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Fit the standard XGBoost recipe on the masked train arm; return
    illicit-class probabilities for every node (windowed evaluation slices them)."""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr = train_mask.cpu().numpy()
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
    return torch.from_numpy(xgb.predict_proba(scaler.transform(x))[:, 1])


def run_windowed_leakage(
    data: Data,
    timesteps: NDArray[Any],
    model_name: str = "sage",
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    val_start: int = 30,
    shutdown: int = 43,
) -> dict[str, Any]:
    """Per-window (pre/post shutdown) PR-AUC for the temporal and random arms."""
    ts = torch.as_tensor(np.asarray(timesteps))
    t_train = data.train_mask & (ts < val_start)
    t_val = data.train_mask & (ts >= val_start)
    t_test = data.test_mask
    y = data.y

    windows = {
        "pre_shutdown": t_test & (ts < shutdown),
        "post_shutdown": t_test & (ts >= shutdown),
    }
    results: dict[str, dict[str, list[float]]] = {
        w: {"temporal": [], "random": [], "inflation": []} for w in windows
    }
    prevalence = {w: round(float((y[m] == 1).float().mean()), 4) for w, m in windows.items()}

    for seed in seeds:
        r_train, r_val, r_test = random_masks(y, seed=seed)
        if model_name == "xgboost":
            p_t = _xgb_full_probs(data, t_train, seed=seed)
            p_r = _xgb_full_probs(data, r_train, seed=seed)
        else:
            probs_t: list[torch.Tensor] = []
            probs_r: list[torch.Tensor] = []
            _fit_with_masks(data, model_name, t_train, t_val, t_test, seed=seed, out_probs=probs_t)
            _fit_with_masks(data, model_name, r_train, r_val, r_test, seed=seed, out_probs=probs_r)
            p_t, p_r = probs_t[0], probs_r[0]
        for w, w_mask in windows.items():
            # temporal arm: all labeled nodes of the window; random arm: the window's
            # nodes that fell into the random test set (unseen by that model).
            m_t = w_mask
            m_r = w_mask & r_test
            pr_t = evaluate_binary(y[m_t].cpu().numpy(), p_t[m_t].cpu().numpy()).pr_auc
            pr_r = evaluate_binary(y[m_r].cpu().numpy(), p_r[m_r].cpu().numpy()).pr_auc
            results[w]["temporal"].append(pr_t)
            results[w]["random"].append(pr_r)
            results[w]["inflation"].append(pr_r - pr_t)

    def agg(vals: list[float]) -> dict[str, Any]:
        return {
            "mean": round(statistics.mean(vals), 4),
            "sstd": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "per_seed": [round(v, 4) for v in vals],
        }

    out: dict[str, Any] = {"model": model_name, "seeds": list(seeds), "prevalence": prevalence}
    for w in windows:
        out[w] = {k: agg(v) for k, v in results[w].items()}
    # The mechanism's prediction: post-shutdown inflation ratio vs pre.
    pre = out["pre_shutdown"]["inflation"]["mean"]
    post = out["post_shutdown"]["inflation"]["mean"]
    out["prediction"] = {
        "pre_inflation": pre,
        "post_inflation": post,
        "post_greater_than_pre": bool(post > pre),
    }
    return out
