"""Recompute the statistical summary (docs/results/stats_summary.json) from the
per-seed arrays. t critical values keyed by DEGREES OF FREEDOM (df = n-1).
Run: uv run python scripts/compute_stats.py"""

from __future__ import annotations

import json
import math
import statistics
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "docs/results"

# two-sided 95% t critical values, keyed by df
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
# two-sided 90% (for TOST), keyed by df
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132}


def sstd(v: list[float]) -> float:
    return statistics.stdev(v) if len(v) > 1 else 0.0


def ci(v: list[float], table: dict[int, float]) -> list[float]:
    n = len(v)
    m = statistics.mean(v)
    h = table[n - 1] * sstd(v) / math.sqrt(n) if n > 1 else 0.0
    return [round(m - h, 4), round(m + h, 4)]


def p_sign(v: list[float]) -> float:
    n = len(v)
    tobs = statistics.mean(v)
    cnt = sum(
        1 for s in product([1, -1], repeat=n)
        if statistics.mean([a * b for a, b in zip(s, v)]) >= tobs - 1e-12
    )
    return cnt / 2 ** n


def agg(v: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(v), 4),
        "sstd": round(sstd(v), 4),
        "ci95": ci(v, T95),
        "ci90_tost": ci(v, T90),
        "per_seed": [round(x, 4) for x in v],
    }


out: dict = {"note": "sample std (ddof=1); t critical values at df=n-1"}
lm = json.loads((R / "leakage_multi.json").read_text())
tf_lm = R / "leakage_transformer.json"
if tf_lm.exists():
    lm["transformer"] = json.loads(tf_lm.read_text())["transformer"]
out["leakage_multi"] = {}
for m, r in lm.items():
    pr = r["pr_auc"]
    inf = pr["inflation"]["per_seed"]
    out["leakage_multi"][m] = {
        "temporal": agg(pr["temporal"]["per_seed"]),
        "random": agg(pr["random"]["per_seed"]),
        "inflation": {**agg(inf), "p_sign_flip_gt0": p_sign(inf)},
    }
dc = json.loads((R / "decompose.json").read_text())
tf_dc = R / "decompose_transformer.json"
if tf_dc.exists():
    dc.update(json.loads(tf_dc.read_text()))
out["decompose"] = {
    m: {k: agg(c["per_seed"]) for k, c in r["components"].items()} for m, r in dc.items()
}
it_path = R / "inductive_temporal.json"
if it_path.exists():
    it = json.loads(it_path.read_text())
    out["inductive_temporal"] = {
        m: {k: agg(v["per_seed"]) for k, v in r.items()} for m, r in it["models"].items()
    }
sage_inf = lm["sage"]["pr_auc"]["inflation"]["per_seed"]
xgb_inf = lm["xgboost"]["pr_auc"]["inflation"]["per_seed"]
d = [a - b for a, b in zip(sage_inf, xgb_inf)]
out["sage_minus_xgb_inflation"] = {**agg(d), "p_sign_flip_gt0": p_sign(d)}

# Windowed analyses: raw arms + prevalence-normalized log-ratio test per model.
out["windowed"] = {}
for name, path in [("sage", "leakage_windowed.json"), ("xgboost", "leakage_windowed_xgb.json")]:
    wp = R / path
    if not wp.exists():
        continue
    w = json.loads(wp.read_text())
    block: dict = {"prevalence": w["prevalence"]}
    for win in ("pre_shutdown", "post_shutdown"):
        block[win] = {k: agg(w[win][k]["per_seed"]) for k in ("temporal", "random", "inflation")}
    norm = {
        (win, arm): [
            (v - w["prevalence"][win]) / (1 - w["prevalence"][win])
            for v in w[win][arm]["per_seed"]
        ]
        for win in ("pre_shutdown", "post_shutdown")
        for arm in ("temporal", "random")
    }
    lr = [
        math.log(norm[("post_shutdown", "random")][i] / norm[("post_shutdown", "temporal")][i])
        - math.log(norm[("pre_shutdown", "random")][i] / norm[("pre_shutdown", "temporal")][i])
        for i in range(len(w["seeds"]))
    ]
    block["normalized_logratio_post_minus_pre"] = {**agg(lr), "p_sign_flip_gt0": p_sign(lr)}
    abs_d = [
        b - a
        for a, b in zip(
            w["pre_shutdown"]["inflation"]["per_seed"], w["post_shutdown"]["inflation"]["per_seed"]
        )
    ]
    block["absolute_inflation_post_minus_pre"] = {**agg(abs_d), "p_sign_flip_gt0": p_sign(abs_d)}
    out["windowed"][name] = block
het = []
for f in ("leakage_hetero_pre_split.json", "leakage_hetero_s43_pre_split.json",
          "leakage_hetero_s44_pre_split.json"):
    h = json.loads((R / f).read_text())
    het.append((h["temporal"]["pr_auc"], h["random"]["pr_auc"]))
out["hetero_pre_split"] = {
    "temporal": agg([a for a, _ in het]),
    "random": agg([b for _, b in het]),
    "inflation": agg([b - a for a, b in het]),
}
(R / "stats_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("SAGE inflation CI95:", out["leakage_multi"]["sage"]["inflation"]["ci95"])
print("SAGE-XGB diff CI95:", out["sage_minus_xgb_inflation"]["ci95"])
for m in ("gcn", "sage", "gat"):
    print(f"{m} MP-leak TOST90:", out["decompose"][m]["message_passing_leak"]["ci90_tost"])
print("hetero inflation CI95:", out["hetero_pre_split"]["inflation"]["ci95"])
