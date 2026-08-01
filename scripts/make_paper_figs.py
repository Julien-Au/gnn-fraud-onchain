"""Generate the paper's figures from docs/results/*.json (no in-figure titles:
the LaTeX captions carry the description). Run: uv run python scripts/make_paper_figs.py"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# --- Figure 1: per-timestep rolling backtest -------------------------------
bt = json.loads((ROOT / "docs/results/temporal_backtest.json").read_text())
steps = [r["t"] for r in bt["per_timestep"]]
prs = [r["pr_auc"] for r in bt["per_timestep"]]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.plot(steps, prs, marker="o", ms=4, color="#4361ee")
ax.axvline(43, ls=":", lw=1, color="#6c757d")
ax.text(43.15, 0.30, "dark-market\nshutdown", fontsize=8, color="#6c757d")
ax.set_xlabel("test time step")
ax.set_ylabel("PR-AUC")
ax.set_ylim(0, 0.42)
fig.tight_layout()
fig.savefig(FIGS / "backtest.pdf")
plt.close(fig)

# --- Figure 2: gap decomposition (bars = mean, dots = seeds) ---------------
dc = json.loads((ROOT / "docs/results/decompose.json").read_text())
tf_path = ROOT / "docs/results/decompose_transformer.json"
if tf_path.exists():
    dc.update(json.loads(tf_path.read_text()))
comps = ["base_rate", "message_passing_leak", "distribution_access"]
labels = {
    "base_rate": "base rate",
    "message_passing_leak": "MP leakage",
    "distribution_access": "distribution\naccess",
}
models = ["gcn", "sage", "gat", "transformer", "xgboost"]
colors = {
    "gcn": "#e76f51",
    "sage": "#4361ee",
    "gat": "#2a9d8f",
    "transformer": "#e9a820",
    "xgboost": "#7d5ba6",
}
present = [m for m in models if m in dc]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
width = 0.8 / max(len(present), 1)
x = np.arange(len(comps))
for i, m in enumerate(present):
    offs = (i - (len(present) - 1) / 2) * width
    for j, c in enumerate(comps):
        if c not in dc[m]["components"]:
            continue
        seeds = dc[m]["components"][c]["per_seed"]
        mean = dc[m]["components"][c]["mean"]
        ax.bar(x[j] + offs, mean, width * 0.85, color=colors[m], alpha=0.45,
               label=m if j == 0 else None)
        ax.scatter([x[j] + offs] * len(seeds), seeds, s=12, color=colors[m], zorder=3)
ax.axhline(0, lw=0.8, color="black")
ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in comps])
ax.set_ylabel("PR-AUC gap component")
ax.legend(fontsize=8, ncol=5)
fig.tight_layout()
fig.savefig(FIGS / "decompose.pdf")
plt.close(fig)

print("figures regenerated (no in-figure titles)")
