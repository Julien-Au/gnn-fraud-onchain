"""Plots for experiment results (baselines, and later GNNs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def plot_baseline_metrics(rows: dict[str, dict[str, float]], path: str | Path) -> None:
    """Grouped bar chart of PR-AUC and minority F1 per model."""
    models = list(rows.keys())
    pr = [float(rows[m]["pr_auc"]) for m in models]
    f1 = [float(rows[m]["f1"]) for m in models]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, pr, width, label="PR-AUC", color="#4361ee")
    ax.bar(x + width / 2, f1, width, label="F1 (illicit)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Non-graph baselines on Elliptic (temporal test)")
    ax.legend()
    for i, (p, f) in enumerate(zip(pr, f1, strict=False)):
        ax.text(i - width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, f, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
