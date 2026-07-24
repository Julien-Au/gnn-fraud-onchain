"""Plots for experiment results (baselines, and later GNNs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def plot_baseline_metrics(
    rows: dict[str, dict[str, float]],
    path: str | Path,
    title: str = "Non-graph baselines on Elliptic (temporal test)",
) -> None:
    """Grouped bar chart of PR-AUC and minority F1 per model."""
    models = list(rows.keys())
    pr = [float(rows[m]["pr_auc"]) for m in models]
    f1 = [float(rows[m]["f1"]) for m in models]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models)), 4))
    ax.bar(x - width / 2, pr, width, label="PR-AUC", color="#4361ee")
    ax.bar(x + width / 2, f1, width, label="F1 (illicit)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    for i, (p, f) in enumerate(zip(pr, f1, strict=False)):
        ax.text(i - width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, f, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_leakage_multi(results: dict[str, dict[str, dict[str, float]]], path: str | Path) -> None:
    """Grouped bars: temporal vs random PR-AUC per model (leakage inflation)."""
    models = list(results.keys())
    temporal = [float(results[m]["temporal"]["pr_auc"]) for m in models]
    random_ = [float(results[m]["random"]["pr_auc"]) for m in models]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(models)), 4))
    ax.bar(x - width / 2, temporal, width, label="temporal (honest)", color="#2a9d8f")
    ax.bar(x + width / 2, random_, width, label="random (leaky)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.set_title("SOTA inflation by future leakage (same model, two splits)")
    ax.legend()
    for i, (t, r) in enumerate(zip(temporal, random_, strict=False)):
        ax.text(i - width / 2, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_temporal_backtest(
    steps: list[int], praucs: list[float], path: str | Path, aggregate: float | None = None
) -> None:
    """PR-AUC per test time step - shows how a temporal model degrades over the horizon."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, praucs, marker="o", ms=4, color="#4361ee", label="per-step PR-AUC")
    if aggregate is not None:
        ax.axhline(aggregate, ls="--", lw=1, color="#e76f51", label=f"aggregate {aggregate:.3f}")
    ax.axvline(43, ls=":", lw=1, color="#6c757d")
    ax.text(43.2, 0.02, "dark-market shutdown ~43", fontsize=8, color="#6c757d")
    ax.set_xlabel("test time step")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("EvolveGCN-O: rolling per-time-step test PR-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
