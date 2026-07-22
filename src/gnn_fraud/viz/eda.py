"""Exploratory data analysis plots for the Elliptic graph.

Figures are written to ``docs/media/eda`` (committed) so the README can show
them. All numbers come from the real dataset; nothing here is illustrative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import degree

from gnn_fraud.ingestion.elliptic import GraphStats

# Elliptic spans 49 time steps, indexed 1..49. In the raw CSV the time step is
# column 1 of the (header-less) features file; the class file uses "1"=illicit,
# "2"=licit, "unknown".
NUM_TIMESTEPS = 49


def temporal_label_counts(
    raw_dir: str | Path = "data/raw/elliptic/raw",
) -> tuple[list[int], list[int], list[int]]:
    """Return (timesteps, licit_counts, illicit_counts) per time step.

    Reads the raw Elliptic CSVs PyG already downloaded and groups by time step.
    Fast and deterministic (vs. re-instantiating 49 temporal snapshots). This is
    what reveals the dataset's temporal structure, including the sharp drop in
    illicit activity after a dark-market shutdown around step ~43.
    """
    raw = Path(raw_dir)
    feats = pd.read_csv(
        raw / "elliptic_txs_features.csv", header=None, usecols=[0, 1], names=["txId", "time"]
    )
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv")  # columns: txId, class
    df = feats.merge(classes, on="txId", how="left")

    steps: list[int] = []
    licit: list[int] = []
    illicit: list[int] = []
    for t in range(1, NUM_TIMESTEPS + 1):
        sub = df.loc[df["time"] == t, "class"]
        steps.append(t)
        licit.append(int((sub == "2").sum()))
        illicit.append(int((sub == "1").sum()))
    return steps, licit, illicit


def plot_class_distribution(stats: GraphStats, path: str | Path) -> None:
    """Bar chart of licit / illicit / unknown counts (log scale)."""
    names = list(stats.class_counts.keys())
    counts = [stats.class_counts[n] for n in names]
    colors = {"licit": "#2a9d8f", "illicit": "#e76f51", "unknown": "#adb5bd"}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, counts, color=[colors.get(n, "#4361ee") for n in names])
    ax.set_yscale("log")
    ax.set_ylabel("node count (log scale)")
    ax.set_title("Elliptic: class distribution (heavy imbalance)")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_degree_distribution(data: Data, path: str | Path) -> None:
    """Log-log degree distribution (undirected view)."""
    num_nodes = int(data.num_nodes)
    deg = degree(data.edge_index[0], num_nodes=num_nodes) + degree(
        data.edge_index[1], num_nodes=num_nodes
    )
    deg_np = deg.cpu().numpy().astype(int)
    max_deg = int(deg_np.max())
    bins = np.arange(0, max_deg + 2)
    hist, _ = np.histogram(deg_np, bins=bins)
    nonzero = hist > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(bins[:-1][nonzero], hist[nonzero], s=10, color="#4361ee")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (in + out)")
    ax.set_ylabel("number of nodes")
    ax.set_title("Elliptic: degree distribution (heavy-tailed)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_temporal(steps: list[int], licit: list[int], illicit: list[int], path: str | Path) -> None:
    """Labeled licit vs illicit counts per time step."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, licit, marker="o", ms=3, color="#2a9d8f", label="licit")
    ax.plot(steps, illicit, marker="o", ms=3, color="#e76f51", label="illicit")
    ax.axvline(34.5, ls="--", lw=1, color="#6c757d")
    ax.text(34.6, ax.get_ylim()[1] * 0.7, "train | test split", fontsize=8, color="#6c757d")
    ax.set_xlabel("time step")
    ax.set_ylabel("labeled node count")
    ax.set_title("Elliptic: labeled activity over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def temporal_class_balance(data: Data) -> dict[str, dict[str, int]]:
    """Illicit/licit counts within the built-in train vs test masks.

    Confirms the split is temporal and that both sides keep positives.
    """
    out: dict[str, dict[str, int]] = {}
    for name in ("train_mask", "test_mask"):
        mask = getattr(data, name)
        y = data.y[mask]
        out[name] = {
            "licit": int((y == 0).sum()),
            "illicit": int((y == 1).sum()),
        }
    return out
