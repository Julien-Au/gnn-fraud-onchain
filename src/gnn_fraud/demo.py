"""A small demo: score a real Elliptic subgraph with a trained GraphSAGE.

Trains GraphSAGE on Elliptic, then focuses on a test-period time step, extracts a
legible neighborhood around a genuinely illicit transaction, and renders it with
nodes colored by the model's predicted illicit probability (true-illicit nodes
outlined). Also prints the top flagged transactions. Everything is real: the model,
the scores, and the labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import networkx as nx

# Heavy imports (torch, torch-geometric, networkx, matplotlib) are done lazily
# inside the functions that need them, so ``pick_demo_timestep`` (pure numpy) stays
# importable and testable without the gnn extra.

TEST_MIN_TIMESTEP = 35


def pick_demo_timestep(timesteps: NDArray[Any], illicit: NDArray[Any]) -> int:
    """Pick the test-period time step with the most illicit labeled nodes."""
    best_t, best_count = TEST_MIN_TIMESTEP, -1
    for t in np.unique(timesteps):
        if int(t) < TEST_MIN_TIMESTEP:
            continue
        count = int(illicit[timesteps == t].sum())
        if count > best_count:
            best_count, best_t = count, int(t)
    return best_t


def _label(v: int) -> str:
    return {1: "illicit", 0: "licit"}.get(int(v), "unknown")


def _focus_nodes(graph: nx.Graph, seed: int, max_nodes: int) -> list[int]:
    """BFS from ``seed`` until ``max_nodes`` are collected (a legible neighborhood)."""
    seen = [seed]
    frontier = [seed]
    seen_set = {seed}
    while frontier and len(seen) < max_nodes:
        nxt: list[int] = []
        for node in frontier:
            for nb in graph.neighbors(node):
                if nb not in seen_set:
                    seen_set.add(nb)
                    seen.append(nb)
                    nxt.append(nb)
                    if len(seen) >= max_nodes:
                        break
            if len(seen) >= max_nodes:
                break
        frontier = nxt
    return seen


def _render(
    edge_index: NDArray[Any],
    prob: NDArray[Any],
    y: NDArray[Any],
    timestep: int,
    out: str | Path,
    max_nodes: int = 250,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(len(prob)))
    g.add_edges_from(edge_index.T.tolist())

    illicit_nodes = np.where(y == 1)[0]
    seed = (
        int(illicit_nodes[np.argmax(prob[illicit_nodes])])
        if illicit_nodes.size
        else int(np.argmax(prob))
    )
    nodes = _focus_nodes(g, seed, max_nodes)
    h = g.subgraph(nodes)

    pos = nx.spring_layout(h, seed=42)
    node_prob = [float(prob[n]) for n in h.nodes()]
    borders = ["#d00000" if y[n] == 1 else "#00000022" for n in h.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(h, pos, ax=ax, edge_color="#cccccc", width=0.5)
    nodes_art = nx.draw_networkx_nodes(
        h,
        pos,
        ax=ax,
        node_color=node_prob,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        node_size=90,
        edgecolors=borders,
        linewidths=1.5,
    )
    fig.colorbar(nodes_art, ax=ax, label="predicted illicit probability")
    ax.set_title(
        f"Elliptic time step {timestep}: subgraph scored by GraphSAGE\n"
        "(red outline = truly illicit)"
    )
    ax.axis("off")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run_demo(
    root: str = "data/raw/elliptic",
    timestep: int | None = None,
    epochs: int = 60,
    top_k: int = 10,
    out: str = "docs/media/demo/subgraph.png",
    seed: int = 42,
) -> dict[str, Any]:
    """Train GraphSAGE, score a subgraph, render it, and return a summary."""
    import torch
    from torch_geometric.utils import subgraph

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.gnn_trainer import fit_gnn

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    model, _ = fit_gnn(data, timesteps, "sage", epochs=epochs, seed=seed)

    model.eval()
    with torch.no_grad():
        prob = model(data.x, data.edge_index).softmax(dim=1)[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()

    if timestep is None:
        timestep = pick_demo_timestep(timesteps, (y == 1).astype(int))
    mask = timesteps == timestep
    node_ids = np.where(mask)[0]
    sub_ei, _ = subgraph(
        torch.from_numpy(mask), data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    sub_prob = prob[node_ids]
    sub_y = y[node_ids]

    _render(sub_ei.cpu().numpy(), sub_prob, sub_y, timestep, out)

    order = np.argsort(-sub_prob)[:top_k]
    top = [
        {"local_id": int(i), "score": round(float(sub_prob[i]), 4), "true": _label(sub_y[i])}
        for i in order
    ]
    return {
        "timestep": int(timestep),
        "num_nodes": int(node_ids.size),
        "num_illicit": int((sub_y == 1).sum()),
        "figure": str(out),
        "top": top,
    }
