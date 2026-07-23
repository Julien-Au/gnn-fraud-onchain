"""Command-line entry point.

Kept deliberately thin: the CLI wires user intent to library functions, it does
not contain logic. Heavy imports (torch, torch-geometric) are done lazily
*inside* the commands that need them, so ``gnn-fraud info`` - the smoke command
the green-gate runs - stays fast and dependency-light.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from gnn_fraud import __version__

app = typer.Typer(
    add_completion=False,
    help="Fraud / anomaly detection on on-chain transaction graphs with GNNs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def info() -> None:
    """Print package info. Used as the fast smoke check by the green-gate."""
    table = Table(title="gnn-fraud-onchain", show_header=False)
    table.add_row("version", __version__)
    table.add_row("status", "step 0 - scaffolding")
    console.print(table)


@app.command()
def eda(
    root: str = "data/raw/elliptic",
    out: str = "docs/media/eda",
) -> None:
    """Load Elliptic, print real graph statistics, and save EDA figures.

    Requires the ``gnn`` extra. Downloads the dataset on first use.
    """
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import graph_stats, load_elliptic
    from gnn_fraud.viz import eda as eda_viz

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading Elliptic...[/bold]")
    data = load_elliptic(root)
    stats = graph_stats(data)

    table = Table(title="Elliptic graph statistics (real)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("nodes", f"{stats.num_nodes:,}")
    table.add_row("edges", f"{stats.num_edges:,}")
    table.add_row("node features", str(stats.num_node_features))
    table.add_row("directed", str(stats.is_directed))
    for name, count in stats.class_counts.items():
        table.add_row(f"class: {name}", f"{count:,}")
    table.add_row("labeled nodes", f"{stats.labeled_nodes:,}")
    table.add_row("illicit / labeled", f"{stats.illicit_share_of_labeled:.2%}")
    table.add_row("illicit / all", f"{stats.illicit_share_of_all:.2%}")
    table.add_row("mean degree", f"{stats.mean_degree:.2f}")
    table.add_row("max degree", f"{stats.max_degree:,}")
    table.add_row("isolated nodes", f"{stats.isolated_nodes:,}")
    table.add_row("connected components", f"{stats.num_connected_components:,}")
    table.add_row("largest component", f"{stats.largest_component_fraction:.2%}")
    console.print(table)

    balance = eda_viz.temporal_class_balance(data)
    console.print(f"train/test class balance (temporal split): {balance}")

    console.print("[bold]Rendering figures...[/bold]")
    eda_viz.plot_class_distribution(stats, out_dir / "class_distribution.png")
    eda_viz.plot_degree_distribution(data, out_dir / "degree_distribution.png")
    steps, licit, illicit = eda_viz.temporal_label_counts(Path(root) / "raw")
    eda_viz.plot_temporal(steps, licit, illicit, out_dir / "temporal.png")
    console.print(f"Saved figures to {out_dir}/")


@app.command()
def baselines(
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/baselines.json",
    seed: int = 42,
) -> None:
    """Run non-graph baselines and report imbalanced-aware metrics on test.

    Requires the ``gnn`` extra (data loading); ``boost`` adds XGBoost.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic
    from gnn_fraud.models.baselines import run_baselines

    console.print("[bold]Loading Elliptic and running baselines...[/bold]")
    data = load_elliptic(root)
    results = run_baselines(data, seed=seed)

    table = Table(title="Non-graph baselines on Elliptic (temporal test split)")
    for col in ("model", "PR-AUC", "ROC-AUC", "F1", "precision", "recall"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for name, m in results.items():
        r = m.as_row()
        table.add_row(
            name,
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
        )
    console.print(table)

    out = Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: m.as_row() for name, m in results.items()}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    from gnn_fraud.viz.results import plot_baseline_metrics

    fig_path = Path("docs/media/results/baselines.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_baseline_metrics(payload, fig_path)
    console.print(f"Saved results to {out} and figure to {fig_path}")


@app.command()
def train_gnn(
    models: str = "gcn,sage,gat",
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/gnn.json",
    epochs: int = 200,
    seed: int = 42,
) -> None:
    """Train GNNs (GCN/SAGE/GAT) transductively and compare to the baselines.

    Requires the ``gnn`` extra.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.gnn_trainer import train_gnn as run_train_gnn
    from gnn_fraud.viz.results import plot_baseline_metrics

    data = load_elliptic(root)
    timesteps = node_timesteps(root)

    outcomes = {}
    for name in [m.strip() for m in models.split(",") if m.strip()]:
        console.print(f"[bold]Training {name}...[/bold]")
        out = run_train_gnn(data, timesteps, name, epochs=epochs, seed=seed)
        outcomes[name] = out
        console.print(
            f"  {name}: PR-AUC={out.metrics.pr_auc:.4f} F1={out.metrics.f1:.4f} "
            f"(best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
        )

    table = Table(title="GNNs on Elliptic (temporal test split)")
    for col in ("model", "PR-AUC", "ROC-AUC", "F1", "precision", "recall"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for name, out in outcomes.items():
        r = out.metrics.as_row()
        table.add_row(
            name,
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
        )
    console.print(table)

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: out.metrics.as_row() for name, out in outcomes.items()}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Comparison figure: baselines (if present) + GNNs.
    combined: dict[str, dict[str, float]] = {}
    baselines_json = Path("docs/results/baselines.json")
    if baselines_json.exists():
        combined.update(json.loads(baselines_json.read_text(encoding="utf-8")))
    combined.update(payload)
    fig_path = Path("docs/media/results/comparison.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_baseline_metrics(combined, fig_path, title="Elliptic: baselines vs GNNs (temporal test)")
    console.print(f"Saved results to {out_path} and comparison to {fig_path}")


@app.command()
def train_temporal(
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/temporal.json",
    epochs: int = 200,
    seed: int = 42,
) -> None:
    """Train EvolveGCN-O (temporal GNN) and compare to baselines + static GNNs.

    Requires the ``gnn`` extra.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.temporal_trainer import train_evolvegcn
    from gnn_fraud.viz.results import plot_baseline_metrics

    data = load_elliptic(root)
    timesteps = node_timesteps(root)

    console.print("[bold]Training EvolveGCN-O...[/bold]")
    out = train_evolvegcn(data, timesteps, epochs=epochs, seed=seed)
    console.print(
        f"  evolvegcn: PR-AUC={out.metrics.pr_auc:.4f} F1={out.metrics.f1:.4f} "
        f"(best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
    )

    table = Table(title="EvolveGCN-O on Elliptic (temporal test split)")
    for col in ("model", "PR-AUC", "ROC-AUC", "F1", "precision", "recall"):
        table.add_column(col, justify="right" if col != "model" else "left")
    r = out.metrics.as_row()
    table.add_row(
        "evolvegcn",
        f"{r['pr_auc']:.4f}",
        f"{r['roc_auc']:.4f}",
        f"{r['f1']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
    )
    console.print(table)

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"evolvegcn": r}, indent=2) + "\n", encoding="utf-8")

    # Rebuild the full comparison figure: baselines + static GNNs + temporal.
    combined: dict[str, dict[str, float]] = {}
    for p in ("docs/results/baselines.json", "docs/results/gnn.json"):
        fp = Path(p)
        if fp.exists():
            combined.update(json.loads(fp.read_text(encoding="utf-8")))
    combined["evolvegcn"] = r
    fig_path = Path("docs/media/results/comparison.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_baseline_metrics(combined, fig_path, title="Elliptic: baselines vs GNNs (temporal test)")
    console.print(f"Saved results to {out_path} and comparison to {fig_path}")


@app.command()
def train_hetero(
    root: str = "data/raw/elliptic_pp",
    cache: str = "data/processed/elliptic_pp.pt",
    results_path: str = "docs/results/hetero.json",
    epochs: int = 200,
    seed: int = 42,
) -> None:
    """Train the heterogeneous GNN on Elliptic++ (tx classification).

    Requires the ``gnn`` extra and the downloaded Elliptic++ data.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic_pp import build_hetero
    from gnn_fraud.train.hetero_trainer import train_hetero as run_hetero
    from gnn_fraud.viz.results import plot_baseline_metrics

    console.print("[bold]Building Elliptic++ HeteroData (cached after first run)...[/bold]")
    data = build_hetero(root, cache)
    console.print("[bold]Training heterogeneous GNN...[/bold]")
    out = run_hetero(data, epochs=epochs, seed=seed)
    console.print(
        f"  hetero-sage: PR-AUC={out.metrics.pr_auc:.4f} F1={out.metrics.f1:.4f} "
        f"(best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
    )

    r = out.metrics.as_row()
    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hetero-sage": r}, indent=2) + "\n", encoding="utf-8")

    combined: dict[str, dict[str, float]] = {}
    for p in ("docs/results/baselines.json", "docs/results/gnn.json", "docs/results/temporal.json"):
        fp = Path(p)
        if fp.exists():
            combined.update(json.loads(fp.read_text(encoding="utf-8")))
    combined["hetero-sage"] = r
    fig_path = Path("docs/media/results/comparison.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_baseline_metrics(
        combined, fig_path, title="Elliptic(++): baselines vs GNNs (temporal test)"
    )
    console.print(f"Saved results to {out_path} and comparison to {fig_path}")


@app.command()
def demo(
    root: str = "data/raw/elliptic",
    timestep: int = -1,
    epochs: int = 60,
    top_k: int = 10,
    out: str = "docs/media/demo/subgraph.png",
    seed: int = 42,
) -> None:
    """Score a real Elliptic subgraph with a trained GraphSAGE and render it.

    Requires the ``gnn`` extra. Pass --timestep to fix the time step (default: the
    test-period step with the most illicit transactions).
    """
    from gnn_fraud.demo import run_demo

    console.print("[bold]Training GraphSAGE for the demo (~1 min)...[/bold]")
    summary = run_demo(root, None if timestep < 0 else timestep, epochs, top_k, out, seed)
    console.print(
        f"Time step {summary['timestep']}: {summary['num_nodes']:,} transactions, "
        f"{summary['num_illicit']} truly illicit."
    )
    table = Table(title=f"Top {top_k} flagged transactions (time step {summary['timestep']})")
    table.add_column("rank")
    table.add_column("illicit score", justify="right")
    table.add_column("true label")
    for rank, row in enumerate(summary["top"], 1):
        table.add_row(str(rank), f"{row['score']:.4f}", row["true"])
    console.print(table)
    console.print(f"Saved subgraph figure to {summary['figure']}")


@app.command()
def smoke_train() -> None:
    """Run a tiny 1-epoch smoke train (requires the ``gnn`` extra).

    This is a placeholder until step 3 lands the real models; for now it just
    proves the graph stack imports and a forward/backward pass runs on CPU, so
    the ``--full`` gate and the CI ``gnn`` job have something real to verify.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise typer.Exit(code=1) from exc

    torch.manual_seed(0)
    # Minimal, honest smoke: a linear layer takes one optimisation step. Real
    # message-passing models replace this from step 3 onward.
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    layer = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss = torch.nn.functional.mse_loss(layer(x), y)
    loss.backward()
    opt.step()
    console.print(f"smoke train OK (loss={loss.item():.4f})")


if __name__ == "__main__":
    app()
