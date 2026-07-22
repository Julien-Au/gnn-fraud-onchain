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
