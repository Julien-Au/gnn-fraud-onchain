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
