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
def backtest_temporal(
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/temporal_backtest.json",
    epochs: int = 200,
    val_start: int = 33,
    seed: int = 42,
) -> None:
    """Rolling temporal backtest of EvolveGCN-O: PR-AUC per test time step.

    Gives EvolveGCN a fuller training budget (--val-start closer to 34) and reports
    a per-time-step breakdown, so the temporal degradation is visible, not hidden in
    one aggregate number. Requires the ``gnn`` extra.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.temporal_trainer import fit_evolvegcn, per_timestep_prauc
    from gnn_fraud.viz.results import plot_temporal_backtest

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    console.print(f"[bold]Training EvolveGCN-O (rolling, val_start={val_start})...[/bold]")
    model, snapshots, outcome = fit_evolvegcn(
        data, timesteps, epochs=epochs, seed=seed, val_start=val_start
    )
    per_step = per_timestep_prauc(model, snapshots, 35, 49)

    console.print(
        f"Aggregate test PR-AUC={outcome.metrics.pr_auc:.4f} "
        f"(best epoch {outcome.best_epoch}, val PR-AUC {outcome.best_val_pr_auc:.4f})"
    )
    table = Table(title="EvolveGCN-O rolling backtest (per test time step)")
    for col in ("time step", "illicit", "PR-AUC"):
        table.add_column(col, justify="right")
    for t, n_ill, pr in per_step:
        table.add_row(str(t), str(n_ill), f"{pr:.4f}")
    console.print(table)

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": outcome.metrics.as_row(),
        "per_timestep": [{"t": t, "illicit": n, "pr_auc": round(pr, 4)} for t, n, pr in per_step],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig_path = Path("docs/media/results/temporal_backtest.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_temporal_backtest(
        [t for t, _, _ in per_step],
        [pr for _, _, pr in per_step],
        fig_path,
        aggregate=outcome.metrics.pr_auc,
    )
    console.print(f"Saved backtest to {out_path} and figure to {fig_path}")


@app.command()
def train_hetero(
    root: str = "data/raw/elliptic_pp",
    cache: str = "data/processed/elliptic_pp.pt",
    results_path: str = "docs/results/hetero.json",
    target: str = "tx",
    epochs: int = 200,
    seed: int = 42,
) -> None:
    """Train the heterogeneous GNN on Elliptic++ (--target tx or addr).

    Requires the ``gnn`` extra and the downloaded Elliptic++ data.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic_pp import build_hetero
    from gnn_fraud.train.hetero_trainer import train_hetero as run_hetero
    from gnn_fraud.viz.results import plot_baseline_metrics

    label = f"hetero-sage-{target}"
    console.print("[bold]Building Elliptic++ HeteroData (cached after first run)...[/bold]")
    data = build_hetero(root, cache)
    console.print(f"[bold]Training heterogeneous GNN (target={target})...[/bold]")
    out = run_hetero(data, epochs=epochs, seed=seed, target=target)
    console.print(
        f"  {label}: PR-AUC={out.metrics.pr_auc:.4f} F1={out.metrics.f1:.4f} "
        f"(best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
    )

    r = out.metrics.as_row()
    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({label: r}, indent=2) + "\n", encoding="utf-8")

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
def train_graph_usad(
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/graph_usad.json",
    epochs: int = 100,
    seed: int = 42,
    normal_recent: int = -1,
) -> None:
    """Train GraphUSAD (research: USAD-style adversarial graph autoencoder).

    Unsupervised on licit nodes; evaluates illicit detection on the temporal test.
    Pass --normal-recent N for the drift-aware v2 (normal = latest N train steps).
    Requires the ``gnn`` extra.
    """
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.graph_usad_trainer import train_graph_usad as run_usad

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    recent = None if normal_recent < 0 else normal_recent
    tag = "v2 drift-aware" if recent else "v1"
    console.print(f"[bold]Training GraphUSAD ({tag}, unsupervised, adversarial)...[/bold]")
    out = run_usad(data, timesteps, epochs=epochs, seed=seed, normal_recent_steps=recent)
    console.print(
        f"  graph-usad: PR-AUC={out.metrics.pr_auc:.4f} ROC-AUC={out.metrics.roc_auc:.4f} "
        f"F1={out.metrics.f1:.4f} (best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
    )
    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"graph-usad": out.metrics.as_row()}, indent=2) + "\n", encoding="utf-8"
    )
    console.print(f"Saved result to {out_path}")


@app.command()
def leakage(
    root: str = "data/raw/elliptic",
    model: str = "sage",
    results_path: str = "docs/results/leakage.json",
    seed: int = 42,
) -> None:
    """Demonstrate SOTA inflation: same GNN under the honest temporal vs a leaky random split.

    Requires the ``gnn`` extra.
    """
    import json
    from pathlib import Path

    from gnn_fraud.experiments.leakage import run_leakage_experiment
    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.viz.results import plot_baseline_metrics

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    console.print(f"[bold]Leakage experiment ({model}): temporal vs random split...[/bold]")
    res = run_leakage_experiment(data, timesteps, model, seed=seed)

    table = Table(title=f"Same {model}: honest temporal vs leaky random split")
    for col in ("split", "PR-AUC", "ROC-AUC", "F1", "precision", "recall"):
        table.add_column(col, justify="right" if col != "split" else "left")
    for name in ("temporal", "random"):
        r = res[name]
        label = "temporal (honest)" if name == "temporal" else "random (leaky)"
        table.add_row(
            label,
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
        )
    console.print(table)
    lift = res["random"]["pr_auc"] - res["temporal"]["pr_auc"]
    console.print(
        f"[bold red]Leakage inflation: +{lift:.3f} PR-AUC from the random split alone.[/bold red]"
    )

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    fig_path = Path("docs/media/results/leakage.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_baseline_metrics(
        {"temporal (honest)": res["temporal"], "random (leaky)": res["random"]},
        fig_path,
        title=f"SOTA inflation by future leakage (same {model})",
    )
    console.print(f"Saved to {out_path} and {fig_path}")


@app.command()
def leakage_multi(
    root: str = "data/raw/elliptic",
    models: str = "gcn,sage,gat",
    results_path: str = "docs/results/leakage_multi.json",
    seed: int = 42,
) -> None:
    """Multi-model leakage table: each model under honest temporal vs leaky random split.

    Requires the ``gnn`` extra (``boost`` adds the XGBoost row).
    """
    import json
    from pathlib import Path

    from gnn_fraud.experiments.leakage import run_leakage_multi
    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.viz.results import plot_leakage_multi

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    model_list = tuple(m.strip() for m in models.split(",") if m.strip())
    console.print("[bold]Multi-model leakage: temporal vs random split...[/bold]")
    res = run_leakage_multi(data, timesteps, models=model_list, seed=seed)

    table = Table(title="PR-AUC: honest temporal vs leaky random split (same model)")
    for col in ("model", "temporal", "random", "inflation"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for m, r in res.items():
        t, rd = float(r["temporal"]["pr_auc"]), float(r["random"]["pr_auc"])
        table.add_row(m, f"{t:.4f}", f"{rd:.4f}", f"+{rd - t:.4f}")
    console.print(table)

    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    fig_path = Path("docs/media/results/leakage_multi.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_leakage_multi(res, fig_path)
    console.print(f"Saved to {out_path} and {fig_path}")


@app.command()
def train_graph_usad_dann(
    root: str = "data/raw/elliptic",
    results_path: str = "docs/results/graph_usad_dann.json",
    epochs: int = 100,
    gamma: float = 1.0,
    seed: int = 42,
) -> None:
    """GraphUSAD v3: domain-adversarial (time-invariant) reconstruction. Requires ``gnn``."""
    import json
    from pathlib import Path

    from gnn_fraud.ingestion.elliptic import load_elliptic, node_timesteps
    from gnn_fraud.train.graph_usad_dann_trainer import train_graph_usad_dann as run_dann

    data = load_elliptic(root)
    timesteps = node_timesteps(root)
    console.print("[bold]Training GraphUSAD v3 (domain-adversarial, time-invariant)...[/bold]")
    out = run_dann(data, timesteps, epochs=epochs, gamma=gamma, seed=seed)
    console.print(
        f"  graph-usad-dann: PR-AUC={out.metrics.pr_auc:.4f} ROC-AUC={out.metrics.roc_auc:.4f} "
        f"F1={out.metrics.f1:.4f} (best epoch {out.best_epoch}, val PR-AUC {out.best_val_pr_auc:.4f})"
    )
    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"graph-usad-dann": out.metrics.as_row()}, indent=2) + "\n", encoding="utf-8"
    )
    console.print(f"Saved result to {out_path}")


@app.command()
def leakage_hetero(
    root: str = "data/raw/elliptic_pp",
    cache: str = "data/processed/elliptic_pp.pt",
    target: str = "addr",
    results_path: str = "docs/results/leakage_hetero.json",
    seed: int = 42,
) -> None:
    """Leakage on the Elliptic++ hetero task (--target addr|tx): temporal vs random split.

    Requires the ``gnn`` extra and the Elliptic++ data.
    """
    import json
    from pathlib import Path

    from gnn_fraud.experiments.leakage import run_hetero_leakage
    from gnn_fraud.ingestion.elliptic_pp import build_hetero

    data = build_hetero(root, cache)
    console.print(f"[bold]Elliptic++ leakage ({target}): temporal vs random...[/bold]")
    res = run_hetero_leakage(data, target=target, seed=seed)
    t, r = float(res["temporal"]["pr_auc"]), float(res["random"]["pr_auc"])
    table = Table(title=f"Elliptic++ {target}: honest temporal vs leaky random split")
    for col in ("split", "PR-AUC", "F1"):
        table.add_column(col, justify="right" if col != "split" else "left")
    table.add_row("temporal (honest)", f"{t:.4f}", f"{res['temporal']['f1']:.4f}")
    table.add_row("random (leaky)", f"{r:.4f}", f"{res['random']['f1']:.4f}")
    console.print(table)
    console.print(f"[bold red]Inflation: +{r - t:.3f} PR-AUC.[/bold red]")
    out_path = Path(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    console.print(f"Saved to {out_path}")


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
