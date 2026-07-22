# gnn-fraud-onchain

**Fraud / anomaly detection on on-chain transaction graphs with Graph Neural Networks.**

A portfolio-grade, fully open and reproducible study: from non-graph baselines
(including an unsupervised autoencoder, a nod to [USAD](https://dl.acm.org/doi/10.1145/3394486.3403392))
to GNNs (GCN, GraphSAGE, GAT), a temporal variant, and a **heterogeneous /
relational** formulation aimed at the "one model across many schemas" question.

> Status: **step 0 - scaffolding.** The engineering loop (green-gate + CI +
> agent playbook) is in place; data, models and results land in the steps below.
> No result is reported until it is real and reproduced. See
> [`docs/loops/`](docs/loops/) for how this repo maintains itself.

## Why this project

Detecting illicit activity on a blockchain is naturally a **graph** problem:
accounts and transactions are nodes, value flows are edges, and fraud hides in
the *structure* (mixing patterns, peeling chains, fan-in/fan-out), not just in
per-node features. This repo learns that structure with GNNs and holds itself to
research-grade rigor: real public data with cited provenance, temporal
train/val/test splits (no leakage), and metrics that respect heavy class
imbalance (PR-AUC, minority-class F1).

## Roadmap

| Step | What | Status |
|------|------|--------|
| 0 | Scaffolding, green-gate, CI, agent playbook | in progress |
| 1 | Ingestion + graph EDA (Elliptic via PyTorch Geometric) | todo |
| 2 | Baselines: LogReg / XGBoost + an unsupervised autoencoder (USAD nod) | todo |
| 3 | GNNs: GCN -> GraphSAGE -> GAT, plus a temporal variant | todo |
| 4 | Heterogeneous graph (Elliptic++) + relational-foundation-model framing | todo |
| 5 | Rigorous evaluation, experiment tracking, demo (CLI / Streamlit) | todo |

## Data

Real, public data only, with provenance and license documented in
[`data/DATA_CARD.md`](data/DATA_CARD.md). The default pipeline uses the
**Elliptic Bitcoin** dataset packaged in PyTorch Geometric (no scraping);
the heterogeneous track uses **Elliptic++**. An optional extension ingests real
Ethereum data via official APIs (Etherscan / BigQuery), rate-limit-respecting,
with secrets kept in `.env` (never committed).

## Quickstart

```bash
# 1. Install uv (https://docs.astral.sh/uv/), then:
uv sync                 # dev environment (lint/typecheck/test tooling)
uv sync --extra gnn     # add torch + torch-geometric (the graph stack)

# 2. Run the green-gate (what CI runs, locally, in seconds):
bash scripts/verify.sh          # lint + typecheck + tests + smoke CLI
bash scripts/verify.sh --full   # also the torch/PyG smoke train

# 3. Explore the CLI:
uv run gnn-fraud info
```

## Reproducibility & engineering

- **Green-gate** (`scripts/verify.sh`): ruff + mypy + pytest + smoke, mirrored by
  GitHub Actions CI. Nothing merges on a red gate. *Fix the code, never the test.*
- **uv** for a locked, reproducible environment; fixed seeds; one YAML per experiment.
- **Agent playbook** in [`docs/loops/`](docs/loops/): how this repo is built and
  maintained with self-verifying Claude Code loops (plan -> implement ->
  adversarial self-review -> note).

## Learning in the open

- [`LEARNING_NOTES.md`](LEARNING_NOTES.md) - interview-ready notes on every concept
  (message passing, over-smoothing, heterogeneous/temporal graphs, imbalanced metrics).
- [`docs/from-usad-to-gnns.md`](docs/from-usad-to-gnns.md) - the narrative from
  temporal autoencoders (USAD, KDD 2020) to GNNs on relational / graph data.

## License

MIT - see [`LICENSE`](LICENSE).
