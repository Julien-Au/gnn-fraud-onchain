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
| 0 | Scaffolding, green-gate, CI, agent playbook | done |
| 1 | Ingestion + graph EDA (Elliptic via PyTorch Geometric) | done |
| 2 | Baselines: LogReg / XGBoost + an unsupervised autoencoder (USAD nod) | done |
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

## Data at a glance (step 1 EDA)

Measured with `uv run --extra gnn gnn-fraud eda` on the PyG Elliptic build
(numbers, not estimates; see [`data/DATA_CARD.md`](data/DATA_CARD.md)):

- **203,769** transaction nodes, **234,355** edges, 165 features, directed.
- Labels: 42,019 licit / 4,545 illicit / 157,205 unknown -> only 46,564 labeled;
  illicit is **9.76% of labeled, 2.23% of all** (severe imbalance).
- Sparse and heavy-tailed: mean degree 2.30, max 473, zero isolated nodes.
- **49 connected components for 49 time steps**: edges barely cross time, so the
  graph is nearly a disjoint union of per-time-step subgraphs.

| Class imbalance | Degree (heavy-tailed) | Activity over time |
|---|---|---|
| ![classes](docs/media/eda/class_distribution.png) | ![degree](docs/media/eda/degree_distribution.png) | ![temporal](docs/media/eda/temporal.png) |

The temporal panel shows the built-in train/test boundary and the sharp drop in
illicit activity after a dark-market shutdown (~step 43) - exactly the kind of
distribution shift a temporal split must respect and a random split would hide.

## Results so far (step 2 baselines)

Non-graph reference on node features, evaluated on the **temporal** test split
(reproduce with `uv run --extra gnn --extra boost gnn-fraud baselines`; raw
numbers in [`docs/results/baselines.json`](docs/results/baselines.json)):

| Model | PR-AUC | ROC-AUC | F1 (illicit) | Precision | Recall |
|---|---|---|---|---|---|
| Logistic regression | 0.288 | 0.881 | 0.351 | 0.235 | 0.693 |
| **XGBoost** | **0.799** | 0.928 | **0.817** | 0.990 | 0.695 |
| Autoencoder (USAD-style) | 0.038 | 0.213 | 0.122 | 0.065 | 1.000 |

![baselines](docs/media/results/baselines.png)

Reading these honestly:
- **XGBoost is the bar to beat** (PR-AUC 0.80). Note PR-AUC 0.80 vs ROC-AUC 0.93
  for the same model - ROC always looks rosier under imbalance, which is why
  PR-AUC is the primary metric here.
- **The unsupervised autoencoder fails (ROC-AUC 0.21, below random).** Trained on
  licit-only train nodes, it does *not* separate illicit at test time - illicit
  nodes actually reconstruct *better* than test-period licit ones. The most
  likely cause is **temporal distribution shift** (the licit "normal" learned on
  early time steps drifts after the dark-market shutdown), so feature-space
  reconstruction error stops tracking illicitness. This negative result is
  reported as-is; it is exactly the motivation for adding graph structure and
  temporal modeling (steps 3-4). A USAD-style adversarial refinement is a noted
  future variant, not a claim.

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
