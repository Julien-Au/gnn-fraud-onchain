# Autonomy log

Append-only log of what each step / loop iteration did. One entry per increment.
State lives here (and in git), not only in a session.

---

## Step 0 - Scaffolding & workflow

**Plan.** Stand up the engineering loop before any ML: project layout, uv build,
green-gate, CI, agent contract (CLAUDE.md), data card, learning notes, loops
playbook. Prove the gate is green with a trivial-but-real typed module (config
loader + CLI) and tests.

**Implemented.**
- `pyproject.toml` (uv, ruff, mypy strict, pytest; `gnn`/`boost`/`demo` extras).
- `scripts/verify.sh` green-gate + `.github/workflows/ci.yml` (fast job + gnn job).
- `src/gnn_fraud/`: `config.py` (typed `ExperimentConfig`), `cli.py` (`info`,
  `smoke-train`), submodule packages.
- `tests/`: config + CLI tests (7 tests).
- Docs: `README.md`, `CLAUDE.md`, the research log, `data/DATA_CARD.md`,
  `docs/loops/`, `.env.example`, MIT `LICENSE`.

**Self-review (adversarial).** No data or results reported (nothing to fabricate
yet). No secrets. Torch kept out of the fast gate so CI stays honest and fast.
Config is frozen + validated; tests cover the missing-field and roundtrip paths.

**Gate.** `bash scripts/verify.sh` -> green (ruff, mypy, 7 pytest, smoke CLI).

**Next.** Step 1 - ingestion of the Elliptic dataset via PyTorch Geometric +
graph EDA (degrees, components, class imbalance, temporal structure).

---

## Step 1 - Ingestion + graph EDA (Elliptic)

**Plan.** Load Elliptic via PyG, compute real graph statistics, and render EDA
figures. Explain the graph data model and imbalance/temporal structure before
any modeling. Update DATA_CARD with measured (not recited) numbers.

**Implemented.**
- `ingestion/elliptic.py`: `load_elliptic()` + generic, unit-testable
  `graph_stats()` (counts, class balance, degree, isolated, connected components).
- `viz/eda.py`: class-distribution, degree-distribution, and per-time-step
  activity plots + `temporal_class_balance()`.
- `cli.py eda`: prints a real stats table and writes figures to `docs/media/eda/`.
- `tests/test_ingestion.py`: `graph_stats` on a synthetic graph (skipped without
  the gnn extra); gnn tier of the gate/CI now runs pytest with the extra.

**Measured (real).** 203,769 nodes / 234,355 edges / 165 features; 42,019 licit,
4,545 illicit (9.76% of 46,564 labeled), 157,205 unknown; mean degree 2.30,
max 473, 0 isolated; **49 components for 49 time steps** (graph ~ disjoint per
time step). Temporal split keeps positives on both sides (3,462 / 1,083 illicit).

**Self-review (adversarial).** No model results reported. Numbers come from
`gnn-fraud eda`, reproducible. Temporal split is PyG's built-in (early -> train),
never randomized. Replaced a slow 49x temporal-dataset loop with a single pandas
read of the raw CSV (same numbers, deterministic). Hit a real race: running an
`--extra gnn` command and `verify.sh` (which `uv sync` strips the extra)
concurrently uninstalled torch mid-run - do not run those in parallel.

**Gate.** `bash scripts/verify.sh --full` (to run before commit).

**Next.** Step 2 - non-graph baselines: LogReg / XGBoost on node features + an
unsupervised autoencoder (a la USAD), with the temporal split and PR-AUC / F1.

---

## Step 2 - Non-graph baselines

**Plan.** Establish an honest reference without graph structure: logistic
regression, XGBoost, and an unsupervised autoencoder (USAD nod). Temporal split,
scaler fit on train only, threshold chosen on train, PR-AUC primary.

**Implemented.**
- `eval/metrics.py`: `evaluate_binary` (PR-AUC, ROC-AUC, F1/precision/recall,
  confusion) + `best_f1_threshold`; unit-tested in the fast tier.
- `train/split.py`: `labeled_temporal_split` (surfaces PyG's masks as arrays).
- `models/autoencoder.py`: MLP autoencoder trained on normal rows (USAD instinct).
- `models/baselines.py`: `run_baselines` (logreg + optional xgboost + AE).
- `cli.py baselines`: prints metrics, writes `docs/results/baselines.json` and a
  figure. `viz/results.py` renders the bar chart.
- Tests: metrics (fast) + a `run_baselines` smoke on a synthetic graph (gnn tier).
- mypy: narrow override for `models.autoencoder` (subclasses torch `nn.Module`,
  which is Any without torch stubs in the fast gate).

**Measured (real, temporal test).** logreg PR-AUC 0.288 / F1 0.351; xgboost
PR-AUC **0.799** / F1 **0.817**; autoencoder PR-AUC 0.038 / **ROC-AUC 0.213
(below random)**.

**Self-review (adversarial).** Numbers are reproducible via the CLI. The AE's
worse-than-random result is reported as-is and analyzed (temporal distribution
shift), not hidden or inverted with test knowledge. No leakage: scaler and
threshold both fit on train. XGBoost is the honest bar for the GNN to beat.

**Gate.** `bash scripts/verify.sh --full` green (fast + gnn tier tests + smoke).

**Next.** Step 3 - the first GNNs: GCN -> GraphSAGE -> GAT (transductive on the
full graph incl. unknown nodes for message passing), compared against XGBoost,
then a temporal variant.

---

## Step 3 - First GNNs (GCN / GraphSAGE / GAT)

**Plan.** Implement GCN -> GraphSAGE -> GAT (2-layer, transductive), a trainer with
a *temporal* validation slice (latest train steps) and early stopping, compare to
XGBoost. Explain what each architecture changes.

**Implemented.**
- `models/gnn.py`: GCN, GraphSAGE, GAT + `build_model` factory.
- `ingestion/elliptic.py`: `node_timesteps()` (aligned to node order, asserted).
- `train/gnn_trainer.py`: transductive full-batch training, temporal val split
  (with an alignment assertion vs the built-in masks), class-weighted loss,
  early stopping on val PR-AUC, threshold chosen on val.
- `cli.py train-gnn` + `viz/results.py` comparison chart.
- Tests: model factory, temporal-mask partition, and a 3-model train smoke (gnn
  tier). mypy override extended to `models.gnn`; ruff ignores N812 (torch's `F`).

**Measured (real, temporal test).** GCN PR-AUC 0.294 / SAGE **0.488** / GAT 0.332,
vs XGBoost **0.799**. Vanilla GNNs do not beat the tabular baseline (documented
Elliptic result). Analyzed: aggregated features already in XGBoost's input;
per-time-step disconnection limits the receptive field; temporal shift; low mean
degree makes attention unhelpful.

**Self-review (adversarial).** No leakage: temporal val carved from train only,
threshold chosen on val, test untouched. Negative result reported as-is and
explained, not inflated. Seeds fixed. Alignment of timesteps to node order is
asserted at runtime.

**Compute.** Local machine was saturated by an unrelated user job, so training ran
on a throwaway Hetzner cpx42 box (code rsynced, deps via uv), results retrieved,
and the box destroyed. No standing infrastructure.

**Gate.** Fast gate green locally; gnn-tier tests run in CI.

**Next.** Step 4 - a temporal GNN (EvolveGCN-style) and the heterogeneous graph
(Elliptic++), with the relational-foundation-model framing.

---

## Step 3b - Temporal GNN (EvolveGCN-O)

**Plan.** Address step 3's temporal weakness with EvolveGCN-O, implemented from
scratch (the reference lib pulls compiled torch-sparse and would not install).

**Implemented.**
- `models/temporal.py`: `EvolveGCNO` (GRU-evolved square weight + GCN propagation)
  + `EvolveGCN` (proj -> 2 EvolveGCN-O layers -> classifier). Faithful to Pareja
  et al. 2020 (verified against the reference forward logic via WebFetch).
- `train/temporal_trainer.py`: per-time-step snapshots, sequential BPTT, leakage-safe
  temporal val, early stopping. `cli train-temporal` + comparison figure.
- Tests: weight actually evolves, snapshot coverage, train smoke (gnn tier).

**Measured + diagnosed.** Strict split (train 1-29 / val 30-34 / test 35-49): test
PR-AUC **0.069** (ROC 0.552). Ran a trajectory diagnostic (loss/val/test across LR
and gradient clipping): train loss decreases, val ~0.3, test collapses to ~0.1 in
all configs -> a real generalization failure across the post-shutdown shift, not a
bug. Reported with two caveats: (1) harder task than the transductive static GNNs;
(2) EvolveGCN's canonical protocol is rolling, not one far split (queued as a
fairer experiment).

**Self-review (adversarial).** The number is suspicious (below base rate), so it
was diagnosed before reporting rather than published blind. No leakage (loss never
touches val/test snapshots; weights evolve through them using structure only).
Result reported as-is; the fairness caveats are stated, not used to hide it.

**Gate.** Fast gate green; temporal tests validated locally with the gnn extra.

**Next.** Step 4 - (a) a fair rolling temporal evaluation, and (b) the
heterogeneous Elliptic++ graph (pending the user's green light on the data source;
license is not explicitly stated in the repo - to confirm before download).

---

## Step 4 - Heterogeneous GNN on Elliptic++

**Plan.** Build a HeteroData (tx + addr, four relations) from Elliptic++, train a
HeteroConv GraphSAGE on the same tx-classification task/split, and see whether the
relational structure helps.

**Implemented.**
- `ingestion/elliptic_pp.py`: parse ~2 GB CSVs -> HeteroData (203,769 tx / 822,942
  addr; tx-tx, addr-tx, tx-addr, addr-addr), features cleaned + z-scored, cached.
- `models/hetero.py`: `HeteroGNN` (HeteroConv + per-relation SAGEConv; chose
  HeteroConv over `to_hetero` after hitting a torch.fx tracing bug).
- `train/hetero_trainer.py`: ToUndirected, lazy-param warmup, temporal val,
  early stopping. `cli train-hetero` + comparison figure. Smoke test (gnn tier).

**Measured (real, same tx test split).** heterogeneous SAGE **PR-AUC 0.586** /
F1 0.538, vs tx-only SAGE 0.488 and XGBoost 0.799. **+20% relative over the tx-only
GNN - the biggest graph-driven gain so far.** The relational structure helps.

**Self-review (adversarial).** Comparable by construction (identical tx labels,
features scaling aside, and temporal split). Positive result but not overstated:
still below XGBoost; val >> test gap (temporal shift) reported; address feature
aggregation documented as a modeling choice. Hit a real NaN bug (unnormalized
wallet features) - diagnosed and fixed with cleaning + z-scoring, not silenced.

**Compute.** 2 GB parse locally -> 403 MB cached HeteroData; training was too slow
locally (30-min timeout), so ran on a throwaway Hetzner cpx42 (code + cache
rsynced), retrieved results, destroyed the box.

**Next.** Step 5 - fair rolling temporal eval; address-level classification on the
same graph; the short USAD->GNN write-up and a demo.

---

## Step 5 - Final deliverables (write-up, model cards, demo)

**Plan.** Complete the brief's stated deliverables that use existing results (no new
training runs): the research write-up, model cards, and a demo.

**Implemented.**
- `docs/from-usad-to-gnns.md`: the full narrative from USAD to relational GNNs, with
  the measured results arc and the relational-foundation-model bridge.
- `docs/model-cards.md`: one card per model (intent, inputs, metrics, limitations).
- `demo.py` + `cli demo`: trains GraphSAGE, scores a test-period subgraph around a
  real illicit transaction, renders it (nodes colored by predicted illicit
  probability, true-illicit outlined), prints the top flagged transactions.
  Refactored `gnn_trainer` to expose `fit_gnn` (returns the fitted model) so the
  demo reuses the real training path. Pure helper `pick_demo_timestep` unit-tested.
- README: demo section + figure + links to the write-up, model cards, quiz.

**Real demo output.** Auto-picked time step 42 (7,140 tx, 239 illicit, just before
the dark-market shutdown); rendered `docs/media/demo/subgraph.png`.

**Self-review (adversarial).** No new numbers invented; the write-up/cards quote the
already-measured results. Demo shows real model scores and true labels (top flags
are mostly "unknown", reported honestly, since 77% of nodes are unlabeled).

**Gate.** Fast gate green; demo run produced a real figure.

**Next (optional extras, not required deliverables).** Fair rolling temporal eval
(EvolveGCN's fair shot); address-level classification on the same hetero graph.

---

## Step 5 extras - rolling backtest, address task, Docker

**Implemented.**
- Rolling temporal backtest: `fit_evolvegcn` (exposes the model) + `per_timestep_prauc`
  + `cli backtest-temporal` + `plot_temporal_backtest`. Aggregate PR-AUC 0.100;
  per-step figure shows the collapse at the shutdown (steps 44-46 ~0.01).
- Address-level task: `elliptic_pp` now adds address labels + first-seen temporal
  split; `HeteroGNN`/`train_hetero` take a `target` node type; `cli train-hetero
  --target addr`. Result: PR-AUC 0.456 / F1 0.529 on 92,451 test addresses.
- Docker: `Dockerfile` (uv-based, core image + `--build-arg EXTRAS`), `.dockerignore`,
  CI `docker` job (build + run `gnn-fraud info`). Verified locally.
- Tests: `per_timestep_prauc` and `train_hetero(target="addr")` smoke (gnn tier).

**Self-review (adversarial).** Rolling result reported as-is (still a failure) with
the per-step figure making the reason visible. Address task labeled honestly as a
*different* task (not comparable to the tx numbers). Heavy runs on a throwaway box,
destroyed after (confirmed zero residual servers).

**Gate.** Fast gate green; gnn-tier tests green; Docker image builds and runs.

**Status.** Steps 0-5 plus all three optional extras are done. The repo is a
complete, honest, CI-gated study.
