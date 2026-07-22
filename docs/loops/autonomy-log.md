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
- Docs: `README.md`, `CLAUDE.md`, `LEARNING_NOTES.md`, `data/DATA_CARD.md`,
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
