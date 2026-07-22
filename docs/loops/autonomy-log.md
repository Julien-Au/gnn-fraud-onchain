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
