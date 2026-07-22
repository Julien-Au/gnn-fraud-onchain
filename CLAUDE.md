# CLAUDE.md - working agreement for agents on gnn-fraud-onchain

This project is a portfolio-grade, fully reproducible study of **fraud / anomaly
detection on on-chain transaction graphs with Graph Neural Networks**. This file
tells any Claude Code agent (interactive or inside a loop) how to work here so it
does not have to re-derive conventions.

## What this project is

- **Language**: Python >= 3.10, typed (mypy strict on `src/gnn_fraud`).
- **Env / build**: `uv` (locked, reproducible). Core deps are pure-Python; the
  graph stack (`torch`, `torch-geometric`) is the `gnn` extra so the fast gate
  stays light.
- **ML stack**: PyTorch + PyTorch Geometric. Baselines with scikit-learn / XGBoost.
- **Structure**: `src/gnn_fraud/{ingestion,models,train,eval,viz}`, `tests/`,
  `configs/` (one YAML per experiment), `notebooks/`, `demo/`.

## The green-gate (self-verification - never skip)

Before committing or opening a PR, the change MUST pass:

```bash
bash scripts/verify.sh          # uv sync + ruff + mypy + pytest + smoke CLI (fast, no torch)
bash scripts/verify.sh --full   # also installs the gnn extra + a 1-epoch smoke train
```

CI (`.github/workflows/ci.yml`) runs the same fast gate on every PR, plus a
separate `gnn` job that installs CPU torch/PyG and runs the smoke train.

**Fix the code, never the test.** A red gate is fixed at its cause. Deleting or
skipping a test, loosening an assertion, or silencing an error to get green is
forbidden - it improves the scoreboard, not the code.

## Non-negotiable research integrity

This is the core of the project's credibility (it will face a research jury):

- **Real, public data only.** Every source's provenance and license is documented
  in `data/DATA_CARD.md`. No synthetic or fabricated data unless explicitly labeled
  as a unit-test fixture.
- **No invented results, ever.** Numbers in the README / notes come from a real,
  reproducible run with a fixed seed. If a result is weak, we say so and analyze it.
- **No leakage.** Splits are **temporal** where the data has time (train < val <
  test in time). This is checked, not assumed.
- **Imbalanced-aware metrics.** Report PR-AUC and minority-class F1 with the
  confusion matrix - never accuracy alone.

## Code conventions

- Typed Python; `mypy` strict must pass. Avoid `Any` where avoidable.
- `ruff` lint + format must pass (`ruff format`, not black).
- English-only for code, comments, docstrings, and committed docs.
  (`LEARNING_NOTES.md` may contain the author's study notes; keep them English.)
- **Do not use em-dashes or en-dashes; use a regular hyphen.**
- **Conventional Commits** (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, ...).
- Keep PRs / commits focused; add or update tests with the change.
- Fix random seeds everywhere randomness enters (config `seed`).

## Where things live

- `src/gnn_fraud/ingestion/` - build `Data` / `HeteroData` graphs from sources.
- `src/gnn_fraud/models/` - `gcn`, `sage`, `gat`, temporal, hetero, autoencoder.
- `src/gnn_fraud/train/` - training loop, temporal split.
- `src/gnn_fraud/eval/` - PR-AUC, minority F1, confusion matrix.
- `src/gnn_fraud/viz/` - graph visualizations.
- `configs/` - one YAML per experiment (dataset, model, seed, hyperparams).
- `data/DATA_CARD.md` - provenance + license of every source.
- `docs/loops/` - how this repo is built/maintained by autonomous loops.
- `scripts/verify.sh` - the green-gate.

## Secrets & external APIs

- **Never commit secrets.** Keys live in `.env` (gitignored); `.env.example`
  documents the shape. The `curl`/`wget` deny in `.claude/settings.json` is
  defense-in-depth; the real control is: never print, commit, or transmit
  `.env` / keys anywhere, and never add code that sends them off-box.
- Respect every API's ToS and rate limits (Etherscan, BigQuery). No aggressive
  scraping and no bypassing protections. Document limits in `data/DATA_CARD.md`.

## Git / PR etiquette

- Push to GitHub via the `github-oss` SSH remote (account `Julien-Au`).
- Never commit directly to `main` once the repo has an issue/PR flow; one branch
  per task: `feat/<slug>` or `fix/<slug>`.
- Do not force-push; do not `git reset --hard` shared history (both denied in
  `.claude/settings.json`).

## Tutor mode (this repo is also a learning artifact)

The author is deliberately learning graph learning here. For each step: explain
the concept and the trade-off *before* coding, keep `LEARNING_NOTES.md` current,
and end each step with a couple of comprehension questions.
