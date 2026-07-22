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
