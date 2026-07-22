#!/usr/bin/env bash
#
# verify.sh - the gnn-fraud-onchain green-gate.
#
# This is the self-verification step every commit and every autonomous loop
# must pass before claiming a task is done (see docs/loops/). It mirrors the CI
# "quality" job so a loop can catch its own regressions locally, in seconds.
#
#   PASS  -> exit 0, the working tree is safe to commit / open a PR from.
#   FAIL  -> exit non-zero, fix the reported step before retrying.
#
# Tiers:
#   (default)  lint (ruff) + typecheck (mypy) + unit tests (pytest) + smoke CLI
#   --full     also runs the gnn extra: import torch/PyG + a 1-epoch smoke train
#              (needs `uv sync --extra gnn`; heavier, mirrors the CI "gnn" job)
#
# Everything runs inside the uv-managed environment so it is reproducible.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

fail() { echo ""; echo "GREEN-GATE FAILED at: $1"; exit 1; }
step() { echo ""; echo "==> $1"; }

echo "gnn-fraud-onchain green-gate - uv $(uv --version 2>/dev/null || echo '??')"

step "sync dev environment"
uv sync --quiet || fail "uv sync"

step "lint (ruff)"
uv run ruff check src tests || fail "lint"

step "format check (ruff)"
uv run ruff format --check src tests || fail "format"

step "typecheck (mypy)"
uv run mypy || fail "typecheck"

step "unit tests (pytest)"
uv run pytest || fail "unit tests"

step "smoke CLI"
uv run gnn-fraud info >/dev/null || fail "smoke CLI"

if [ "$FULL" = "1" ]; then
  step "sync gnn extra (torch + torch-geometric)"
  uv sync --extra gnn --quiet || fail "uv sync --extra gnn"
  step "smoke train (1 epoch on a tiny subgraph)"
  uv run --extra gnn gnn-fraud smoke-train || fail "smoke train"
fi

echo ""
echo "GREEN-GATE PASSED - safe to commit and open a PR."
