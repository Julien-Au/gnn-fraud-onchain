#!/usr/bin/env bash
# Build the anonymized supplementary archive for double-blind review.
#
# Allowlist copy + identity scrub + verification sweep. Fails loudly if any
# identifying string survives. Usage:
#   bash scripts/make_supplementary.sh [output.zip]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-supplementary.zip}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
STAGE="$(mktemp -d)/supplementary"
mkdir -p "$STAGE"

# Allowlist: code, tests, configs, scripts, per-seed results, data card.
# Excluded by construction: .git, LICENSE, pyproject author fields, CLAUDE.md,
# paper/, docs/loops/, docs/reviews/, README.md (replaced below).
cp -r "$ROOT/src" "$ROOT/tests" "$ROOT/configs" "$ROOT/scripts" \
      "$ROOT/Dockerfile" "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$STAGE/"
# This script's own grep/sed patterns would trip the sweep; it is build tooling,
# not review material.
rm -f "$STAGE/scripts/make_supplementary.sh"
mkdir -p "$STAGE/docs" && cp -r "$ROOT/docs/results" "$STAGE/docs/"
cp "$ROOT/data/DATA_CARD.md" "$STAGE/"

cat > "$STAGE/README_ANONYMOUS.md" <<'EOF'
# Supplementary material (anonymized)

Full source code and per-seed results for the submission
"How Much of the Elliptic Leaderboard Is Real?".

## Reproduce

```bash
uv sync --extra gnn --extra boost
bash scripts/verify.sh            # lint + typecheck + tests
uv run gnn-fraud leakage-multi --models gcn,sage,gat,transformer --seeds 42,43,44,45,46
uv run gnn-fraud decompose --model sage --seeds 42,43,44,45,46
uv run gnn-fraud recipe
python scripts/compute_stats.py   # regenerates docs/results/stats_summary.json
```

Every number in the paper maps to a file in `docs/results/` (per-seed values
included). Elliptic downloads automatically via PyTorch Geometric; Elliptic++,
DGraph-Fin and AMLworld are gated by their authors (see DATA_CARD.md).
EOF

# Identity scrub (names, emails, handles, affiliations).
grep -rl -iE 'julien|audibert|deepfi|gymcoach|ja@|Julien-Au' "$STAGE" 2>/dev/null | while read -r f; do
  sed -i -E 's/(Julien[- ]?Au[a-z]*|julien|audibert|deepfi|gymcoach|ja@[a-z.]+)/anonymous/Ig' "$f"
done

# Verification sweep: fail if anything identifying remains.
if grep -ri -l -E 'julien|audibert|deepfi|gymcoach|ja@' "$STAGE" | grep -q .; then
  echo "FATAL: identifying strings survived the scrub:" >&2
  grep -ri -l -E 'julien|audibert|deepfi|gymcoach|ja@' "$STAGE" >&2
  exit 1
fi

( cd "$(dirname "$STAGE")" && zip -rq "$OUT" "$(basename "$STAGE")" )
echo "OK: $OUT ($(du -h "$OUT" | cut -f1)) - scrub verified clean"
