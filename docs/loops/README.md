# Loops: how this repo is built and maintained

This project is built with **self-verifying Claude Code loops**, the same shape
used to maintain [GymCoach](https://github.com/Julien-Au/gymcoach). The point is
not "an AI wrote it" - it is that the *process* is legible: every change is
planned, implemented in small steps, adversarially self-reviewed, verified by a
green-gate, and logged. The repo tells two stories at once: the GNN work, and how
I pilot agents to do rigorous engineering.

## The four rules we never break

1. **A loop is only as good as its feedback.** Everything self-verifies
   (`scripts/verify.sh`) before claiming success. Nothing lands on a red gate.
2. **The reusable unit is a skill, not a prompt.** Repeated work graduates into a
   `.claude/skills/` skill (added as the workflow stabilizes).
3. **The loop must halt.** Max steps, no-progress detection, explicit stop
   conditions - no infinite loops, no billing surprises.
4. **Durability is explicit.** State lives in git and on GitHub, not only in a
   session. A crash loses nothing.

## The per-step loop (this project's cadence)

Each roadmap step (see the README) is one turn of the loop:

```
plan  ->  implement (small commits)  ->  adversarial self-review of the diff  ->  green-gate  ->  note
```

- **plan** - explain the concept and the trade-off first (tutor mode), decide the
  smallest useful increment.
- **implement** - typed code + tests, one focused commit at a time.
- **adversarial self-review** - re-read the diff as a hostile reviewer: leakage?
  fabricated numbers? unfixed seed? weakened test? (This is where research
  integrity is enforced.)
- **green-gate** - `bash scripts/verify.sh` must pass.
- **note** - update the research log and log the step in `autonomy-log.md`.

## Files in this folder

- [`00-concept.md`](00-concept.md) - what a self-verifying loop is and why it fits
  a research-grade repo.
- `autonomy-log.md` - append-only log of what each step did (created as steps land).

## Integrity guardrails specific to this repo

Because a jury will read this, the self-review step explicitly checks:

- **No fabricated results.** Every reported number traces to a reproducible run.
- **No leakage.** Temporal splits where the data has time.
- **Imbalanced-aware metrics.** PR-AUC + minority F1 + confusion matrix.
- **Cited data.** Provenance/license in `data/DATA_CARD.md`; secrets only in `.env`.
