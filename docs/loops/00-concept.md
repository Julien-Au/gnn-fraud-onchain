# 00 - What a self-verifying loop is

A "loop" here is a small, repeatable cycle where the agent produces work, an
automatic gate verifies it, and only verified work is kept. It is the difference
between "an agent that types code" and "an agent you can trust with a task".

## The six ingredients (mapped to real tools)

| Ingredient | In this repo |
|---|---|
| A task | one roadmap step (README) or one issue |
| A skill/prompt | tutor-mode plan + implementation convention (CLAUDE.md) |
| An action surface | Claude Code tools (edit files, run `uv`, run tests, git) |
| **A feedback gate** | `scripts/verify.sh` (ruff + mypy + pytest + smoke) + CI |
| Memory / durability | git history, the research log, `docs/loops/autonomy-log.md` |
| A halting rule | small commits, explicit stop conditions, human sign-off per step |

## Why this shape suits a *research* repo

The failure mode a jury worries about is not "the code is ugly" - it is "the
number is not real". The gate is where that is caught:

- The gate runs the **tests** (including a check that the temporal split does not
  leak), so a change that introduces leakage fails locally before it is committed.
- The gate is **cheap** (seconds), so it runs on every increment, not just at the
  end - regressions surface immediately.
- CI re-runs the gate on a clean machine, so "works on my machine" is not enough.

## Where it graduates

Early steps run interactively (I watch each diff). As conventions stabilize, the
repeated parts graduate into `.claude/skills/` and the whole thing can run as a
`/loop`, exactly as in GymCoach's `docs/loops/`. The trust boundary never moves:
nothing lands on a red gate, and reported results always trace to a real run.
