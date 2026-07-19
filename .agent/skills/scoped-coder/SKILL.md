---
name: scoped-coder
description: Implement a bounded bybit_grid_bot work block after its scope, acceptance criteria, and write-set are approved. Use for focused Python, tests, or documentation changes; do not use for planning, review, or verification.
---

# Scoped Coder

Implement the smallest coherent change in an approved work block. This skill is
for the Coder stage only; it does not authorize scope changes, commits, deploys
or configuration changes.

## Inputs required

- ticket and objective;
- explicit write-set and out-of-scope list;
- acceptance criteria and targeted checks;
- approved plan/spec for non-trivial code work.

If any input is absent or a required file falls outside the write-set, stop and
return a scope question to the Orchestrator.

## Workflow

1. Read `AGENTS.md`, memory-bank context and ticket artifacts; inspect
   `git status --short` and preserve unrelated dirty files.
2. Trace the affected public API and existing tests before editing.
3. Implement only the approved change. Keep DEX LP code in `src/lp/` unless
   the plan explicitly names another boundary.
4. Add focused pytest coverage for logic changes; add a regression test for a
   bug fix. Use deterministic fixtures rather than live providers.
5. Run the narrowest relevant test command, then an import or module
   smoke-check when applicable.
6. Report changed files, exact commands, results, skipped checks and risks.

## Guardrails

- Use `logging`, not `print`, and never log secrets.
- Do not edit `requirements.txt`, CI, deploy/VPS files, `.env`, credentials,
  schemas or order-execution behavior without explicit Owner approval.
- Do not update task completion or memory-bank status; that happens after
  independent verification.
