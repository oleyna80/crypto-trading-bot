---
name: ssot-sync-closeout
description: Synchronize ticket and memory-bank records after a bybit_grid_bot work block has a READY verification verdict. Use for documentation closeout only; do not use to declare incomplete or unverified work complete.
---

# SSOT Sync Closeout

Close a verified work block without altering code or concealing uncertainty.
Use only after the Verifier has supplied a `READY` verdict with evidence.

## Inputs required

- ticket identifier and accepted scope;
- final changed-file list and diff summary;
- AC-by-AC verification evidence and review verdict;
- remaining risks, follow-ups and proposed next ticket.

## Workflow

1. Update the ticket tasklist/checklist: mark only verified items complete and
   leave blocked or deferred items explicit.
2. Add a concise `memory_bank/progress.md` entry with outcome, commands,
   residual risks and next action.
3. Add an ADR-style decision only when an accepted architecture/provider/
   strategy decision was made; link its ticket and consequences.
4. Prepare a Conventional Commit suggestion, but do not stage, commit, push or
   deploy without explicit Owner approval.

## Boundaries

Do not rewrite historical decisions, backfill passing status for skipped checks
or modify implementation files. If the verifier did not return `READY`, report
the missing evidence rather than updating completion records.
