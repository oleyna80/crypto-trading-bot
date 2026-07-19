# Codex Spec-Driven Development Workflow

Use this workflow for features, bug fixes, strategy changes, integrations and
any task that can alter trading behavior. Keep roles separate between stages.

## Stage 0 — Plan (Orchestrator)

1. Read `AGENTS.md`, the memory bank and current ticket artifacts.
2. Check `git status --short`; preserve unrelated dirty work.
3. State objective, expected result, write-set, out of scope, AC, risks and
   proposed tests.
4. Use bounded read-only analyst tasks only where they reduce uncertainty.

Output: an approved work block or a clear Owner question.

## Stage 1 — Spec (Orchestrator / Docs Analyst)

For a non-trivial code task, create or update a plan/spec/tasklist in `docs/`.
Capture interfaces, failure behavior, rollback concerns and verification
commands. Request Owner approval before any risky or protected change.

Output: implementation-ready, approved ticket artifacts.

## Stage 2 — Implementation (Coder)

Make the smallest coherent change inside the write-set, add focused tests and
run the narrowest relevant checks. Do not revise requirements, broaden scope,
commit or deploy while coding.

Output: diff summary, changed files, exact commands and unresolved risks.

## Stage 3 — Review (Reviewer)

Read-only compare the diff with AC, architecture and safety boundaries. Check
provider parsing, error handling, logging, secrets and unrelated-file drift.

Output: findings with file/line evidence and a verdict: approved, changes
requested or blocked.

## Stage 4 — Verification (Verifier)

Independently run agreed tests and smoke checks; reconcile the result with AC.
Do not repair code in this role. If checks fail, return to a new Coder stage.

Output: `READY`, `BLOCKED` or `UNVERIFIED` verdict with evidence.

## Stage 5 — SSOT closeout (Orchestrator / Docs Analyst)

After `READY`, update the tasklist and memory bank, record accepted
architecture decisions, and propose a Conventional Commit message. Commit,
push and deployment always require explicit Owner approval.
