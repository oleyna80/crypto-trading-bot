# Agent Configuration

This directory is the project-local operating layer for Codex. It complements
the repository-level contract in `AGENTS.md`; when rules differ, `AGENTS.md`
wins.

## Read order

1. `AGENTS.md`
2. `memory_bank/context.md`, `memory_bank/progress.md`, `memory_bank/decisions.md`
3. Active-ticket PRD, plan and tasklist in `docs/`
4. The minimal relevant file in `.agent/rules/`, `.agent/workflows/` or
   `.agent/skills/`

## Codex role model

- **Orchestrator:** defines AC and a bounded work block; uses read-only
  specialists when helpful.
- **Coder:** one writer for the implementation stage and only within the
  approved write-set.
- **Reviewer:** read-only review of the diff and requirements.
- **Verifier:** independent read-only evidence for AC and tests.

The same Codex session may move through these roles in separate stages, but
must not combine implementation with its review or verification stage.

## Safety boundaries

- Do not make code changes without an approved task scope.
- Keep DEX LP work in `src/lp/` unless an approved plan says otherwise.
- Keep operational secrets in `.env`; commit only examples/templates.
- Do not change deployment, VPS, database, dependencies or production
  configuration without explicit Owner approval.

## Skills

Use `.agent/skills/README.md` for selection and provenance. Skills provide
repeatable procedures; they do not grant broader authority than `AGENTS.md`.
