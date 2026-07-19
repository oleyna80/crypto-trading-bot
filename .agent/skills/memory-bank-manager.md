# Memory Bank Manager

Use this skill to keep `memory_bank/` concise, evidence-based and current.

## Files

- `context.md`: durable project facts, architecture and conventions.
- `progress.md`: verified completed work, next task and known blockers.
- `decisions.md`: ADR-style decisions with rationale and consequences.

## Rules

- Read all three files before starting a scoped task.
- Update progress only after the Verifier returns `READY`; include checks run
  and any residual risk.
- Add a decision when an accepted choice changes architecture, provider
  behavior, strategy rules or deployment assumptions.
- Keep ticket-level implementation detail in `docs/` and preserve links to
  source files and ticket IDs.

## History policy

- Do not delete historical decisions; mark superseded entries explicitly.
- Do not erase prior status without a replacement reference.
- Do not record an assumption as verified project fact.
