# Spec-Driven Development Enforcer

Use the Agentic SDLC for non-trivial, risky, multi-domain or production-impacting
work: **Plan → Spec → Implementation → Review → Verification**.

## Non-negotiable constraints

- Do not implement code until the ticket has an approved scope, explicit AC
  and an implementation plan/spec when the task is non-trivial.
- Use `docs/.active_ticket` when present; user-specified ticket takes priority.
- Keep each stage in one role: implementation is not review or verification.
- Only the active Coder may write repository files in an implementation stage.
- Do not change unrelated files; stop if requirements conflict or scope must
  expand.
- Treat `memory_bank/*` as required context, not optional notes.

## Required pre-code checklist

1. Confirm ticket ID and `git status --short`.
2. Read memory-bank context plus PRD, plan and tasklist when present.
3. State objective, write-set, out of scope, AC and risks.
4. Identify targeted tests and smoke checks.
5. Obtain any required Owner approval, then begin implementation.
