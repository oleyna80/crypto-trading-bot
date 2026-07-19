# Project-local skills

This directory contains reusable procedures for Codex working in this
repository. They supplement, but cannot override, `AGENTS.md`.

## Selecting a skill

Read only the skill needed for the current role and stage:

- discover a boundary before planning: `architecture-discovery/`;
- implement an approved work block: `scoped-coder/`;
- inspect a diff: `reviewer/`;
- prove AC independently: `verifier/`;
- investigate a defect: `systematic-debugging/`;
- triage security concerns: `security-audit-triage/`;
- synchronize verified state: `ssot-sync-closeout/`.

The existing flat `*.md` files are legacy project procedures and remain usable
where their DEX/LP-specific guidance applies. New Codex-native procedures use
`<skill-name>/SKILL.md` and may include `agents/openai.yaml` metadata.

## Provenance and maintenance

Selected procedures were adapted, not copied verbatim, from the local
`agentic-sdlc-framework`. `.agent/skills.lock.yml` records the exact source
revision. Update a skill only with a clear project need, preserve its narrow
authority, and validate it with the skill-creator validator before handoff.
