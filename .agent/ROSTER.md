# Agent Roster

## Core execution roles

| Role | Responsibility | File permission |
| --- | --- | --- |
| Orchestrator | Defines objective, AC, work block, risks and stage hand-offs. | Read-only; may make explicitly approved documentation-only updates. |
| Coder | Implements the approved plan and focused tests. | One Coder only; approved write-set only. |
| Reviewer | Reviews diff against spec and architecture. | Read-only. |
| Verifier | Runs independent acceptance and smoke checks. | Read-only. |

## Read-only specializations

Product, Architecture, Frontend, Backend, Design, Security, QA and Docs
Analysts may be assigned a narrow investigation. Their brief must state scope,
out of scope, expected evidence and that no repository files may be changed.

## Skill mapping

### Codex-native procedures

- Architecture discovery: `.agent/skills/architecture-discovery/SKILL.md`
- Scoped implementation: `.agent/skills/scoped-coder/SKILL.md`
- Review: `.agent/skills/reviewer/SKILL.md`
- Verification: `.agent/skills/verifier/SKILL.md`
- Debugging: `.agent/skills/systematic-debugging/SKILL.md`
- Security triage: `.agent/skills/security-audit-triage/SKILL.md`
- SSOT closeout: `.agent/skills/ssot-sync-closeout/SKILL.md`

### Existing project procedures

- Technical discovery: `.agent/skills/technical-discovery.md`
- Task decomposition: `.agent/skills/task-decomposition.md`
- DEX range operations: `.agent/skills/clmm-range-ops.md`
- Risk/security/Telegram/memory-bank procedures: matching flat files in
  `.agent/skills/`.
