---
name: Technical Discovery
description: Технический анализ кодовой базы и исследование решений
---

# Skill: Technical Discovery

## Description
Анализ существующего кода, tech stack и исследование решений для новых фич.

## Triggers
- "исследуй", "проанализируй", "RFC", "audit tech stack"

## Inputs
- Task card или запрос
- `AGENTS.md`
- `.agent/workflows/sdd-protocol.md`
- `memory_bank/context.md`, `memory_bank/progress.md`, `memory_bank/decisions.md`
- Целевые файлы для анализа

## Modes

### Mode A: Codebase Analysis
1. **`rg`** — быстрый поиск паттернов и импортов
2. **`find`** — инвентаризация структуры
3. **`sed` / `cat`** — чтение ключевых участков кода
4. **`pytest -k <pattern>`** — локализация тестового контура (если нужно)

### Mode B: Solution Research
1. **Audit Tech Stack** — проверь `requirements.txt`, `.env`, existing DBs
2. **Web Search** — docs, GitHub, best practices
3. **Compare Options** — минимум 2 варианта (A vs B)

## Output
Файл `docs/research/RFC-<feature>.md` или `docs/research/<task-id>-research.md`:

```markdown
# Research Report: <Topic>

## Context
Что исследовалось и почему.

## Findings
### 1. <Topic>
- File: `path/to/file.py`
- Observation: ...
- Evidence: (code snippet)

## Options Analysis
| Option | Pros | Cons |
|--------|------|------|
| A | ... | ... |
| B | ... | ... |

## Gaps / Risks
## Recommendation
```

## Constraints
- **No code changes** — только анализ
- **Evidence-based** — каждый finding с ссылкой на код
