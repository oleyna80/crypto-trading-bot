---
name: Code Review
description: Анализ кода на соответствие архитектуре, безопасность и качество
---

# Skill: Code Review

## Description
Детальный анализ кода на соответствие conventions, безопасность и maintainability.

## Triggers
- "ревью", "review", "audit", "security review"

## Inputs
- Измененные файлы (diff или полный код)
- `.agent/conventions.md`
- `AGENTS.md`
- `memory_bank/context.md`, `memory_bank/decisions.md`
- План/Spec для контекста

## Checklist

### Architecture
- [ ] Модули соблюдают boundaries (core pure, services async)
- [ ] Модели/контракты данных согласованы (dataclass/pydantic/typed dict)
- [ ] DI используется правильно
- [ ] Нет циклических зависимостей

### Code Quality
- [ ] Type hints везде
- [ ] Понятные имена
- [ ] DRY соблюдается
- [ ] Нет dead code

### Security
- [ ] Секреты только в .env
- [ ] Input validated
- [ ] Sensitive data не в логах

### Performance
- [ ] Async I/O правильно
- [ ] Нет N+1 queries
- [ ] Кэширование где нужно

## Output
Отчет `docs/reports/<ticket>-code.review.md` (или комментарий в PR, если отдельные отчеты не ведутся):

```markdown
# Code Review: <TASK-ID>

## Summary
- Files reviewed: N
- Issues: X critical, Y warnings, Z suggestions
- Verdict: APPROVED | NEEDS_CHANGES | BLOCKED

## Critical Issues (must fix)
## Warnings (should fix)
## Suggestions (nice to have)
```

## Severity Levels
- **Critical** → BLOCKED
- **Warning** → NEEDS_CHANGES
- **Suggestion** → APPROVED (with notes)
