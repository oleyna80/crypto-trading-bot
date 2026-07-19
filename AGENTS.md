# Repository Guidelines

Операционные правила для Codex в `bybit_grid_bot`.

## Режим и роли

Для нетривиальных, рискованных, многофайловых, архитектурных, security- или
production-задач применять **Plan → Spec → Implementation → Review →
Verification**. В начале и финале указывать stage, objective, role, expected
result, scope, actions, changed files, checks, risks и next action.

| Роль | Полномочия |
| --- | --- |
| Orchestrator | Задаёт objective, AC, write-set, риски и hand-off; не реализует код в той же стадии. |
| Coder | Единственный writer стадии Implementation; меняет только write-set. |
| Reviewer | Read-only проверка diff, требований и рисков. |
| Verifier | Независимая read-only проверка AC и команд. |

Product, Architecture, Frontend, Backend, Design, Security, QA и Docs Analyst
не получают write-доступ автоматически. У subagent-задания обязательны role,
scope, out of scope, expected output и file-change permission.

## Контекст и планирование

Тикет: запрос пользователя → `docs/.active_ticket` → первый незакрытый
tasklist. Перед правками проверить `git status --short`, прочитать
`memory_bank/context.md`, `progress.md`, `decisions.md`, затем PRD, plan и
tasklist тикета. SSOT: `AGENTS.md` → `roadmap.md` → `memory_bank/*` →
`docs/`.

До нетривиальной реализации сформулировать objective, write-set, out of scope,
AC, проверки и риски. Для тривиальной документации допустим путь:
understand → scoped change → check → report.

## Согласования и безопасность

Запросить Owner approval до изменений зависимостей, CI/CD, инфраструктуры,
конфигурации/секретов, схемы БД, deploy, order/payment behavior, архитектуры
вне scope, удаления, commit или push. Остановиться при неясном требовании,
scope drift, чужих dirty files или неуспешной проверке. Не stage-ить чужие
изменения и не коммитить `.env`, ключи, токены, build output или credentials.

## Код и тесты

- `config/` — настройки; `models/`, `services/` — legacy; `src/` — актуальные
  strategy/analysis/API/DEX-LP; `web_ui/` и `src/api/` — UI/API; `tests/` — тесты.
- Python, Markdown, YAML/JSON: PEP 8, black-совместимый line length 88 только
  в затронутых строках; без массового reformat.
- Использовать `logging`: `INFO` этапы, `WARNING` fallback, `ERROR` сбои,
  `logger.exception(...)` в `except`; не логировать секреты.
- Бизнес-логика требует теста, багфикс — regression test. Минимум: затронутый
  `pytest`-scope и import/module smoke-check; явно фиксировать пропуски.

## Закрытие и skills

После `READY` обновить tasklist и `memory_bank/progress.md`; принятое
архитектурное решение — `memory_bank/decisions.md`. В финале дать AC status,
команды, diff summary, риски и Conventional Commit suggestion; commit/push —
только по явному разрешению.

Skills находятся в `.agent/skills/`; сначала читать `.agent/README.md` и
выбирать минимально нужный `SKILL.md`. Они не расширяют полномочия этого файла;
происхождение Codex-native skills зафиксировано в `.agent/skills.lock.yml`.
