# Codex Agent Development Workflow

Статус: ACTIVE
Последнее обновление: 2026-07-19

## 1. Назначение

Этот документ описывает Codex-native процесс для `bybit_grid_bot`: локальная
разработка в WSL, независимые review/verification и контролируемый handoff.
Он не заменяет `AGENTS.md`; при конфликте приоритет у корневого контракта.

Проект остаётся в read-only/paper режиме для DEX LP Stage 1: сигналы и
рекомендованные диапазоны допустимы, автоматическое on-chain исполнение — нет.

## 2. Роли

| Роль | Задача | Изменение файлов |
| --- | --- | --- |
| Orchestrator | Формирует objective, AC, write-set, риски и hand-off между стадиями. | Нет, кроме явно одобренной документации. |
| Coder | Реализует один утверждённый work block и его тесты. | Да, только один Coder и только write-set. |
| Reviewer | Проверяет diff относительно AC и архитектуры. | Нет. |
| Verifier | Независимо подтверждает AC командами и smoke-check. | Нет. |

При необходимости Orchestrator назначает read-only Product, Architecture,
Backend, Security, QA или Docs Analyst. В задаче всегда указать role, scope,
out of scope, expected output и file-change permission.

## 3. Рабочий цикл

### Stage 0 — Plan

Прочитать `AGENTS.md`, `memory_bank/*`, активный тикет и проверить
`git status --short`. Зафиксировать objective, expected result, write-set,
out of scope, AC, риски и тестовые команды. Нетривиальная задача не переходит
к коду без согласованных scope и AC.

### Stage 1 — Spec

Для feature, интеграции, изменения стратегии, миграции или потенциально
торгового поведения подготовить/обновить PRD, plan и tasklist в `docs/`.
Описать интерфейсы, отказоустойчивость, rollback и проверку. До зависимостей,
конфигурации, схемы, deploy, order/payment behavior или архитектуры вне scope
обязательно запросить Owner approval.

### Stage 2 — Implementation

Единственный Coder вносит минимальный coherent diff в write-set, добавляет
детерминированные pytest-тесты и запускает узкий тестовый scope плюс import/
module smoke-check. Нельзя менять чужие dirty files, расширять scope,
commit/push или deploy.

### Stage 3 — Review

Reviewer работает read-only: сопоставляет diff с AC, проверяет scope drift,
public API, ошибки, logging, секреты, provider payloads и разделение parser →
decision → range/alert. Вердикт: `approved`, `changes requested` или
`blocked`; findings содержат `file:line`, влияние и точечную рекомендацию.

### Stage 4 — Verification

Verifier повторно читает AC и запускает согласованные checks независимо от
Coder. Вердикт: `READY` (есть passing evidence), `BLOCKED` (проверка не
проходит) или `UNVERIFIED` (недостаёт среды/доказательств). Verifier не
исправляет код.

### Stage 5 — SSOT closeout

Только после `READY` обновить tasklist и `memory_bank/progress.md`; для
принятого решения об архитектуре/провайдере/стратегии добавить ADR в
`memory_bank/decisions.md`. Сформировать Conventional Commit suggestion.
Stage, commit, push и deployment выполняются только при явном Owner approval.

## 4. Локальные skills

Выбирать минимально необходимый skill из `.agent/skills/`:

- `architecture-discovery/` — доказательная подготовка границ до plan;
- `scoped-coder/` — approved implementation work block;
- `reviewer/` и `verifier/` — раздельные read-only gates;
- `systematic-debugging/` — воспроизводимый defect + regression test;
- `security-audit-triage/` — evidence-based security triage;
- `ssot-sync-closeout/` — обновление SSOT после `READY`.

Legacy flat skills применяются для DEX/LP-процедур, где они релевантны.
Происхождение новых процедур закреплено в `.agent/skills.lock.yml`.

## 5. Локальные проверки и handoff

Типовой минимум для изменённой Python-логики:

```bash
pytest -q tests/<target_test>.py
python3 -c "from src.<module> import <symbol>; print('OK')"
```

Не использовать live provider credentials или внешние транзакции для unit/
smoke проверок. В отчёте обязательны changed files, AC status, commands and
results, skipped checks, risks, next action и предложенный commit message.

GitHub/VPS — отдельные Owner-approved операции: не применять `git add -A`,
commit, push, `pip install` на VPS или deploy по умолчанию.

## 6. Источники истины

При конфликте:

1. `AGENTS.md`;
2. `roadmap.md`;
3. `memory_bank/*`;
4. ticket artifacts в `docs/specs/`, `docs/plans/`, `docs/tasklist/`;
5. `.agent/*`.

`memory_bank/*` — каноничный оперативный контекст. `docs/memory-bank/*`, если
используется, не должен расходиться по статусу тикета или принятому решению.
