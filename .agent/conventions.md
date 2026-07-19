# Project Conventions

## Purpose

Rules for Codex work in `bybit_grid_bot`.

## Scope

- Legacy: `models/`, `services/`
- Active strategy code: `src/`
- UI/API: `web_ui/`, `src/api/`

## Coding Rules

- Python 3.12; type hints are required for public functions.
- Do not change public signatures without updating the approved spec/plan.
- Use `logging`, not `print`, in production paths.
- Add new DEX LP logic under `src/lp/*`; do not break the grid path.
- Keep network/provider parsing, strategy decisions and alert delivery
  separate; parsing must not silently open positions or send alerts.

## Testing Rules

- Add pytest coverage for non-trivial logic and a regression test for a defect.
- Prefer deterministic fixtures to live provider calls.
- Run the narrowest relevant test command first, then broaden only when needed.
- State which tests were not run and why; a skipped check is not a pass.

## Safety Rules

- Secrets stay in `.env`, never in git.
- Do not change deployment, CI, dependencies or runtime configuration without
  explicit approval.
- Record the approved write-set before editing. If a required file is outside
  it, stop and ask for a scope update.
- Review and verification are read-only; do not fix findings in either role.
