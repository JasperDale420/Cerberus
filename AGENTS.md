# AGENTS.md

Project-specific guide for AI coding agents working in Cerberus.

## What Cerberus Is

Cerberus is a multi-strategy intraday trading system for US equities.

Plain-English flow:
1. Pull market data and options flow data.
2. Build features and scan symbols.
3. Let strategies generate trade ideas.
4. Apply risk rules.
5. Submit paper or live orders.
6. Persist trades and analytics in SQLite.

## Repo Layout (High Level)

- `src/main.py`: main CLI entrypoint.
- `src/engine/`: execution engine, orders, risk, positions.
- `src/strategies/`: strategy implementations.
- `src/scanner/`: symbol selection and ranking.
- `src/data/`: data clients and feature pipeline.
- `src/backtest/`: simulation and report generation.
- `src/analysis/`, `src/analytics/`: persistence and performance analytics.
- `src/api/`: FastAPI endpoints for backtest and WFO outputs.
- `config/`: YAML runtime settings.
- `tests/`: unit, integration, contract, and e2e tests.
- `docs/`: architecture, runbook, and supporting docs.

## Standard Commands

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run tests:
```bash
pytest -q
```

Lint and type-check:
```bash
ruff check .
mypy .
```

Run app:
```bash
python -m src.main --mode paper
python -m src.main --healthcheck
```

Run API:
```bash
uvicorn src.api.backtest_api:app --port 8002
```

Docker:
```bash
docker build -t empire/cerberus:latest .
docker compose up -d cerberus-trader
```

## Non-Negotiables

- Keep paper mode as the default unless explicitly asked otherwise.
- Never weaken risk guardrails in `src/engine/risk.py`.
- Add tests before production code changes (TDD cycle).
- Use structured logging and include context fields at failure boundaries.
- Update `CHANGELOG.md` whenever behavior or docs change.

## Safety-Critical Paths

Handle with extra care:
- `src/engine/orders.py`
- `src/engine/risk.py`
- `src/engine/execution.py`
- `src/engine/position_manager.py`
- `src/main.py`

For changes in these areas, explicitly test risk limits and paper/live mode behavior.

## Documentation Rules

- Keep required docs present and current:
  - `README.md`
  - `CHANGELOG.md`
  - `AGENTS.md`
  - `docs/ARCHITECTURE.md`
  - `docs/RUNBOOK.md`
  - `docs/API_REFERENCE.md`
- Add new operational learnings to docs instead of leaving tribal knowledge in code comments.
