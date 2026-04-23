# AGENTS.md

Project-specific coding instructions for Cerberus.

## Project Overview

Cerberus is a multi-strategy intraday trading system for US equities. It supports paper and live execution, risk gating, strategy routing by market regime, and backtesting + analytics.

## Architecture

Cerberus uses a vertical-slice flow:

1. Data ingestion and feature pipeline
2. Scanner + strategy evaluation
3. Risk validation and position sizing
4. Order execution and fill handling
5. Persistence and analytics

Primary paths:

- `src/main.py` — CLI entrypoint and orchestration
- `src/engine/` — execution loop, risk manager, orders, positions
- `src/strategies/` — strategy implementations and activation policies
- `src/data/` — market data and provider clients
- `src/backtest/` and `src/analytics/` — simulation and reporting
- `src/api/` — FastAPI backtest results endpoints

## Development Commands

```bash
python -m pip install -r requirements.txt
uv run pytest -q
ruff check .
mypy .
python -m src.main --healthcheck
python -m src.main --mode paper
uv run uvicorn src.api.backtest_api:app --port 8002
```

## Safety-Critical Areas

Use extra caution when editing these files:

- `src/engine/orders.py`
- `src/engine/risk.py`
- `src/engine/execution.py`
- `src/engine/position_manager.py`

Never weaken these safeguards without explicit instruction:

- Paper defaults (`--mode paper`, `ALPACA_PAPER=true`)
- Daily loss limits
- Position/notional caps
- Kill-switch behavior

## Logging and Error Handling

- Use the project logger shim in `src/core/logger.py`
- Keep logs structured and include context keys
- Include `exc_info=True` when logging exceptions
- Fail fast for invalid config, missing credentials, and critical dependency startup failures

## Testing Workflow

TDD is required:

1. Write failing test first (RED)
2. Implement minimum code (GREEN)
3. Refactor while tests stay green

Quality gate for completed changes:

```bash
pytest -q && ruff check . && mypy .
```

## Documentation Rules

- Keep `CHANGELOG.md` updated in `[Unreleased]`
- Keep docs under `docs/` in canonical uppercase names where standardized
- Maintain `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, and `docs/API_REFERENCE.md` for this service
