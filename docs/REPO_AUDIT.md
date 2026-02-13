# Repository Audit: Cerberus Trading System

## Executive Summary
The Cerberus trading system is a highly modular, professional-grade trading framework built with a vertical-slice architecture. It demonstrates strong reproducibility in backtesting, robust risk management, and a clean separation between strategy logic and execution.

## Key Findings

### 1. Architecture & Reproducibility
- **Vertical Slice Architecture**: The codebase is well-organized into clear domains (strategies, engine, backtest, universe).
- **Reproducibility**: Backtests are fully reproducible using offline datasets. A successful 3-day reproduction run was executed with consistent results.
- **Environment**: Requires `PYTHONPATH=.` for execution from the root directory.

### 2. Backtest Integrity & Data Sanity
- **Invariants**: realized PnL and trade matching (FIFO) were validated against the `BacktestAnalyzer` logic.
- **Session Handling**: The `BacktestRunner` currently flattens positions at the timestamp of the *last available bar* for a given date. In the provided Jan 2024 dataset, this includes after-hours bars up to 00:59:00 UTC (19:59:00 ET).
- **Recommendation**: Implement a formal "Session Definition" that allows for strict 16:00 ET flattening, regardless of after-hours data availability.

### 3. Bias Assessment
- **Look-ahead Bias**: `GapFillStrategy` was audited and found to strictly use historical bars (`symbol_state.bars`). Indicators and gap calculations depend only on data available at the time of the bar.
- **Survivorship Bias**: Universe selection via `UniverseBuilder` uses simulation-time volume ranking (looking back from the start of the backtest), which is a common and acceptable practice to avoid look-ahead, provided the candidate list is representative of the time.

### 4. Performance Metrics (Jan 02 - Jan 05, 2024)
- **Total Trades**: 12
- **Win Rate**: 66.67%
- **Profit Factor**: 3.65
- **Max Drawdown**: 0.02%
- **Net PnL**: $73.41 (on $100k initial cash)

## Conclusion
The system is in a "Ready" state for further development. The infrastructure is solid, and the backtest engine provides high-fidelity simulation. The immediate next best project is to enhance the session management to support more granular market hours control.

## Repository Hygiene Policy (2026-02-12)

This section is the operational policy for keeping the repo clean and stable before market sessions.

### Source of Truth vs Generated Files

- **Source of truth (commit to git):**
  - `src/`, `tests/`, `config/`, `scripts/`, `docs/`, dependency manifests.
- **Generated/local-only (never commit):**
  - `.claude-flow/`, `.swarm/`, `.scannerwork/`
  - runtime logs (`*.log`, `logs/tests/full_test_output*.txt`)
  - generated outputs under `artifacts/` and `results/`

### Pre-Market Log Reset Checklist

1. Archive runtime and old generated logs to:
   - `artifacts/archive/pre-market-YYYY-MM-DD/`
2. Truncate active runtime log path:
   - `logs/cerberus.log`
3. Restart runtime services:
   - `cerberus_trader`
   - `cerberus_scheduler` (if used)
4. Verify health and smoke checks:
   - `docker ps` shows healthy trader
   - `python scripts/smoke_gateway_heber_integration.py` passes
5. Confirm fresh runtime events:
   - Primary source is `docker logs cerberus_trader` (JSON stdout logging)

### Retention Rule

- Keep only the latest 24 hours in active `logs/`, `results/`, and non-archive `artifacts/`.
- Move older generated files into timestamped archive folders under `artifacts/archive/`.
