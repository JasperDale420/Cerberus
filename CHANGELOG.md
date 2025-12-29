# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **H4 Logic Audit**: Add comprehensive fill input validation to `position_manager.on_fill()` to prevent position corruption from malformed broker data. Guards against negative quantities, zero/NaN/Inf values, and invalid types. ([#a994e0e](https://github.com/JasperDale420/Cerberus/commit/a994e0e))
- **H2 Logic Audit**: Document safe R-multiple calculation with division by zero protection for breakeven stops (initial_risk = 0). Added comprehensive test suite covering all edge cases. ([#3ec7125](https://github.com/JasperDale420/Cerberus/commit/3ec7125))
- **H3 Logic Audit**: Standardize position side comparisons to use enum (`Side.LONG`/`Side.SHORT`) consistently instead of string comparisons (`pos.side.value == "long"`). Makes code more type-safe and refactoring-friendly. ([#df32aa3](https://github.com/JasperDale420/Cerberus/commit/df32aa3))
- **H1 Logic Audit**: Add timestamp-based reconciliation race condition prevention. Position now tracks `last_updated` timestamp, and reconciliation skips positions modified within last 2 seconds to prevent fill data loss during async broker reconciliation. ([#74a1979](https://github.com/JasperDale420/Cerberus/commit/74a1979))
- **M1 Logic Audit**: Clear feature cache on market regime changes, not just daily boundaries. Prevents strategies from using stale regime-sensitive indicators (VWAP, RSI, etc.) for hours after regime transitions. Cache now clears 2-6 times per day instead of once. ([#2b88542](https://github.com/JasperDale420/Cerberus/commit/2b88542))
- **M2 Logic Audit**: Track positions carried forward at session rollover for observability. RiskManager now logs number of overnight positions at session boundaries to help diagnose position limit issues. ([#7e2a665](https://github.com/JasperDale420/Cerberus/commit/7e2a665))
- **M3 Logic Audit**: Prioritize target over stop when both exit conditions trigger on same bar. More trader-friendly since target is the better exit. Updated test to verify new behavior. ([#5514691](https://github.com/JasperDale420/Cerberus/commit/5514691))
- **M4 Logic Audit**: Skip position reconciliation for symbols with pending orders. Prevents partial fill state corruption during mid-fill broker queries. ([#dd857ee](https://github.com/JasperDale420/Cerberus/commit/dd857ee))
- **M5 Logic Audit**: Already fixed - MAE/MFE tracking happens before broker_managed_exits check, so updates on every bar
- **M6 Logic Audit**: Added optional est_exit_commission parameter to update_unrealized_pnl() for more accurate net PnL (subtracts estimated exit costs)

### Added
- **Error Logging Improvements**: Comprehensive audit and enhancement of error logging across the codebase
  - Added `exc_info=True` to 16 critical ERROR-level logs for full stack traces in production debugging
  - Added DEBUG-level logging to 5 silent exception handlers for best-effort operation visibility
  - Expanded ErrorCode enum from 15 to 50+ codes organized by category (Config, Analytics, Alpaca, Engine, Scanner, Risk, Orders, Agent, Database, Backtest)
  - Improved production debugging capability, observability, and error categorization for operational monitoring
  - Commits: `5eb2db6`, `b7b7788`, `61fcd7b`


### Added
- **Repository Hygiene (PR #1)**: Added project identity files for open-source readiness
  - LICENSE file (MIT License) for legal clarity
  - SECURITY.md with vulnerability disclosure policy and trading-specific security guidelines
  - .env.example template with safe defaults and comprehensive documentation
  - Updated README.md to reference LICENSE, SECURITY.md, and .env.example
- **Repository Hygiene (PR #2)**: Reorganized root-level utilities for clarity
  - Created `tools/` directory with comprehensive README
  - Moved `verify_architecture.py`, `verify_deepseek.py`, `paper_live_harness.py` to tools/
  - Archived obsolete `codereview_notes.md` to artifacts/archive/
- **Repository Hygiene (PR #3)**: Added operational maturity tooling
  - Created `docs/runbook.md` with 6 failure scenarios, diagnostics, and recovery procedures
  - Implemented `src/core/health.py` with database/API/system health checks
  - Added `--healthcheck` CLI flag for operational readiness verification
  - Updated README.md with healthcheck usage documentation
- **Strategies**: Implemented full suite of 8 remediation strategies:
    - VWAP Mean Reversion
    - Opening Range Breakout (ORB)
    - Trend Pullback
    - Failed Breakout Fade
    - VWAP Trend Rider
    - Index Mean Reversion
    - Flow-Confirmed Momentum
    - Gap-Fill Scalper
- **Scanner**: Implemented `ScannerProfile` interface and specific profiles for all 8 strategies. Filters based on technicals (ADX, RSI, BB) and Option Flow (Unusual Whales Z-Score).
- **Pipeline**: Added comprehensive feature generation:
    - `prior_day_high`, `prior_day_low`
    - `bb_upper`, `bb_lower`, `price_zscore`
    - `flow_zscore`, `call_put_ratio` (Unusual Whales)
    - `premarket_volume` calculation
- **Architecture**:
    - `Agent` meta-loop for daily analysis and config updates.
    - `Analytics` layer for trade statistics and efficiency auditing.
    - `Scheduler` integration for automated functionality.
- **Testing**: Added unit tests for all strategies (`tests/test_strategy_*.py`).
- **Docker**: Added `Dockerfile`, `.dockerignore`, `docker-compose.yml` and `make` targets (`up`, `down`, `logs`) for full containerized orchestration.
- **Scheduler**: Added internal `APScheduler` implementation (`src/scheduler.py`) to replace external Chronos dependency. Run via `python -m src.main --scheduler`.

### Changed
- **Scanner Core**: Fixed duplicate watchlist entry bug and added sorting by score.
- **Pipeline**: Removed hardcoded `premarket_volume`; now calculates from intraday data.
- **Config**: Extended `config.yaml` to support all new strategies and parameters.
- **Agent**: Updated Stage 3 System and User prompts to be "self-annealing" and PRD-aligned, prioritizing incremental refinement over radical changes.
- **Config**: Added `unusual_whales.enabled` flags to toggle external flow data integration (disabled by default).

### Fixed
- **Pre-commit**: Resolved all Ruff linting errors, Mypy type-check failures, and Black formatting inconsistencies across the codebase.
- **Data Pipeline**: Fix incorrect usage of `zip(strict=False)` and unused variables.
- **Testing**: Fix mock type injection errors in unit tests.
