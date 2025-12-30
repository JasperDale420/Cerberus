# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **H1 Logic Audit (New)**: Fix `VWAPReversionStrategy.on_bar()` unbound variable crash when price is within VWAP bands. Initialize `signal = None` before conditional blocks.
- **H2 Logic Audit (New)**: Fix `FlowMomentumStrategy` threshold logic that allowed weak flow signals. Now properly rejects all signals below `min_flow_zscore`.
- **H3 Logic Audit (New)**: Fix `BacktestAnalyzer._calculate_drawdown()` inconsistency with unrealized PnL. Peak tracking now based only on closed trades for consistency; added clarifying docstring.
- **Agent.run_cycle_with_db**: Now calls `apply_actions()` after persisting actions to DB, ensuring `strategies.auto.yaml` is automatically written.
- **Agent.apply_actions**: Fully implements REDUCE_RISK and DISABLE_STRATEGY actions: writes config to `strategies.auto.yaml`, supports both strategy-level and regime-specific overrides, adds floor at 0.0 when risk drops below threshold.
- **E2E Test Risk Values**: Update `test_prd_vertical_slice_success_metric.py` to use valid risk values within the new RiskConfig validation limits ($10k daily loss, not $1M).
- **M1 Logic Audit (New)**: Add DEBUG-level logging to silent exception handlers in `PositionManager` for MAE/MFE tracking and max-hold check failures. Improves observability without breaking trading.
- **M2 Logic Audit (New)**: Fix `RiskManager` positions_carried_forward tracking to capture count BEFORE session rollover reset for accurate logging.
- **M3 Logic Audit (New)**: Raise `Scanner` watchlist cap from 30 to 50 with documented PRD recommendation. Configurable limit with clearer warning message.
- **M4 Logic Audit (New)**: Add robust date extraction in `ExecutionEngine._update_symbol_state()` with clock fallback when bar_time is unusable. Prevents stale feature cache on date parsing failures.
- **M5 Logic Audit (New)**: Add `bar_duration_minutes` config parameter to `BaseStrategy` for accurate cooldown calculation across different timeframes. Defaults to 1.0 for backward compatibility.
- **L1 Logic Audit (New)**: Add named constant `QTY_EPSILON = 1e-7` in `BacktestAnalyzer` to replace magic number for floating-point quantity comparisons. Documents purpose and prevents potential infinite loops.
- **L3 Logic Audit (New)**: Add named constants `STOP_BUFFER_LONG` (0.99) and `STOP_BUFFER_SHORT` (1.01) in `FlowMomentumStrategy` for emergency stop buffer calculations. Documents the 1% buffer purpose.
- **CI Fix**: Add `asyncio_mode = "auto"` to pytest configuration in `pyproject.toml`. Enables pytest-asyncio to detect and run async test functions.
- **CI Fix**: Add missing `pytest-asyncio` dependency to `requirements.txt`. CI environment was missing this package, causing async tests to fail with "async def functions are not natively supported".

### Changed
- **SonarQube Refactoring**: Refactored `FlowMomentumStrategy.on_bar()` by extracting `_validate_flow_direction()`, `_get_average_volume()`, and `_build_signal()` helper methods. Reduced cognitive complexity from 26 to ~12.
- **SonarQube Refactoring**: Refactored `PositionManager.on_fill()` by extracting 8 helper methods: `_extract_fill_data()`, `_extract_risk_config()`, `_get_entry_context()`, `_apply_costs_to_position()`, `_open_new_position()`, `_increase_position()`, `_calculate_pnl()`, `_build_closed_trade_info()`, `_reduce_or_close_position()`. Reduced cognitive complexity from ~72 to ~15.
- **SonarQube Refactoring**: Refactored `PositionManager.on_bar()` by extracting 4 helper methods: `_update_mae_mfe()`, `_check_max_hold_exit()`, `_check_stop_target_exit()`, `_create_exit_intent()`. Reduced cognitive complexity from 40 to ~12.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._update_symbol_state()` by extracting 4 helpers: `_get_or_create_symbol_state()`, `_extract_current_date()`, `_handle_index_bar_update()`, `_update_session_vwap()`. Reduced complexity from 40 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._update_indicator_cache()` by extracting 5 helpers: `_collect_indicator_periods()`, `_update_ema_indicators()`, `_update_rsi_indicators()`, `_update_vol_sma_indicators()`, `_update_bb_indicators()`. Reduced complexity from 31 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_signal()` by extracting 5 helpers: `_bind_signal_logger()`, `_log_risk_failure()`, `_persist_signal()`, `_store_pending_entry()`, `_get_max_hold_seconds()`. Reduced complexity from 26 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_positions()` by extracting 3 helpers: `_should_skip_reconcile()`, `_reconcile_single_position()`, `_handle_position_mismatch()`. Reduced complexity from 33 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._execute_signal_intents()` by extracting 2 helpers: `_should_halt_trading_for_db()`, `_submit_single_intent()`. Reduced complexity from 23 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine.on_fill()` by extracting 3 helpers: `_normalize_fill_correlation_id()`, `_process_fill_with_position_manager()`, `_handle_closed_trade()`. Reduced complexity from 19 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine.flatten_all()` by extracting 5 helpers: `_flatten_cancel_orders()`, `_flatten_close_positions()`, `_flatten_confirm_state()`, `_flatten_reset_local_state()`, `_flatten_handle_result()`. Reduced complexity from 17 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._refresh_strategy_engine()` by extracting 2 helpers: `_get_regime_strategies()`, `_is_strategy_enabled()`. Reduced complexity from 25 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_scan_removals()` by extracting 2 helpers: `_cleanup_orders_for_symbol()`, `_get_pending_order_ids()`. Reduced complexity from 21 to ~8.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._process_scan_additions()` by extracting 3 helpers: `_build_scan_meta()`, `_determine_flow_bias()`, `_enrich_meta_from_features()`. Reduced complexity from 21 to ~6.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_orders()` by extracting 2 helpers: `_sync_open_orders_to_state()`, `_cancel_stale_orders()`. Reduced complexity from 29 to ~10.
- **SonarQube Refactoring**: Refactored `ExecutionEngine._reconcile_db_orders()` by extracting 4 helpers: `_update_open_order_statuses()`, `_update_closed_order_statuses()`, `_mark_stale_orders_cancelled()`, `_apply_reconcile_status()`. Reduced complexity from 34 to ~8.
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
- **L5 Logic Audit**: Added Pydantic field_validators to RiskConfig for bounds checking: max_daily_loss (0-$100k), max_risk_per_trade (0-$10k), max_open_positions (0-100), risk_mode (normal/reduced/off). ([#a4dd8f0](https://github.com/JasperDale420/Cerberus/commit/a4dd8f0))

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
