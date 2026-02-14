# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Multi-Asset Support** (2026-02-14):
  - Introduced `MarketSession` abstraction to support both US Equity (9:30-16:00 ET) and Crypto (24/7) trading sessions.
  - Added `CERBERUS_ASSET_CLASS` setting (`us_equity` or `crypto`).
  - Added `get_crypto_bars` and `get_crypto_trades` to `CentralApiClient`.
  - Updated `UniverseBuilder` to route data requests based on asset class.
  - Made `GatewayStreamClient` asset-class-aware: selects `crypto_bars` or `stock_bars` feeds dynamically.
  - WebSocket subscription subscribes to crypto universe symbols when `CERBERUS_ASSET_CLASS=crypto`.
  - Added `cerberus-crypto` Docker service (profile: `crypto`) for headless 24/7 operation.
  - Added `LiveBarBuffer` to feed real-time WebSocket bars into the scanner, bypassing 15-min delayed REST endpoints after initial seed.

- **Backtest Data Provisioning via Data-Gateway + Heber** (2026-02-13):
  - Added backfill methods to `CentralApiClient`: `request_backfill`, `get_backfill_status`, `wait_for_backfill`, `cancel_backfill`.
  - Added `BacktestDataProvisioner` orchestrator supporting chunked backfills and Gateway fallback.
  - Added `--data-source` parameter to `BacktestRunner` (`alpaca`, `gateway`, `heber`).
  - Added backfill configuration settings to `settings.py`.
  - Added 22 unit tests covering backfill methods and provisioner flow.

### Changed

- `main.py`: Refactored main loop to use `MarketSession` instead of hardcoded market hour checks.
- `api_client.py`: Added retry logic for 404s on backfill polling.

### Fixed
- Avoided fetching GEX when flow data is empty; added context logging when flow/GEX fetch fails.
- Handled None close values in feature close extraction to prevent type errors.
- Respected data backend when deciding whether to start Alpaca streams.

### Fixed

- (2026-02-14): FeaturePipeline close extraction now handles None/invalid values with warning fallback.
- (2026-02-14): `_should_start_alpaca_stream()` now accounts for `data_backend` to match gateway/legacy behavior.
- **Critical: Zero-Trade Pipeline Fix** (2026-02-13):
  - Root cause: `_should_start_alpaca_stream()` returned `False` in `gateway+noop` mode, preventing bar WebSocket stream from starting. Without bars, `on_bar()` never fired — zero signals, zero trades.
  - Synced local `main.py` with Docker image (session control, strategy registry, market-hours helpers).
  - Gateway bar stream now always starts when `data_backend=gateway`, independent of order executor.
  - `_should_start_alpaca_stream()` scoped to only control direct Alpaca streams (executor=alpaca).
  - Added `--order-executor gateway` with `GatewayOrderExecutor` for Data-Gateway routing.
  - Changed `docker-compose.yml` default from `--order-executor noop` to `--order-executor gateway`.
- **Hard stop validation now fails fast** (2026-02-14):
  - Invalid `hard_stop_time` formats/ranges now log an error and raise to avoid silent misconfiguration.
- **Gateway/legacy stream gating guardrails** (2026-02-14):
  - `_should_start_alpaca_stream()` now respects `data_backend` and avoids unexpected argument errors in gateway mode tests.
- **Feature pipeline close extraction guard** (2026-02-14):
  - `_extract_closes()` now treats `None` close values as `0.0` instead of raising.
- **CI tooling config typing cleanup** (2026-02-14):
  - Cast CI config loaders to satisfy mypy return type expectations.
- **Gateway stream bar coercion** (2026-02-14):
  - Normalize gateway bar fields with safe float coercion to avoid None/typing errors.
- **Snapshot persistence call fix** (2026-02-14):
  - Feature snapshot persistence now uses keyword arguments to match snapshot manager signature.
- **Alpaca historical response guards** (2026-02-14):
  - Use safe `data` extraction when Alpaca SDK responses are dicts or objects.
- **Gateway retry backoff typing** (2026-02-14):
  - Tightened retry backoff parsing/casts to keep mypy clean.

### Added

- (2026-02-14): GapFill config now supports `min_or_volume` (opening range volume filter).
- Automated hourly report generated (2026-02-13 04:03 UTC).
- **ORB opening-range filter** (2026-02-14):
  - Added `min_or_range_pct` to skip narrow opening ranges and avoid low-signal breakouts.

- **Cerberus/Data-Gateway/Heber integration gate tooling** (2026-02-11):
  - Added one-command integration smoke script:
    - `scripts/smoke_gateway_heber_integration.py`
  - Added unit coverage for smoke gate checks:
    - `tests/unit/test_smoke_gateway_heber_integration_unit.py`

- Added local Claude/Swarm workspace tooling assets and skill bundles:
  - `.claude/` helpers, settings, and skill definitions
  - `.claude-flow/` agent/task state files
  - `.swarm/` runtime state files
  - `CLAUDE.md` and `vectors.db` local support artifacts

- **Data-Gateway/Heber Phase 1 Completion** (2026-02-10):
  - Enhanced dual-read parity logging with comprehensive comparison:
    - Bar value comparison (OHLCV) with percentage difference tracking
    - Trades count parity logging for gateway vs legacy
    - Flow count parity logging for gateway vs legacy
    - GEX data parity confirmation logging
    - Success logs for confirmed parity across all data types
  - Startup environment validation for gateway/heber modes:
    - Added `validate_startup_mode()` method to `Settings` class
    - Added `validate_startup_settings()` function for main entry point validation
    - Validates required env vars based on configured backend mode:
      - Gateway mode: requires `CERBERUS_GATEWAY_URL` and `CERBERUS_GATEWAY_KEY`
      - Heber mode: requires `CERBERUS_HEBER_CATALOG_URL`
      - Legacy/dual+failover: requires Alpaca credentials
    - Integrated validation into `src/main.py` startup sequence
  - Comprehensive gateway/failover integration tests:
    - Created `tests/integration/test_gateway_failover_integration.py` with 11 test scenarios
    - Created `tests/unit/test_startup_validation_unit.py` with 14 validation tests

### Changed

- Universe static file parsing now strips inline `#` comments before symbol extraction (2026-02-14).

- Feature pipeline now reuses extracted close prices per symbol to avoid duplicate passes.
  - Test coverage for: legacy mode, gateway mode, dual mode, failover behavior, parity logging
  - Tightened startup validation test precision:
    - Gateway required-field test now uses explicit empty URL value for deterministic assertions
    - Added a focused unit test confirming custom gateway URL only flags missing gateway key

### Documentation

- Added integration planning docs for Cerberus migration to Data-Gateway + Heber:
  - `docs/cerberus-data-gateway-heber-architecture.md`
  - `docs/cerberus-data-gateway-heber-migration-roadmap.md`
  - `docs/cerberus-data-gateway-heber-implementation-checklist.md`
- Added Data-Gateway/Heber runtime variable reference updates to:
  - `docs/environment-variables.md`
  - `.env.example`
- Comprehensive documentation audit and remediation completed.
- Rewrote `README.md` to match current runtime architecture, commands, and modules.
- Reworked `docs/architecture.md` with updated system/data-flow diagrams and module map.
- Added `docs/environment-variables.md` as source-of-truth env var reference.
- Updated `.env.example` to include current runtime vars and APCA aliases.
- Updated `docs/runbook.md`, `docs/order_flow.md`, and `docs/strategy_guide.md` for current interfaces/CLI behavior.
- Updated `CONTRIBUTING.md`, `TESTING.md`, and `SECURITY.md` for current workflows.
- Removed stale auto-generated `codebase.md`.

### Changed

- Deduplicated Stage 3 approval checks into shared helper used by weekly report and proposals.
- ORB stop loss now caps risk using `stop_loss_pct` to avoid oversized stops on wide ranges (2026-02-14).

- **Gateway-first trading execution path** (2026-02-13):
  - Set gateway-first runtime defaults in `src/core/settings.py`:
    - `CERBERUS_DATA_BACKEND` default is now `gateway`
    - `ALPACA_PAPER` default is now `true`
  - Updated main runtime order routing in `src/main.py`:
    - `--order-executor` now defaults to `gateway`
    - Added `gateway` executor option alongside `alpaca` and `noop`
    - Blocked direct `alpaca` order execution when gateway data mode is active
  - Added gateway trading adapters in `src/data/api_client.py`:
    - `submit_alpaca_order`
    - `get_alpaca_orders`
    - `cancel_alpaca_order`
  - Added `GatewayOrderExecutor` in `src/engine/orders.py` to route submissions/cancels through Data-Gateway.
  - Updated defaults/docs:
    - `.env.example` now sets `CERBERUS_DATA_BACKEND=gateway`
    - `docs/environment-variables.md` defaults updated for `CERBERUS_DATA_BACKEND` and `ALPACA_PAPER`
  - Added coverage:
    - `tests/unit/test_gateway_order_executor_unit.py`
    - `tests/contract/test_central_api_client_contract.py` order-submit contract
    - `tests/unit/test_startup_validation_unit.py` gateway/paper defaults assertion
  - Added gateway live-stream ingestion path:
    - New `src/data/gateway_stream.py` WebSocket client for `ws://.../ws` auth + `stock_bars` subscriptions.
    - Updated `src/main.py` to stream bars via Data-Gateway when `CERBERUS_DATA_BACKEND=gateway|dual`, while retaining Alpaca stream for legacy mode.
    - Added unit coverage in `tests/unit/test_gateway_stream_client_unit.py`.

- **Central API retry classification for gateway integration** (2026-02-11):
  - Added status-aware retry policy in `src/data/api_client.py`:
    - no retry for `401/403`
    - retry for `429`, `5xx`, timeout, and transport errors
    - support for `Retry-After` with exponential backoff fallback
  - Updated checklist progress in:
    - `docs/cerberus-data-gateway-heber-implementation-checklist.md`

- Added Phase 1 integration scaffolding for Data-Gateway/Heber:
  - Extended runtime settings in `src/core/settings.py` with backend mode and Gateway/Heber config.
  - Upgraded `src/data/api_client.py` to Data-Gateway v1 routes and `X-Gateway-Key` support while preserving LLM chat compatibility.
  - Expanded `src/core/health.py` to check Data-Gateway and Heber connectivity, including gateway-mode credential handling.
  - Updated contract tests in `tests/contract/test_central_api_client_contract.py` for route and header expectations.
- Wired gateway-backed fetching in runtime data path:
  - Added Data-Gateway adapters in `src/data/api_client.py` for Alpaca trades and UW GEX.
  - Enabled `src/data/fetcher.py` to route bars/trades/flow/gex through Data-Gateway when `CERBERUS_DATA_BACKEND=gateway|dual`, with failover control via `CERBERUS_FAILOVER_TO_LEGACY`.
  - Added lightweight dual-mode parity diagnostics for bar-count mismatch in `src/data/fetcher.py`.
  - Injected `CentralApiClient` into `FeaturePipeline` from `src/main.py`.
  - Added new contract coverage for `get_alpaca_trades` and `get_uw_gex`.
- Extended gateway-backed universe sourcing:
  - Added Data-Gateway screener adapters (`most_actives`, `movers`) in `src/data/api_client.py`.
  - Updated `src/scanner/universe.py` to use Data-Gateway for dynamic volume/screener sources in gateway mode, with optional legacy failover.
  - Injected `CentralApiClient` into `UniverseBuilder` in `src/main.py`.
  - Added contract coverage for `get_alpaca_most_actives` and `get_alpaca_movers`.
- **HTTP Client: requests → httpx** — Migrated `scripts/update_universe_lists.py` from `requests` to `httpx`
- **Multi-Axis Regime Migration**: Replaced legacy BULL/BEAR/CHOP regime classification with full 5-axis multi-axis regime system
  - `Signal.regime` field removed, now uses `Signal.regime_tags: Dict[str, str]` and `Signal.regime_confidence: Dict[str, float]`
  - `Position.regime_at_entry` replaced with `Position.regime_tags_at_entry: Dict[str, str]`
  - `ClosedTradeInfo` now stores `regime_tags_at_entry/exit` dicts with 5 axes
  - Trades record full regime context: `{trend, vol, liquidity, risk, session}` at entry and exit
  - Removed legacy regime config checks from `RiskManager`
  - Updated `base.py._create_signal()` to populate regime_tags from `MarketState.regime_snapshot`
- **VXX-Based Risk Axis**: Risk axis now properly uses VXX momentum (rising VXX = RISK_OFF, falling = RISK_ON)
  - Added `update_vol(bar)` to `MarketContextService` and `MarketStateManager`
  - Wired VXX bar processing in both `BacktestRunner` and `ExecutionEngine` for parity
  - Risk distribution improved from 84% neutral to 44% neutral / 40% risk_off / 16% risk_on

### Added

- **Backtest Parity Improvements**: Enhanced backtest realism with configurable simulation settings
  - Volume-aware partial fills: `partial_fill_mode` (none|fixed|volume_aware) with `partial_fill_rate` for liquidity modeling
  - Volume-impact slippage: `slippage_mode` (fixed|volume_impact) with `slippage_impact_mult` for market impact simulation
  - ATR-based spread: `spread_mode` (fixed|atr_based) for volatility-sensitive spread modeling
  - Flow strategy gating: `disable_flow_strategies` config to skip flow-dependent strategies in backtest
  - New `backtest:` config section in `config.yaml` with all realism settings
  - Unit tests in `tests/unit/test_backtest_parity_unit.py` (13 new tests)
- **Dynamic Ticker Discovery**: True live-parity stock discovery using Alpaca Screener API
  - `AlpacaClient.get_most_actives()` - Fetch top volume stocks
  - `AlpacaClient.get_movers()` - Fetch top gainers/losers
  - `UniverseBuilder` screener dynamic source with configurable `most_actives_top_n` and `movers_top_n`
  - `scripts/capture_screener_snapshot.py` - Daily snapshot capture for future historical replay
  - Setup guide: `docs/screener_snapshot_setup.md`

### Added

- **Config: pydantic-settings for runtime env vars** (2026-02-09)
  - Created `src/core/settings.py` with `Settings(BaseSettings)` for Alpaca credentials
  - Migrated `health.py` from `os.getenv` to settings with `resolved_*` property helpers
  - Supports both `ALPACA_*` and `APCA_*` naming conventions via resolved properties
  - Added `pydantic-settings>=2.0` dependency

### Added

- **Alpha Overhaul Phase 4: Order Flow & Microstructure**:
  - `Trade Flow Imbalance (TFI)`: High-fidelity microstructure edge using Tick Test (Lee-Ready).
  - `Net Gamma Exposure (GEX)`: Integrated Unusual Whales greek exposure API for MM pinning analysis.
- **Alpha Overhaul Phase 5: Statistical & Regime Alpha**:
  - `Fractional Differentiation`: achieved stationarity while preserving memory ($d \approx 0.4$).
  - `Hurst Exponent`: R/S analysis for regime classification (MR vs Trending).
- **Alpha Overhaul Phase 6: Meta-Labeling & Probabilistic Execution**:
  - `Signal Enrichment`: Every `Signal` now carries a `feature_snapshot` representing the full alpha context at time of generation.
  - `MetaLabeler`: Implementation of a heuristic v1 vetteur using Hurst, TFI, and GEX to reject low-probability trades.
  - `Database Persistence`: Enhanced `signals` table schema to log feature snapshots for future model training.
- **Alpha Overhaul Phase 7: Automated Parameter Tuning & Walk-Forward**:
  - `Dynamic Parameter Updates`: Refactored `BaseStrategy` and all strategies to support `update_params` for runtime parameter injection.
  - `GridSearchOptimizer`: Modular parameter search with custom scoring functions.
  - `WalkForwardManager`: Rolling window stability checks to prevent overfitting.
  - `Stage2Tuner Integration`: Enhanced agent tuning with walk-forward validation.
- **Historical Replay Data Architecture**:
  - `ExternalSnapshot`: Layer 1 table for raw API data (GEX, flow) capture.
  - `FeatureSnapshot`: Layer 2 table for computed features at point-in-time.
  - `DailyUniverse`: Tracks which symbols passed filtering each day.
  - `SnapshotManager`: Orchestrates capture of external API data and computed features.
  - `ReplayProvider`: Provides historical data from snapshots for offline backtesting.
  - `FeaturePipeline Integration`: Auto-persists snapshots when `snapshots.enabled: true`.
- **Statistical Dependencies**: Added `statsmodels==0.14.4` to `requirements.txt`.
- **Agent Stage 2 Pipeline Integration** (PRD 9.2): Parameter tuning now runs as part of daily agent cycle in `run_cycle_with_db()`. Strategies not disabled by Stage 1 are evaluated for parameter improvements.
- **Agent Stage 3 Weekly Analysis** (PRD 9.3): New `run_weekly_analysis()` method generates weekly performance reports with LLM-powered feature/model recommendations. Reports saved to `artifacts/weekly_reports/`.
- **Scheduler Weekly Job**: Added Friday 16:30 ET job for automatic Stage 3 weekly analysis.
- **Stage 2 Search Space Config**: Added parameter search space for `vwap_reversion`, `orb`, `gap_fill`, `vwap_trend_rider`, and `index_mean_reversion` strategies.
- **Scanner Profiles**: Added `VixSpikeFadeProfile` and `MomentumContinuationProfile` to complete scanner coverage for all active strategies
- **Multi-Axis Regime Schema** (PRD Regime Upgrade Patch §7):
  - Trade table: Added `regime_tags_entry_json` and `regime_tags_exit_json` columns for full regime context
  - RegimeHistory table: Added `model_version`, `trend`, `vol_regime`, `liquidity`, `risk`, `session`, `vol_of_vol`, `liquidity_score`, `risk_score`, `confidence_json` columns
- **Signal Fusion Core (Phase 2)**: Added propagation of `atr`, `orb_high`, `orb_low`, `dof_score`, and `relative_strength` from features to strategy execution metadata.
- **Relative Strength (RS) Calculation**: Implemented benchmark-anchored RS calculation in `BacktestFeaturePipeline` and `FeaturePipeline`.
- **Directional Options Flow (DOF) Support**: Added skeleton and metadata mapping for DOF/UW flow features in `SymbolFeatures`.

### Fixed

- **Stability and quality gate fixes** (2026-02-10):
  - Restored compatibility imports for archived strategies via `src/strategies/failed_breakout.py` and `src/strategies/trend_pullback.py`.
  - Reinstated legacy CHOP-only guards for standalone `FailedBreakoutStrategy` and `VWAPReversionStrategy` tests.
  - Fixed `RiskManager` regime-disable rejection behavior (`REGIME_DISABLED`) and preserved rejection reason precedence when qty/risk resolves to zero.
  - Fixed signal DB persistence JSON serialization by sanitizing datetime-containing payloads before insert.
  - Normalized agent regime placeholders (e.g. `{}`) to `chop` for Stage 1 action targeting.
  - Added Stage 2 evaluator callable compatibility for deterministic test evaluators.
  - Restored flow-metric backward compatibility to 5-tuple output and kept DOF scoring in pipeline enrichment.
  - Hardened pair scanner for non-datetime price indexes and replaced statsmodels OLS dependency with numpy least-squares in pair stats/half-life calculations.
  - Made weekly scheduler job opt-in (`enable_weekly_analysis`) to keep daily-only default behavior backward compatible.

- **Critical: TechnicalFeatures Missing Field**: Added missing `last_updated` field to `TechnicalFeatures` constructor in `calculator.py`.
- **Critical: Alpaca Trade Stream Handler**: Fixed async handler compatibility with latest Alpaca SDK in `alpaca.py`.
- **Critical: Zero Signal Backtest Bug**: Fixed `BacktestFeaturePipeline` lookup window (extended to 24h) and data parity issue that prevented signal generation.
- **Critical: Scanner Return Contract**: Restored `Scanner.scan()` to return all evaluation candidates, preventing global strategy routing failures.
- **Critical: SymbolFeatures Dataclass Error**: Fixed `TypeError: non-default argument follows default argument` by reordering fields in `SymbolFeatures`.
- **Strategy Execution Fix**: Fixed missing `time` import in `FusionStrategyV1.py`.
- **Backtest Determinism**: Updated `Scanner` cache TTL to use simulated `scan_time` instead of `datetime.now()`.

### Changed

- **Backtest Feature Parity**: Aligned `BacktestFeaturePipeline` SymbolFeatures construction with live `FeaturePipeline` to ensure consistent alpha signal availability.

- **Disabled Partial Exits** (`config/backtest_5yr/config.yaml`):
  - Set `partial_exits.enabled: false` - analysis showed early profit-taking harmed overall performance
  - Trades held >60 min had 51% WR and +$101K profit vs early exits losing money
  - Reduced slippage from 5.0 to 2.0 bps for more realistic execution modeling

### Removed

- **Archived Strategies** (`src/strategies/archived/`):
  - Moved `failed_breakout.py` to archive - 5-year backtest showed no profitable edge in any regime
  - Moved `trend_pullback.py` to archive - 5-year backtest showed -$1.8M loss with no profitable conditions
  - Removed from isolation config generator (`scripts/generate_isolation_configs.py`)

### Performance

- **Backtest Engine Optimizations** - ~7x speedup for large backtests:
  - **Order List Indexing** (`src/backtest/mock_executor.py`): Added `_pending_by_symbol` dict for O(1) order lookup per bar instead of O(N) scanning through all pending orders
  - **Sync Core Loop** (`src/backtest/runner.py`): Extracted synchronous `_process_loop_event_core()` from async wrapper to eliminate async overhead for CPU-bound bar processing
  - **Lazy Event Stream Merge** (`src/backtest/runner.py`): Replaced eager `list.sort()` with `heapq.merge()` for O(N) lazy merging of pre-sorted per-symbol bar streams
  - Verified: All 27 backtest tests pass; 5-year 25M bar backtest completes in ~8.5 minutes vs ~1 hour previously

### Fixed

- **Critical: Backtest Session Filters Now Enforced** (`src/engine/strategy_engine.py`):
  - Fixed `scanner_bypass=True` causing ALL activation policy checks to be skipped
  - Session filters (e.g., `session: [opening, midday]`) now properly block premarket trades
  - Before: 97% of trades occurred in premarket despite config filters
  - After: 0% premarket trades, proper RTH-only execution

### Added

- **Backtest Session VWAP Injection** (`src/backtest/runner.py`):
  - Added `_vwap_state` tracking dict for cumulative TPV/volume per symbol per session
  - Session VWAP calculated and injected as `bar.vwap` attribute for VWAP-based strategies
  - Enables `vwap_trend_rider` and `vwap_reversion` in offline backtests

- **Backtest Gap Calculation** (`src/backtest/runner.py`):
  - Added `_prev_day_closes` tracking for gap percentage calculation
  - `gap_pct` injected into `symbol_state.meta` for `gap_fill` strategy

- **Enhanced Regime Analysis Script** (`scripts/analyze_regime_ci.py`):
  - Wilson score 95% confidence intervals for statistically robust win rate estimation
  - Confidence ratings based on sample size (insufficient/low/medium/high)
  - High-confidence deployment recommendations (CI lower > 25%)

### Documentation

- **PRD.md Audit Update (Dec 2025)**: Comprehensive alignment of PRD with implementation
  - Added Section 12: Advanced Exit System (trailing stops, partial profits, regime-aware stops)
  - Added Section 13: Backtesting Engine (volume-aware fills, slippage modeling, ATR spreads)
  - Added strategies 9 (Momentum Continuation) and 10 (VIX Spike Fade) to Section 7.2
  - Marked PRD Regime Upgrade Patch as IMPLEMENTED
- **README.md**: Updated capabilities to include 5-axis regime system, advanced exits, and backtesting engine
- **architecture.md**: Updated Key Design Principles with 5-axis regime system replacing legacy routing

### Changed

- **WebSocket Resilience Hardening** (`src/data/alpaca.py`):
  - Added explicit `feed` parameter (`DataFeed.SIP`/`IEX`) to prevent IEX fallback on premium accounts
  - Added jitter to exponential backoff (0-0.5s random)
  - Added terminal error detection (`connection limit exceeded`) to enable REST fallback
  - Limited retries to 5 with 30s max backoff
  - Stored `config_loader` for feed configuration access
  - Added resilience constants from KI: `HEARTBEAT_TIMEOUT_SEC=120`, `FIRST_BAR_TIMEOUT_SEC=10`
- **Multi-Axis Regime Migration**: Replaced legacy BULL/BEAR/CHOP regime classification with full 5-axis multi-axis regime system
  - `Signal.regime` field removed, now uses `Signal.regime_tags: Dict[str, str]` and `Signal.regime_confidence: Dict[str, float]`
  - `Position.regime_at_entry` replaced with `Position.regime_tags_at_entry: Dict[str, str]`
  - `ClosedTradeInfo` now stores `regime_tags_at_entry/exit` dicts with 5 axes
  - Trades record full regime context: `{trend, vol, liquidity, risk, session}` at entry and exit
  - Removed legacy regime config checks from `RiskManager`
  - Updated `base.py._create_signal()` to populate regime_tags from `MarketState.regime_snapshot`
- **VXX-Based Risk Axis**: Risk axis now properly uses VXX momentum (rising VXX = RISK_OFF, falling = RISK_ON)
  - Added `update_vol(bar)` to `MarketContextService` and `MarketStateManager`
  - Wired VXX bar processing in both `BacktestRunner` and `ExecutionEngine` for parity
  - Risk distribution improved from 84% neutral to 44% neutral / 40% risk_off / 16% risk_on

### Added

- **Backtest Parity Improvements**: Enhanced backtest realism with configurable simulation settings
  - Volume-aware partial fills: `partial_fill_mode` (none|fixed|volume_aware) with `partial_fill_rate` for liquidity modeling
  - Volume-impact slippage: `slippage_mode` (fixed|volume_impact) with `slippage_impact_mult` for market impact simulation
  - ATR-based spread: `spread_mode` (fixed|atr_based) for volatility-sensitive spread modeling
  - Flow strategy gating: `disable_flow_strategies` config to skip flow-dependent strategies in backtest
  - New `backtest:` config section in `config.yaml` with all realism settings
  - Unit tests in `tests/unit/test_backtest_parity_unit.py` (13 new tests)
- **Dynamic Ticker Discovery**: True live-parity stock discovery using Alpaca Screener API
  - `AlpacaClient.get_most_actives()` - Fetch top volume stocks
  - `AlpacaClient.get_movers()` - Fetch top gainers/losers
  - `UniverseBuilder` screener dynamic source with configurable `most_actives_top_n` and `movers_top_n`
  - `scripts/capture_screener_snapshot.py` - Daily snapshot capture for future historical replay
  - Setup guide: `docs/screener_snapshot_setup.md`

### Fixed

- **H1 Logic Audit (New)**: Fix `VWAPReversionStrategy.on_bar()` unbound variable crash when price is within VWAP bands. Initialize `signal = None` before conditional blocks.
- **H2 Logic Audit (New)**: Fix `FlowMomentumStrategy` threshold logic that allowed weak flow signals. Now properly rejects all signals below `min_flow_zscore`.
- **H3 Logic Audit (New)**: Fix `BacktestAnalyzer._calculate_drawdown()` inconsistency with unrealized PnL. Peak tracking now based only on closed trades for consistency; added clarifying docstring.
- **Agent.run_cycle_with_db**: Now calls `apply_actions()` after persisting actions to DB, ensuring `strategies.auto.yaml` is automatically written.
- **Agent.apply_actions**: Fully implements REDUCE_RISK and DISABLE_STRATEGY actions: writes config to `strategies.auto.yaml`, supports both strategy-level and regime-specific overrides, adds floor at 0.0 when risk drops below threshold.
- **E2E Test Risk Values**: Update `test_prd_vertical_slice_success_metric.py` to use valid risk values within the new RiskConfig validation limits ($10k daily loss, not $1M).
- **M1 Logic Audit (New)**: Add DEBUG-level logging to silent exception handlers in `PositionManager` for MAE/MFE tracking and max-hold check failures. Improves observability without breaking trading.
- **M2 Logic Audit (New)**: Fix `RiskManager` positions_carried_forward tracking to capture count BEFORE session rollover reset for accurate logging.
- **Momentum Strategy Target Fix**: Changed target calculation in `MomentumContinuationStrategy` from bar-range-based to risk-based (`abs(entry-stop) * risk_reward`) for consistent R:R ratio.
- **M3 Logic Audit (New)**: Raise `Scanner` watchlist cap from 30 to 50 with documented PRD recommendation. Configurable limit with clearer warning message.
- **M4 Logic Audit (New)**: Add robust date extraction in `ExecutionEngine._update_symbol_state()` with clock fallback when bar_time is unusable. Prevents stale feature cache on date parsing failures.
- **M5 Logic Audit (New)**: Add `bar_duration_minutes` config parameter to `BaseStrategy` for accurate cooldown calculation across different timeframes. Defaults to 1.0 for backward compatibility.
- **L1 Logic Audit (New)**: Add named constant `QTY_EPSILON = 1e-7` in `BacktestAnalyzer` to replace magic number for floating-point quantity comparisons. Documents purpose and prevents potential infinite loops.
- **L3 Logic Audit (New)**: Add named constants `STOP_BUFFER_LONG` (0.99) and `STOP_BUFFER_SHORT` (1.01) in `FlowMomentumStrategy` for emergency stop buffer calculations. Documents the 1% buffer purpose.
- **CI Fix**: Add `asyncio_mode = "auto"` to pytest configuration in `pyproject.toml`. Enables pytest-asyncio to detect and run async test functions.
- **CI Fix**: Add missing `pytest-asyncio` dependency to `requirements.txt`. CI environment was missing this package, causing async tests to fail with "async def functions are not natively supported".
- **Memory Audit H1**: Add LRU eviction to Scanner `_feature_cache` using OrderedDict with configurable `feature_cache_maxsize` (default 1000). Prevents unbounded memory growth in long-running sessions.
- **Memory Audit H2**: Add LRU eviction to DataFetcher `_bars_cache` using OrderedDict with configurable `bars_cache_maxsize` (default 500). Evicts oldest entries when limit exceeded.
- **Memory Audit M1**: Convert ExecutionEngine `closed_trades` from unbounded list to bounded deque with maxlen=5000. Keeps last 5000 trades in multi-day runs.
- **Dead Code Removal**: Remove unused code: `ScanningError`, `run_scan_symbols()`, `run_scan_async()`, `_safe_float()`. ~150 lines removed.
- **Dead Code Removal**: Remove unused `data/models.py` (Trade/Quote classes) and test file. ~30 lines removed.
- **Indicator Consolidation**: Refactor `_compute_atr()` and `_compute_adx()` in `calculator.py` to use `RollingATR` and `RollingADX` incremental indicators. Removes ~50 lines of duplicate Wilder smoothing code.

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

## 2026-01-05

### Fixed

- **Critical ConfigLoader Bug**: Fixed issue where specific config files (e.g., `config_vwap_trend_rider.yaml`) were being ignored - ConfigLoader now loads the specific file AFTER suite files to properly override settings
- **Strategy Isolation Configs**: Added ORB to `generate_isolation_configs.py` - now generates all 7 strategy configs

### Changed

- `src/core/config.py`: ConfigLoader.load_config() now tracks specific file paths and loads them after the suite files to allow proper overrides
- `scripts/generate_isolation_configs.py`: Added ORB strategy configuration with OPEN_ACTIVATION filters
