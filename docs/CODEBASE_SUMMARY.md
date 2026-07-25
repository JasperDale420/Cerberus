# Codebase Summary

A package-by-package index of the Cerberus source tree, with the most important entry points called out. Use this as the first stop when navigating the repo. For runtime flow diagrams, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Top-Level Layout

```
Cerberus/
├── src/                      # All runtime code
├── tests/                    # pytest suite (unit / integration / contract / e2e / smoke)
├── scripts/                  # CLI runners (backtest, WFO, optimization, label generation)
├── tools/                    # Standalone operator tools (e.g. paper_live_harness.py)
├── config/                   # YAML config suite + universe lists
├── docs/                     # This documentation
├── artifacts/                # Generated WFO/backtest artifacts (gitignored content)
├── data/                     # Snapshots, replay bars, ad-hoc data
├── logs/                     # Daily rotating structlog output (gitignored)
├── state_export/             # Host-visible export of the Docker named-volume state (cerberus.db, ledger.db)
├── docker-compose.yml        # `cerberus-trader` + `cerberus-snapshot` + optional `cerberus-scheduler`
├── Dockerfile                # Python 3.12 image
├── Makefile                  # Common make targets (test, lint, type-check)
├── pyproject.toml            # Hatch + uv project
├── PRD.md                    # Historical PRD (large, includes the 5-axis regime upgrade patch)
└── README.md                 # Quick-start
```

## src/ Package Tour

### `src/main.py`

CLI entry point. Argparse defines `--mode {paper,live}`, `--order-executor {gateway,alpaca,noop}`, `--config`, `--run-once`, `--run-agent`, `--eod[/--eod-date]`, `--scheduler`, `--healthcheck`. `_build_strategy_registry()` builds a dict of every known strategy class; only the strategies listed (`enabled: true`) in `config/strategies.yaml` actually get registered into the running `ExecutionEngine` — see the Strategy Registry section below. `_build_strategy_registry()` also dynamically loads any `BaseStrategy` subclass dropped under `src/strategies/graduated/`.

### `src/scheduler.py`

`CerberusScheduler` — APScheduler-based daily session launcher (Mon–Fri cron). Used as the parent process when running under Docker `cerberus-scheduler` or launchd.

### `src/core/`

Shared primitives. Do not import provider/strategy code here.

| Module | Purpose |
|---|---|
| `config.py` | `ConfigLoader` — merges the YAML suite + `APP_*` env overrides |
| `settings.py` | Pydantic Settings for credentials, gateway URLs, Heber paths; `validate_startup_settings()`, `validate_runtime_execution_requirements()` |
| `domain.py` | Enums (`Regime`, `OrderSide`, `OrderType`) and dataclasses (`Bar`, `Signal`, `OrderIntent`, `Position`, `MarketState`, `SymbolState`) |
| `errors.py` | `CerberusError` + `ErrorCode` enum |
| `logger.py` | `StructuredLogger` wrapper that delegates to `empire_core.logger` (service name `"cerberus"`) |
| `http_client.py` | Shared httpx client factory (matches Empire HTTP client pattern) |
| `indicators.py`, `indicators_fast.py` | Rolling EMA / RSI / SMA / Std (fast variant uses numpy) |
| `health.py` | `run_healthcheck()` — validates DB, credentials, gateway, Heber catalog |
| `ledger_adapter.py` | Bridge to `empire_core.ledger` for audit-trail writes |
| `time_utils.py` | Timezone-aware helpers (ET market hours) |

### `src/strategies/`

Each file is a `BaseStrategy` subclass exposing a `name = "..."` class attribute and implementing `on_bar()` / `generate_signal()`. `base.py` provides cooldown, hard-stop time, HMM regime gate, overnight-handling, and signal construction helpers. `config_models.py` holds the Pydantic `StrategyActivationPolicy` model.

#### Strategy Registry

Only strategies listed with `enabled: true` in `config/strategies.yaml` are actually registered into the live/paper trading loop. As of this writing that's 13: `vwap_reversion`, `orb`, `index_mean_reversion`, `flow_momentum`, `gap_fill`, `vwap_trend_rider`, `vix_spike_fade`, `momentum_continuation`, `regime_trend_up`, `regime_bear`, `regime_adaptive`, `regime_flat`, `autoresearch_strategy`. `config/strategies.auto.yaml` (agent Stage 2 output) can add/override on top of this — treat `config/strategies.yaml` + `config/strategies.auto.yaml` together as the source of truth, not any hardcoded list (including this one).

`_build_strategy_registry()` in `src/main.py` also defines a much larger set of strategies that exist in code but are **not** currently in `strategies.yaml`, so they don't run: `mean_reversion_pro`, `trend_rider_pro`, `flow_alpha`, `orb_v2`, `pair_trading_v2`, `rsi_bounce`, `momentum_fade`, `daily_momentum`, `daily_mean_reversion`, `daily_vol_fade`, `regime_adaptive_momentum`, `fusion_v1`, `pair_trading`, `trend_pullback`, `failed_breakout`, `order_flow_imbalance`, `intraday_momentum`, plus anything under `src/strategies/graduated/` (loaded dynamically by filename) and `src/strategies/research_archive/` (not loaded at all).

**Subdirectories** — `archived/` (not registered), `graduated/` (dynamically loaded), `research_archive/` (out-of-band, not loaded).

### `src/engine/` — Execution Core

| Module | Purpose |
|---|---|
| `execution.py` | `ExecutionEngine` — orchestrates the full trading lifecycle (data → strategies → risk → orders → fills → positions → DB) |
| `strategy_engine.py` | `StrategyEngine` + `StrategyActivationPolicy` — gates signals by 5-axis regime |
| `risk.py` | `RiskManager` — daily-loss, position, notional, per-strategy caps; wires Kelly / CPPI / CVaR / HRP sizers |
| `orders.py` | `OrderExecutor` — Alpaca, gateway, and noop adapters |
| `position_manager.py` | Position tracking, trailing stops, partial exits, EOD flatten |
| `market.py` | `MarketStateManager` |
| `kelly.py` | Kelly Criterion sizer (Wasserstein DRO robust mode) |
| `cppi.py` | CPPI drawdown-controlled sizer |
| `cvar_sizer.py` | CVaR-based position sizer |
| `hrp.py` | Hierarchical Risk Parity cross-strategy allocator |
| `adaptive_sizer.py` | Regime-adaptive sizing multipliers |
| `health.py` | Per-engine health probes |

### `src/analysis/` — Regime + Signal Analytics

`regime.py` is `MarketContextService` — the 5-axis classifier (trend / vol / liquidity / risk / session). Companion modules: `bocpd.py` (Bayesian Online Changepoint Detection), `entropy.py`, `vrp.py` (Variance Risk Premium), `gex.py` (gamma exposure), `iv_surface.py`, `momentum_crash.py`. `db.py` is the SQLAlchemy + SQLite layer for `cerberus.db`; `analytics.py` is `AnalyticsEngine` (daily aggregation).

### `src/regime_models/hmm/` — HMM Regime Sidecar

Optional regime-detection package. Additive only — the rule-based engine in `analysis/regime.py` stays in place.

- `service.py` — `HmmRegimeService` (train, predict, shadow/primary mode)
- `adapters.py` — `PomegranateDenseHmmAdapter`
- `features.py` — OHLCV → HMM feature pipeline (deterministic)
- `labeling.py` — Hidden state → regime label mapping
- `config.py` — Nested Pydantic config

Bootstrap with `scripts/bootstrap_hmm_regime.py`. Artifacts land in `artifacts/regime_models/hmm/`.

### `src/data/`

| Module | Purpose |
|---|---|
| `client.py` | `UnifiedDataClient` — REST + WebSocket to Data-Gateway |
| `heber_read_client.py` | Direct Heber parquet reads for historical bars |
| `pipeline.py` | `FeaturePipeline` — indicator computation per bar |
| `unusual_whales.py` | UnusualWhalesClient (REST, optional WS) |
| `atlas_reader.py` | Atlas factor bridge (Gold layer → live signals) |
| `replay_provider.py` | Historical bar replay for backtests |
| `snapshot_manager.py` | Screener / GEX / flow snapshot capture |
| `alpaca.py` | Legacy direct Alpaca client (excluded from mypy) |

### `src/scanner/`

`core.py` — `Scanner` (universe selection, candidate ranking). `streaming_scanner.py` — `StreamingScanner` (event-driven candidate refresh). `pair_scanner.py`, `ranking.py`, `validation.py`, `universe.py` round out the package.

### `src/agent/` — Offline EOD Agent

3-stage offline pipeline (`uv run python -m src.main --eod`):

1. **Stage 1** — rolling-stats health checks, risk-budget adjustments (deterministic).
2. **Stage 2** — grid search + walk-forward validation; writes `config/strategies.auto.yaml`.
3. **Stage 3** (`stage3.py`) — LLM-generated code/parameter proposals, gated by `CERBERUS_STAGE3_APPROVED` env var. `approval.py`, `llm.py`, `bars_provider.py`, `core.py`, `models.py` support these stages.

### `src/backtest/`

Replay runner + deterministic mock execution. Pluggable `fill_models/` directory (`fixed` BPS vs `volume_aware`). `data_quality.py` runs pre-backtest validation. `result_store.py` writes JSON artifacts consumed by the FastAPI service.

### `src/analytics/`

Post-backtest and WFO toolkit. Walk-forward optimization (Optuna harness), Monte Carlo bootstrap, benchmark comparison (alpha/beta/IR vs SPY), meta-labeling, parameter sensitivity (Spearman), diagnostics engine (regime mismatches, time-of-day edge).

### `src/portfolio/`

Cross-strategy aggregation. `signal_aggregator.py`, `risk_budget.py`, `allocator.py`, `performance.py`. See `src/portfolio/README.md`.

### `src/quant/`

Quant primitives: cointegration tests, statistical filters, regime statistics, volatility models. Used by pair-trading and the HMM service.

### `src/api/`

`backtest_api.py` — FastAPI app serving backtest/WFO results to EmpireUI. Port 8002 by default. See [`API_REFERENCE.md`](API_REFERENCE.md).

### `src/config/`

`models.py` — Pydantic models for the YAML config tree (`RiskConfig`, `StrategyConfig`, etc.). Imported by `RiskManager` and others.

## tools/ and scripts/ Highlights

| Path | Purpose |
|---|---|
| `tools/paper_live_harness.py` | Paper-live integration harness (`--scenario happy\|failure\|risk`, `--inject-signal`) — see [`../DEVELOPER_NOTES.md`](../DEVELOPER_NOTES.md) |
| `scripts/run_backtest.py` | Single backtest run |
| `scripts/run_wfo.py`, `run_wfo_*.py` | Walk-forward optimization variants |
| `scripts/optimize_strategy.py` | Optuna parameter sweep for a single strategy |
| `scripts/bootstrap_hmm_regime.py` | Train initial HMM regime model |
| `scripts/paper_live_test.py` | Separate script-level paper/live comparison harness with its own `KILL_SWITCH`/`PAPER_LIVE` env vars |
| `scripts/analyze_wfo_results.py`, `wfo_dashboard.py` | WFO post-processing |
| `scripts/download_bars.py`, `download_bars_parquet.py`, `backfill_bars.py` | Bar ingestion helpers |
| `scripts/label_*.py` | Label-set generation (regime, liquidity, macro, earnings, sessions) |
| `scripts/run_holdout.py`, `run_oos_validation.py` | Holdout / OOS validation runners |
| `scripts/check_ledger_health.sh` | Ledger integrity check |
| `scripts/com.cerberus.main.paper.plist`, `com.cerberus.paper.plist` | macOS launchd templates |

## tests/ Layout

| Path | Marker | Purpose |
|---|---|---|
| `tests/unit/` | `unit` | Fast, isolated tests (no I/O, no network) |
| `tests/integration/` | `integration` | Real DB / file I/O / component-to-component |
| `tests/e2e/` | `e2e` | Full system flow |
| `tests/smoke/` | (varies) | Lightweight startup checks |
| `tests/strategies/` | mixed | Per-strategy unit tests (one file per strategy) |
| `tests/data/` | mixed | Data-pipeline fixtures and adapters |
| `tests/conftest.py` | — | Safe defaults: `ALPACA_API_KEY=test`, `ALPACA_PAPER=True`, `DATA_INGESTION_URL=http://central.test` |
| `tests/test_*.py` | mixed | Per-module suites (engine, scanner, agent, regime, etc.) |

Coverage gate: `--cov-fail-under=68`.

## Configuration Files (`config/`)

See [`CONFIGURATION_GUIDE.md`](CONFIGURATION_GUIDE.md) for full details.

- `config.yaml` — base runtime
- `strategies.yaml` — per-strategy enable + activation policies
- `strategies.auto.yaml` — agent Stage 2 overrides (auto-generated)
- `risk.yaml` — RiskManager limits (loss caps, position caps, notional caps)
- `scanner.yaml` — universe scanner thresholds
- `universe.yaml` + `universe_nasdaq100.txt`, `universe_sp500.txt` — tradeable universe
- `logging.yaml` — log handlers
- `backtest_*.yaml` — backtest profiles (smoke, 5yr, jan2024, portfolio, autoresearch, etc.)

## Key Dependencies

| Package | Role |
|---|---|
| `empire-core` | Shared logger, errors, HTTP client, ledger (editable monorepo dep) |
| `empire-schemas` | Shared data contracts (EventEnvelope, NormalizedBar) |
| `empire-gateway-client` | Data-Gateway SDK |
| `alpaca-py` `>=0.43.2` | Broker SDK |
| `unusualwhales-python-client` `>=5.0.1` | Vendored flow data client (in-repo dir) |
| `pydantic` / `pydantic-settings` `>=2` | Domain + settings models |
| `pyarrow`, `pandas`, `pandas-ta`, `numpy` | Bar processing / indicators |
| `sqlalchemy` `>=2` | `cerberus.db` ORM |
| `APScheduler` `>=3.11` | Daily session launcher |
| `httpx` `>=0.27,<0.29` | All HTTP (REST + WS upgrades) |
| `structlog` `>=24.1` | JSON logging via `empire_core.logger` |
| `optuna` `>=4.7` | WFO parameter search |
| `pomegranate` `>=1.1.2` | HMM regime sidecar |
| `arch`, `statsmodels`, `filterpy` | Volatility, time-series, Kalman utilities |
| `fastapi`, `uvicorn` | Backtest API |
| `tenacity` `>=9.1.4` | Retry decorators |
| `pytz` | Market timezones |

Dev: `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`, `bandit`, `detect-secrets`, `pre-commit`. Note: `requirements.txt` at repo root is a separately pinned full dependency snapshot ("Dec 2025 audit") — the project is uv/`pyproject.toml`-managed day to day; check both if you touch dependencies to avoid drift.

## SQLite Stores

- `cerberus.db` — primary analytics DB (signals, orders, fills, trades, regime snapshots, daily aggregates). Touched by `src/analysis/db.py`. Symlinked to `state_export/cerberus.db` in the current Docker setup.
- `ledger.db` — append-only trade audit ledger written via `empire_core.ledger`. Very large (multi-GB); do not edit by hand. Symlinked to `state_export/ledger.db`.
- `cerberus_backup_*.db`, `ledger_backup_*.db` — manual point-in-time backups.

## Related Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — diagrams & runtime flow
- [`CONFIGURATION_GUIDE.md`](CONFIGURATION_GUIDE.md) — full config suite
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker, launchd, scheduler
- [`../TESTING.md`](../TESTING.md) — pytest + backtest + WFO
- [`API_REFERENCE.md`](API_REFERENCE.md) — FastAPI endpoints
- [`../AGENTS.md`](../AGENTS.md) / [`../CLAUDE.md`](../CLAUDE.md) — Empire patterns enforced here
