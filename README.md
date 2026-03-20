# Cerberus Trading System

Cerberus is an intraday algorithmic trading engine for US equities. It supports paper/live modes, multi-strategy signal generation, pre-trade risk checks, execution via Alpaca, and SQLite-backed analytics.

- Python: `>=3.12`
- Package/deps: `pip` + `requirements.txt`
- Primary entrypoint: `python -m src.main`

## Quick Start

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
2. Configure environment:
```bash
cp .env.example .env
# Fill in API credentials
```
3. Run health check:
```bash
python -m src.main --healthcheck
```
4. Run paper mode:
```bash
python -m src.main --mode paper
```

## Runtime Modes

| Mode | Command | Purpose |
|---|---|---|
| Paper loop | `python -m src.main --mode paper` | Continuous paper trading |
| Live loop | `python -m src.main --mode live` | Real execution (high risk) |
| One-shot | `python -m src.main --mode paper --run-once` | Validate startup + initial scan |
| Scheduler | `python -m src.main --scheduler` | Persistent APScheduler process |
| EOD | `python -m src.main --eod` | Run daily aggregation + agent then exit |
| Healthcheck | `python -m src.main --healthcheck` | Validate DB and credentials |

## Architecture

Cerberus follows a vertical-slice pipeline:

```mermaid
flowchart LR
  A[Alpaca + Unusual Whales] --> B[Data / Feature Pipeline]
  B --> C[Scanner]
  C --> D[Strategy Engine]
  D --> E[Risk Manager]
  E --> F[Order Executor]
  F --> G[Broker Fills]
  G --> H[Position Manager]
  H --> I[(SQLite: cerberus.db)]
  I --> J[Analytics / Agent]
```

## Module Map

| Path | Responsibility |
|---|---|
| `src/main.py` | CLI entrypoint and process orchestration |
| `src/engine/` | Execution, order routing, position/risk management |
| `src/strategies/` | Signal logic (ORB, VWAP, gap, flow, pair, fusion, etc.) |
| `src/scanner/` | Universe selection, ranking, scanner profiles |
| `src/data/` | Alpaca/UW clients, feature pipeline, snapshots |
| `src/backtest/` | Offline replay runner + mock executor |
| `src/analysis/` | DB schema, persistence, reporting |
| `src/analytics/` | Optimization, walk-forward, meta-labeling utilities |
| `src/backtest/fill_models/` | Pluggable fill simulation (fixed BPS, volume-aware) |
| `src/backtest/result_store.py` | JSON result persistence for dashboard API |
| `src/api/` | FastAPI backtest API for EmpireUI dashboard |
| `src/agent/` | Stage-based analysis/tuning/report generation |
| `config/` | Runtime config overlays (`config.yaml`, `risk.yaml`, etc.) |
| `tests/` | Unit/integration/contract/e2e/smoke tests |

## Strategies

Runtime strategy registry in `src/main.py` currently supports:

- `vwap_reversion`
- `orb`
- `vwap_trend_rider`
- `index_mean_reversion`
- `flow_momentum`
- `gap_fill`
- `vix_spike_fade`
- `momentum_continuation`
- `fusion_v1`
- `pair_trading`

Archived strategies live under `src/strategies/archived/` and are not registered by default.

## Configuration

Config is merged by `src/core/config.py` from (in order):

- `config/config.yaml`
- `config/strategies.yaml`
- `config/risk.yaml`
- `config/scanner.yaml`
- `config/universe.yaml`
- `config/logging.yaml`
- optional `config/strategies.auto.yaml`
- optional explicit `--config` file/directory override
- env var overrides via `APP_*`

See:
- Environment vars: [`docs/environment-variables.md`](docs/environment-variables.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)

## Hidden Markov Regime Sidecar

Cerberus now has a dedicated HMM sidecar package under `src/regime_models/hmm/`.

- It is additive, not destructive.
- The current rule-based regime engine stays in place.
- The HMM layer is intended for shadow-mode A/B testing first.
- Default config keeps it off until you explicitly train and enable it.

Core pieces:

- `src/regime_models/hmm/config.py`: nested config for runtime, features, and training
- `src/regime_models/hmm/features.py`: deterministic OHLCV-to-feature preparation
- `src/regime_models/hmm/service.py`: train, predict, and save/load HMM artifacts
- `scripts/bootstrap_hmm_regime.py`: bootstrap a model from CSV or parquet OHLCV

Quick start:

```bash
python -m pip install -r requirements.txt
python scripts/bootstrap_hmm_regime.py --config config/config.yaml --input /path/to/bars.parquet
make test-hmm
```

Artifacts are written under `artifacts/regime_models/hmm/` by default.

## Backtesting

Run backtest:
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10
```

Offline deterministic replay:
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10 --offline-bars-dir /path/to/jsonl_bars
```

### Realism Controls

- **Pluggable fill models**: Choose between fixed-BPS slippage or volume-aware market impact via `backtest.fill_model` config (`fixed` or `volume_aware`)
- **Per-strategy overnight handling**: Configure `allow_overnight`, `max_hold_days`, and `overnight_stop_mult` per strategy instead of global EOD flattening
- **Data quality checks**: Pre-backtest validation for gaps, zero-volume bars, price outliers, and stale prices. Enable via `analytics.data_quality` in config

### Post-Backtest Analytics

- **Benchmark comparison**: Alpha, beta, information ratio, up/down capture ratios vs SPY
- **Monte Carlo simulation**: Bootstrap resampling for probability of loss/ruin, equity confidence intervals, Sharpe distribution. Enable via `analytics.monte_carlo.enabled: true`
- **Diagnostics engine**: Strategy ranking, regime mismatch detection, time-of-day edge analysis, hold/exit analysis. Enable via `analytics.diagnostics.enabled: true`
- **Parameter sensitivity**: Spearman rank correlation analysis of Optuna trial params vs objective scores (auto-runs after WFO)

### Results API

Backtest results are persisted as JSON and served via FastAPI:
```bash
uvicorn src.api.backtest_api:app --port 8004
```

Endpoints: `/api/backtest/runs`, `/api/backtest/runs/{id}/equity`, `/api/backtest/runs/{id}/trades`, `/api/backtest/runs/{id}/monte-carlo`, `/api/backtest/runs/{id}/regime-splits`

## Development

Common commands:

```bash
make test
make test-ci
make test-unit
make test-integration
make test-contract
make test-e2e
make lint
make type-check
make security
```

## Docker

Build:
```bash
docker build -t empire/cerberus:latest .
```

Run paper trader:
```bash
docker compose up -d cerberus-trader
```

Run scheduler profile:
```bash
docker compose --profile scheduler up -d cerberus-scheduler
```

## Operations and Safety

- Default safety: use paper mode until explicitly ready for live mode.
- Validate runtime with `--healthcheck` before market hours.
- Review operational procedures in [`docs/runbook.md`](docs/runbook.md).
- Security policy in [`SECURITY.md`](SECURITY.md).
- Testing conventions in [`TESTING.md`](TESTING.md).
