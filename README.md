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

## Backtesting

Run backtest:
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10
```

Offline deterministic replay:
```bash
python scripts/run_backtest.py --config config/config.yaml --start-date 2024-01-03 --end-date 2024-01-10 --offline-bars-dir /path/to/jsonl_bars
```

Backtest realism controls are under the `backtest:` section in config (partial fills, slippage mode, spread mode, flow-strategy gating, strict session flatten options).

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
