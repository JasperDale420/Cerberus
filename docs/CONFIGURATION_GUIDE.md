# Configuration Guide

Cerberus has two layered configuration systems:

1. **YAML suite** under `config/` — strategies, risk, scanner, universe, logging.
2. **Pydantic Settings** in `src/core/settings.py` — credentials, service URLs, runtime backends.

Both are merged at startup. `APP_*` env vars override any YAML key.

> See [`environment-variables.md`](environment-variables.md) for the per-variable matrix. The CLI defaults in `src/main.py` are described in [`../README.md`](../README.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

## YAML Suite

`ConfigLoader` in `src/core/config.py` merges files in this fixed order:

```
1. config/config.yaml            # base runtime
2. config/strategies.yaml        # per-strategy enable + activation
3. config/risk.yaml              # RiskManager limits
4. config/scanner.yaml           # universe scanner thresholds
5. config/universe.yaml          # tradeable universe
6. config/logging.yaml           # log handlers
7. config/strategies.auto.yaml   # OPTIONAL — agent Stage 2 overrides
8. --config <path> override      # OPTIONAL — file or directory
9. APP_* env var overrides       # final word
```

Later layers override earlier ones key-by-key. Lists are replaced, not merged.

### Backtest variants

| File | Purpose |
|---|---|
| `backtest_smoke.yaml` | Fast smoke test (~1 week) |
| `backtest_jan2024/*.yaml` | Jan 2024 calibration sweep |
| `backtest_5yr/*.yaml` | Long-horizon validation |
| `backtest_23_25.yaml` | Calendar slice 2023–2025 |
| `backtest_v2.yaml` | V2 strategy benchmark |
| `backtest_portfolio.yaml` | Portfolio-aggregation run |
| `backtest_trp_sortino.yaml` | TRP / Sortino objective |
| `autoresearch_2025.yaml` | Autoresearch driver config |

### Universe files

- `universe.yaml` — primary universe spec (referenced symbols, sector groupings)
- `universe_sp500.txt` — line-delimited S&P 500 symbols
- `universe_nasdaq100.txt` — line-delimited Nasdaq-100 symbols
- `offline_symbols.txt` — symbols available in local replay bar set

## `config/risk.yaml` (Safety-Critical)

These are the hard ceilings enforced by `RiskManager` (`src/engine/risk.py`), verified directly against the file. The defaults are conservative; do not raise them without explicit user instruction.

```yaml
risk:
  max_daily_loss: 500.0           # USD — halt new entries on breach
  max_risk_per_trade: 50.0        # USD — (entry - stop) * qty cap
  max_open_risk: 200.0            # USD — sum of open per-trade risk
  max_trades_per_day: 20
  max_open_positions: 5
  max_positions_per_strategy: 3
  max_notional_per_order: 5000.0
  max_notional_per_symbol: 5000.0
  time_in_force: "day"            # day, gtc, ioc, fok
  risk_mode: "normal"             # normal | reduced | off
  commission_per_share: 0.005     # USD per share
  min_commission: 1.0             # USD minimum per fill leg
  slippage_bps: 1.0               # bps per notional per fill leg
```

Additional optional keys consumed by `src/config/models.py:RiskConfig`:
- `kelly:` — `KellySizer` config (cap fraction, Wasserstein DRO, lookback)
- `cppi:` — `CPPISizer` config (floor, multiplier, cushion calc)
- `hrp:` — `HRPAllocator` config (lookback, linkage)
- `position_mismatch_mode: halt` — halts trading on broker/local divergence

## `config/strategies.yaml`

Defines which strategies are enabled and their activation policy + parameters. As of this writing it enables 13 strategies — see [`../README.md`](../README.md) for the current list, since this changes over time. Example shape:

```yaml
strategies:
  vwap_reversion:
    enabled: true
    cooldown_bars: 5
    bar_duration_minutes: 1
    activation:
      trend: [flat, down]
      vol: [normal, high]
      liquidity: [good]
      session: [midday, power_hour]
    params:
      lookback: 20
      z_threshold: 2.0
```

Note: as of May 2026 no strategy in this file gates on the `risk` axis (see [`ARCHITECTURE.md`](ARCHITECTURE.md#5-axis-regime-gate) for why) — an empty/omitted `risk:` list means "no constraint on this axis," it does not mean the axis is ignored elsewhere.

`strategies.auto.yaml` is written by the EOD agent Stage 2 (parameter tuning). It is loaded **after** `strategies.yaml`, so any auto-generated overrides win unless an explicit `--config` overrides them. Because `config/` is mounted read-only (`:ro`) into the `cerberus-trader` container (see [`DEPLOYMENT.md`](DEPLOYMENT.md)), confirm the EOD agent's writes are actually landing on disk in your deployment rather than assuming Stage 2 tuning is taking effect.

## `config/scanner.yaml`

Universe scanner thresholds (price, volume, volatility, spread). Consumed by `src/scanner/core.py` and `src/scanner/streaming_scanner.py`.

## `config/logging.yaml`

Maps log levels and rotating-file handler paths. The structlog stack itself is configured by `empire_core.logger`; this file feeds the underlying stdlib handlers.

## Pydantic Settings (`src/core/settings.py`)

Runtime env vars hydrated by `pydantic-settings`. The full matrix lives in [`environment-variables.md`](environment-variables.md). Highlights:

### Broker credentials

| Var | Default | Notes |
|---|---|---|
| `ALPACA_API_KEY` / `APCA_API_KEY_ID` | — | Required for live or paper |
| `ALPACA_SECRET_KEY` / `APCA_API_SECRET_KEY` | — | Required for live or paper |
| `ALPACA_PAPER` | `True` | **Do not flip to False without explicit instruction** |
| `ALPACA_BASE_URL` / `APCA_API_BASE_URL` | — | Override broker REST root |

### Data-Gateway + Heber

| Var | Default | Notes |
|---|---|---|
| `CERBERUS_GATEWAY_URL` / `DATA_INGESTION_URL` | `http://localhost:8080` | Data-Gateway base URL |
| `CERBERUS_GATEWAY_KEY` / `GATEWAY_API_KEY` | — | API key, sent as `X-Gateway-Key` |
| `CERBERUS_GATEWAY_TIMEOUT_SECONDS` | `30` | Request timeout |
| `CERBERUS_DATA_BACKEND` | `gateway` | `legacy` \| `gateway` \| `dual` |
| `CERBERUS_STORAGE_BACKEND` | `sqlite` | `sqlite` \| `heber` \| `dual` |
| `CERBERUS_DUAL_READ_COMPARE` | `false` | Side-by-side legacy vs gateway compare |
| `CERBERUS_FAILOVER_TO_LEGACY` | `true` | Allow fallback to legacy on errors |
| `CERBERUS_HEBER_CATALOG_URL` | — | Heber catalog API |
| `CERBERUS_HEBER_DATA_ROOT` / `HEBER_DATA_ROOT` | — | Heber parquet root (read-only) |

### Unusual Whales

| Var | Default |
|---|---|
| `UW_API_TOKEN` | empty |
| `UW_BASE_URL` | `https://api.unusualwhales.com` |
| `UNUSUAL_WHALES_FLOW_URL_TEMPLATE` | empty |

### Agent gating

| Var | Default | Purpose |
|---|---|---|
| `CERBERUS_STAGE3_APPROVED` | empty | **Required** before Stage 3 LLM proposals run |

### Empire shared logging

| Var | Default |
|---|---|
| `EMPIRE_LOG_LEVEL` | `INFO` |
| `EMPIRE_LOG_FORMAT` | `json` |
| `EMPIRE_LOG_DIR` | `./logs` |

## `APP_*` Override Pattern

Any nested YAML key can be overridden at the env-var level using:

```
APP_<SECTION>_<KEY>=value
APP_RISK_MAX_DAILY_LOSS=1000
APP_LOG_LEVEL=DEBUG
APP_SCANNER_INTERVAL_MINUTES=1
```

`ConfigLoader` does type coercion for booleans (`true`/`false`/`1`/`0`) and numerics.

## `.env` Files

`.env.example` ships with the repo as a template. `.env` is `.gitignore`'d. The Docker Compose service `cerberus-trader` mounts `.env` via `env_file:` rather than baking secrets into the image.

## Startup Validation

Before the trading loop starts, `src/main.py` calls:

- `validate_startup_settings()` — required credentials present, sane defaults
- `validate_runtime_execution_requirements(order_executor=..., mode=...)` — refuses `--mode live` without explicit env signal

Both raise `ValueError`, which `main.py` logs and re-raises (fail-fast).

## Healthcheck

```bash
uv run python -m src.main --healthcheck
```

Runs `src/core/health.py:run_healthcheck()` — validates DB writable, credentials present, Data-Gateway reachable, Heber catalog reachable (if configured). Used as the Docker `cerberus-trader` healthcheck (`interval: 5m`, `timeout: 60s`, `retries: 3`).

## Related Docs

- [`environment-variables.md`](environment-variables.md) — per-variable matrix
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker + launchd
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — config merge diagram
- [`RUNBOOK.md`](RUNBOOK.md) — operational incidents
