# CLAUDE.md

Cerberus is a multi-strategy algorithmic trading system for US equities. It runs multiple strategies concurrently, routes them by market regime, and manages risk through a unified execution engine.

## Commands

```bash
uv sync                                    # install deps (includes editable empire-core, empire-schemas, empire-gateway-client)
uv run pytest                              # all tests
uv run pytest -m unit                      # fast unit tests only
uv run pytest -m integration               # integration tests (DB, file I/O)
uv run pytest -m contract                  # boundary tests for external APIs
uv run pytest -m e2e                       # full system flow
uv run pytest --cov=src --cov-fail-under=68  # with coverage gate
ruff check .                               # lint
ruff format .                              # auto-format
mypy                                       # type check (excludes src/data/alpaca.py, src/data/pipeline.py)

# Run trading (paper mode is default)
uv run python -m src.main --mode paper --order-executor noop
uv run python -m src.main --mode paper --order-executor gateway   # routes orders via Data-Gateway
uv run python -m src.main --mode paper --order-executor alpaca    # direct Alpaca broker orders

# Scheduler (persistent daily session launcher)
uv run python -m src.main --scheduler

# EOD agent (offline analytics + strategy tuning)
uv run python -m src.main --eod
uv run python -m src.main --eod --eod-date 2026-03-19

# Healthcheck
uv run python -m src.main --healthcheck

# Backtest API (serves results to EmpireUI)
uv run uvicorn src.api.backtest_api:app --port 8002

# HMM regime model bootstrap
uv run python scripts/bootstrap_hmm_regime.py --config config/config.yaml --input <bars_csv>

# Walk-forward optimization
uv run python scripts/run_wfo.py
```

## Architecture

### Package Layout

```
src/
├── main.py                  # CLI entry point (argparse: --mode, --order-executor, --config, --eod, --scheduler)
├── scheduler.py             # APScheduler-based daily session launcher (Mon-Fri cron)
├── core/
│   ├── config.py            # ConfigLoader — merges YAML suite + env var overrides (APP_* prefix)
│   ├── settings.py          # Pydantic Settings — Alpaca creds, Data-Gateway URL, Heber config
│   ├── domain.py            # Enums (Regime, Side, OrderType) + dataclasses (Bar, Signal, Position, MarketState)
│   ├── errors.py            # CerberusError + ErrorCode enum
│   ├── logger.py            # StructuredLogger wrapper → empire_core.logger (service name: "cerberus")
│   ├── http_client.py       # Shared httpx client factory
│   ├── indicators.py        # Rolling EMA/RSI/SMA/Std
│   └── ledger_adapter.py    # Bridge to empire_core.ledger (trade audit trail)
├── strategies/
│   ├── base.py              # BaseStrategy ABC — cooldown, hard stop, HMM gate, overnight handling
│   ├── config_models.py     # Pydantic models for activation policies from YAML
│   └── <strategy>.py        # ~30 strategy implementations (see Strategy Registry below)
├── engine/
│   ├── execution.py         # ExecutionEngine — orchestrates data flow, strategy eval, order mgmt
│   ├── strategy_engine.py   # StrategyEngine + StrategyActivationPolicy (multi-axis regime routing)
│   ├── risk.py              # RiskManager — daily loss limits, position limits, notional caps
│   ├── orders.py            # OrderExecutor — Alpaca SDK order submission
│   ├── position_manager.py  # Position tracking, trailing stops, partial exits
│   ├── market.py            # MarketStateManager
│   ├── kelly.py             # Kelly Criterion sizer (Wasserstein DRO robust mode)
│   ├── cppi.py              # CPPI drawdown-controlled sizer
│   ├── cvar_sizer.py        # CVaR-based position sizer
│   ├── hrp.py               # Hierarchical Risk Parity cross-strategy allocator
│   └── adaptive_sizer.py    # Regime-adaptive position sizing
├── analysis/
│   ├── regime.py            # MarketContextService — 5-axis regime (trend/vol/liquidity/risk/session)
│   ├── bocpd.py             # Bayesian Online Changepoint Detection
│   ├── entropy.py           # Entropy analyzer
│   ├── vrp.py               # Variance Risk Premium
│   ├── gex.py               # Gamma exposure analysis
│   ├── iv_surface.py        # Implied volatility surface
│   ├── momentum_crash.py    # Momentum crash detector
│   └── db.py                # SQLite analytics DB (cerberus.db)
├── regime_models/hmm/       # HMM-based regime detection (pomegranate)
│   ├── service.py           # HmmRegimeService — fit/predict/shadow-compare
│   ├── adapters.py          # PomegranateDenseHmmAdapter
│   ├── features.py          # OHLCV → HMM feature engineering
│   └── labeling.py          # Hidden state → regime label mapping
├── data/
│   ├── client.py            # UnifiedDataClient — REST + WebSocket to Data-Gateway
│   ├── heber_read_client.py # Direct Heber parquet reads (historical bars)
│   ├── pipeline.py          # FeaturePipeline — indicator computation
│   ├── unusual_whales.py    # UnusualWhales flow data client
│   ├── atlas_reader.py      # Atlas factor bridge (Gold layer → live signals)
│   ├── replay_provider.py   # Historical bar replay for backtesting
│   └── snapshot_manager.py  # GEX/flow snapshot capture
├── scanner/                 # Universe scanning and symbol ranking
├── agent/                   # Offline EOD agent (3-stage: health → tuning → LLM proposals)
├── backtest/                # Backtest runner, fill models (fixed/volume-aware), stats
├── analytics/               # Walk-forward optimization, Optuna harness, Monte Carlo, meta-labeler
├── portfolio/               # Signal aggregation, risk budgeting, HRP allocation
├── quant/                   # Cointegration, filters, regime stats, volatility models
└── api/                     # FastAPI backtest results API (port 8002)
```

### Configuration

Dual config system:

1. **YAML suite** (`config/`): `ConfigLoader` merges `config.yaml`, `strategies.yaml`, `risk.yaml`, `scanner.yaml`, `universe.yaml`, `logging.yaml` in order. `strategies.auto.yaml` applies agent-generated overrides on top. Env vars with `APP_` prefix override any YAML key.

2. **Pydantic Settings** (`src/core/settings.py`): Runtime env vars for credentials and service URLs. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALPACA_API_KEY` / `APCA_API_KEY_ID` | — | Alpaca broker credentials |
| `ALPACA_SECRET_KEY` / `APCA_API_SECRET_KEY` | — | Alpaca broker secret |
| `ALPACA_PAPER` | `True` | Paper mode (always default True) |
| `CERBERUS_GATEWAY_URL` / `DATA_INGESTION_URL` | `http://localhost:8080` | Data-Gateway URL |
| `CERBERUS_GATEWAY_KEY` / `GATEWAY_API_KEY` | — | Data-Gateway API key (required) |
| `CERBERUS_HEBER_DATA_ROOT` / `HEBER_DATA_ROOT` | — | Heber parquet root for direct reads |
| `CERBERUS_STAGE3_APPROVED` | — | Gate for Stage 3 LLM agent proposals |

### Strategy Registry

Active V2 strategies: `mean_reversion_pro`, `trend_rider_pro`, `flow_alpha`, `orb_v2`, `pair_trading_v2`, `rsi_bounce`, `momentum_fade`.

Legacy strategies (still registered): `vwap_reversion`, `orb`, `vwap_trend_rider`, `index_mean_reversion`, `flow_momentum`, `gap_fill`, `vix_spike_fade`, `momentum_continuation`, `fusion_v1`, `pair_trading`, `trend_pullback`, `failed_breakout`, `order_flow_imbalance`, `intraday_momentum`.

All strategies extend `BaseStrategy` (ABC) and implement `generate_signal()`.

### Multi-Axis Regime System

`MarketContextService` classifies market state across 5 orthogonal axes:
- **Trend**: UP / DOWN / FLAT
- **Volatility**: LOW / NORMAL / HIGH / SHOCK
- **Liquidity**: GOOD / THIN / STRESSED
- **Risk**: RISK_ON / NEUTRAL / RISK_OFF
- **Session**: PREMARKET / OPENING / MIDDAY / POWER_HOUR / CLOSE

Each strategy declares a `StrategyActivationPolicy` (via `activation:` in YAML) specifying which regime combinations it should trade in. `StrategyEngine` gates signals through these policies.

Optional HMM-based regime detection runs in `shadow` or `primary` mode alongside the rule-based system.

### Position Sizing

Multiple sizers available, selected per-trade: Kelly Criterion (with Wasserstein DRO), CPPI (drawdown-controlled), CVaR-based, and HRP cross-strategy allocation. Regime-adaptive multipliers adjust sizing based on volatility state.

### EOD Agent (3-Stage Pipeline)

1. **Stage 1** — Deterministic health/risk adjustments based on rolling trade stats
2. **Stage 2** — Offline parameter tuning via grid search + walk-forward validation (writes `strategies.auto.yaml`)
3. **Stage 3** — LLM-generated code proposals (gated by `CERBERUS_STAGE3_APPROVED` env var)

### Data Flow

All market data flows through Data-Gateway (port 8080) — no direct API calls to Alpaca/providers. `UnifiedDataClient` handles both REST (historical) and WebSocket (real-time bars/quotes/trades). Heber parquet reads available for historical backtesting via `heber_read_client.py`.

## Safety-Critical Code

Extra caution required when modifying:

- **`src/engine/orders.py`** — Order submission to Alpaca broker
- **`src/engine/risk.py`** — RiskManager enforces daily loss limits, position caps, notional limits
- **`src/engine/execution.py`** — ExecutionEngine orchestrates the full trading loop
- **`src/engine/position_manager.py`** — Position tracking and exit logic
- **`src/main.py`** — CLI defaults (`--mode paper`, `--order-executor gateway`)

Safety invariants:
- Paper mode (`--mode paper`) is the default — never change this
- `--order-executor noop` simulates without submitting orders
- `ALPACA_PAPER=True` is the default in Settings — never change this
- `risk.yaml` hard dollar ceilings: `max_daily_loss`, `max_open_positions`, `max_notional_per_order`
- RiskManager rejects signals that would breach any limit
- `position_mismatch_mode: halt` stops trading on broker/local position divergence

## Test Markers

```
unit         — fast, isolated, no I/O or network
integration  — real DB, file I/O, or component interactions
contract     — boundary tests for external APIs/protocols
e2e          — full system flow
slow         — tests >1s, opt-in
```

Coverage gate: 68% minimum (`--cov-fail-under=68`).

Test conftest sets safe defaults: `ALPACA_API_KEY=test`, `ALPACA_SECRET_KEY=test`, `ALPACA_PAPER=True`, `DATA_INGESTION_URL=http://central.test`.

## Commit & Changelog Discipline

- **Commit often** — make small, atomic commits after each logical change. Do not accumulate large uncommitted diffs across multiple files.
- **Update the changelog** — every commit that changes behavior, fixes a bug, or adds a feature must have a corresponding entry in `CHANGELOG.md`. If no `CHANGELOG.md` exists, create one.
- Changelog format: `## [Unreleased]` section at the top with entries grouped by `Added`, `Changed`, `Fixed`, `Removed`.
- Write changelog entries from the user's perspective — describe *what changed*, not implementation details.

---

## Karpathy Coding Guidelines

_Source: [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) — behavioral guidelines to reduce common LLM coding mistakes. Bias toward caution over speed; for trivial tasks, use judgment._

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Data Analysis Review

Any data analysis presented as a conclusion — backtest results, strategy performance claims, Optuna/WFO output, dataset QA findings, or other statistical/quantitative findings — must be adversarially reviewed before being presented to the user. Use one of:

- **An Opus subagent** (`Agent` with `model: "opus"`), explicitly instructed to challenge the methodology — look for overfitting, look-ahead/leakage, cherry-picked windows, confounds, and unsupported causal claims. Not a proofread pass.
- **A Codex adversarial review** (`/codex:adversarial-review`, or the `codex` skill run in review mode) using the strongest available GPT model (currently `gpt-5.6-terra`) at high/xhigh reasoning effort.

Report the adversarial review's findings alongside the analysis itself, not as a separate follow-up step.
