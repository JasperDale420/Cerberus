# Cerberus Repo Familiarization — Strategy Config Tuning Prep

## Architecture Overview

Cerberus is an intraday algorithmic trading engine for US equities. The pipeline flows:

```
Data (Alpaca/UW) → Feature Pipeline → Scanner → Strategy Engine → Risk Manager → Order Executor → Broker → Position Manager → SQLite → Analytics
```

## Configuration System

Config is loaded by [ConfigLoader](file:///Users/jacobmcmillan/Empire/Cerberus/src/core/config.py) via deep-merge of YAML files in this precedence order:

1. `config/config.yaml` — global settings, risk, backtest, universe, agent tuning
2. `config/strategies.yaml` — per-strategy params + activation regimes
3. `config/risk.yaml` — overrides risk section (smaller limits for live)
4. `config/scanner.yaml`, `config/universe.yaml`, `config/logging.yaml`
5. `config/strategies.auto.yaml` — agent-generated overrides
6. `--config` CLI override (file or directory)
7. `APP_*` env var overrides

## Strategies — 10 Active

| # | Strategy | File | Config Model | Key Tunable Params |
|---|----------|------|-------------|-------------------|
| 1 | **vwap_reversion** | [vwap_reversion.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/vwap_reversion.py) | `VWAPReversionConfig` | `band_sigma`, `rsi_len`, `rsi_oversold/overbought`, `risk_reward`, `time_window_start/end`, `max_hold_minutes`, `confirmation` |
| 2 | **orb** | [orb.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/orb.py) | `ORBConfig` | `orb_minutes`, `risk_reward`, `stop_loss_pct`, `min_gap_pct`, `min_flow_zscore`, `stop_buffer_atr_mult` |
| 3 | **vwap_trend_rider** | [vwap_trend_rider.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/vwap_trend_rider.py) | `VWAPTrendRiderConfig` | `ema_fast`, `ema_slow`, `vol_mult`, `risk_reward`, `min_trend_score` |
| 4 | **index_mean_reversion** | [index_mean_reversion.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/index_mean_reversion.py) | `IndexMeanReversionConfig` | `bb_len`, `bb_std`, `stop_pct`, `symbols` |
| 5 | **flow_momentum** | [flow_momentum.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/flow_momentum.py) | `FlowMomentumConfig` | `min_flow_zscore`, `vol_mult`, `risk_reward` |
| 6 | **gap_fill** | [gap_fill.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/gap_fill.py) | `GapFillConfig` | `min_gap`, `max_gap`, `risk_reward`, `or_time_minutes`, `weak_trend_max_score` |
| 7 | **vix_spike_fade** | [vix_spike_fade.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/vix_spike_fade.py) | `VixSpikeFadeConfig` | `vix_spike_pct`, `vix_absolute`, `index_drop_pct`, `reversion_target`, `stop_buffer`, `symbols` |
| 8 | **momentum_continuation** | [momentum_continuation.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/momentum_continuation.py) | `MomentumContinuationConfig` | `breakout_lookback`, `vol_mult`, `close_position`, `risk_reward`, `ema_fast/slow`, `max_trades_per_session` |
| 9 | **fusion_v1** | [fusion_v1.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/fusion_v1.py) | `FusionStrategyConfig` | `orb_minutes`, `min_dof_score`, `min_flow_bias`, `min_relative_strength`, `atr_period`, `stop_atr_mult`, `target_atr_mult`, `entry_window_start/end` |
| 10 | **pair_trading** | [pair_trading.py](file:///Users/jacobmcmillan/Empire/Cerberus/src/strategies/pair_trading.py) | *(no typed config model)* | `entry_zscore`, `exit_zscore` |

Plus 2 stub strategies (`failed_breakout`, `trend_pullback`) that are registered but minimal.

## Regime Activation System

Every strategy has a 5-axis **activation policy** defined in `strategies.yaml`:

| Axis | Values | Purpose |
|------|--------|---------|
| **session** | `opening`, `midday`, `power_hour` | Time-of-day session regime |
| **trend** | `up`, `down`, `flat` | Market trend direction |
| **vol** | `low`, `normal`, `high`, `shock` | Volatility regime |
| **liquidity** | `good`, `thin` | Market liquidity state |
| **risk** | `risk_on`, `neutral`, `risk_off` | Risk appetite regime |
| **min_confidence** | 0.0–1.0 | Minimum confidence across all axes |

The [StrategyEngine](file:///Users/jacobmcmillan/Empire/Cerberus/src/engine/strategy_engine.py) checks `StrategyActivationPolicy.is_active()` on each bar to gate strategy execution.

## Risk Configuration

Two levels of risk config exist:

| Parameter | `config.yaml` (paper/backtest) | `risk.yaml` (live) |
|-----------|-------------------------------|-------------------|
| `max_daily_loss` | $2,000 | $500 |
| `max_risk_per_trade` | $500 | $50 |
| `max_open_risk` | $2,000 | $200 |
| `max_trades_per_day` | 100 | 20 |
| `max_notional_per_order` | $50,000 | $5,000 |
| `max_notional_per_symbol` | $50,000 | $5,000 |
| `slippage_bps` | 0 | 1.0 |
| `commission_per_share` | $0 | $0.005 |

## Backtest Infrastructure

- Per-strategy isolation configs exist in [config/backtest_5yr/](file:///Users/jacobmcmillan/Empire/Cerberus/config/backtest_5yr) for: `vwap_reversion`, `orb`, `gap_fill`, `index_mean_reversion`, `momentum_continuation`, `vwap_trend_rider`, `fusion_v1`
- Each isolation config enables **only** the target strategy, disables all others, and widens risk limits
- Activation policies are broadened (all sessions/trends/vol/risk) with `min_confidence: 0.0` so the strategy gets maximum signal opportunity for tuning
- Backtest runner: `python scripts/run_backtest.py --config <config> --start-date <start> --end-date <end>`

## Agent Tuning System (Stage 2)

The agent's Stage 2 tuning defines search spaces in `config.yaml`:

```yaml
search_space:
  vwap_reversion: { band_sigma: [1.5, 2.0, 2.5], max_hold_minutes: [30, 60, 90] }
  orb: { orb_minutes: [10, 15, 20], risk_reward: [1.5, 2.0, 2.5] }
  gap_fill: { min_gap: [0.01, 0.015, 0.02], max_gap: [0.08, 0.10, 0.12] }
  vwap_trend_rider: { min_trend_score: [1.0, 1.5, 2.0], vol_mult: [1.0, 1.2, 1.5] }
  index_mean_reversion: { bb_std: [1.5, 2.0, 2.5], stop_pct: [0.003, 0.005, 0.007] }
```

## What's Needed for Config Tuning

When we tune strategies, for each strategy we can adjust:

1. **Signal parameters** — thresholds that control entry/exit conditions (band widths, RSI levels, volume multiples, etc.)
2. **Risk/reward ratios** — `risk_reward`, `stop_loss_pct`, ATR-based stops
3. **Timing windows** — `time_window_start/end`, `orb_minutes`, `max_hold_minutes`
4. **Activation policies** — which regime conditions allow the strategy to trade
5. **Cooldown** — `cooldown_bars` (inherited from BaseStrategy)

The base strategy also applies a **regime volatility multiplier** to stop distances:

- `low`: 0.8x, `normal`: 1.0x, `high`: 1.2x, `shock`: 1.5x
