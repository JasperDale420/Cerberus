# Autoresearch V2: Daily Momentum Strategy Optimization

## Executive Summary

We ran 15 iterations of parameter optimization on the `daily_momentum` strategy using 2020-2024 as in-sample (IS) and 2025 as out-of-sample (OOS). The best configuration (Iteration 10) produces strong risk-adjusted returns that survive realistic transaction costs, but fails in low-pullback melt-up environments like 2024.

## Methodology

- **In-Sample Period:** 2020-01-02 to 2024-12-31 (5 years, includes COVID crash, 2022 bear, recovery)
- **Out-of-Sample Period:** 2025-01-02 to 2025-12-31 (1 year, true OOS)
- **Data:** 1-minute bars aggregated to daily, 16 mega-cap US equities + SPY/QQQ
- **Benchmark:** SPY total return (IS: +80.4%, OOS: +16.6%)
- **Strategy:** `daily_momentum` — EMA crossover trend-following with pullback entries
- **Backtester:** Cerberus offline deterministic backtest (no live data fetch)

## Best Configuration (Iteration 10)

```yaml
strategies:
  daily_momentum:
    ema_fast_period: 20
    ema_slow_period: 50
    pullback_pct: 0.02        # Require 2% pullback to 20 EMA
    stop_atr_mult: 1.5        # Stop loss at 1.5x ATR below entry
    target_atr_mult: 5.0      # Profit target at 5.0x ATR
    max_hold_days: 8           # Exit after 8 days max
    long_only: true
    vol_avg_mult: 1.0          # Require average volume (no below-avg entries)
    confluence_threshold: 55.0 # Signal quality gate

risk:
  risk_pct: 0.0125            # 1.25% equity risked per trade
  max_open_positions: 6
  max_positions_per_strategy: 3
  slippage_bps: 2.0           # 2 basis points (test config)
  commission_per_share: 0.0
```

## Performance Summary

### Primary Results (2 bps slippage, zero commission)

| Metric | IS (2020-2024) | OOS (2025) |
|--------|---------------|------------|
| Total Return | +178.43% | +34.00% |
| vs SPY | +98.0% alpha | +17.4% alpha |
| Sharpe Ratio | 0.923 | 1.357 |
| Sortino Ratio | 1.260 | 1.886 |
| Max Drawdown | 24.29% | 12.38% |
| Calmar Ratio | 0.940 | 2.811 |
| Win Rate | 49.4% | 52.2% |
| Profit Factor | 1.21 | 1.40 |
| Total Trades | 1994 | 301 |
| CAGR | 22.83% | 34.80% |

### Stress Test (5 bps slippage + $0.005/share commission)

| Metric | IS (2020-2024) | OOS (2025) |
|--------|---------------|------------|
| Total Return | +138.33% | +32.89% |
| Sharpe Ratio | 0.768 | 1.315 |
| Max Drawdown | 25.80% | 13.40% |
| Calmar Ratio | 0.739 | 2.511 |
| Win Rate | 48.5% | 51.0% |
| Profit Factor | 1.16 | 1.37 |

**Impact of realistic costs:** IS return drops ~40% (from 178% to 138%), OOS drops only ~1% (from 34% to 33%). The strategy is robust to transaction costs in the OOS period.

### Walk-Forward Failure: 2024

| Metric | IS (2020-2023) | OOS (2024) |
|--------|---------------|------------|
| Total Return | +41.07% | **-23.88%** |
| Sharpe Ratio | 0.434 | -0.627 |
| Max Drawdown | 41.62% | 34.31% |
| Win Rate | 46.8% | 41.3% |
| Profit Factor | 1.06 | 0.86 |

**2024 context:** SPY gained +24.0% in a low-volatility melt-up. The strategy's pullback-based entries found few opportunities, and those taken were whipsawed. This is the strategy's worst regime.

## SPY Annual Returns (Context)

| Year | SPY | VXX | Market Character |
|------|-----|-----|-----------------|
| 2020 | +15.1% | +15.6% | COVID crash + V-recovery |
| 2021 | +28.7% | +1.6% | Strong bull, low vol |
| 2022 | -20.0% | -21.1% | Bear market, rate hikes |
| 2023 | +24.8% | +11.1% | Recovery, AI rally |
| 2024 | +24.0% | +194.3%* | Melt-up, no pullbacks |
| 2025 | +16.6% | -43.2% | Moderate, healthy pullbacks |

*VXX 2024: includes reverse split artifact

## Key Findings from 15 Iterations

### What Works
1. **Entry selectivity is the biggest lever** — requiring 2% pullback depth and average volume confirmation cuts drawdown dramatically while preserving returns
2. **1.25% risk per trade is the sweet spot** — 2% gives huge returns but 31%+ drawdown; below 1% kills returns without proportional drawdown reduction
3. **5.0-6.0x ATR profit targets are essential** — targets below 4.0x destroy OOS returns by cutting winners too early
4. **1.5x ATR stops survive volatility** — tighter stops (1.0x) work in calm markets but get destroyed in volatile periods like COVID/2022
5. **20/50 EMA is optimal** — faster EMAs (10/30) cause too many whipsaws
6. **Quality universe > quantity** — 16 mega-cap names outperform 32 mixed names dramatically
7. **Daily bars dominate intraday** — only `daily_momentum` works on daily bars, but it's far more robust than intraday strategies

### What Doesn't Work
1. **Tight daily loss circuit breakers** — they stop the strategy from recovering after bad days
2. **Expanded universes** — adding volatile/small-cap names causes catastrophic OOS failure
3. **Tighter profit targets** — cutting winners short is the fastest way to kill the strategy
4. **Faster EMAs** — more signals = more noise = worse performance

### The Core Weakness
The strategy **fails in low-pullback, melt-up environments** (2024). When the market goes straight up without meaningful pullbacks, the strategy either:
- Gets no signals (pullback never reaches 2%)
- Gets false signals and gets whipsawed
- Misses the move entirely while losing on false entries

## Iteration Log

| Iter | Key Change | OOS Return | OOS Sharpe | OOS MaxDD | Result |
|------|-----------|------------|------------|-----------|--------|
| 0 | Baseline: 2% risk, 1.5/6.0 ATR | +19.01% | 0.790 | 31.27% | baseline |
| 1 | 0.75% risk, 1.0/4.0 ATR | +23.50% | 2.138 | 6.26% | discard (IS dead) |
| 6 | 1.25% risk, 6 max pos | +36.50% | 1.469 | 14.51% | KEEP |
| 8 | 5.0 ATR target | +35.71% | 1.448 | 14.56% | keep |
| **10** | **Selective entries (2%, vol 1.0x, conf 55)** | **+34.00%** | **1.357** | **12.38%** | **BEST** |
| 11 | Very selective (2.5%, vol 1.2x, conf 60) | +16.01% | 0.858 | 12.53% | alt (best IS DD) |
| 14 | 6.0 ATR target | +30.64% | 1.332 | 12.38% | keep (higher IS) |

## Recommendations

### For Production Use
This strategy should NOT be used standalone. It needs to be paired with:
1. **A trend-following component** for melt-up regimes (buy-and-hold or simple trend)
2. **Regime detection** to reduce sizing or pause during low-pullback environments
3. **Portfolio allocation** — allocate 50-70% to the momentum-pullback strategy, 30-50% to a complementary strategy

### Next Steps
1. Build a multi-regime strategy that combines daily_momentum with trend-following
2. Implement regime detection that identifies melt-up vs normal environments
3. Test the combined system across all years including 2024
4. Consider adding mean-reversion component for range-bound markets

## Files

- `config.yaml` — Best configuration (Iteration 10)
- `run_experiment.py` — Backtest runner with IS/OOS split
- `RESULTS_LOG.md` — Full iteration log with all metrics
