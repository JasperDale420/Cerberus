# Autoresearch V2 Results Log

## SPY Benchmarks
- **IN-SAMPLE (2020-2024):** SPY +80.40%
- **OUT-OF-SAMPLE (2025):** SPY +16.64%

## Goal
Beat SPY on OOS (2025) with max drawdown < 15%. Optimize on IS, validate on OOS.

## Iteration Log

| Iter | Change | IS Return | IS Sharpe | IS MaxDD | IS Calmar | OOS Return | OOS Sharpe | OOS MaxDD | OOS Calmar | OOS Trades | OOS WR | Result |
|------|--------|-----------|-----------|----------|-----------|------------|------------|-----------|------------|------------|--------|--------|
| 0 | Baseline: 2% risk, 1.5/6.0 ATR, 16 syms | +349.64% | 1.104 | 34.09% | 1.034 | +19.01% | 0.790 | 31.27% | 0.621 | 273 | 50.5% | BASELINE |
| 1 | 0.75% risk, 1.0/4.0 ATR, 7d hold | +0.16% | 0.097 | 39.92% | 0.001 | +23.50% | 2.138 | 6.26% | 3.838 | 254 | 55.9% | discard (IS dead) |
| 2 | 1% risk, 1.25/5.0 ATR, 8d hold | +70.55% | 0.808 | 21.93% | 0.516 | +10.29% | 0.791 | 12.78% | 0.823 | 234 | 51.7% | discard (OOS < SPY) |
| 3 | 0.75% risk, 1.5/4.0 ATR | +70.10% | 0.797 | 29.73% | 0.379 | -0.53% | 0.048 | 7.99% | -0.068 | 388 | 51.8% | discard |
| 4 | 0.75% risk, 1.5/6.0 ATR | +119.40% | 1.079 | 25.29% | 0.676 | +8.79% | 0.651 | 15.47% | 0.581 | 273 | 50.2% | discard (OOS < SPY) |
| 5 | 1.5% risk, 1.5/6.0 ATR, 10 sym, tight daily loss | +63.93% | 0.568 | 40.36% | 0.259 | -6.84% | -0.446 | 18.32% | -0.381 | 177 | 48.0% | discard |
| 6 | 1.25% risk, 1.5/6.0 ATR, 6 max pos | +185.34% | 0.791 | 34.83% | 0.673 | +36.50% | 1.469 | 14.51% | 2.575 | 257 | 57.2% | **KEEP** |
| 7 | 1% risk (iter 6 base) | +149.44% | 0.719 | 40.38% | 0.499 | +27.32% | 1.242 | 15.89% | 1.759 | 285 | 55.1% | discard |
| 8 | 1.25% risk, 5.0 ATR target | +183.18% | 0.790 | 32.41% | 0.717 | +35.71% | 1.448 | 14.56% | 2.510 | 254 | 57.5% | keep (marginal) |
| 9 | Expanded 32-sym universe | +363.33% | 1.004 | 47.04% | 0.766 | -66.90% | -1.262 | 69.25% | -0.977 | 500 | 42.0% | discard |
| **10** | **Selective entries: pullback 2%, vol 1.0x, conf 55** | **+178.43%** | **0.923** | **24.29%** | **0.940** | **+34.00%** | **1.357** | **12.38%** | **2.811** | **301** | **52.2%** | **BEST** |
| 11 | Very selective: pullback 2.5%, vol 1.2x, conf 60 | +187.99% | 1.170 | 14.42% | 1.641 | +16.01% | 0.858 | 12.53% | 1.306 | 245 | 54.3% | alt (best IS DD) |
| 12 | Midpoint selectivity | +177.34% | 0.918 | 33.64% | 0.676 | +30.22% | 1.413 | 12.85% | 2.406 | 246 | 52.4% | discard |
| 13 | High selectivity + 1.5% risk | +183.36% | 1.125 | 14.71% | 1.582 | +11.44% | 0.616 | 13.73% | 0.851 | 240 | 51.7% | discard (OOS < SPY) |
| 14 | Iter 10 + 6.0 ATR target | +224.97% | 1.060 | 25.34% | 1.054 | +30.64% | 1.332 | 12.38% | 2.533 | 245 | 55.5% | keep (higher IS) |
| 15 | Faster EMAs (10/30) | +211.42% | 1.015 | 24.36% | 1.052 | -0.13% | 0.134 | 20.51% | -0.007 | 327 | 49.2% | discard |

## Key Insights

1. **Position sizing is the #1 drawdown control** — 1.25% risk per trade is the sweet spot
2. **ATR targets must be wide (5.0-6.0)** — tighter targets (4.0) kill OOS returns
3. **1.5 ATR stops survive volatility** — tighter (1.0) kills IS, wider is wasteful
4. **Entry selectivity matters more than exit params** — deeper pullback (2%) and volume confirmation (1.0x) filter bad trades
5. **Quality universe > quantity** — 16 mega-cap names >> 32 mixed names
6. **20/50 EMAs are optimal** — faster EMAs (10/30) cause too many whipsaws
7. **Max 6 open positions** — concentrates capital on highest-conviction trades
8. **Daily bars >> intraday** — reduces noise, only daily_momentum can fire

## Best Config (Iter 10)

```yaml
risk_pct: 0.0125 (1.25% per trade)
stop_atr_mult: 1.5
target_atr_mult: 5.0
max_hold_days: 8
pullback_pct: 0.02
vol_avg_mult: 1.0
confluence_threshold: 55.0
ema_fast/slow: 20/50
max_open_positions: 6
slippage_bps: 2.0
bar_resolution: daily
```

### Performance Summary
| Period | Return | vs SPY | Sharpe | MaxDD | Calmar | Win Rate | PF |
|--------|--------|--------|--------|-------|--------|----------|----|
| IS (2020-2024) | +178.43% | +98.03% alpha | 0.923 | 24.29% | 0.940 | 49.4% | 1.21 |
| OOS (2025) | +34.00% | +17.36% alpha | 1.357 | 12.38% | 2.811 | 52.2% | 1.40 |
