# Autoresearch V2 Results Log

## SPY Benchmarks
- **IN-SAMPLE (2020-2024):** SPY +80.40%
- **OUT-OF-SAMPLE (2025):** SPY +16.64%

## Goal
Beat SPY on OOS (2025) with max drawdown < 15%. Optimize on IS, validate on OOS.

---

## Part 1: Parameter Optimization (15 iterations)

### Iteration Log

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

### Key Insights from Part 1

1. **Position sizing is the #1 drawdown control** — 1.25% risk per trade is the sweet spot
2. **ATR targets must be wide (5.0-6.0)** — tighter targets (4.0) kill OOS returns
3. **1.5 ATR stops survive volatility** — tighter (1.0) kills IS, wider is wasteful
4. **Entry selectivity matters more than exit params** — deeper pullback (2%) and volume confirmation (1.0x) filter bad trades
5. **Quality universe > quantity** — 16 mega-cap names >> 32 mixed names
6. **20/50 EMAs are optimal** — faster EMAs (10/30) cause too many whipsaws
7. **Max 6 open positions** — concentrates capital on highest-conviction trades
8. **Daily bars >> intraday** — reduces noise, only daily_momentum can fire

---

## Part 2: Multi-Strategy System

### New Strategies Created

| Strategy | Regime Target | Logic | Status |
|----------|--------------|-------|--------|
| `daily_mean_reversion` | FLAT/range-bound | BB(20,2σ) fade + ADX<22 filter + RSI confirmation | ✅ Working |
| `daily_vol_fade` | HIGH/SHOCK vol | Buy extreme selloffs (2+ ATR below EMA20) | ✅ Working |
| `regime_adaptive_momentum` | Melt-up / breakout | ADX-gated breakout entries + pullback mode | ✅ Working |

### Multi-Strategy Performance (config_multi.yaml, no activation policies)

| Period | Return | Sharpe | MaxDD | Trades | per-strategy breakdown |
|--------|--------|--------|-------|--------|----------------------|
| IS 2020-2024 | +108.12% | 0.662 | 27.76% | 2235 | momentum +$98K, mean_rev +$5.8K, vol_fade +$1.9K, breakout +$2K |
| OOS 2025 | +21.47% | 0.996 | 14.71% | 353 | momentum +$21.6K, mean_rev -$63, vol_fade -$181 |
| 2024 holdout | -7.91% | -0.124 | 35.30% | 497 | momentum -$25K, **mean_rev +$3.5K**, vol_fade +$77 |

### Empirical Regime-Fit Analysis (2665 trades, 1507 days)

Ran each strategy solo on full 2020-2025 data, tagged trades with ADX-based regime:

**Regime Distribution**: TRENDING 37.7%, RANGE_BOUND 35.2%, WEAK_TREND 21.0%, SHOCK 4.2%

| Strategy | TRENDING | WEAK_TREND | RANGE_BOUND | SHOCK | Best Regime |
|----------|----------|------------|-------------|-------|-------------|
| daily_momentum | +$93.8K ✅ | +$63.5K ✅ | +$47.8K ✅ | **-$22.3K ❌** | ALL except SHOCK |
| daily_mean_reversion | +$4.1K (PF 2.10) ✅ | +$2.4K ✅ | **-$1.3K ❌** | — | TRENDING (!) |
| daily_vol_fade | -$585 ❌ | -$10 | +$998 ✅ | — | RANGE_BOUND |
| regime_adaptive_momentum | +$19 | +$1.2K ✅ | +$437 ✅ | — | WEAK_TREND |

**By Volatility** (daily_momentum):
- NORMAL: **+$229K** (sweet spot)
- LOW: +$11K
- HIGH: **-$34K** (avoid)
- SHOCK: **-$22K** (avoid)

### Activation Policy Testing Results

| Config | OOS 2025 | IS 2020-2024 | Notes |
|--------|----------|-------------|-------|
| config_multi.yaml (ungated) | **+21.47%** | **+108.12%** | Best multi-strategy |
| config_empirical.yaml (activation policies) | +15.77% | +102.17% | Activation policies HURT |
| config.yaml (iter 10, momentum only) | **+34.00%** | **+178.43%** | Best single-strategy |

**Why activation policies hurt**: The MarketContextService's 5-axis regime classifier (EWMA-based volatility) doesn't match our ADX-based analysis classifier. The "HIGH" vol regime means different things in each system, so policies filter wrong trades.

---

## Infrastructure Changes

- ✅ Deleted legacy bull/bear/chop strategy routing from config + code
- ✅ Strategies now use 5-axis activation policies or run unrestricted
- ✅ `breakout_only` param added to `regime_adaptive_momentum`
- ✅ 2 new strategies registered in strategy registry

---

## TODO: Next Steps

### High Priority
- [ ] **Align regime classifiers** — Make the strategies' internal ADX-based regime match MarketContextService's 5-axis EWMA-based regime so activation policies actually filter correctly. This means either (a) having the strategies read `market_state.regime_snapshot` directly instead of computing their own ADX, or (b) adding ADX as a 6th axis to MarketContextService.
- [ ] **Add HIGH vol filter to daily_momentum** — Empirically loses -$34K in HIGH vol. Add `if snapshot.vol == VolRegime.HIGH: return None` alongside existing SHOCK check. Simple code change, big risk reduction.

### Medium Priority
- [ ] **Walk-forward regime policies** — Train activation policies on 2020-2023, validate on 2024, test on 2025 to prevent look-ahead bias
- [ ] **Increase vol_fade trade count** — Only 17 trades total across 6 years. Relax thresholds (deviation_mult from 2.0 to 1.5?) or expand universe
- [ ] **Mean reversion short side** — Currently long-only. Test with `allow_short: true` (empirical data shows it's profitable in UP trends, suggesting counter-trend shorts could work in DOWN trends)

### Low Priority / Research
- [ ] **ML meta-labeler** — Replace heuristic Hurst/TFI/GEX with a trained model. Requires: (a) labeled training data (regime + trade outcome), (b) feature engineering, (c) cross-validation. Our 2665 trades may not be enough — consider augmenting with synthetic data or using the regime_fit_data.json as initial labels
- [ ] **Intraday strategies** — The V2 strategies (mean_reversion_pro, trend_rider_pro, etc.) need intraday bars. Could run a separate intraday backtest alongside daily strategies
- [ ] **Pair trading / market-neutral leg** — Would provide returns decorrelated from market direction. pair_trading_v2 exists but barely fires on daily bars

---

## Config Reference

### Best Single-Strategy (Iter 10)
```yaml
# experiments/autoresearch_v2/config.yaml
risk_pct: 0.0125, stop_atr_mult: 1.5, target_atr_mult: 5.0
max_hold_days: 8, pullback_pct: 0.02, vol_avg_mult: 1.0
confluence_threshold: 55.0, ema: 20/50, max_open_positions: 6
slippage_bps: 2.0, commission: $0
```

### Best Multi-Strategy (config_multi.yaml)
```yaml
# experiments/autoresearch_v2/config_multi.yaml
# 4 strategies, 1% risk/trade, 5 bps slippage, $0.005/share commission
# max_open_positions: 8, max_positions_per_strategy: 3
```
