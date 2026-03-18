# Quant Strategy Upgrade Design

**Date:** 2026-03-18
**Goal:** Upgrade all 7 Cerberus strategies from basic technical analysis (avg 2.4/5 math level) to rigorous quantitative strategies (target 4+/5), add portfolio optimization layer, and build validation framework to prevent overfitting.

**Approach:** Bottom-up — build shared quant primitives, upgrade each strategy in-place, add portfolio layer on top.

---

## Phase 1: Quant Foundation Layer (`src/quant/`)

Shared math primitives used by all strategies. No strategy implements its own GARCH or Kalman — everything goes through this layer.

### `src/quant/filters.py` — Adaptive Estimation
- **Kalman Filter** (via `filterpy`): State-space estimation for dynamic hedge ratios, adaptive VWAP tracking, and mean estimation that handles regime shifts. Replaces EMA smoothing.
- **EWMA with regime-aware decay**: Decay factor adapts to vol regime (fast in SHOCK, slow in LOW vol).

### `src/quant/cointegration.py` — Mean-Reversion Validation
- **Engle-Granger two-step test** (via `statsmodels`): Tests cointegration before trading spreads. Returns test stat, p-value, ECM coefficients.
- **Johansen test**: Multi-leg cointegration for 3+ symbol baskets.
- **Rolling cointegration monitor**: Continuously checks if relationship is valid mid-trade, triggers exit on breakdown.

### `src/quant/volatility.py` — Volatility Modeling
- **GARCH(1,1)** (via `arch`): Conditional volatility forecasting. Replaces rolling std with forward-looking vol estimates.
- **Realized volatility estimators**: Parkinson (high-low), Garman-Klass (OHLC), Yang-Zhang (drift-adjusted).
- **VRP calculator**: Formalized realized vs implied vol gap.

### `src/quant/statistics.py` — Hypothesis Testing
- **Variance ratio test** (Lo-MacKinlay): Generalized from rsi_bounce for all mean-reversion strategies.
- **Hurst exponent**: Gates strategy activation — H<0.5 for mean-reversion, H>0.5 for trend-following.
- **CUSUM detector**: Statistical breakout significance testing (replaces naive "close > high + buffer").
- **Granger causality**: Validates that flow signals predict price, not just correlate.
- **ADF test wrapper**: Stationarity testing for cointegration and mean-reversion.

### `src/quant/regime.py` — Regime-Switching Models
- **Markov regime-switching** (via `statsmodels.tsa.regime_switching`): 2-3 state model with transition probabilities. Strategies use filtered probability for adaptive thresholds.
- **Adaptive threshold engine**: Takes base threshold, scales by regime state, vol forecast, and Hurst exponent. Replaces all hardcoded thresholds.

### `src/quant/sizing.py` — Enhanced Position Sizing
- **Optimal f with drawdown constraint**: Kelly variant constraining max drawdown probability.
- **Signal-strength scaling**: Size proportional to confluence score and regime confidence.

### `src/quant/validation.py` — Anti-Overfitting
- **Anchored walk-forward**: Expanding train window, fixed OOS window (20 days), minimum 8 windows.
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014): Adjusts Sharpe for number of trials.
- **Combinatorial Purged Cross-Validation** (CPCV): Purged K-fold respecting time-series structure with embargo periods.
- **CUSUM drift detector**: Flags strategies whose live returns deviate from backtest expectation.
- **IC decay tracking**: Monitors information coefficient over 30-day windows, alerts on zero-trend.

---

## Phase 2: Strategy Upgrades

### mean_reversion_pro (2.5 → 4/5)
- Replace rolling z-score with **GARCH-conditional z-score**
- Add **Engle-Granger cointegration test** against sector ETF as entry gate
- Make OU half-life a **hard gate** (skip if half-life > max hold period)
- Add **Hurst exponent filter** (H < 0.45 required)
- Replace hardcoded confluence threshold with **adaptive threshold** from regime-switching model

### trend_rider_pro (2 → 3.5/5)
- Replace EMA-20 pullback with **Kalman filter mean estimate**
- Add **Hurst exponent gate** (H > 0.55 required)
- Replace hardcoded ADX threshold with **Markov regime-switching filtered probability** (P(trending) > 0.7)
- Replace fixed ATR stop/target with **GARCH-forecasted vol** scaling
- Add **autocorrelation significance test** on pullback returns

### flow_alpha (2.5 → 4/5)
- Replace static signal weights with **rolling IC-weighted combination**
- Add **Granger causality test** (re-evaluate weekly)
- Add **VPIN toxicity gate** (reuse from mean_reversion_pro)
- Replace fixed z-score normalization with **GARCH-conditional normalization**

### orb_v2 (1.5 → 3.5/5)
- Replace naive breakout with **CUSUM test** for statistical significance
- Replace hardcoded range window with **Markov regime-switching optimal window**
- Add **BOCPD changepoint probability** as breakout confidence multiplier
- Add **variance ratio test** (VR > 1.0 = trending = breakout valid)
- Replace fixed volume gate with **volume relative to GARCH-forecasted vol**

### pair_trading_v2 (3.5 → 5/5)
- Add **Engle-Granger cointegration test** as hard entry gate (p < 0.05)
- Replace EMA hedge ratio with **true Kalman filter** state-space model
- Replace rolling std z-score with **GARCH-conditional z-score** on spread
- Add **OU half-life gate** (half-life < max hold)
- Add **rolling correlation monitor** (exit if correlation < 0.6 mid-trade)
- Add **Johansen test** for optional 3-leg baskets

### rsi_bounce (4 → 4.5/5)
- Replace rolling z-score with **GARCH-conditional z-score**
- Add **BOCPD structural break awareness** (suppress entries during regime breaks)
- Add **higher-order moment filters** (skip on extreme kurtosis)
- Make thresholds **adaptive** via regime-switching engine

### momentum_fade (1.5 → 3.5/5)
- Replace arbitrary VWAP deviation with **GARCH-conditional z-score of price vs VWAP**
- Add **momentum exhaustion model** (velocity + acceleration — fade when velocity high, acceleration negative)
- Replace fixed volume surge with **volume relative to intraday seasonal profile**
- Add **Hurst exponent gate** (H < 0.5 required)
- Add **entropy filter** (skip fades in high-entropy random markets)

---

## Phase 3: Portfolio Optimization Layer (`src/portfolio/`)

### `src/portfolio/signal_aggregator.py` — Signal Conflict Resolution
- **IC-weighted signal combination**: Weight each strategy's signal by trailing information coefficient.
- **Directional conflict resolution**: Net weighted signal below confidence threshold → no trade.
- **Strategy correlation penalty**: Discount correlated signals to avoid concentration.

### `src/portfolio/allocator.py` — Cross-Strategy Capital Allocation
- **Risk-parity allocation**: Capital inversely proportional to realized vol. Rebalanced daily at EOD.
- **Drawdown-aware throttling**: Halve allocation if trailing DD > 1.5x historical max DD.
- **Correlation-adjusted gross exposure**: Total exposure scales down when cross-strategy correlations rise.

### `src/portfolio/risk_budget.py` — Portfolio-Level Risk
- **Portfolio VaR/CVaR**: Aggregate tail risk using GARCH-forecasted vol.
- **Marginal risk contribution**: Reject new positions if marginal CVaR contribution exceeds budget.
- **Cross-strategy concentration limits**: Max 40% risk to one strategy, max 25% to one symbol.

### `src/portfolio/performance.py` — Portfolio Analytics
- **Strategy attribution**: Brinson-style P&L decomposition.
- **Rolling Sharpe/Sortino per strategy**: Used by allocator for reweighting.
- **Correlation matrix monitoring**: Daily update, alert on correlation spikes.

---

## Phase 4: Validation & Anti-Overfitting

### Walk-Forward Protocol
- Anchored walk-forward with expanding window, 20-day OOS, minimum 8 windows.
- Deflated Sharpe Ratio > 1.0 required for any parameter change to go live.
- CPCV with embargo periods for parameter optimization.

### Regime-Conditional Backtesting
- Per-regime Sharpe decomposition across all 5 axes.
- Worst-regime stress test: flag if worst-regime DD > 2x average DD.

### Live Monitoring
- CUSUM drift detector on strategy returns vs backtest expectation.
- IC decay tracking over 30-day windows.
- Automatic throttling: 2 consecutive failed WFO windows → halve allocation; 3 → disable.

---

## Integration

### New Dependencies
```
statsmodels  — cointegration, regime-switching, Granger causality, ADF
arch         — GARCH(1,1) conditional volatility
filterpy     — Kalman filter state-space estimation
```

### Execution Pipeline Change
```
Strategy signals → SignalAggregator (NEW) → RiskManager (existing) → PortfolioRiskBudget (NEW) → OrderExecutor (existing)
```

### Database Additions
- `strategy_ic_daily` — daily IC per strategy
- `portfolio_risk_snapshots` — hourly VaR/CVaR, correlation matrix, concentration

### File Structure
```
src/quant/           — NEW: shared math primitives (7 modules)
src/portfolio/       — NEW: multi-strat portfolio layer (4 modules)
src/strategies/      — MODIFIED: 7 strategies upgraded in-place
src/engine/          — MODIFIED: execution.py and risk.py integrate portfolio layer
src/backtest/        — MODIFIED: WFO, CPCV, deflated Sharpe added
src/analysis/schema.py — MODIFIED: 2 new tables
```
