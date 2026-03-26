# Backtest & WFO Analytics Gap Analysis
## Institutional-Grade Benchmarking

*Research Date: 2026-03-22*

## Methodology

Benchmarked our current analytics against:
- **pyfolio** (Quantopian/Stefan Jansen) — industry-standard tear sheet library
- **QuantConnect LEAN** — institutional backtesting platform (70+ KPIs)
- **de Prado (2018)** — *Advances in Financial Machine Learning* (AQR Capital)
- **Bailey & Lopez de Prado (2014)** — Deflated Sharpe Ratio, PBO framework
- **BackQuant** — institutional analytics platform

---

## Current State Inventory

### What We Have

| Module | Metrics | Status |
|--------|---------|--------|
| **backtest_report.py** | Sharpe, Sortino, Calmar, max DD, win rate, profit factor, avg win/loss, total PnL, exposure %, CAGR | Done |
| **report_card.py** | MAE/MFE, Omega, Tail ratio, VaR, CVaR, Ulcer Index, Gain-to-pain, Rolling Sharpe/WinRate, Drawdown catalog (top 5), Trade clustering, Cost sensitivity, Time breakdowns, Entry/exit efficiency, Skew/Kurtosis (PnL), Statistical tests (t-test, Ljung-Box, Jarque-Bera, runs test, bootstrap CI, permutation test) | Done |
| **benchmark.py** | Alpha, beta, information ratio, up/down capture ratios, return attribution vs SPY | Done |
| **monte_carlo.py** | Bootstrap PnL paths, P(loss), P(ruin), confidence intervals, percentile equity bands | Done |
| **diagnostics.py** | Strategy ranking, regime mismatch, time-of-day edge, hold duration analysis | Done |
| **param_sensitivity.py** | Spearman rank correlation of Optuna params vs objective | Done |
| **data_quality.py** | Gap detection, zero volume, price outliers, staleness | Done |
| **fill_models/** | Fixed BPS + volume-aware slippage with participation rate impact | Done |
| **optuna_harness.py** | Deflated Sharpe Ratio, holdout validation dataclass | Done |

**Revised coverage assessment: ~60-65% of institutional standard.**

---

## Gap Analysis: What's Missing

### Tier 1 — HIGH IMPACT, LOW EFFORT (should definitely add)

#### 1.1 Rolling Sortino + Rolling Beta
**What:** Extend existing rolling metrics (currently Sharpe + win rate only) with rolling Sortino and rolling beta vs benchmark over configurable windows (63d, 126d, 252d).
**Why:** Rolling Sharpe alone doesn't capture downside risk evolution or market sensitivity drift. pyfolio and QuantConnect both display rolling 6mo/12mo beta. Strategy that was market-neutral but drifted to beta=1.2 is a hidden risk.
**Effort:** Small — extend existing `compute_rolling_metrics()` in report_card.py.
**References:** pyfolio `rolling_sharpe()`, QuantConnect rolling statistics panel.

> **Note:** Rolling Sharpe, rolling win rate, Calmar, Omega, Tail ratio, VaR, CVaR, Ulcer Index, Skew, Kurtosis, Ljung-Box, drawdown catalog, cost sensitivity already exist in `report_card.py`.

#### 1.2 Return Autocorrelation (Lag-by-Lag)
**What:** Serial correlation of DAILY EQUITY RETURNS at lags 1-5, with significance flags. Different from existing Ljung-Box on trade PnLs.
**Why:** Positive autocorrelation in daily equity returns is a curve-fitting red flag — real alpha shouldn't be predictable from lagged returns. Ljung-Box on trade PnLs tests trade-level dependence; this tests equity-curve-level dependence.
**Effort:** Small — `statsmodels.tsa.stattools.acf()` on daily returns.
**References:** de Prado (2018) Ch. 14, Lo (2002) "The Statistics of Sharpe Ratios".

#### 1.3 Turnover & Cost Drag %
**What:** Annualized turnover rate, transaction cost drag as % of gross returns.
**Why:** A strategy with 200% annual turnover and 2bps slippage loses 4% to friction. Existing cost_sensitivity sweep tests break-even but doesn't express cost as a percentage of gross returns.
**Effort:** Small — computed from trade records we already have.
**References:** QuantConnect turnover metrics, Grinold & Kahn (2000) *Active Portfolio Management*.

#### 1.4 Daily Return Distribution Metrics
**What:** Skew, Kurtosis, VaR, CVaR computed on DAILY PORTFOLIO RETURNS (not trade PnLs).
**Why:** Existing report_card computes these on trade-level PnL distribution. Daily return distribution tells a different story — captures the effects of position sizing, correlation, and time in market.
**Effort:** Small — apply same formulas from report_card.py to daily_returns array.
**References:** pyfolio `SIMPLE_STAT_FUNCS`.

### Tier 2 — HIGH IMPACT, MODERATE EFFORT (strong additions)

#### 2.1 Probability of Backtest Overfitting (PBO)
**What:** Bailey, Borwein, Lopez de Prado, Zhu (2014/2015) — Combinatorially Symmetric Cross-Validation (CSCV) to estimate the probability that the best in-sample configuration underperforms OOS.
**Why:** This is THE gold standard for detecting overfitting in systematic strategies. Goes beyond Deflated Sharpe (which we have) by actually computing the probability that your optimization selected a lucky configuration. If PBO > 0.5, your backtest is likely overfit.
**Formula:** Partition N observations into S equal subsets. For each of C(S, S/2) train/test combinations, find optimal IS strategy and measure OOS rank. PBO = fraction of combinations where IS-optimal strategy ranks below median OOS.
**Effort:** Moderate — need CSCV infrastructure, but the math is straightforward. R package `pbo` exists; Python implementation ~200 lines.
**References:** Bailey et al. (2014) "The Probability of Backtest Overfitting", SSRN #2326253.

#### 2.2 Factor Attribution (Fama-French + Momentum)
**What:** Regress strategy daily returns against market factors: Mkt-RF, SMB, HML, UMD (momentum). Extract alpha, factor loadings, R².
**Why:** Answers "is my strategy actually generating alpha, or just taking disguised beta?" A strategy with Sharpe 1.5 but beta=1.3 and alpha=0% is just leveraged market exposure. Factor data is free from Ken French's website.
**Effort:** Moderate — need to download factor return series and run rolling OLS. ~150 lines + data pipeline.
**References:** Fama & French (1993), Carhart (1997), QuantConnect factor model tutorials.

#### 2.3 Strategy Cross-Correlation Matrix
**What:** Pairwise correlation of daily returns across all active strategies. Diversification ratio = σ(equal-weight portfolio) / Σ(individual σ) weighted.
**Why:** Running 10 strategies that are 0.9 correlated is no better than running 1. This is how portfolio managers decide capital allocation. Also surfaces regime-dependent correlation (strategies that decorrelate in stress = valuable).
**Effort:** Moderate — need multi-strategy results in a single run, then correlation matrix computation.
**References:** Choueifaty & Coignard (2008) "Toward Maximum Diversification", Roncalli (2013).

#### 2.4 IS/OOS Degradation Distribution
**What:** Instead of a single OOS Sharpe, track the distribution of IS-to-OOS ratios across all WFO windows. Plot histogram. Compute mean, median, std.
**Why:** A strategy where IS Sharpe is 3.0 and OOS is 1.5 in every window (ratio=0.5, stable) is MUCH better than one where ratios are [0.1, 0.9, 0.2, 0.8] (unstable). We currently compute average OOS metrics but don't surface the per-window degradation pattern.
**Effort:** Low-moderate — data already exists in WFO results, just need extraction and visualization.
**References:** de Prado (2018) Ch. 12, QuantConnect WFO documentation.

### Tier 3 — MODERATE IMPACT, HIGHER EFFORT (nice to have)

#### 3.1 Minimum Backtest Length (MinBTL)
**What:** Given observed Sharpe, skew, kurtosis, and number of trials, compute the minimum number of observations needed for the Sharpe to be statistically significant.
**Why:** If your MinBTL is 5 years but your backtest is 2 years, your results aren't statistically meaningful regardless of how good they look.
**Formula:** MinBTL = 1 + (1 - γ₃·SR + (γ₄/4)·SR²) × (z_α / SR)²
where γ₃ = skew, γ₄ = excess kurtosis, z_α = critical value.
**Effort:** Small code, but interpreting and surfacing results requires UX thought.
**References:** Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier", SSRN #1821643.

#### 3.2 Capacity Estimation
**What:** Given average daily dollar volume of traded instruments and strategy turnover, estimate maximum AUM before market impact degrades returns by >X%.
**Why:** A strategy that works with $100K but fails at $10M isn't useful for scaling. Answers "how much money can this strategy manage?"
**Formula:** capacity ≈ max_participation_rate × avg_daily_volume × avg_holding_period / turnover_multiple
**Effort:** Moderate — need volume data per instrument (available from Heber bars) and participation rate modeling (already in VolumeAwareFillModel).
**References:** Korajczyk & Sadka (2004) "Are Momentum Profits Robust to Trading Costs?"

#### 3.3 Drawdown Analytics (Duration + Recovery)
**What:** Top-N drawdowns with: peak date, trough date, recovery date, duration (peak→trough), recovery time (trough→recovery), underwater period. Plot drawdown underwater chart.
**Why:** Max drawdown alone doesn't tell you if it recovered in 2 days or 6 months. Duration and recovery time are critical for position sizing and risk management.
**Effort:** Small-moderate — straightforward cumulative max computation with event tracking.
**References:** pyfolio `gen_drawdown_table()`, QuantConnect drawdown period analysis.

#### 3.4 Probabilistic Sharpe Ratio (PSR)
**What:** P(true Sharpe > benchmark Sharpe) given observed returns, accounting for non-normality.
**Formula:** PSR = Φ((SR̂ - SR*) × √(n-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4 · SR̂²))
**Why:** A Sharpe of 1.5 over 50 trades means nothing. A Sharpe of 1.5 over 500 trades with PSR > 95% is real. Complements the DSR we already have.
**Effort:** Small — single function, <30 lines.
**References:** Bailey & Lopez de Prado (2012), QuantConnect PSR metric.

#### 3.5 White's Reality Check / SPA Test
**What:** Test whether the best strategy's performance is significantly better than a universe of alternatives, after accounting for data snooping.
**Why:** When you test 100 parameter combinations, the best one will look good by chance. White's Reality Check / Hansen's SPA test corrects for this more rigorously than DSR.
**Effort:** Moderate — bootstrap-based, ~200 lines, needs access to all trial results.
**References:** White (2000) "A Reality Check for Data Snooping", Hansen (2005) "A Test for Superior Predictive Ability".

---

## Priority Recommendation

### Phase 10: Analytics Enrichment (Recommended Next)

**Task 10.1 — Rolling Sortino + Beta + Daily Return Stats** (Tier 1.1 + 1.4)
Extend `report_card.py` rolling metrics with Sortino and beta. Add daily return distribution (skew, kurtosis, VaR, CVaR) to `backtest_report.py`.
~100 lines total.

**Task 10.2 — Return Autocorrelation + Turnover** (Tier 1.2 + 1.3)
New `src/analytics/return_diagnostics.py`: daily return autocorrelation (lag 1-5 with significance), annualized turnover rate, cost drag as % of gross returns.
~120 lines, needs `statsmodels`.

**Task 10.3 — PSR + MinBTL** (Tier 3.4 + 3.1)
New `src/analytics/statistical_tests.py`: Probabilistic Sharpe Ratio, Minimum Backtest Length.
~60 lines, pure math (no new deps).

**Task 10.4 — IS/OOS Degradation Distribution** (Tier 2.4)
Enhance `optuna_harness.py` WFO results: per-window IS/OOS Sharpe ratio array, mean/median/std of degradation ratios.
~80 lines.

**Task 10.5 — PBO (Probability of Backtest Overfitting)** (Tier 2.1)
New `src/analytics/pbo.py`: Combinatorially Symmetric Cross-Validation (CSCV) implementation.
~200 lines. The strongest single addition for overfitting detection.

**Task 10.6 — Factor Attribution** (Tier 2.2)
New `src/analytics/factor_attribution.py`: Fama-French 4-factor (Mkt-RF, SMB, HML, UMD) rolling regression.
~200 lines + Fama-French data download utility.

**Task 10.7 — Strategy Correlation Matrix** (Tier 2.3)
New `src/analytics/correlation.py`: pairwise strategy daily returns, correlation matrix, diversification ratio.
~100 lines.

**Task 10.8 — Wire into Runner + API + Dashboard**
Wire new analytics into backtest runner, add API endpoints, add EmpireUI components for rolling charts, factor loadings, correlation heatmap.

### Deferred (revisit later)
- Capacity estimation (needs more volume data infrastructure)
- White's SPA test (complex, marginal benefit over PBO + DSR)
- Multi-regime correlation (correlation conditioned on regime state — needs more data)
- Combinatorial Purged Cross-Validation (CPCV) — full de Prado CV framework (significant effort)

---

## References

1. Bailey, D. H. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2).
2. Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5).
3. Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
4. Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance." *Journal of Finance*, 52(1).
5. Choueifaty, Y. & Coignard, Y. (2008). "Toward Maximum Diversification." *Journal of Portfolio Management*, 35(1).
6. de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
7. Fama, E. F. & French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1).
8. Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill, 2nd ed.
9. Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics*, 23(4).
10. Korajczyk, R. A. & Sadka, R. (2004). "Are Momentum Profits Robust to Trading Costs?" *Journal of Finance*, 59(3).
11. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4).
12. Man Group (2023). "Covering Your Tail: The Case for Expected Shortfall." Man Institute.
13. White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5).
14. pyfolio (Quantopian/Stefan Jansen). Portfolio and risk analytics in Python. github.com/stefan-jansen/pyfolio-reloaded.
15. QuantConnect. Backtest Report Documentation. quantconnect.com/docs/v2/cloud-platform/backtesting/report.
