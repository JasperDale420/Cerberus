# Task — Regime Labeling Pipeline Critique

Critique the regime-labeling pipeline used to train Cerberus autoresearch.

Repository: Cerberus (multi-strategy intraday algo trading system at `/Users/jacobmcmillan/Empire/Cerberus`). Autoresearch loops iterate strategies and use regime-labeled training data to validate signal edge.

## Two distinct regime classifiers exist

**Labeler A — Training-data labeler** — `scripts/label_regime_dataset.py`, output at `data/regime_labeled/<SYMBOL>_daily_regime.parquet` (69 symbols, 103,803 daily bars, 2020-01-01 → 2026-03-19). Used to feed autoresearch.
- Trend: dual-SMA crossover (fast=10, slow=40, flat_band=1%)
- Vol: trailing 30-day annualized realized vol with absolute thresholds: LOW<8%, HIGH≥20%, SHOCK≥50%
- Tuned via grid search against SPY ground-truth periods only
- Declared accuracy: trend 71.5%, vol 92.3%, combined 81.9%

**Labeler B — Live regime engine** — `src/analysis/regime.py` (MarketContextService).
- Trend: Hurst exponent (≥0.55 trending) + 60-day cumulative log-return sign
- Vol: EWMA variance ratio (short span=10, long span=120), z=sqrt(short/long); SHOCK z≥3.0, HIGH z≥1.5, LOW z≤0.7
- Risk axis: 5-day cumulative return (>0.5%=RISK_ON, <-0.5%=RISK_OFF)
- Adds 3 axes (liquidity, risk, session) the labeler does NOT produce

## Empirical findings from the labeled parquets

1. **SPY's labels look reasonable** — 2024 Bull: 256/271 NORMAL, 14 LOW, 1 HIGH, 0 SHOCK. COVID Crash: 13 SHOCK + 11 HIGH + 7 NORMAL. 2021 Steady Bull: 232 NORMAL + 16 LOW + 7 HIGH out of 255 days.

2. **Cross-symbol calibration is broken** — fraction of days HIGH-or-SHOCK by symbol:
   - SPY 19.9%, JNJ 15.8%, COST 37.3%, QQQ 38.3%, WMT 39.9%, JPM 51.8%
   - **Median across 68 symbols: 94.3%**
   - TSLA 98.6%, AMD 98.5%, NIO/MARA/AMC/MRVL/RIOT/NET ~98.2%
   - Only 3 of 68 symbols ever hit LOW vol on ≥5% of days

3. **Aggregate distribution is artifact-dominated** — 103,803 symbol-days: only 0.32% LOW, 67.6% HIGH/SHOCK. SPY's true 2020-2026: ~9% LOW / ~80% NORMAL / ~18% HIGH / ~2% SHOCK.

4. **Methodology drift** — autoresearch trains on Labeler A; live runtime uses Labeler B (Hurst+EWMA z). Different boundary behavior, hysteresis, and trend definition.

5. **Risk axis silently dropped** — `optimized_params.json` notes "Risk axis dropped — 5-day return proxy too noisy (58% accuracy)". Labeled training data has no risk axis. Live engine emits one and many strategy activation policies key off it (e.g., `RISK_ON`-only entries). Autoresearch can't have learned anything about risk-axis interactions.

6. **Ground truth is small + SPY-only + in-sample** — 8 hand-picked SPY periods total, no held-out OOS, no alternative-asset validation. The 81.9% combined accuracy is in-sample on the same SPY periods used to tune.

## Questions

- Are the labels usable as-is, or must autoresearch results conditioned on them be re-evaluated?
- Highest-leverage fix: (a) per-symbol vol thresholds (percentile-based on each symbol's own history), (b) replace Labeler A entirely with Labeler B's Hurst+EWMA so train==live, (c) add the Risk axis to the labeled dataset, (d) something else?
- Hidden assumption that might invalidate the critique — e.g., maybe autoresearch only uses SPY labels and per-symbol parquets are unused?
- Concrete dangers — false confidence in autoresearch's recent regime-conditioned wins?

Generate, critique, refine. Converge on concrete next-step recommendations and a prioritized fix list.
