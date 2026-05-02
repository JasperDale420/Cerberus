# Critique: Cerberus Regime Labeling Pipeline

## 1. Are the labels usable as-is?

**No — the verdict is conditional on a single cheap fact about the autoresearch driver, and under the most likely reading the labels are toxic for non-SPY symbols and any regime-conditioned autoresearch result against the broader 68-symbol universe must be re-evaluated.**

Two coherent readings of "regime label" exist, and usability flips between them:

- **Interpretation X — regime as a *market-wide* state.** A single coordinate the strategy uses to decide whether its edge applies. Every symbol on day D should carry the same vol regime, because the claim is about the market.
- **Interpretation Y — regime as a *symbol-local* state.** A per-symbol z-score-like feature: "is this name unusually volatile relative to its own history?"

Labeler A's absolute thresholds (LOW<8%, HIGH≥20%, SHOCK≥50% on trailing 30-day annualized realized vol) are literally true under Y for any individual symbol — TSLA *is* above 20% realized vol on 98.6% of days. The pathology only matters if autoresearch consumes those labels under X.

**The most likely use is X.** Autoresearch is a strategy validator; live `MarketContextService` references market-wide axes (RISK_ON, HIGH-vol session); Finding 4 says train uses Labeler A while live uses Labeler B, and Labeler B is structured for market-wide states (Hurst on a benchmark return). The cross-symbol distribution is also bimodal in a way only an X-misuse explains: SPY/JNJ/COST/QQQ/WMT/JPM sit between 16% and 52% HIGH-or-SHOCK; the median of the remaining 62 symbols is 94.3% with a tight cluster near 98% (TSLA 98.6%, AMD 98.5%, NIO/MARA/AMC/MRVL/RIOT/NET ~98.2%). That gap is not a smooth function of underlying volatility — it is an absolute threshold cutting through the realized-vol distribution at the seam separating "index-like" from "single-name-like." Under Y you would expect smooth dispersion, not a cliff.

Under the X reading, "strategy X earns Sharpe Y in HIGH regime" computed on a labeler that calls 94% of single-name days HIGH is approximately the unconditional result with extra steps. The conditioning variable carries near-zero information for non-SPY symbols. Defensible exceptions: SPY-only or SPY/QQQ/JNJ/COST backtests, where the absolute thresholds happen to match the asset's natural vol scale. Everything else is toxic.

## 2. Highest-leverage fix

**Lead fix: redefine `vol_regime` as a single market-wide state shared across all symbols, derived from a benchmark vol estimator (VIX level or SPY's trailing 30-day realized vol), and propagated to every symbol on day D. Per-symbol relative vol should be added as a separate `symbol_vol_pctile` column — not used to replace the market-wide label. Then add the Risk axis only after building honest OOS ground truth.**

Defense:

- This is the root-cause fix because it directly inverts the X-misuse: every symbol agrees on the day's vol state, the 94.3%-median pathology collapses, and the labeled `vol_regime` becomes definitionally aligned with what live `MarketContextService` is computing (Hurst+EWMA on a benchmark, not per name). Cross-strategy ensemble routing, activation policies, and HRP allocation all remain well-defined because "HIGH" denotes the same market state everywhere.
- Idiosyncratic vol stays addressable via a separate column. Strategies that need "is TSLA unusual for itself today" use `symbol_vol_pctile`; strategies that need "is the market in a vol shock" use the shared `vol_regime`. Two orthogonal axes, both well-defined.

Why not the alternatives:

- **Per-symbol percentile-only thresholds destroy semantics.** Switching to "LOW = bottom quartile of each symbol's own trailing 252d vol" balances the per-symbol histograms but silently re-interprets "HIGH" 68 different ways. Activation policies, ensemble routing, and any cross-strategy comparison break, and the failure mode is undetectable from outside — worse than status quo, where the discrepancy is at least visible.
- **Wholesale swap to Labeler B (Hurst+EWMA z) to enforce train==live.** Right *direction*, but Labeler A and Labeler B are *both* effectively unvalidated — Labeler A's 81.9% accuracy is in-sample on the same 8 SPY periods used to tune (Finding 6), and Labeler B has no published ground-truth comparison at all. Convergence on an unvalidated labeler is no better than divergence between two unvalidated ones; in fact it is worse, because divergence is at least detectable. Adopt B's outputs only after both labelers are scored on a common, OOS, multi-asset ground-truth set.
- **Adding the Risk axis** (Finding 5) is a real gap and must be closed, but on a *separate* axis after a defensible proxy is built. The 5-day-return proxy was rejected at 58% accuracy on a ground truth Finding 6 already shows is unreliable; a 58% number from an unreliable yardstick is not a reliable rejection. Define what ground truth a Risk-axis classifier must beat first.

## 3. Hidden assumption that might invalidate the critique

**Autoresearch may consume only `SPY_daily_regime.parquet` and broadcast it across symbols.** If so, the 67 broken non-SPY parquets are dead weight on disk, the calibration finding is a distraction, and Findings 2 and 3 are irrelevant — only SPY's labels enter training, and SPY's labels look reasonable per Finding 1.

Two cheap verifications, ≈30 minutes total:

1. Grep the autoresearch driver for the join key — `merge(..., on=["symbol","date"])` vs `merge(..., on="date")` — and the file path pattern (`SPY_daily_regime` only vs `<symbol>_daily_regime` glob).
2. Inspect any saved autoresearch artifact that breaks down performance by regime — does it carry per-symbol regime histograms or only a market-wide one?

A weaker but separate invalidator: autoresearch may use regime labels as a *feature* (input to a model that learns its own conditioning) rather than as a *filter* (subset bars by regime before computing edge). A tree-based model can learn "HIGH for TSLA" is near-constant and weight it accordingly; filter use cannot.

## 4. Concrete dangers

1. **False confidence in regime-conditioned wins on non-SPY symbols.** Under the X-reading, any "edge in HIGH regime" claim on the 62 high-vol-cluster symbols is effectively unconditional with extra labelling. If a strategy has been promoted to live based on this, it is being deployed under a regime hypothesis that the data never actually tested.
2. **Train/live regime divergence (Finding 4).** A strategy validated on Labeler A's regime space hits production governed by Labeler B. Boundary behavior, hysteresis, and trend definition differ. Because both labelers are unvalidated against honest OOS ground truth, this divergence is hard to bound.
3. **Risk-axis blind spot (Finding 5).** Strategies whose activation keys off `RISK_ON` have training data that did not contain the gating dimension. Their live performance is forward-test-only with no prior. Highest-severity silent failure — nothing in a backtest report would surface it.
4. **In-sample tuning bias (Finding 6).** The headline 81.9% accuracy is a model scored on its own training set. Confidence intervals assuming "labeler is ~82% accurate" are mis-specified, and the rejection of the Risk-axis proxy at 58% rests on the same in-sample yardstick.
5. **Universe composition skew disguised as regime.** Under the X-misuse, the labels effectively partition by *symbol class* (mega-cap index ETF vs. high-vol single name) rather than by regime. Autoresearch may be inadvertently learning symbol selection disguised as regime conditioning.

## Prioritized fix list

1. **P0 — Read the autoresearch driver and confirm whether it joins per-symbol or SPY-only regime parquets** (≈30 min). This single fact selects between Section 1's verdicts and gates every subsequent fix.
2. **P0, conditional on P0a returning "per-symbol joins" — quarantine and re-evaluate** all regime-conditioned autoresearch results from the past N iterations. Tag any "HIGH-conditioned edge" claim on non-SPY symbols as unverified pending re-run on a corrected labeler. Skip if SPY-only.
3. **P1 — Redefine `vol_regime` in `scripts/label_regime_dataset.py` as a market-wide state** (benchmark-derived from VIX or SPY-realized-vol) propagated to all symbols. Add a separate `symbol_vol_pctile` column for idiosyncratic vol use cases. Re-emit all 69 parquets.
4. **P1 — Build an honest, multi-asset, OOS regime ground-truth set** before claiming labeler accuracy. Hand-label at least 4 non-SPY periods (a high-vol single name, a cyclical sector ETF, a defensive, a commodity-linked) plus a held-out SPY year not in the original 8 tuning periods. Score Labeler A and Labeler B on the same set; only then is "train==live convergence on Labeler B" defensible.
5. **P2 — Add the Risk axis only when a defensible proxy beats a stated bar.** Bar: correctly labels each of Mar 2020, Jan 2022, Mar 2023 (SVB), Aug 2024 (yen carry) as RISK_OFF on t±2 and labels ≥80% of unambiguous calm periods as RISK_ON. Until then, runtime can keep emitting Risk axis with `RISK_ON`-gated strategies disabled in autoresearch validation.
6. **P2 — Document the regime contract** in a single source-of-truth file: which axes exist, what estimator each uses, threshold derivation, and how train and live are kept in sync. Findings 4 and 5 are symptoms of an unwritten contract.
7. **P3 — Add a labeler health check** that fails CI if (a) `vol_regime` is *not* constant across symbols on each date — the X-invariant — or (b) fewer than three of {LOW, NORMAL, HIGH, SHOCK} appear in any rolling 252-day window. Do not use "median HIGH/SHOCK across universe < threshold" — that metric is circular under both threshold regimes and would have been gamed by the percentile fix.
