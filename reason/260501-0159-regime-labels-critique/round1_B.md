# Critique: Cerberus Regime Labeling Pipeline

## 1. Are the labels usable as-is?

**The verdict is conditional on a single fact about the autoresearch driver that the evidence pack does not pin down.** Two coherent readings of "regime label" exist, and usability flips between them:

- **Interpretation X — regime as a *market-wide* state.** A single coordinate the strategy uses to decide whether its edge applies. Every symbol on day D should carry the same vol regime, because the claim is about the market, not the name.
- **Interpretation Y — regime as a *symbol-local* state.** A per-symbol feature analogous to a z-score: "is this name unusually volatile relative to its own history?"

Labeler A's absolute thresholds (LOW<8%, HIGH≥20%, SHOCK≥50% on trailing 30-day annualized realized vol, per the task brief) are plausibly correct under Y for any individual symbol — TSLA labelled HIGH on 98.6% of days is literally true; TSLA's realized vol *is* above 20% on 98.6% of days. The problem is whether autoresearch consumes those labels under X (broken) or Y (defensible per-symbol but unable to support cross-sectional regime conditioning).

**Most likely use is X.** Autoresearch is a strategy validator; live `MarketContextService` references market-wide axes (RISK_ON, HIGH-vol session); Finding 4 says train uses Labeler A while live uses Labeler B, and Labeler B is structured for market-wide states (Hurst on a benchmark return). Under that reading: **labels are not usable as-is for non-SPY symbols, and any regime-conditioned autoresearch claim against the broader 68-symbol universe must be re-evaluated.** Under the alternative-Y reading, labels are roughly defensible per symbol but autoresearch then has no cross-sectional regime signal at all and the "regime conditioning" framing is being misused upstream. Either way, something is broken; the single decisive check is in Question 3, and a 30-minute investigation collapses the conditional.

Supporting the X reading: the cross-symbol distribution is bimodal. SPY/JNJ/COST/QQQ/WMT/JPM sit between 16% and 52% HIGH-or-SHOCK; the median of the remaining 62 symbols is 94.3% with a tight cluster near 98%. That gap is not a smooth function of underlying volatility — it is the signature of an absolute threshold cutting through the realized-vol distribution at a point separating "index-like" from "single-name-like." Under Interpretation Y you would expect smooth dispersion across symbols, not a cliff.

## 2. Highest-leverage fix

**Lead fix: redefine `vol_regime` as a single market-wide state shared across all symbols, computed from a benchmark or vol index — not from each symbol's own series.** Per-symbol relative vol *should be added as a separate feature* but must not replace the market-wide label, because that would silently break the train==live contract with `MarketContextService` and make cross-strategy ensemble routing ill-defined.

Concretely:

- The per-symbol parquet's `vol_regime` column is derived from a single market-wide vol estimator (VIX level, or SPY's trailing 30d realized vol scaled to comparable units) and propagated to every symbol on day D. This collapses the 94.3%-median pathology because all symbols agree on the day's vol state. It also matches what live `MarketContextService` does — Hurst+EWMA computed on a benchmark, not per name.
- Idiosyncratic vol can stay in the parquet as a separate column (e.g., `symbol_vol_pctile` = the symbol's realized vol's rank within its own trailing 252d distribution). Strategies that need "is TSLA unusual for itself today" use that column; strategies that need "is the market in a vol shock" use the shared regime column. Both axes are well-defined and orthogonal.

Why not the alternatives:

- **Per-symbol percentile-only thresholds destroy semantics.** If TSLA's "HIGH" means "top quartile of TSLA's own vol" while SPY's "HIGH" means the same for SPY, regime labels no longer denote the same market state across symbols. Activation policies, ensemble routing, and any cross-strategy comparison silently re-interpret "HIGH" 68 different ways. Worse than status quo because the failure mode is undetectable from outside.
- **Wholesale swap to Labeler B (Hurst+EWMA z) to enforce train==live.** Right *direction*, but Labeler A and Labeler B are *both* effectively unvalidated — Labeler A's 81.9% accuracy is in-sample on the same 8 SPY periods used to tune (Finding 6), and Labeler B has no published ground-truth comparison at all. Convergence on an unvalidated labeler is no better than divergence between two unvalidated labelers; in fact it is worse, because divergence is at least detectable. Adopt B's outputs only after both labelers are scored on a common, OOS, multi-asset ground-truth set.
- **Adding the Risk axis** (Finding 5) is a real gap and must be closed, but on a *separate* axis. The 5-day-return proxy was rejected at 58% accuracy on a ground truth Finding 6 already shows is unreliable; a 58% number from an unreliable yardstick is not a reliable rejection. The right response is to define what ground truth a Risk-axis classifier must beat — not to rush in a slightly more elaborate version of the rejected proxy.

## 3. Hidden assumption that might invalidate the critique

**Autoresearch may consume only SPY's regime labels and broadcast them across symbols.** If so, the 67 broken non-SPY parquets are dead weight on disk and the calibration finding is a distraction.

If the driver opens `SPY_daily_regime.parquet`, joins by date alone, and ignores per-symbol parquets, labels are effectively under Interpretation X with SPY as the benchmark — and SPY's labels are defensible (Finding 1). The 94.3% median pathology only matters if those parquets are being read.

Two cheap verifications:

1. Grep the autoresearch driver for the join key — `merge(..., on=["symbol","date"])` vs `merge(..., on="date")` — and the file path pattern (`SPY_daily_regime` only vs `<symbol>_daily_regime` glob).
2. Inspect any saved autoresearch artifact that breaks down performance by regime — does it carry per-symbol regime histograms or only a market-wide one?

A weaker but separate invalidator: autoresearch may use regime as a *feature* (input to a model that learns its own conditioning) rather than as a *filter* (subset bars before computing edge). A tree-based model can learn "HIGH for TSLA" is near-constant and weight it appropriately; filter use cannot.

## 4. Concrete dangers

1. **False confidence in regime-conditioned wins on non-SPY symbols.** Under the X-reading, any "edge in HIGH regime" claim on the 62 high-vol-cluster symbols is effectively unconditional with extra labelling. The conditioning variable carries near-zero information there.
2. **Train/live regime divergence (Finding 4).** A strategy validated on Labeler A's regime space hits production governed by Labeler B. Even when Labeler A agrees with reality on SPY, the boundary behavior, hysteresis, and trend definition differ from Labeler B. The fact that both labelers are unvalidated against honest OOS ground truth makes this divergence hard to bound.
3. **Risk-axis blind spot (Finding 5).** Strategies whose activation keys off `RISK_ON` have training data that did not contain the gating dimension. Their live performance is forward-test-only with no prior. Highest-severity silent failure — nothing in a backtest report would surface it.
4. **In-sample tuning bias (Finding 6).** The headline 81.9% accuracy is a model scored on its own training set. Confidence intervals assuming "labeler is ~82% accurate" are mis-specified, and the rejection of the Risk-axis proxy at 58% rests on the same in-sample yardstick.

## Prioritized fix list

1. **P0 — Read the autoresearch driver and confirm whether it joins per-symbol or SPY-only regime parquets** (≈30 minutes). This single fact selects between the two Section 1 verdicts and gates every subsequent fix.
2. **P1, conditional on P0 returning "per-symbol joins" — quarantine and re-evaluate** past iterations of regime-conditioned autoresearch on non-SPY symbols. Tag any "edge in HIGH regime" claim as unverified pending re-run on a corrected labeler. If P0 returns "SPY-only," skip this entirely.
3. **P1 — Redefine `vol_regime` as a market-wide state** (benchmark-derived: VIX or SPY-realized-vol) propagated to all symbols. Add a separate `symbol_vol_pctile` column. Re-emit all 69 parquets. Verify `vol_regime` matches across symbols on every date.
4. **P1 — Build an honest, multi-asset, OOS regime ground-truth set** before claiming labeler accuracy. Hand-label at least 4 non-SPY periods (a high-vol single name, a cyclical sector ETF, a defensive, a commodity-linked) and a held-out SPY year not in the original 8 tuning periods. Score Labeler A and Labeler B on the same set; only then is "train==live convergence on Labeler B" defensible.
5. **P2 — Add the Risk axis only when a defensible proxy beats a stated bar.** Bar: correctly labels each of Mar 2020, Jan 2022, Mar 2023 (SVB), Aug 2024 (yen carry) as RISK_OFF on t±2, and labels ≥80% of unambiguous calm periods as RISK_ON. Until then, runtime can keep emitting Risk axis with `RISK_ON`-gated strategies disabled.
6. **P2 — Document the regime contract** in one source-of-truth file: axes, estimator per axis, threshold derivation, train/live convergence rule. Findings 4 and 5 are symptoms of an unwritten contract.
7. **P3 — Add a labeler health check** asserting (a) `vol_regime` is constant across symbols on each date — the X-invariant — and (b) at least three of {LOW, NORMAL, HIGH, SHOCK} appear in any rolling 252-day window. Do not use "median HIGH/SHOCK across universe < threshold" — that metric is circular under both threshold regimes.
