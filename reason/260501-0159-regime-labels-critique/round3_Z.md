# Critique: Cerberus Regime Labeling Pipeline

## 1. Are the labels usable as-is?

**No, on the broad universe — and the strong prior is that they are poisoning autoresearch, not merely wasted.** The 69-parquet output layout (`data/regime_labeled/<SYMBOL>_daily_regime.parquet`) is itself near-conclusive evidence of per-symbol joining: nobody emits 68 redundant copies of a SPY-only series. The burden of proof has shifted — not "toxic if joined per-symbol" but "toxic unless we prove only SPY is consumed."

Three independent calibration failures, any one of which invalidates regime-conditioned conclusions on the non-SPY universe:

- **HIGH/SHOCK is a symbol-class proxy, not a market state.** Median symbol is HIGH-or-SHOCK 94.3% of days; for 65 of 68 names `vol_regime` carries no time-varying information. TSLA at 98.6% is constant relative to the labeler's resolution.
- **LOW is unreachable for ~96% of the universe.** Only 3 of 68 symbols spend ≥5% of days in LOW. Mean-reversion, theta-selling, and carry strategies — whose edge concentrates in low vol — have that branch silently deleted from training for 65 symbols.
- **The threshold scheme is a category error, not a tuning miss.** Absolute thresholds (8%/20%/50% annualized) are coherent only against a market-level benchmark (SPY, VIX). Applied to single-name vol whose distribution is shifted right by 2-5x, the universe must collapse into the upper bins. Re-tuning globally cannot fix it; the design is wrong.

Three concerns survive even if SPY-only consumption is confirmed: (i) train (SMA + absolute thresholds) vs. live (Hurst + EWMA z) estimator divergence, (ii) Risk axis in live gating, absent from training, (iii) the 81.9% accuracy is not just in-sample — its ground truth uses the same threshold scheme as the labeler, so it tests SMA-vs-rule-generator agreement, not whether the rules describe market physics.

## 2. Highest-leverage fix

**The single highest-leverage action is the 30-minute P0 audit (Section 3, item 1) — without it, every subsequent decision is uninformed. Conditional on confirming per-symbol joining (high prior), the lead fix is:** redesign Labeler A around a **market-state column** (one shared value per date, derived from VIX or SPY 30d realized vol with the existing absolute thresholds) plus a **per-symbol z-score** (continuous, published mapping to LOW/NORMAL/HIGH/SHOCK with thresholds derived from each symbol's trailing 252d distribution). Document the contract explicitly in a `REGIME_CONTRACT.md` checked in next to the labeler.

Why this beats the listed alternatives:

- **Per-symbol percentile alone (option a).** Fixes Failures 1 and 3 — defensive ETF and meme stock both reach LOW similar fractions, which is the goal. The objection that this "silently redefines HIGH 68 ways" only holds without a written contract; a documented per-symbol z-score is not invisible. What it *does* lose is the market signal — that all single-name z-scores spiked together. Adding a shared market column on top fixes that.
- **Swap to Labeler B (option b).** Right direction on train/live convergence (Finding 4), but B is also unvalidated and adds Hurst — noisy on daily bars over short windows and itself regime-dependent. Convergence on an unvalidated estimator is not strictly better than divergence; divergence is at least a detectable signal.
- **Add Risk axis (option c).** Necessary but does nothing about Failures 1-3. Sequence after vol-axis redesign.

Redesign is gated on OOS multi-asset ground truth (P1 below). Choosing labelers without that is the vibes-driven decision that produced this state.

## 3. Hidden assumptions that might invalidate the critique

1. **Consumption pattern (load-bearing, priors strong).** Per-symbol join means critique stands; SPY-only join collapses the verdict to "delete unused parquets and document the contract." 69-file layout is strong prior evidence for per-symbol joining — confirm by greping the autoresearch driver and `src/agent/`. Cost: 30 min. Until done, conservative stance is quarantine, not "probably fine."

2. **Filter-mode vs. feature-mode.** `df[df.vol_regime == "HIGH"]` — including results-reporting `groupby(regime)` — inherits miscalibration in full. A model fed `vol_regime` as one feature *might* attenuate, but tree models can recover the bias via interactions with symbol identifiers (constant-feature problem becomes leakage). Same grep resolves it.

3. **Strategies actually gated on broken states.** Without auditing `activation:` blocks in `config/strategies.yaml`, the *specific* harm of LOW-vol blindness is not yet tied to a named strategy. The general harm — autoresearch cannot learn what it cannot observe — stands.

4. **81.9% accuracy is doubly circular.** Beyond in-sample tuning, the SPY ground-truth proportions in Finding 3 (~9% LOW / ~80% NORMAL / ~18% HIGH / ~2% SHOCK) almost certainly use the labeler's own threshold scheme. The 81.9% tests whether SMA agrees with a rule generator sharing those thresholds — not whether the thresholds describe market physics. Any uncertainty estimate built on it is mis-specified beneath naive overfitting.

## 4. Concrete dangers

1. **Symbol-class skew disguised as regime conditioning.** Most insidious failure. Labels partition non-SPY symbols by *unconditional volatility class* (mega-cap defensive vs. high-beta single name) rather than time-varying market state. Autoresearch learns *symbol selection* while reporting *regime conditioning*. Regime-breakdown plots populate plausibly — but the partition is not what its label claims.
2. **False confidence in HIGH/SHOCK edge on non-SPY symbols.** "Edge in HIGH regime on TSLA/AMD/NIO/MARA" is approximately the unconditional return on those names, since they are HIGH 98%+ of the time. Strategies promoted on this basis are untested against the hypothesis they claim.
3. **LOW-vol blindness on 65 symbols.** Mean-reversion and theta-selling strategies cannot be validated outside the 3 LOW-reaching symbols. They look unviable on the wrong names or viable for the wrong reason — both wrong-direction errors biasing the promotion pipeline.
4. **Train/live divergence with unbounded magnitude.** Validated on dual-SMA + absolute thresholds, deployed under Hurst + EWMA z. Both unvalidated on a common OOS set; the gap is unbounded.
5. **Risk-axis silent failure.** 5-day-return proxy rejected at 58% — barely above coin flip — yet live engine emits it and `RISK_ON`-gated strategies fire on it. Backtest edge is unconditional-on-risk; live edge conditional on near-noise. No standard report surfaces this gap.
6. **Doubly-circular accuracy reporting.** The 81.9% cannot detect whether the threshold scheme is wrong, because ground truth shares those thresholds. Uncertainty propagation built on it is mis-specified.

## Prioritized fix list

1. **P0 — Read the autoresearch driver and `src/agent/` consumers.** ~30 min. Confirm per-symbol vs. SPY-only joining; note filter-vs-feature use; list every `activation:` block keyed off `vol_regime` or `risk_state`. Selects between Section 1's verdicts; gates everything below.
2. **P0 conditional — Quarantine non-SPY regime-conditioned promotions.** If per-symbol joins confirmed (high prior): tag every "edge in HIGH/SHOCK on non-SPY symbol" finding from the last 90 days as unverified; block live promotions until P2 lands. Cheaper than recalculating, reversible.
3. **P1 — Build defensible OOS multi-asset regime ground truth.** Hand-label held-out periods not in the original 8 SPY tuning windows, plus one period each for a high-vol single name (TSLA), defensive name (JNJ), sector ETF, and commodity-linked ETF. Score Labeler A *and* Labeler B against this set. Until this exists, picking a labeler is choosing on vibes.
4. **P1 — Risk axis: ship a real classifier or stop emitting it.** A 58% classifier driving live `RISK_ON` gates is worse than no classifier — strategies are conditioning on noise. Either ship a defensible proxy (must classify Mar 2020, Jan 2022, Mar 2023 SVB, Aug 2024 yen-carry as RISK_OFF on t±2 and ≥80% of unambiguous calm windows as RISK_ON) or remove the axis from `MarketContextService` and have `RISK_ON`-gated strategies treat it as constant. Currently it is the worst of both worlds.
5. **P2 — Redesign Labeler A: market-state column + per-symbol z-score column.** Both with documented thresholds. Re-emit all 69 parquets; do not preserve bare `vol_regime` as alias — force every consumer to pick one explicitly. After P1 ground truth, decide whether absolute-threshold market state or Hurst+EWMA z best matches truth, and use the same estimator in train and live.
6. **P2 — Document the regime contract.** `REGIME_CONTRACT.md` next to the labeler: axes, estimators, threshold derivation, hysteresis, train/live invariant. Methodology drift is the symptom; the absent contract is the disease.
7. **P3 — Aggregate-distribution CI checks on every labeler emit.** Fail the build if (a) market-state column differs across symbols on any common date, (b) cross-universe time-in-LOW for the per-symbol column differs from a stated reference (e.g., 5-15% range) by more than a factor, (c) >50% of universe spends >80% of days in any single bin of the per-symbol column, (d) train and live emit mismatched axes for any common date in the last 30 days. Per-symbol forced distributions would fire false positives on a defensive utility ETF; the pathology is universe-aggregate, the check should be too.
