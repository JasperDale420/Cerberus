# Critique: Cerberus Regime Labeling Pipeline

[Synthesis approach: A2 commits to a verdict ("No, labels are unusable") with crisp Three-Failure framing and the strongest fix proposal — two named columns that prevent silent rebasing. B2 is more rigorously honest about epistemic gating: the toxic vs. wasted-disk verdict literally hinges on consumption pattern, the SPY-truth circularity is a real flaw A2 misses, and B2's CI checks correctly target universe-aggregate not per-symbol. I take A2's structural fix and danger taxonomy, B2's epistemic conditioning and circularity insight, and resolve their direct contradiction on parallel-columns by defending A2's position with B2's audit requirement attached.]

## 1. Are the labels usable as-is?

**No — not for non-SPY symbols, with one cheap-to-check escape that downgrades the verdict from toxic to wasted disk.** The pipeline fails on cross-symbol calibration in three independent ways, each sufficient to invalidate regime-conditioned autoresearch on the broad universe:

- **Failure 1 — Cross-symbol semantics collapse.** Median symbol is HIGH-or-SHOCK 94.3% of days; for 65 of 68 names `vol_regime` is effectively constant.
- **Failure 2 — LOW unreachable for almost the entire universe.** Only 3 of 68 symbols spend >=5% of days in LOW. Mean-reversion, carry, and theta-selling — whose edge concentrates in low-vol — have that branch deleted from training for 65 symbols.
- **Failure 3 — SHOCK is asymmetric across symbols.** SPY hits 50% annualized vol only in real crises (~2% of days). TSLA's normal noise approaches it. Backtest claims of edge in SHOCK on TSLA-class names are studying the unconditional return.

The escape (Section 3) is a consumption pattern question — SPY-only broadcast vs. per-symbol joins — answerable by reading one driver file. SPY's own labels look approximately reasonable; if those are the only labels actually consumed, the 68 broken parquets are inert. Until that read is done, the asymmetric cost favors quarantining recent regime-conditioned non-SPY promotion decisions immediately.

Three concerns survive both readings: (i) train/live estimator divergence, (ii) missing Risk axis at training despite live gating, (iii) the 81.9% accuracy has a circularity flaw deeper than naive in-sample bias (Section 3, point 4).

## 2. Highest-leverage fix

**Lead fix: replace the single `vol_regime` column with two explicitly-named columns — `vol_regime_market` (one shared state per date, derived from VIX or SPY 30d realized vol) and `vol_regime_symbol` (per-symbol percentile against that name's own trailing 252d distribution). Force every consumer to pick one by name. This is gated on a measurement prerequisite: build defensible OOS multi-asset ground truth before deciding the train/live convergence question.**

Why this beats each alternative in isolation:

- **Per-symbol percentile only (option a).** Solves Failure 1 but silently redefines "HIGH" 68 different ways. Activation policies firing on `vol_regime == HIGH` would trade TSLA's 60th-self-percentile vol the same way they trade SPY's — radically different states. Replaces visible miscalibration with invisible semantic break: strictly worse than status quo, because at least the current pathology is detectable.
- **Swap to Labeler B's Hurst+EWMA z (option b).** Right *direction* on Finding 4, but A and B are both unvalidated. A's 81.9% is in-sample on the same 8 SPY periods used to tune. B has no published number at all. Convergence on an unvalidated estimator is no better than divergence between two — divergence is at least a detectable signal.
- **Add Risk axis (option c).** Real gap, close it as a separate axis after a defensible proxy clears a stated bar. The 5-day-return rejection at 58% came from the *same* in-sample SPY-only set that produced the 81.9% — an unreliable rejection from an unreliable yardstick.

**On parallel columns:** the objection that two columns "double the train/live consistency surface" is wrong. The current single-column design *already* has a hidden second meaning — broken on 65 symbols — and the train/live gap stems from missing contract documentation, not column count. A bare `vol_regime` rename cannot represent that two semantically different things are being asked of one label. Two named columns force the activation-policy audit to happen; one column hides it.

The measurement gate is non-negotiable: do not re-emit anything before honest OOS multi-asset ground truth exists. Choosing labelers on vibes is what produced this state.

## 3. Hidden assumptions that might invalidate the critique

1. **Consumption pattern (load-bearing).** `merge(..., on=["symbol","date"])` against per-symbol parquets means the critique stands; `merge(..., on="date")` against SPY-only frame collapses the toxic verdict to "delete unused parquets and document the contract." Resolve by reading the autoresearch driver and `src/agent/`. Cost: minutes.

2. **Filter-mode vs. feature-mode consumption.** A `df[df.vol_regime == "HIGH"]` filter inherits miscalibration in full. A model fed `vol_regime` as one feature *might* attenuate, but near-constant features can still corrupt tree models via interactions with symbol identifiers. A `groupby` on regime in results-reporting code is filter-mode and inherits the toxicity in full. Same grep resolves it.

3. **Strategies actually gated on broken states.** Without auditing `activation:` blocks in `config/strategies.yaml`, LOW-vol blindness has not been shown to delete any specific strategy's edge. The general harm — autoresearch cannot learn what it cannot observe — stands; the specific harm needs evidence.

4. **SPY truth proportions share the labeler's threshold scheme.** The cited "true" SPY distribution uses the same LOW<8% / HIGH>=20% / SHOCK>=50% scheme as the labeler. The 81.9% in-sample number is therefore not testing whether the *thresholds* are correct — only whether the SMA estimator agrees with a label generator using the same thresholds. The labeler could be wrong about market physics in a way no SPY-derived number could detect.

## 4. Concrete dangers

1. **Symbol-class skew disguised as regime conditioning.** The most insidious failure. Labels effectively partition non-SPY symbols by their unconditional volatility class (mega-cap defensive vs. high-beta single name) rather than by time-varying market state. Autoresearch is learning *symbol selection* while reporting *regime conditioning*. Regime-breakdown plots populate plausibly — but the underlying partition is not what it claims.
2. **False confidence in regime-conditioned wins on non-SPY symbols.** Any "edge in HIGH regime" claim on the high-vol-cluster names is approximately the unconditional result. If a strategy was promoted on this conditioning, its live performance is being judged against a hypothesis the data never tested.
3. **LOW-vol blindness.** Strategies whose edge concentrates in LOW vol cannot be validated for any name outside the 3 LOW-reaching symbols. They look unviable on the wrong names or viable for the wrong reason — both wrong-direction errors.
4. **Train/live divergence with unknown magnitude.** Strategies validated on dual-SMA + absolute thresholds, deployed under Hurst + EWMA z. Both labelers unvalidated on a common OOS set, so the gap is unbounded.
5. **Risk-axis silent failure.** `RISK_ON`-gated strategies trained on data with no risk-axis dimension. Backtest edge unconditional-on-risk; live edge conditional-on-risk. No standard report surfaces this.
6. **Circular in-sample tuning bias.** The 81.9% is not just in-sample — its ground-truth labels share the labeler's threshold scheme. Any uncertainty estimate propagated from this number is mis-specified at a deeper level than naive overfitting.

## Prioritized fix list

1. **P0 — Read the autoresearch driver. ~30 min.** Confirm per-symbol vs. SPY-only joining; note filter-vs-feature use. Selects between Section 1's verdicts; gates everything below.
2. **P0 conditional — Quarantine non-SPY regime-conditioned results** if per-symbol joins confirmed. Tag every "edge in HIGH/SHOCK on non-SPY symbol" as unverified; block promotions until P1 completes.
3. **P1 — Build honest multi-asset OOS regime ground truth.** Hand-label a held-out SPY year not in the original 8 tuning periods, plus one period each for: high-vol single name, defensive name, sector ETF, commodity-linked. Score Labeler A *and* Labeler B against this set. Until this number exists, choosing between estimators is choosing on vibes.
4. **P1 — Emit two columns** (`vol_regime_market`, `vol_regime_symbol`). Re-emit all 69 parquets. Update consumer code and every `activation:` block to reference one explicitly — do not preserve bare `vol_regime` as alias. After P1 ground truth, decide whether Labeler A or B feeds each column.
5. **P2 — Risk axis behind a stated empirical bar.** Required: correctly classify Mar 2020, Jan 2022, Mar 2023 (SVB), Aug 2024 (yen carry) as RISK_OFF on t±2; >=80% of unambiguous calm windows as RISK_ON. Until then, runtime keeps emitting Risk axis but `RISK_ON`-gated strategies run in shadow during autoresearch validation.
6. **P2 — Document the regime contract** in a single source-of-truth file: axes, estimators, threshold derivation, hysteresis, train/live consistency invariant. Methodology drift is the symptom; the absent contract is the disease.
7. **P3 — Aggregate-distribution CI labeler health checks.** Fail the build if (a) `vol_regime_market` is not identical across symbols on a given date, (b) cross-universe aggregate time-in-LOW differs from SPY's by more than a stated factor (current gap: 0.32% vs. 9%), (c) >50% of universe spends >80% of days in any single bin, (d) train and live emit mismatched axes for any common date in the last 30 days. Per-symbol forced distributions would fire false positives on a defensive utility ETF that genuinely sits NORMAL all year — the pathology is universe-aggregate, the check should be too.
