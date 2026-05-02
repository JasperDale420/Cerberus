# Critique: Cerberus Regime Labeling Pipeline

**Evidence (from task brief).** 103,803 daily symbol-bars. SPY's per-year histograms approximately track plausible truth (2024: 256/271 NORMAL, 14 LOW, 1 HIGH, 0 SHOCK). Cross-sectional median symbol: 94.3% time in HIGH-or-SHOCK. Only 3 of 68 symbols reach LOW on >5% of days. Aggregate: 0.32% LOW vs. 67.6% HIGH/SHOCK. SPY's measured truth: ~9/80/18/2. Labeler tuned by grid search on 8 hand-picked SPY periods, no held-out OOS, declared 81.9% combined accuracy on the same tuning set. Live runtime uses a different estimator (Hurst + EWMA z) than training (dual-SMA + absolute thresholds). Risk axis dropped from labeler ("5-day return proxy too noisy, 58%") but live engine still emits one; several activation policies key off it.

---

## 1. Are the labels usable as-is?

**The verdict depends on a single unanswered question: does autoresearch consume the per-symbol parquets, or only `SPY_daily_regime.parquet` broadcast across the universe?** Until that is read out of the autoresearch driver, the strongest defensible verdict is conditional. Asserting "toxic" before reading the consumer is overreach; asserting "fine" ignores genuine evidence of miscalibration.

**Scenario X — autoresearch joins per-symbol regime to per-symbol bars.** SPY's labels look approximately calibrated against the cited SPY truth, but every other symbol's `vol_regime` carries near-zero information: at a median 94.3% time-in-HIGH-or-SHOCK and only 3 of 68 names ever touching LOW, the column is degenerate per-symbol. Any "edge in HIGH regime on TSLA" claim is being computed on a state that fires on essentially every TSLA bar — that is the unconditional return relabeled. Recent regime-conditioned promotion decisions on non-SPY names need to be quarantined.

**Scenario Y — autoresearch reads only SPY's parquet and broadcasts on `date`.** SPY's labels appear roughly reasonable. The 68 broken non-SPY parquets are wasted disk, the calibration pathology never enters training, and the practical fix is to delete the bad files and document the contract. The miscalibration findings remain scientifically embarrassing but operationally inert.

The asymmetric cost favors checking. Either way, three concerns survive both readings: (i) train/live estimator divergence, (ii) the missing Risk axis at training time despite live gating, (iii) in-sample-only accuracy numbers.

## 2. Highest-leverage fix

**The highest-leverage fix is not (a), (b), or (c) standalone — it is a *measurement* fix that gates the redesign: confirm consumption pattern, then build defensible OOS multi-asset ground truth, and only then choose between options. Adopting any redesign before this measurement work is choosing one unvalidated estimator over another.**

Why each option is currently undermotivated in isolation:

- **Option (a) — per-symbol percentile thresholds.** The right *direction* for cross-symbol calibration: a per-symbol 30d realized-vol percentile against that name's own trailing window would convert elevated absolute vol into a state that varies meaningfully over time. The serious cost is semantic: `vol_regime == HIGH` then means "high relative to itself" rather than "high relative to the market," and every activation policy reading that label is silently rebased per symbol. The policy YAMLs must be re-audited against the new meaning. Discrete bins vs. continuous percentile is a secondary tradeoff: bins preserve interface compatibility but lose information at boundaries; pairs and vol-mean-reversion logic may want the continuous z.

- **Option (b) — replace Labeler A with Labeler B's Hurst+EWMA z.** Solves the methodology-drift gap directly. The cost is that B has no published accuracy number on any ground-truth set — converging two unvalidated estimators is no better than running them divergent, since divergence is at least a detectable signal.

- **Option (c) — add Risk axis.** The 5-day-return proxy was rejected at 58% on the same in-sample SPY-only set that produced the 81.9% combined number. That rejection is unreliable for the same reason the acceptance is. A defensible Risk-axis labeler needs a stated bar (correct classification of major selloffs) and validation off-SPY before integration.

Recommended sequencing: do (a) *or* (b), not both as parallel columns. Splitting `vol_regime` into a market-derived column and a per-symbol percentile column doubles the train/live consistency surface while validation work is outstanding — adding a second axis under the same uncertainty makes consistency worse. Pick one labeler, one definition; force consuming activation policies to be honest about which they want; validate before changing anything else.

## 3. Hidden assumptions that might invalidate the critique

1. **Consumption pattern (load-bearing).** `merge(..., on=["symbol","date"])` against per-symbol parquets means the critique stands; `merge(..., on="date")` against a SPY-only frame collapses Scenario X. Resolve by reading `/Users/jacobmcmillan/Empire/Cerberus/src/agent/` and the autoresearch loop's data-prep stage. Cost: minutes.

2. **Filter-mode vs. feature-mode consumption.** Even under Scenario X, harm depends on *how* the column is used. A `df[df.vol_regime == "HIGH"]` filter inherits the miscalibration in full. A model fed `vol_regime` as one feature *might* attenuate the problem under some model classes, but this should not be assumed — near-constant features can still corrupt tree models via interactions with symbol identifiers, and "feature-importance regularization will save us" is wishful thinking unless inspected. Same grep resolves it.

3. **Strategies actually gated on the broken states.** This critique deliberately does not claim that any specific Cerberus strategy is gated on `vol_regime == LOW` for non-SPY symbols. Without auditing `activation:` blocks in `config/strategies.yaml`, the LOW-vol blindness in the labels has not been shown to delete any specific strategy's edge. The general harm — autoresearch cannot learn what it cannot observe — stands; the specific harm needs evidence.

4. **SPY truth proportions.** The cited "true" SPY distribution shares the same threshold scheme as the labeler. The 81.9% in-sample number is therefore not testing whether the thresholds are *correct* — only whether the SMA-based estimator agrees with a label generator using the same thresholds. All SPY-tuned numbers in the report inherit this circularity.

## 4. Concrete dangers

1. **Symbol-class skew disguised as regime conditioning.** The most insidious failure. If labels effectively partition non-SPY symbols by their unconditional volatility class (mega-cap defensive vs. high-beta single name) rather than by time-varying market state, autoresearch is learning *symbol selection* while reporting *regime conditioning*. Regime-breakdown plots populate plausibly — but the underlying partition is not what it claims. Promotion gates reading "edge X% in HIGH regime" are then approximately reading "edge X% on volatile names" with no time-varying content.

2. **False confidence in regime-conditioned promotion decisions.** Conditional on Scenario X, any strategy promoted on "outperforms in HIGH regime" against the broad universe is being judged against a hypothesis the data cannot test. The training data does not contain enough non-HIGH bars per non-SPY symbol to support the conditional claim.

3. **Train/live divergence with unknown magnitude.** Strategies validated on dual-SMA + absolute-threshold labels are deployed under Hurst + EWMA z + relative-threshold labels. Boundary behavior, hysteresis, and trend definition all differ. Both labelers are unvalidated against a common OOS set, so the gap is unbounded.

4. **Risk-axis silent failure.** Strategies with `RISK_ON`-only entry conditions were trained on data containing no risk-axis dimension. Their backtest edge is unconditional-on-risk; their live edge is conditional-on-risk. Standard backtest reports will not surface this.

5. **In-sample tuning bias.** The 81.9% headline accuracy is a model scored on its own training set. Any downstream uncertainty estimate propagated from it is mis-specified.

## Prioritized fix list

1. **P0 — Read the autoresearch driver.** Locate the regime-label join, confirm per-symbol vs. SPY-only consumption, note filter-vs-feature use. Selects between Section 1's two scenarios; gates everything below.
2. **P0 conditional — Quarantine non-SPY regime-conditioned results** if Scenario X holds. Tag every "edge in HIGH/SHOCK on non-SPY symbol" claim as unverified; block promotions until P1 completes.
3. **P1 — Build honest OOS multi-asset ground truth.** Hand-label a held-out SPY year not in the original 8 tuning periods, plus one period each for a high-vol single name, a defensive name, and a sector ETF. Score Labeler A *and* Labeler B against this set. Until this exists, choosing between estimators is choosing on vibes.
4. **P1 — Decide one labeler, one definition.** After P1: adopt Labeler B end-to-end (option b) or replace the absolute-threshold vol component with per-symbol percentile bins on Labeler A (option a). Do not run both as parallel columns. Re-emit all 69 parquets; update activation policies to match.
5. **P2 — Risk axis behind a stated empirical bar.** Define acceptance criteria before integration: correctly classify Mar 2020, Jan 2022, Mar 2023 (SVB), Aug 2024 (yen carry) as RISK_OFF on t±2; ≥80% of unambiguous calm windows as RISK_ON. Until then, `RISK_ON`-gated strategies run in shadow during autoresearch validation.
6. **P2 — Document the regime contract** in a single source-of-truth file: axes, estimators, threshold derivation, hysteresis rules, train/live consistency invariant. Methodology drift is the symptom; the absent contract is the disease.
7. **P3 — Aggregate-distribution CI checks (not per-symbol forced distributions).** Fail the build if (a) cross-universe aggregate time-in-LOW differs from SPY's by more than a stated factor (current gap: 0.32% vs. 9%), (b) >50% of the universe spends >80% of days in a single regime bin, (c) train and live emit mismatched axes for any common date in the last 30 days. Per-symbol forced distributions would fire false positives on a defensive utility ETF that genuinely sits in NORMAL all year. The pathology is universe-aggregate; the check should be too.
