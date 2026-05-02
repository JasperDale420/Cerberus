# Critique: Cerberus Regime Labeling Pipeline

## 1. Are the labels usable as-is?

**No.** The pipeline is broken in three independent ways, any one of which invalidates regime-conditioned autoresearch on the non-SPY universe. The interpretive ambiguity (market-wide vs. symbol-local regime) is real but does not rescue the labels — both readings fail empirically.

Thresholds (LOW<8%, NORMAL 8–20%, HIGH 20–50%, SHOCK≥50% on 30d annualized realized vol) were tuned on SPY and encode an SPY-shaped vol distribution as universal physics. Three failures follow:

- **Failure 1 — Cross-symbol semantics collapse.** Median symbol is HIGH-or-SHOCK 94.3% of days; for 65 of 68 names `vol_regime` is effectively constant. A constant feature carries zero information. The symbol-local reading does not save it either — symbol-local thresholds should produce roughly equal time in each bin per symbol, not the observed cluster at one extreme.
- **Failure 2 — LOW is unreachable for almost the entire universe.** Only 3 of 68 symbols spend ≥5% of days in LOW. LOW vol is where mean-reversion and carry strategies typically have their cleanest edge — the labeler has deleted that branch from the training distribution for 65 symbols.
- **Failure 3 — SHOCK is asymmetric across symbols.** SPY hits 50% annualized vol only in real crises (≈2% of days, per Finding 1's COVID tally). TSLA's normal daily noise approaches it. The same threshold means "1-in-50-year tail" for SPY and "Tuesday" for TSLA. A backtest claiming edge in SHOCK on TSLA-class names is studying TSLA's unconditional return, not a tail event.

One hidden-assumption escape (Section 3) could downgrade this from "toxic" to "wasted disk." Absent that escape, every non-SPY label is compromised and recent regime-conditioned promotion decisions need to be quarantined.

## 2. Highest-leverage fix

**Lead fix: replace the single `vol_regime` column with two explicitly-named columns — `vol_regime_market` (one shared state per date, derived from VIX or SPY 30d realized vol) and `vol_regime_symbol` (per-symbol percentile against that name's own trailing 252d distribution). Force every consumer to pick one by name. Defer the train/live convergence question and the Risk axis until honest OOS ground truth exists.**

Why this beats the alternatives:

- **Per-symbol percentile-only thresholds (option a).** Solves Failure 1 but silently redefines "HIGH" 68 different ways. Activation policies firing on `vol_regime == HIGH` would trade TSLA's 60th-self-percentile vol the same way they trade SPY's — radically different states. Replaces a visible miscalibration with an invisible semantic break, strictly worse than status quo. Two named columns avoid this by forcing the choice explicit at every consumer site.
- **Swap to Labeler B — Hurst+EWMA z (option b).** Right *direction* on Finding 4, but A and B are both unvalidated. A's 81.9% is in-sample on the same 8 SPY periods used to tune (Finding 6). B has no published number at all. Convergence on an unvalidated estimator is no better than divergence between two — arguably worse, since divergence is at least a detectable signal. Adopt B only after both are scored on a common OOS multi-asset ground-truth set.
- **Add the Risk axis (option c).** Real gap (Finding 5), close it as a separate axis after a defensible proxy exists. The 5-day-return proxy was rejected at 58% on the *same in-sample SPY-only ground truth* that produced 81.9% — an unreliable rejection from an unreliable yardstick.

Two discrete-bucketed columns rather than one market-wide column plus a raw percentile: vol-mean-reversion and idiosyncratic-pairs strategies genuinely need per-symbol state, and a discrete bin keeps interface-compatibility with existing state-based gating. Both columns share `LOW/NORMAL/HIGH/SHOCK` vocabulary but mean what their suffix says.

## 3. Hidden assumption that might invalidate the critique

**Autoresearch may consume only `SPY_daily_regime.parquet` and broadcast its labels across every symbol on the join.** Under that case, the 68 broken non-SPY parquets sit on disk unused, the calibration pathology never enters training, and Findings 2–3 are spectator data. SPY's labels themselves (Finding 1) look approximately reasonable and roughly track the cited Heber-truth proportions for 2020–2026.

Two cheap verifications, well under an hour combined:

1. Grep the autoresearch driver for the join. `merge(..., on=["symbol","date"])` or a per-symbol parquet glob means per-symbol labels enter training — critique stands. `merge(..., on="date")` against a SPY-only frame means market-wide broadcast — critique downgrades to "delete unused parquets and document the contract."
2. Inspect a recent autoresearch artifact (any walk-forward report or by-regime breakdown). Per-symbol regime histograms in the output → per-symbol labels in the input.

A weaker secondary escape: regime labels may be consumed as a *feature* (model input) rather than a *filter* (subsetting bars before computing edge). Tree models would learn that `vol_regime` is near-constant for TSLA-class names and downweight it. Filter-based use cannot self-correct this way. Worth checking the consumer code for `df[df.vol_regime=="HIGH"]` patterns vs. `model.fit(X_with_regime_col)` patterns — a `groupby` on regime in any results-reporting code is filter-mode and inherits the toxicity in full.

## 4. Concrete dangers

1. **False confidence in regime-conditioned wins on non-SPY symbols.** Any "edge in HIGH regime" claim on the high-vol-cluster names is approximately the unconditional result. If a strategy was promoted to live based on this conditioning, its live performance is being judged against a hypothesis the data never tested.
2. **Universe-composition skew disguised as regime.** The labels effectively partition by *symbol class* (mega-cap ETF/defensive vs. high-vol single name) rather than by market state. Autoresearch may be learning symbol selection while reporting regime conditioning. This is the most insidious failure mode because it produces plausible-looking regime breakdown plots that are actually symbol-cluster breakdowns.
3. **LOW-vol blindness.** Strategies whose edge concentrates in LOW vol environments (carry, low-vol mean reversion, theta selling on stable underliers) cannot be validated for any name outside the 3 LOW-reaching symbols. Either they look unviable on the wrong names or they look viable for the wrong reason — both are wrong-direction errors.
4. **Train/live regime divergence (Finding 4).** Strategy validated on Labeler A, deployed under Labeler B. Boundary behavior, hysteresis, trend definition all differ. Both labelers are unvalidated, so the size of the gap is unbounded.
5. **Risk-axis silent failure (Finding 5).** Strategies gated on `RISK_ON` at runtime have training data with no risk-axis dimension. Their backtest edge is unconditional-on-risk; their live edge is conditional-on-risk. No backtest report will surface this.
6. **In-sample tuning bias (Finding 6).** 81.9% is a model scored on its own training set. Any confidence interval propagated from that number to downstream uncertainty estimates is mis-specified.

## Prioritized fix list

1. **P0 — Read the autoresearch driver. ≈30 min.** Confirm per-symbol vs. SPY-only joining. Selects between Section 1's "toxic" verdict and Section 3's "wasted disk" verdict. Gates everything below.
2. **P0 conditional — Quarantine recent regime-conditioned autoresearch results** if P0 returns "per-symbol joins." Tag every "HIGH-conditioned edge" on non-SPY symbols as unverified pending re-run on a corrected labeler. No promotions out of this state.
3. **P1 — Emit two columns** (`vol_regime_market`, `vol_regime_symbol`) from `scripts/label_regime_dataset.py`. Market-wide derived from VIX or SPY 30d realized vol; symbol-wide derived from per-symbol trailing-252d percentile bins. Re-emit all 69 parquets. Update consumer code to reference one explicitly — do not preserve the bare `vol_regime` name as an alias, that is the trap from Section 2's "option a."
4. **P1 — Build honest multi-asset OOS regime ground truth** before claiming labeler accuracy. Hand-label ≥4 non-SPY periods (one each: high-vol single name, cyclical sector ETF, defensive, commodity-linked) plus a held-out SPY year not in the original 8 tuning periods. Score Labeler A and Labeler B against this set. Only after this number exists is "converge to Labeler B" defensible.
5. **P2 — Add Risk axis only after a defensible proxy clears a stated bar.** Required: correctly labels Mar 2020, Jan 2022, Mar 2023 (SVB), Aug 2024 (yen carry) as RISK_OFF on t±2 and ≥80% of unambiguous calm periods as RISK_ON. Until then, runtime keeps emitting Risk axis but `RISK_ON`-gated strategies must run in shadow during autoresearch validation.
6. **P2 — Document the regime contract** as a single source-of-truth file: which axes exist, which estimator each uses, threshold derivation, hysteresis rules, and the train/live consistency invariant. Findings 4 and 5 are symptoms of an unwritten contract.
7. **P3 — CI labeler health check.** Fail the build if (a) `vol_regime_market` is *not* identical across symbols on a given date, (b) any single symbol's `vol_regime_symbol` distribution has fewer than three of {LOW, NORMAL, HIGH, SHOCK} on a 252-day rolling window, or (c) cross-universe median time in any one bin exceeds 70%. The third condition would have caught the current pathology in CI on day one.
