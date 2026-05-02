# Critique: Cerberus Regime Labeling Pipeline

## 1. Are the labels usable as-is?

**No. Labeler A's per-symbol parquets are not usable for autoresearch validation, and any regime-conditioned wins computed against them must be re-evaluated.**

The decisive evidence is Finding 2 plus Finding 3. SPY's distribution looks defensible (Finding 1: 2024 Bull = 256/271 NORMAL; COVID Crash = 13 SHOCK + 11 HIGH + 7 NORMAL), but that defensibility evaporates the moment you cross-validate against the rest of the universe. With a **median of 94.3% HIGH/SHOCK across 68 non-SPY symbols** and aggregate distribution of 67.6% HIGH/SHOCK vs. SPY's true ~20% (Finding 3), the labels do not encode "market vol regime." They encode "this symbol is not SPY." TSLA at 98.6%, AMD at 98.5%, and a cluster around 98% for NIO/MARA/AMC/MRVL/RIOT/NET show this is not noise — it is a structural threshold mismatch: absolute-vol thresholds (LOW<8%, HIGH≥20%, SHOCK≥50%) calibrated on SPY's volatility distribution misclassify the entire universe of higher-vol single-name equities, where 30-day annualized realized vol routinely sits above 20% as the *baseline*.

Any autoresearch result of the form "strategy X earns Sharpe Y in HIGH regime" computed on a labeler that calls 94% of single-name days HIGH is not a regime-conditioned result — it is approximately the unconditional result with extra steps. The conditioning variable carries almost no information for non-SPY symbols.

**Conditions under which the labels could still be used:** SPY-only (or SPY/QQQ/JNJ/COST — symbols whose HIGH/SHOCK fraction is in a sane range, sub-40%) backtests where the absolute thresholds happen to match the asset's natural volatility scale. For everything else, the labels are toxic.

## 2. Highest-leverage fix

**(a) Per-symbol percentile-based vol thresholds, layered with (c) adding the Risk axis. Do not pursue (b) replacing Labeler A with Labeler B wholesale.**

Defense:

- **(a) is the root cause fix.** The pathology in Findings 2 and 3 is *exclusively* a calibration problem with absolute thresholds. Switching to per-symbol percentile thresholds (e.g., LOW = bottom quartile of a symbol's own trailing 252-day vol distribution, HIGH = top quartile, SHOCK = top 5%) directly inverts the failure mode. Each symbol then has a roughly balanced regime distribution by construction, which is what autoresearch actually needs as a conditioning variable. This fix is also cheap — it is a one-script edit to `scripts/label_regime_dataset.py`.
- **(c) is the second fix because Finding 5 reveals a silent contract violation.** The training labels lack the Risk axis, but live strategy activation policies key off `RISK_ON`. Autoresearch therefore validates strategies in a label space that does not even contain the dimension live execution gates on. The 5-day return proxy was rejected at 58% accuracy, but 58% of *what ground truth?* — given Finding 6 (8 SPY-only hand-picked periods, in-sample), that 58% is itself unreliable. A Risk-axis proxy is required even if imperfect; binary RISK_ON/RISK_OFF using e.g., SPY 5-day return + VIX level + breadth is a serviceable starting point.
- **(b) is wrong as the lead fix.** Replacing Labeler A with Labeler B's Hurst+EWMA z-score machinery to enforce train==live (Finding 4) sounds principled, but Labeler B itself has not been validated against ground truth at all. Adopting it wholesale propagates whatever miscalibration it carries into the training data. Train==live is necessary; train==live==broken is worse than the current state because at least now the discrepancy is detectable.

The correct sequencing: fix (a) and (c) on Labeler A, then evaluate Labeler B against the same SPY ground truth periods to see whether it is even competitive, then converge the two over time. Train==live is a Q3 goal, not a Q1 emergency response.

## 3. Hidden assumption that might invalidate the critique

**The strongest invalidator: autoresearch may consume only SPY labels (or a tiny SPY/QQQ subset), and the 67 broken per-symbol parquets are dead weight on disk.**

If autoresearch joins per-symbol bars only against SPY's regime column (treating regime as a *market-wide* state rather than a symbol-specific state), then Findings 2 and 3 are irrelevant — only the SPY labels enter the training set, and SPY labels look reasonable per Finding 1.

This must be verified before declaring an emergency. Two checks:

1. Open the autoresearch driver and grep for how it joins regime-labeled parquets to bar data — does it read each `<SYMBOL>_daily_regime.parquet` and join by symbol+date, or does it read only `SPY_daily_regime.parquet` and broadcast?
2. Check whether autoresearch results carry a per-symbol regime histogram. If they do, the broken labels are entering training. If they do not, only SPY's labels are.

A weaker invalidator: autoresearch might be using regime labels purely as a *feature* (input to a model that learns its own conditioning) rather than as a *filter* (subset bars by regime before computing edge). Feature use is more robust to miscalibration — a tree-based model can learn that "HIGH regime for TSLA" is a near-constant and weight it accordingly. Filter use is catastrophic.

## 4. Concrete dangers

1. **False confidence in regime-conditioned wins.** Any reported result of the form "strategy X has positive edge in regime Y" where Y is HIGH or SHOCK across a non-SPY universe is, given Finding 2, computed on essentially the unconditional sample. The "edge in HIGH" is just edge, period. If the strategy has been promoted to live based on this, it is being deployed under a regime hypothesis that the data never actually tested.
2. **Train/live regime divergence (Finding 4).** Strategy promoted in Labeler A's regime-space hits production governed by Labeler B. A strategy validated as "edge in HIGH vol" by Labeler A may rarely or never see HIGH vol from Labeler B (Hurst+EWMA z-score has different distribution). Live behavior is then untested.
3. **Risk-axis blind spot (Finding 5).** Strategies with `RISK_ON`-only activation have *zero* validation data — their training labels never contained the gate variable. Live performance of such strategies is pure forward-testing without prior. This is the highest-severity hidden danger because the failure mode is silent.
4. **In-sample tuning bias (Finding 6).** The 81.9% combined accuracy is meaningless as a quality metric — it is the score of a model evaluated on its training set. Any belief that "the labeler is 82% accurate" influencing downstream confidence intervals is mis-specified.
5. **Universe survivorship/composition skew.** Because the labels effectively partition by *symbol class* (mega-cap index ETF vs. high-vol single name) rather than by regime, autoresearch may be inadvertently learning symbol selection disguised as regime conditioning.

## Prioritized fix list

1. **P0 — Verify which labels autoresearch actually consumes** (1 hour). If only SPY, downgrade everything else to P2. If per-symbol, proceed.
2. **P0 — Quarantine and re-evaluate all regime-conditioned autoresearch results from the past N iterations.** Tag any "HIGH-conditioned edge" claim on non-SPY symbols as unverified pending re-run.
3. **P1 — Implement per-symbol percentile-based vol thresholds in `scripts/label_regime_dataset.py`** (LOW = ≤25th percentile of trailing 252-day vol, HIGH = ≥75th, SHOCK = ≥95th, computed per symbol). Re-emit all 69 parquets. Verify median HIGH/SHOCK fraction lands in 25–35% range across the universe.
4. **P1 — Add a Risk axis to Labeler A.** Use a defensible binary proxy (5-day SPY return sign + VIX above/below trailing median) rather than the rejected single-feature 58%-accuracy version. Validate against any discrete risk-off events 2020–2026 (Mar 2020, Jan 2022, Mar 2023 SVB, Aug 2024 yen carry).
5. **P1 — Build a real held-out OOS ground-truth set.** Add at least 4 non-SPY hand-labeled periods (1 high-vol single-name, 1 cyclical sector ETF, 1 defensive, 1 commodity-linked). Compute accuracy OOS, not on the tuning periods.
6. **P2 — Validate Labeler B against the same expanded ground truth.** If Labeler B underperforms Labeler A, fix Labeler B before considering convergence. If it outperforms, plan a controlled migration with a shadow period.
7. **P2 — Document the regime contract** in a single source-of-truth file: which axes exist, what proxy each uses, what the threshold derivation is, and how train and live are kept in sync. Findings 4 and 5 are symptoms of an unwritten contract.
8. **P3 — Add a regression test** that fails CI if any symbol's HIGH/SHOCK fraction in the labeled parquet exceeds 50% over a multi-year sample. The current breakage would have been caught by such a test on day one.
