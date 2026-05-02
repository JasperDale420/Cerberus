# P0 Audit Memo — Regime-Label Consumption Patterns in Cerberus Autoresearch

**Date:** 2026-05-01
**Audit cost:** ~30 min (as predicted)
**Verdict on the reason-loop's hardest question:** Both consumption patterns coexist. The "SPY-only" escape hatch does NOT save us — the per-symbol broken parquets ARE consumed by the backtest runner, which is the primary autoresearch evaluator.

---

## Q1 — Consumption pattern: per-symbol vs SPY-only?

**Answer: BOTH. Two parallel consumption paths.**

### Path A — Autoresearch driver: SPY-only window classification
[scripts/cerberus_autoresearch.py:64-141](scripts/cerberus_autoresearch.py:64) — `classify_window_regime()` reads ONLY `data/regime_labeled/SPY_daily_regime.parquet` (line 76) to assign each WFO OOS window a "dominant regime" tag (e.g., `UP+NORMAL`, `DOWN+HIGH`). Falls back to recomputing from SPY 1-min bars if the parquet is missing. Used to classify *windows*, not bars.

This path is benign — SPY's labels look reasonable (we already verified: 2024 Bull = 256/271 NORMAL, etc.), and using SPY's market regime to tag a WFO window is a defensible aggregate description.

### Path B — Backtest runner: per-symbol bar-level injection
[src/backtest/runner.py:346-383](src/backtest/runner.py:346) — `_load_regime_labels()` loops `for symbol in sorted(symbols)` and reads `<symbol>_daily_regime.parquet` for every symbol in the WFO universe.

Then [src/backtest/runner.py:1161-1188](src/backtest/runner.py:1161) — at each bar, it injects the day's labels into `symbol_state.meta["regime_labels"]` for that symbol:
```python
if regime_labels:
    _rl_df = regime_labels.get(row.symbol)  # per-symbol parquet
    if _rl_df is not None and ...:
        _sym_state.meta["regime_labels"] = {
            "regime_trend": ..., "regime_vol": ..., ...
        }
```

This is the toxic path. Strategies that read `symbol_state.meta["regime_labels"]` get the per-symbol Labeler A output, which is broken on 65 of 68 non-SPY names.

### Implication

The reason-loop's optimistic escape ("maybe autoresearch only uses SPY") is **falsified**. The per-symbol parquets ARE consumed at bar resolution by the backtest runner that autoresearch invokes via `WalkForwardOptimizer`. The SPY-only Path A is just for window-level reporting.

---

## Q2 — Filter mode vs feature mode?

**Answer: HARD FILTER MODE in the autoresearch-relevant strategy family.**

The `daily_research_v*` strategy family — which appears to be the lineage autoresearch is iterating on — uses `regime_vol` as a hard early-return filter:

| File | Filter expression | Effect |
|------|-------------------|--------|
| [daily_research_v9c.py:88](src/strategies/daily_research_v9c.py:88) | `if regime_vol in ("SHOCK", "HIGH"): return None` | Skips signal entirely |
| [daily_research_v9d.py:107](src/strategies/daily_research_v9d.py:107) | `if regime_vol in ("SHOCK", "HIGH"): return None` | Skips signal entirely |
| [daily_research_v10c.py:85](src/strategies/daily_research_v10c.py:85) | `if regime_vol in ("SHOCK", "HIGH"): return None` | Skips signal entirely |

**Direct consequence on the broken universe:** for the 65 non-SPY symbols where `regime_vol` is HIGH-or-SHOCK 94%+ of days, these strategies generate signals only on the residual ~6% of days. For the 8 worst-pinned symbols (TSLA/AMD/NIO/MARA/AMC/MRVL/RIOT/NET, all 98%+ HIGH/SHOCK), strategies fire on at most 1.4–1.9% of days. Whatever edge autoresearch reports for these strategies is computed on a tiny, label-selected residual that's effectively orthogonal to the strategy's intended logic.

In addition: [daily_research_v9d.py:152-168](src/strategies/daily_research_v9d.py:152) uses `regime_trend` to switch threshold parameters per UP/DOWN/FLAT — a softer feature-mode use, but conditioned on the same broken Labeler A axis.

**Pre-trade vs post-trade tagging:** [src/backtest/runner.py:560-575](src/backtest/runner.py:560) also stamps `entry_regime`/`exit_regime` onto every TradeRecord for post-trade analysis ([src/analytics/regime_stats.py](src/analytics/regime_stats.py) consumes these). So even strategies that *don't* filter on regime have their results sliced by Labeler A's broken vol axis in reports.

---

## Q3 — Activation blocks keying on regime axes?

**Answer: 20 strategies have `activation:` blocks; nearly all key on vol/risk. BUT — these activation blocks check Labeler B (live `MarketContextService`), not Labeler A.** Two regime systems are running side-by-side.

| Strategy | `vol:` allow-list | `risk:` allow-list | Concern |
|----------|------------------|---------------------|---------|
| `vwap_reversion` | `[normal, high]` | `[neutral, risk_off]` | Allows HIGH; risk axis active |
| `orb` | `[normal, high, shock]` | `[risk_on, neutral]` | Permissive |
| `index_mean_reversion` | `[normal, high]` | `[neutral, risk_off]` | SPY/QQQ only — likely OK |
| `flow_momentum` | `[normal, high, shock]` | `[risk_on, neutral]` | Permissive |
| `gap_fill` | `[normal, high]` | `[neutral]` | Allows HIGH |
| **`vwap_trend_rider`** | **`[low, normal]`** | **`[risk_on]`** | **Most restrictive — requires LOW/NORMAL vol AND RISK_ON. Both on Labeler B's axis.** |
| `vix_spike_fade` | `[high, shock]` | `[risk_off]` | SPY/QQQ only |
| `momentum_continuation` | `[normal, high, shock]` | `[risk_on]` | Permissive |
| `regime_trend_up` | `[low, normal, high]` | `[risk_on, neutral, risk_off]` | All risk states |
| `regime_bear` | `[low, normal, high, shock]` | `[risk_on, neutral, risk_off]` | No filter |
| `regime_adaptive` | `[low, normal, high, shock]` | `[risk_on, neutral, risk_off]` | No filter |
| `autoresearch_strategy` | (truncated; need full read) | — | — |

### Critical structural finding

There are **TWO regime systems running in parallel** with DIFFERENT axes and DIFFERENT estimators:

| System | Source | Axes | Estimator | Used by |
|--------|--------|------|-----------|---------|
| **Labeler A (training)** | `data/regime_labeled/*.parquet` | trend, vol (no risk, no liquidity, no session) | SMA + absolute thresholds | `daily_research_v*` strategies, post-trade `regime_stats` |
| **Labeler B (live)** | `MarketContextService` (`src/analysis/regime.py`) | trend, vol, liquidity, risk, session | Hurst + EWMA z | All `activation:` blocks, position sizers (`_apply_regime_volatility_multiplier`) |

The `daily_research_v*` strategies are special — they read **both** systems: activation gating from Labeler B, hard filters from Labeler A. That's the worst combination because:
- Their backtest behavior is governed by Labeler A's filter (broken on 65 of 68 names)
- Their live behavior is governed by Labeler B's activation (different boundary behavior, different axes)
- Train/live divergence is therefore **strategy-specific and unbounded** for this family.

---

## Q4 — SPY ground truth: independently labeled or threshold-derived?

**Answer: independently human-labeled (label values, not thresholds). The reason-loop's "tautological" suspicion is partly resolved — it's standard in-sample threshold tuning, not literal threshold-self-validation.**

[scripts/regime_grid_search.py:53-118](scripts/regime_grid_search.py:53) — `GROUND_TRUTH` is a Python list of 8 hand-curated periods (COVID Crash, COVID Recovery, 2021 Bull, 2022 H1 Bear, 2022 H2 Range, 2023 H1 Recovery, 2023 Q3 Correction, 2024 Bull) with hand-coded `trend`/`vol`/`risk` allow-lists, e.g.:
```python
{"label": "COVID Crash", "start": "2020-02-01", "end": "2020-03-31",
 "trend": ["DOWN"], "vol": ["HIGH", "SHOCK"], "risk": ["RISK_OFF"]}
```

The grid search [scripts/regime_grid_search.py:240-269](scripts/regime_grid_search.py:240) iterates over thresholds and scores the labeler's output against these allow-lists. So:

- **Label vocabulary** (LOW/NORMAL/HIGH/SHOCK, UP/DOWN/FLAT) is fixed and shared between truth and labeler.
- **Threshold values** that map realized vol → label name are tuned against the human labels.
- The 81.9% accuracy is therefore standard **in-sample threshold tuning**, not tautological self-validation.

**The reason-loop's deeper concern (R2's "the cited 'true' SPY distribution uses the same threshold scheme") was over-stated.** Reduce that danger from "doubly circular" to "naive in-sample bias."

The remaining concern: ground truth allow-lists are *permissive* (e.g., "COVID Crash vol = ['HIGH', 'SHOCK']" — anything in either bucket counts as correct), so 81.9% on 8 periods is a low-resolution score. Out-of-sample (a held-out year, non-SPY assets) remains required.

---

## Operational implications

### What the audit confirms
1. **Toxic path is active.** Per-symbol Labeler A labels ARE consumed by the backtest runner that autoresearch invokes. The SPY-only escape is falsified.
2. **Hard filters dominate the autoresearch lineage.** `daily_research_v9c/v9d/v10c` early-return on `regime_vol in ("SHOCK","HIGH")` — for high-vol single names this means the strategy fires on <2% of days, all label-selected.
3. **Train/live divergence is structural, not just numeric.** Labeler A and Labeler B emit different axes with different estimators. `daily_research_v*` strategies depend on both simultaneously.
4. **81.9% accuracy claim is in-sample but not tautological.** Less catastrophic than R2 suggested, but still SPY-only and in-sample.

### What changes in the reason-loop priorities
- **P0 quarantine fires.** Mark recent regime-conditioned non-SPY findings provisional in the results store. Specifically: any iteration of `daily_research_v9c/v9d/v10c/v10d` that reported "edge in <regime>" on names beyond SPY/JNJ/QQQ/COST is suspect.
- **The audit also surfaces a NEW class of issues the reason-loop didn't predict:** dual-system strategy family (`daily_research_v*` reads both labelers). This needs to be the highest-priority cleanup target — these strategies' train and live behavior are governed by different regime systems with different axes. Pick one.
- **Reduce the "tautological accuracy" danger** in the reason-loop's findings memo from "deeper than naive overfitting" to "standard in-sample bias on a low-resolution 8-period yardstick."
- **`vwap_trend_rider` is at high risk for a different reason** — it activates on Labeler B's RISK_ON + LOW/NORMAL vol, but for high-beta names in the universe, Labeler B's `vol_regime` (Hurst+EWMA z) likely also pins HIGH; we have NOT verified Labeler B's distribution per-symbol. Adding to P1 list.

### Updated priority list

| Priority | Action | Cost |
|----------|--------|------|
| **P0 (now)** | Mark `daily_research_v9c/v9d/v10c/v10d` regime-conditioned findings on non-SPY symbols as provisional in `artifacts/autoresearch/` | <10 min |
| **P0+ (this week)** | Pick a single regime system for the `daily_research_v*` family. Either (a) replace Labeler A meta injection with Labeler B output via `MarketContextService`, or (b) gate strategies entirely on `regime_labels` and remove activation block reliance. Currently they consult both. | 1-2 days |
| **P1** | Build OOS multi-asset ground truth (4-6 hand-labeled periods covering JNJ/TSLA/sector ETF/commodity ETF + a held-out SPY year). Score Labeler A AND Labeler B against same set. | 4-8 hrs |
| **P1** | Audit Labeler B's per-symbol distribution: does Hurst+EWMA z also pin TSLA-class names to HIGH? If yes, the dual-system fix doesn't help — both labelers share the same "individual-stock vol >> SPY vol" miscalibration. | 1-2 hrs |
| **P1** | MI vs permutation-null test on `regime_vol` against forward returns per symbol. Distinguish "rare but informative" from "constant thus useless." | 2-4 hrs |
| **P1** | Risk-axis ultimatum (build a real classifier hitting Mar-2020/Jan-2022/SVB-2023/yen-Aug-2024 RISK_OFF on t±2; ≥80% calm-window RISK_ON, OR remove the axis from activation policies and have `RISK_ON`-gated strategies treat it as constant). | 1 day |
| **P2** | Redesign Labeler A: market-state column (one shared state per date, calibrated to SPY/VIX) + per-symbol z-score column. Co-published, both named. Estimator chosen by P1 ground truth. | 2-3 days |
| **P2** | `REGIME_CONTRACT.md` documenting axes, estimators, threshold derivation, train/live invariant. The methodology drift is the symptom; absent contract is the disease. | 4 hrs |
| **P3** | Universe-aggregate CI checks: market-state column identical across symbols on any common date; <50% of universe in any single bin >80% of days; train/live agreement on shared dates in last 30 days | 4 hrs |

### What NOT to do
- Do not delete the per-symbol parquets. They're consumed and removing them silently fails the runner.
- Do not adopt per-symbol percentile thresholds without the two-column architecture — would silently re-interpret HIGH 68 different ways.
- Do not "fix" the 81.9% by adding more SPY periods — the OOS multi-asset gap is the load-bearing one.

---

## Files inspected
- [scripts/cerberus_autoresearch.py](scripts/cerberus_autoresearch.py) (autoresearch driver)
- [src/backtest/runner.py:340-420, 1140-1300](src/backtest/runner.py) (regime label loader + injector)
- [src/strategies/daily_research_v9c.py:77-95](src/strategies/daily_research_v9c.py:77), [v9d.py:99-115](src/strategies/daily_research_v9d.py:99), [v10c.py:74-90](src/strategies/daily_research_v10c.py:74), [v10d.py:137-175](src/strategies/daily_research_v10d.py:137) (filter usage)
- [config/strategies.yaml:1-200](config/strategies.yaml) (activation blocks)
- [scripts/regime_grid_search.py:53-118, 240-269](scripts/regime_grid_search.py:53) (ground-truth construction)
- [artifacts/regime_validation/optimized_params.json](artifacts/regime_validation/optimized_params.json)


---

# Addendum — Labeler B per-symbol distribution + live wiring check

**Date:** 2026-05-01 (same session)

## Finding 1 — Labeler B per-symbol IS well-calibrated

Applied [src/analysis/regime.py::_classify_vol](src/analysis/regime.py:497) (EWMA-z, short_span=10, long_span=120, thresholds 0.7/1.5/3.0) to each of 68 symbols' daily bars after a 120-day warmup. Per-symbol distribution table at [reason/260501-0159-regime-labels-critique/labeler_b_per_symbol.csv](reason/260501-0159-regime-labels-critique/labeler_b_per_symbol.csv).

| Bin | Labeler A mean across 68 syms | Labeler B mean across 68 syms |
|-----|-------------------------------|-------------------------------|
| LOW | 0.5% | **29.1%** |
| NORMAL | 16.2% | **66.4%** |
| HIGH | 79.4% | 4.5% |
| SHOCK | 3.9% | 0.0% |
| HIGH+SHOCK | 83.3% | **4.5%** |

**Per-symbol HIGH+SHOCK fraction range:** Labeler B 2.2% – 9.0% (uniform across 68 names). Labeler A 15.8% – 98.6% (bimodal: SPY/JNJ-cluster vs single-name-cluster).

31 of 68 symbols flip from "Labeler A says 95%+ HIGH/SHOCK" to "Labeler B says <10% HIGH/SHOCK with 22-60% LOW representation." TSLA: A=98.6% / B=4.3%. AMD: A=98.5% / B=3.5%. NIO: A=98.2% / B=3.0%. MARA: A=98.2% / B=3.8%.

**The reason-loop's hypothesis that Labeler B might share Labeler A's per-symbol miscalibration is FALSIFIED.** Labeler B's relative-EWMA-z construction is naturally per-symbol-friendly: it asks "is this symbol unusually volatile *for itself* recently?" rather than "is this symbol above 20% absolute realized vol?"

### Caveats

1. **Labeler B's SHOCK bin is empty per-symbol** (mean 0.0%, max 0.2%). z≥3.0 requires 9× variance ratio — extremely rare under EWMA dynamics on daily bars. Either tighten thresholds (z≥2.5 for SHOCK) or drop the bin entirely.
2. **VXX shows 60.6% LOW under Labeler B**, GME 59.1%. Both are correct in the relative sense — VXX has consistent vol-of-vol, GME has long stagnation between rallies — but downstream consumers should know LOW under Labeler B doesn't mean "calm market" the way LOW under Labeler A might be assumed to.
3. **SPY itself shifts** from 19.9% HIGH+SHOCK (Labeler A) to 5.3% HIGH+SHOCK (Labeler B). For SPY's market-state semantics, Labeler A's absolute thresholds may actually be the correct framing.

## Finding 2 — Per-symbol regime labels are silently dead in live mode

**Critical:** searching `src/` for writers of `symbol_state.meta["regime_labels"]` returns exactly two writers, BOTH in [src/backtest/runner.py](src/backtest/runner.py) (lines 575 and 1167). **There is no writer in the live engine path.**

Consequence for `daily_research_v9c/v9d/v10c/v10d` in live mode:

```python
regime_labels = symbol_state.meta.get("regime_labels", {})  # → {}
# ...
regime_vol = regime_labels.get("regime_vol", "NORMAL")  # → "NORMAL"
if regime_vol in ("SHOCK", "HIGH"):  # → False, never fires
    return None
regime_trend = regime_labels.get("regime_trend", "FLAT").upper()  # → "FLAT"
if regime_trend == "DOWN":  # → False, never fires
    ...
```

**The hard-filter and regime-conditional branches are dead code in live.** Strategies validated under "filter HIGH/SHOCK" run with **no filter** in production. The default fallback ("NORMAL", "FLAT") makes this silent — no exception, no log warning. A strategy backtested as "trades only the 1.4% of TSLA days that aren't HIGH/SHOCK" would in live trade **every** TSLA day. Risk profile inverts.

This is a different class of problem than the reason-loop predicted. Not a numeric drift between two estimators with different boundary behavior — a **binary code path divergence**: one branch in backtest, the other in live, switched silently by an empty dict.

### Severity

If `daily_research_v9c/v9d/v10c/v10d` (or any strategy in the autoresearch lineage that consumes `regime_labels`) has been promoted to live, then:
- Backtest Sharpe / win rate / drawdown numbers were computed on a label-selected subset of bars
- Live execution is on the full bar stream
- The two are not comparable

Verify live-promotion status of these strategies before any further work. If none of them have been promoted, the danger is contained to autoresearch reports being misleading. If even one has been promoted, this is a safety-critical issue and should be the immediate priority.

## Updated recommendations

### What changes from the previous priority list

- **P0 (highest)** — Audit live deployment status of `daily_research_v9c/v9d/v10c/v10d` and any strategy reading `symbol_state.meta["regime_labels"]`. If any are running in paper or live, gate them behind a kill switch until a live regime-label writer exists or the regime-filter code path is removed. **NEW; was not in earlier list.**
- **P0+ (this week)** — Pick a single regime mechanism for the autoresearch lineage:
  - Option A: Write a live `regime_labels` source in `engine/` that feeds per-symbol Labeler B (runs `_classify_vol` on a per-symbol EWMA state). Cheap — Labeler B's per-symbol distribution is well-calibrated, no further tuning needed.
  - Option B: Remove `symbol_state.meta["regime_labels"]` reads from strategies and have them depend only on `market_state` axes (Labeler B market-wide). Loses per-symbol relative-vol information.
  - Option C: Remove the regime filters entirely. Cleanest if the filters were never doing useful work — needs an MI test to decide.

  Pick A if backtest results justify the per-symbol filter. Pick C if the autoresearch wins on these strategies are mostly just symbol-class effects (likely, given the 98%-pin pathology under Labeler A).

- **P1** — Build OOS multi-asset ground truth (unchanged, still gates labeler choice).
- **P1** — MI vs permutation null tests (unchanged, but extra valuable now: settles whether the per-symbol filter ever did useful work).
- **P2** — Two-column redesign now has clearer architecture: `vol_regime_market` from Labeler A on SPY (absolute-threshold semantics, "is the market in a vol shock"), `vol_regime_symbol` from Labeler B per-symbol (relative-EWMA-z semantics, "is this symbol unusually volatile for itself"). Both are well-calibrated for their semantics; the issue is only that they were conflated into one column.
- **REMOVE from priority list:** "Audit Labeler B's per-symbol distribution" (P1) — done, well-calibrated. "Tautological accuracy danger" (R2) — over-stated, not literal threshold-self-validation.

### What is now LOW priority (downgraded)

- The "swap A for B" debate — moot. Both are needed, for different semantic columns. The reason-loop's "convergence on an unvalidated estimator" concern doesn't apply when the two estimators serve different purposes.

## Files inspected (addendum)

- [src/analysis/regime.py:31-140, 260-350, 487-518](src/analysis/regime.py:31) (MarketContextService — confirmed per-symbol-relative EWMA-z math, default `index_symbol="SPY"` for live)
- [src/engine/market.py:51, 166](src/engine/market.py:51) (live MarketContextService instantiation — fed market-wide bars, not per-symbol)
- Generated: [reason/260501-0159-regime-labels-critique/labeler_b_per_symbol.csv](reason/260501-0159-regime-labels-critique/labeler_b_per_symbol.csv) (68-symbol Labeler B distribution)
