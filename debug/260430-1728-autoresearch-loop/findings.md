# Autoresearch Loop Debug — Findings

**Session:** debug/260430-1728-autoresearch-loop
**Scope:** scripts/cerberus_autoresearch.py, scripts/autoresearch_driver.sh, scripts/autoresearch_loop.sh, scripts/autoresearch_score*.py, scripts/extract_wfo_insights.py, src/analytics/optuna_harness.py (relevant), program_cerberus*.md
**Mode:** Find all bugs (full scan), 30 iterations, auto-fix after
**Result:** 18 confirmed bugs (2 CRITICAL, 3 HIGH, 8 MEDIUM, 5 LOW), 3 disproven

## Live-run symptoms (Phase 1)
- 5 consecutive iterations (iter 1–4) all scored exactly `-2.0` (gate floor); iter 5 ERROR after 180 min
- iter 6 PIVOT to `regime_bear`, also scored -2.0 (0 trades after regime filter)
- Eval times 94–180 min per iteration
- Driver header shows "Strategy: regime_adaptive" but iters target `regime_trend_up` — strategy-name mismatch
- All comparisons stuck at the `-2.0` floor; no iteration ever crosses the gate

---

## CRITICAL

### Bug #C1: `ratio_vs_spy = strat/spy` flips sign when SPY is negative
- **Location:** `scripts/cerberus_autoresearch.py:401`
- **Evidence:**
  - `ratio = strategy_return_pct / spy_return_pct`
  - Strategy −20% / SPY −10% → ratio = +2.0 (passes 2x SPY headline goal while LOSING money)
  - Strategy +5% / SPY −10% → ratio = −0.5 (fails gate even though strategy genuinely outperformed)
  - The gate at L415-416 fails on `ratio_vs_spy <= 0` — exactly inverted in bear markets
- **Impact:** Headline metric the user uses to evaluate "did we beat 2x SPY" is mathematically wrong. The honest-scoring infra introduced in `2ef480a4` is compromised.
- **Root cause:** Naive division gives the right answer only in same-sign regime. In bear markets, more negative numerator divided by less negative denominator yields a positive ratio that masquerades as outperformance.
- **Suggested fix:** Use growth-factor outperformance: `ratio_vs_spy = (1 + strat_return) / (1 + spy_return) - 1`, gated `> 0` (strategy compounds faster than SPY). Or compute alpha directly: `strategy_return_pct - spy_return_pct`, gated `> 0` and a separate "≥ 2x SPY" multiplier check on growth factors.

### Bug #C2: `strategy_return_pct` is additive but SPY is compounded — apples-to-oranges
- **Location:** `scripts/cerberus_autoresearch.py:373` (strategy) vs `:388-396` (SPY)
- **Evidence:**
  - Strategy: `strategy_return_pct = sum(net_pnl across windows) / 100_000 * 100` — treats $100k as fresh capital each window, then sums PnL.
  - SPY: `spy_total_return *= seg.iloc[-1].close / seg.iloc[0].close` per window — compounds growth factors.
  - Reproduction: 18 windows of identical 5% per-window returns yield strategy=90% (additive), SPY=140.7% (compounded), ratio=0.64 — strategy looks 36% **worse** despite identical returns.
- **Impact:** Systematically penalizes strategies vs SPY across many windows. Score drifts further negative the more windows the WFO has.
- **Root cause:** Two different mathematical models on the two sides of the ratio.
- **Suggested fix:** Compound the strategy too. For each scoring window, compute `window_growth = 1 + (window_pnl / starting_capital_for_window)`, then multiply across windows. Or: rebalance capital to a constant `starting_capital_at_each_window_start = previous_capital * window_growth` so strategy compounds match SPY's compounding.

---

## HIGH

### Bug #H1: `$EVAL_STRATEGY` unbound in protected-files violation branch
- **Location:** `scripts/autoresearch_driver.sh:220`
- **Evidence:** Reproduced in isolation: `set -uo pipefail` + `printf ... "$EVAL_STRATEGY"` (defined later at L272) → `EVAL_STRATEGY: unbound variable; exit=1`
- **Impact:** When the agent modifies a protected file (the very thing the violation check exists to detect), the driver crashes the whole session instead of skipping the iteration and continuing. The defense-in-depth layer is itself a bomb.
- **Suggested fix:** Use `"$STRAT_FILE"` (already in scope from L125) instead of `"$EVAL_STRATEGY"` in the protected-violation TSV row, OR move `EVAL_STRATEGY="$STRAT_FILE"` assignment up to L125-130 before any TSV writes.

### Bug #H2: Phase pivot doesn't reset BEST_SCORE/BEST_COMMIT; bootstrap branch is dead code
- **Location:** `scripts/autoresearch_driver.sh:107` (baseline sets BEST_SCORE), `:137-149` (pivot), `:325-333` (bootstrap requires `BEST_SCORE <= -999`)
- **Evidence:**
  - After baseline at L107, `BEST_SCORE` becomes the baseline composite (typically `-2.0`).
  - The bootstrap branch at L327 requires `BEST_SCORE <= -999` — never true post-baseline.
  - Phase pivot at L137-149 changes `STRAT_FILE` and target regime but leaves `BEST_SCORE` and `BEST_COMMIT` carrying state from the previous phase.
  - Net effect: a new-phase strategy that scores `-2.0` (gate floor) is compared to the old phase's `-2.0` and discarded; if `BEST_COMMIT` happens to point to old-phase code, discards reset to that.
- **Impact:** New regime specialists can't bootstrap; comparisons across phases are incoherent.
- **Suggested fix:** On pivot, reset `BEST_SCORE=-999.0` and `BEST_COMMIT=$BASELINE_COMMIT` (write both files). Remove the dead bootstrap branch or repurpose it to gate the post-pivot first iteration.

### Bug #H3: Baseline runs `$STRATEGY` arg, iterations use `REGIME_PHASES[CURRENT_PHASE]` — strategy mismatch
- **Location:** `scripts/autoresearch_driver.sh:10,89,125`
- **Evidence:** Live driver.log shows `Strategy: regime_adaptive` (user passed arg) but iter 1+ target `regime_trend_up` (`REGIME_PHASES[0]`). Baseline TSV row records `strategy=regime_adaptive composite=-2.0`; subsequent rows record `strategy=regime_trend_up`. The `BEST_SCORE` derived from regime_adaptive's gate-floor is then used to gate regime_trend_up.
- **Impact:** Two different strategies under one comparison. The header "Best:" shown to the agent is from a different strategy.
- **Suggested fix:** Either (a) make baseline run `${REGIME_PHASES[0]}` (same as iter 1), or (b) ignore the user's `$1` arg and derive baseline target from CURRENT_PHASE. Update header echo to show the actual baseline target.

---

## MEDIUM

### Bug #M1: `--target-regime` filter falls open silently when no windows match
- **Location:** `scripts/cerberus_autoresearch.py:346`
- **Evidence:** `if target_regime and regime_stats.get(target_regime):` — when no WFO window classifies as `target_regime`, the conditional is falsy, `scoring_windows` stays as all-windows, and the strategy is silently scored against the wrong window set.
- **Impact:** Phase 2 in the driver targets `FLAT+NORMAL` (advertised as 5% of data). If the dual-SMA classifier never tags a window as FLAT+NORMAL, the regime specialist is scored on all 18 windows instead — guaranteed gate failure for a niche specialist.
- **Suggested fix:** When `target_regime` is set but matches 0 windows, emit a warning line `REGIME_FILTER_NOMATCH target=<regime>` AND force gate failure with `gate_failures.append("regime_filter_no_match")` instead of silently broadening scope.

### Bug #M2: No timeout on `claude -p` agent calls
- **Location:** `scripts/autoresearch_driver.sh:197,236`
- **Evidence:** Eval calls wrap in `timeout 10800` (3h); agent calls have nothing. A hung agent blocks the loop indefinitely.
- **Suggested fix:** Wrap with `timeout 1800 claude -p ...` (30min budget). On timeout, count as a no-commit and let the existing no-commit branch increment discards.

### Bug #M3: Legacy `autoresearch_score*.py` scripts use deprecated weighted-average scoring
- **Location:** `scripts/autoresearch_score.py:24-48`, `autoresearch_score_cv.py:42-67`, `autoresearch_pairs_score.py:25-48`, `autoresearch_trp_sortino.py:38+`
- **Evidence:** Composite formula `0.30*pnl + 0.25*sharpe + 0.20*pf + 0.15*wr + 0.10*trade_score`. The 2026-04-25 honest-scoring fix (`2ef480a4`) replaced this with `ratio_vs_spy` + hard gates in `cerberus_autoresearch.py`, but these legacy scorers were left as live entry points.
- **Impact:** Confusing; future operator can run them and get the gameable old metric. Direct callers in `src/` are zero, but the files remain runnable.
- **Suggested fix:** Delete or rename to `*.py.deprecated`. If kept, replace the formula body with a stub that prints a deprecation notice and exits non-zero.

### Bug #M4: Driver auto-appender uses restrictive activation (missing premarket, close, shock)
- **Location:** `scripts/autoresearch_driver.sh:243`
- **Evidence:** Append template:
  - `session: [opening, midday, power_hour]` (missing `premarket`, `close`)
  - `vol: [low, normal, high]` (missing `shock`)
  - Existing strategies in `config/strategies.yaml` use the full set.
- **Impact:** New strategies created mid-loop are silently locked out of premarket/close/shock conditions. Contradicts the driver's own advice in the agent prompt: "Keep activation permissive."
- **Suggested fix:** Use the full set (`session: [premarket, opening, midday, power_hour, close]`, `vol: [low, normal, high, shock]`).

### Bug #M5: 3-month "holdout" reserved but never validated — advertised anti-overfit guard is inert
- **Location:** `src/analytics/optuna_harness.py:1147-1153,1470` (only returns dates), `program_cerberus.md:8` (claim)
- **Evidence:** `get_holdout_window` returns `{start, end}` strings; the harness output dict carries `holdout_window` but no holdout backtest is ever executed or scored. Searched for `run_holdout`, `validate_on_holdout`, `holdout_metrics` — no callers.
- **Impact:** `program_cerberus.md` advertises "3-month final holdout the agent never sees during iteration" as an anti-overfit guard. The data-exclusion happens (training stops at `holdout_start`), but no validation is run on the held-out range. The advertised guard does nothing.
- **Suggested fix:** Either (a) implement actual holdout validation — re-run the strategy on the held-out range using the per-window best params (or aggregate optimal) and emit `AUTORESEARCH_HOLDOUT` line, OR (b) update `program_cerberus.md` to remove the false claim.

### Bug #M6: Near-zero SPY (|spy_return_pct| ≤ 1%) makes FLAT specialists unscoreable
- **Location:** `scripts/cerberus_autoresearch.py:400-401`
- **Evidence:** `if abs(spy_return_pct) > 1.0: ratio_vs_spy = strategy_return_pct / spy_return_pct` — otherwise `ratio_vs_spy` stays NaN, gate fails with `benchmark_unavailable`, score forced to GATE_FLOOR.
- **Impact:** A FLAT-regime specialist (the very third phase target) is most exposed: across windows where SPY drifts ±1%, the strategy can be richly profitable but scored as `-2.0`.
- **Suggested fix:** When `|spy_return| <= 1%`, fall back to absolute strategy return as the score (e.g., `composite = strategy_return_pct / 100.0`), or use a different benchmark (cash, T-bill rate). Don't force gate floor for a benchmark range that's the specialist's home turf.

### Bug #M7: HISTORY `tail -4 | head -3` drops the most recent iteration from agent visibility
- **Location:** `scripts/autoresearch_driver.sh:165`
- **Evidence:** Reproduced: `tail -4 file | head -3` returns rows N-4..N-2, dropping row N-1 (the most recent prior iter).
- **Impact:** Agent on iter K sees iter K-4..K-2 in HISTORY block; iter K-1 is missing from inline view (LAST_RESULT block does include it separately, but the inline list is wrong). Agent could repeat iter K-1's failed approach because it's not in the "don't repeat" history.
- **Suggested fix:** `tail -3 "$TSV"`.

---

## LOW

### Bug #L1: PROTECTED_FILES missing `program_cerberus_v3.md` and v3 launcher
- **Location:** `scripts/autoresearch_driver.sh:63-68`
- **Evidence:** PROTECTED_FILES = {cerberus_autoresearch.py, extract_wfo_insights.py, autoresearch_driver.sh, program_cerberus.md}. v3 playbook + v3 launcher omitted.
- **Impact:** Harmless under v2 driver (which doesn't run v3 path). Vulnerable if v3 loop is ever invoked.
- **Suggested fix:** Add `program_cerberus_v3.md`, `scripts/autoresearch_loop.sh`, and the legacy scorers to PROTECTED_FILES (or delete the legacy scorers entirely per #M3).

### Bug #L2: Commit message sanitization strips `\n\r` but not tabs (TSV corruption risk)
- **Location:** `scripts/autoresearch_driver.sh:259`
- **Evidence:** `tr -d '\n\r'`. TSV is tab-delimited. A tab in the agent's commit message would push the description column open and corrupt subsequent `awk -F'\t'` parsing in HISTORY.
- **Suggested fix:** `tr -d '\n\r\t'` or `tr '\t\n\r' ' '`.

### Bug #L3: LOC penalty counts docstrings as code lines
- **Location:** `scripts/cerberus_autoresearch.py:271-273`
- **Evidence:** Counter = `sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))`. A 4-line docstring + 3 code lines reports 7 LOC (real code = 3).
- **Impact:** A well-documented strategy can lose its <50 LOC simplicity bonus or trip the >100 LOC penalty just from documentation. Encourages stripping docstrings.
- **Suggested fix:** Use `ast.parse(source)` and count statements, or strip docstrings with a regex pass before counting.

### Bug #L4: Eval logs accumulate without cleanup
- **Location:** `artifacts/autoresearch/logs/`
- **Evidence:** 2,569 files / 61 MB at debug session start; no sweeper. Driver has trial-DB sweeper at L33 (`find .agents/tmp/optuna_dbs/ -name 'trial_*.db*' -mmin +60 -delete`), same pattern would help.
- **Suggested fix:** Add to driver setup: `find artifacts/autoresearch/logs/ -name '*.log' -mmin +1440 -delete 2>/dev/null || true` (purge logs >24h old).

### Bug #L5: Driver hardcodes `--n-trials 5`; harness default is 8 — silent inconsistency
- **Location:** `scripts/autoresearch_driver.sh:89,281` vs `scripts/cerberus_autoresearch.py:170`
- **Suggested fix:** Either (a) make driver `N_TRIALS=${N_TRIALS:-5}` and pass `--n-trials "$N_TRIALS"` (configurable env var), or (b) align the harness default to 5.

---

## Disproven hypotheses

| # | Hypothesis | Why disproven |
|---|-----------|---------------|
| 11 | `git diff --name-only` misses newly added strategy files | Confirmed in isolated repo: `git diff` includes added files. |
| 18 | `compute_regime_diversity_multiplier` is still inert (per pre-fix CHANGELOG note) | The 2026-04-29 fix wired `_trade_to_dict` and `"trades": [...]` into `BacktestReportCard.to_dict()` (`src/backtest/backtest_report.py:662,665-678`). |
| 19 | `WFO_FULL_END=2026-03-19` is stale relative to data | SPY data ends exactly at 2026-03-19. Maintenance hazard but not a current bug. |

---

## Severity tally

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 3 |
| MEDIUM | 7 |
| LOW | 5 |
| **Total** | **17 confirmed** + 1 in-flight (#M5 also affects program_cerberus.md text) |

## Files affected
- `scripts/autoresearch_driver.sh` — 9 bugs (#H1, #H2, #H3, #M2, #M4, #M7, #L1, #L2, #L5)
- `scripts/cerberus_autoresearch.py` — 6 bugs (#C1, #C2, #M1, #M6, #L3, #L5)
- `scripts/autoresearch_score*.py` — 1 family (#M3, 4 files)
- `src/analytics/optuna_harness.py` — 1 bug (#M5)
- `program_cerberus.md` — 1 bug (#M5 — false claim)
- `artifacts/autoresearch/logs/` — 1 bug (#L4 — disk leak)
