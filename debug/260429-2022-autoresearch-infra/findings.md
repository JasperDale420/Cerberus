# Confirmed bugs — autoresearch infrastructure debug

**Scope:** scripts/cerberus_autoresearch.py, scripts/autoresearch_driver.sh, scripts/extract_wfo_insights.py, program_cerberus.md, src/analytics/optuna_harness.py (composite_objective + WFO core), src/backtest/runner.py (harness-callable parts)


## [CRITICAL] Bug #1: `trades` list silently dropped from per-window metrics — regime diversity scoring is inert

- **Location:** `src/backtest/backtest_report.py:582` (`BacktestReportCard.to_dict`) and `src/analytics/optuna_harness.py:808-809` (`run_backtest_for_optimization`)
- **Hypothesis:** `BacktestReportCard.to_dict()` does not include the `trades` list. `run_backtest_for_optimization` returns `dict(result.to_dict())`. The trades list never enters `oos_metrics`, so `extract_wfo_insights.py` and `compute_regime_diversity_multiplier(oos_trades)` always operate on `[]`.
- **Evidence (actual JSON output):**
  ```
  Total windows: 17
  Sum of n_trades:    18087
  Sum of len(trades): 0
  windows with "trades" key: 0/17
  ```
  Per-trade detail is dropped 100% of the time across all WFO windows.
- **Reproduction:** Run any autoresearch eval, then `jq '.oos_metrics[0] | keys' artifacts/autoresearch/<strategy>_latest.json` → no `trades` key.
- **Impact (compounding):**
  1. `extract_wfo_insights.py` always reports either `NO_TRADES` (pre-fix) or only the aggregate fallback (post-fix). Agent loses entry/exit/hold/regime trade-level signal — the most actionable iteration signal.
  2. `optuna_harness.py:1336-1339` calls `compute_regime_diversity_multiplier(oos_metrics.get("trades", []))`. With `[]` input the function returns `multiplier=1.0` (line 543-552 short-circuit). The "regime-diversity multiplier" anti-overfit guard advertised in `program_cerberus.md` is therefore **never applied**. Every iteration's `oos_score` passes through diversity multiplier 1.0 unchanged.
  3. Last run logged `REGIME_DIVERSITY n_regimes_profitable=0 concentration=0.0% penalty=1.0` for *every single window of every iteration* — a tell that this bug is in production.
- **Root cause:** `to_dict()` was designed for "human-readable summary" not for downstream consumers. The trade list is the most important piece of analytics data and was never exposed.
- **Fix:** Add `"trades": [t.to_dict() if hasattr(t, "to_dict") else t.__dict__ for t in self.trades]` to `BacktestReportCard.to_dict()`. Or — more surgical — change `run_backtest_for_optimization` to attach trades manually:
  ```python
  out = dict(result.to_dict())
  out["trades"] = [t.__dict__ for t in result.trades]
  return out
  ```
  Preferred: add it to `to_dict()` so all downstream consumers see consistent data.

## [HIGH] Bug #2: SPY benchmark span includes gaps when `target_regime` filter is active — systematically misleads specialist scoring

- **Location:** `scripts/cerberus_autoresearch.py:384-390` (SPY span computation)
- **Hypothesis:** When `target_regime` is set (e.g. specialists run with `--target-regime UP+NORMAL`), the harness restricts `scoring_windows` to matching windows but then computes the SPY benchmark over `min(window.test_start) .. max(window.test_end)` — covering all the *gap* windows the strategy did not trade. Strategy PnL is restricted; SPY return is not. Ratio is apples-to-oranges.
- **Evidence (real data, regime_trend_up against UP+NORMAL filter):**
  ```
  UP+NORMAL windows:                   13/17
  Span min..max:                       2017-06-02 .. 2025-12-02 (≈8.5 years)
  Gaps inside that span:
    182d gap   (2017-12 → 2018-06,    DOWN/FLAT skipped)
    365d gap   (2021-12 → 2022-12,    bear-2022 skipped)
    182d gap   (2024-12 → 2025-06,    skipped)
  Total gap time:                      ≈2.0 years inside the SPY span
  ```
  SPY's full-span return (including the 2-year gap of bear and recovery) is the denominator. Strategy total_pnl is restricted to the 6.5 trading years. Mathematically incompatible.
- **Reproduction:** Run any specialist eval (`--target-regime UP+NORMAL`); inspect `AUTORESEARCH_BENCHMARK span=...` — span will be wider than the actual scored period.
- **Impact:** `ratio_vs_spy` is biased low for any specialist whose target regime is bullish (gaps are typically bullish), and biased high for bearish specialists. The gate `ratio_vs_spy > 0` can trip on honestly profitable specialists, and a bearish specialist can pass the gate while underperforming SPY-during-its-active-windows. Both directions are wrong.
- **Root cause:** Span-based benchmark is the wrong shape for non-contiguous windows.
- **Fix:** Compound per-window SPY returns instead of using a span:
  ```python
  spy_total_return = 1.0
  for i in scoring_windows:
      seg = spy[(spy.timestamp >= window[i]["test_start"]) & (spy.timestamp <= window[i]["test_end"])]
      if len(seg) > 1:
          spy_total_return *= float(seg.iloc[-1]["close"]) / float(seg.iloc[0]["close"])
  spy_return_pct = (spy_total_return - 1) * 100
  ```
  Strategy side already sums per-window PnL, so this aligns the two. Update `benchmark_span` to report `n=<count>` of scoring windows rather than a misleading min..max.

## [MEDIUM] Bug #3: `loc_penalty` is added to `ratio_vs_spy` as if they share units — inflates barely-positive ratios

- **Location:** `scripts/cerberus_autoresearch.py:414` (`composite_score = float(ratio_vs_spy) + loc_penalty`)
- **Hypothesis:** `ratio_vs_spy` is a unitless ratio (e.g. 0.5 = strategy returned half of SPY; 2.0 = doubled SPY). `loc_penalty` is a fixed offset in absolute units (-2.0 to +0.5). Adding them confounds the "how many SPYs we beat" semantic.
- **Evidence (matrix run):**
  ```
  ratio=2.0  + loc=+0.5 (low LOC)              -> composite=2.50  ← +25% inflation (OK, near goal)
  ratio=0.01 + loc=+0.5 (low LOC, barely +)    -> composite=0.51  ← 51× inflation, misleading
  ratio=2.0  + loc=-1.0 (>120 LOC) cv=0.6      -> composite=0.70  ← 65% deflation, OK
  ratio=2.0  + loc=0.0 cv=2.0                  -> composite=1.00  ← 50% from CV floor
  ratio=inf  (SPY exactly 0%)                  -> composite=inf   ← ungated
  ```
- **Reproduction:** Run any eval where the strategy is barely positive and the file is under 50 LOC. The agent reads `composite_score=0.51` and concludes the strategy is at 51% of the 2x SPY goal, when it's actually only at 1%.
- **Impact:** The agent's iteration pressure is decoupled from the actual SPY goal. A short, low-quality strategy (1% above SPY) scores comparably to a longer, mid-quality strategy (50% above SPY).
- **Root cause:** Mixed-unit arithmetic. Loc penalty was carried over from the old composite formula (Sharpe/PF/Calmar weighted sum) where it was unit-compatible. With ratio_vs_spy as the score, the units no longer match.
- **Fix (preferred):** Apply `loc_penalty` multiplicatively, e.g. `composite = ratio_vs_spy * (1 + loc_penalty * 0.05)` — i.e. ±2.5% per LOC unit, capped at ±20%. Or drop loc_penalty from composite entirely and let it be a tiebreaker/separate signal that the agent can see in the result line but which doesn't move the optimization metric.
- **Sub-issue (LOW):** `if spy_return_pct != 0` does not guard against very small but nonzero SPY returns (e.g. 0.001%) which inflate `ratio_vs_spy` to absurdly large values. Tighten the guard:
  ```python
  if abs(spy_return_pct) > 1.0:  # require at least 1% SPY move to compute ratio honestly
      ratio_vs_spy = strategy_return_pct / spy_return_pct
  ```

## [HIGH] Bug #4: Bootstrap KEEP path enshrines gate-failed iterations as the "best"

- **Location:** `scripts/autoresearch_driver.sh:302-308` (bootstrap branch in keep/discard logic)
- **Hypothesis:** When `BEST_SCORE <= -999` and `TRADES > 30`, the iteration is kept *regardless of composite score* — even if the harness returned `composite_score=-2.0` (gate failure). This poisons `BEST_COMMIT` and the agent's "Score to beat" baseline.
- **Evidence:**
  ```bash
  if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] && [ "$BEST_SCORE" <= -999 ]; then
      KEEP=true
      KEEP_REASON="bootstrap: first $TRADES trades"
      BEST_SCORE="$SCORE"   # e.g. -2.0 (gate failure) becomes new "best"
  fi
  ```
  HAS_TRADES is `TRADES > 30`. The harness `MIN_TRADES_PER_WINDOW * total_windows` is 85+ for the 17-window WFO. A strategy with 31 trades passes the driver's HAS_TRADES check but fails the harness min-trades gate, returning composite=-2.0 — and the bootstrap path keeps it.
- **Reproduction:**
  1. Fresh driver run (BEST_SCORE=-999)
  2. Agent writes a strategy that fires 50 trades but loses money (ratio_vs_spy<0). Harness emits `composite_score=-2.0 score_explanation=gates_failed:ratio_vs_spy<=0`.
  3. Driver's bootstrap KEEPs it. BEST_SCORE→-2.0, BEST_COMMIT→that iter.
  4. Next iters now build on a known-failed strategy and only need to beat -2.0.
- **Impact:** The agent's "best score" lineage is rooted in a strategy the harness already rejected. Iterations build on a broken foundation, simplification rules may prune the only meaningful element, and the agent thinks `Score to beat: -2.0` represents real progress.
- **Root cause:** Bootstrap predates the gate logic. It assumed composite=-999 was the only "no signal" sentinel; doesn't recognize -2.0 (gate failure) as also "no signal".
- **Fix:** Require bootstrap to pass the gate floor:
  ```bash
  # Bootstrap: first strategy with trades escapes -999, but ONLY if the score itself isn't a gate failure
  GATE_FLOOR_LOCAL=-2.0
  if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] \
        && echo "$BEST_SCORE" | awk '{exit ($1 <= -999) ? 0 : 1}' \
        && echo "$SCORE $GATE_FLOOR_LOCAL" | awk '{exit ($1 > $2) ? 0 : 1}'; then
      KEEP=true
      KEEP_REASON="bootstrap: first $TRADES trades, score $SCORE > floor"
      ...
  fi
  ```
  Equivalent test in awk: only bootstrap if `composite_score > -2.0` (strictly above gate floor). Iterations that gate-fail at -2.0 should still be discards even when no prior best exists.

## [MEDIUM] Bug #5: Stale `<strategy>_latest.json` survives crashed evals — agent gets phantom trade analysis

- **Location:** `scripts/cerberus_autoresearch.py:443-447` (JSON write on success path only) + `scripts/extract_wfo_insights.py:20` (reads JSON without freshness check)
- **Hypothesis:** `<strategy>_latest.json` is written ONLY at the end of a successful eval. If the eval crashes (line 248-251 `sys.exit(1)`) or the driver kills it (timeout, SIGKILL), the previous successful eval's JSON remains. The driver still calls `extract_wfo_insights.py` on the next iteration; it reads the stale JSON and feeds outdated trade analysis to the agent's prompt.
- **Evidence:**
  ```
  artifacts/autoresearch/regime_trend_up_latest.json   263 KB  Apr 29 12:11  (lying-iter run)
  artifacts/autoresearch/regime_adaptive_latest.json    140 KB  Apr 28 19:28  (older still)
  ```
  After today's bogus run was reverted via `git pull --ff-only`, the JSON file is still on disk and would be read by the next `extract_wfo_insights` call — feeding the agent insights from a discarded-and-archived strategy lineage.
- **Reproduction:**
  1. Start an eval, kill it mid-WFO (e.g. SIGKILL during window 7/17).
  2. Driver records error and continues.
  3. Next iteration: agent's prompt includes `INSIGHTS=$(uv run python scripts/extract_wfo_insights.py ...)` which reads the previous successful eval's JSON.
  4. Agent forms hypotheses from outdated/foreign trade data.
- **Impact:** Wasted research cycles when the agent reads insights from a different strategy or different commit than what's currently being scored. Worsens after fixing bug #1 (trade list inclusion), since stale data becomes more harmful when it actually contains trade-level detail.
- **Root cause:** The harness write is unconditional-on-success but there's no "in-progress" or "stale" marker. `extract_wfo_insights` blindly reads whatever's at the path.
- **Fix:** Two layers:
  1. **Driver-side, simplest:** before each eval, `rm -f "artifacts/autoresearch/${EVAL_STRATEGY}_latest.json"`. A crashed eval leaves no JSON; `extract_wfo_insights` correctly returns `NO_INSIGHTS`.
  2. **Harness-side, more robust:** stamp the JSON with `meta.commit_sha` + `meta.eval_started` + `meta.eval_completed` keys. `extract_wfo_insights` checks `git rev-parse HEAD` matches `meta.commit_sha` before parsing — bail with `STALE_INSIGHTS` if not.

## [MEDIUM] Bug #6: Driver doesn't verify the agent's commit modified the target strategy file — wasted evals

- **Location:** `scripts/autoresearch_driver.sh:217+` (post-agent commit verification)
- **Hypothesis:** The driver checks file existence (line 160, 217), import success (line 218-224), and config entry presence (line 233). It does NOT verify that the agent's commit changed `src/strategies/${STRAT_FILE}.py`. If the agent commits a no-op edit, an unrelated file, or fails to actually modify the target strategy, the eval still runs.
- **Reproduction:**
  1. Agent reads the strategy file, decides "no change needed" but commits anyway with a description.
  2. Or agent edits a different file by mistake (e.g. it edited `regime_bear.py` while phase is on `regime_trend_up`).
  3. Driver does NOT detect this. `NEW_COMMIT != PREV_COMMIT` is true (a commit was made), so the no-commit short-circuit (line 242) doesn't fire.
  4. Eval runs ~60 minutes on exactly the same strategy code as the prior iteration. Score is identical or near-identical. Discarded.
- **Impact:** Wasted ~30-75 minute evals when the agent's commit didn't touch the target. Compounds when the agent gets stuck and makes repeated near-empty commits.
- **Root cause:** Driver assumes "commit happened" implies "target strategy changed", which doesn't follow.
- **Fix:** After the agent commits, verify the target strategy file is in the diff:
  ```bash
  if ! git diff "$PREV_COMMIT" HEAD --name-only | grep -q "^src/strategies/${STRAT_FILE}\.py$"; then
      echo "[iter $ITER] Agent did not modify ${STRAT_FILE}.py — skipping eval"
      git reset --hard "$BEST_COMMIT"
      printf "%d\t%s\t%s\t-2.0\tno_strategy_change\t0/0\t0\t0.0\t\tagent_no_change\n" "$ITER" "$NEW_COMMIT" "$STRAT_FILE" >> "$TSV"
      CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1)); ITER=$((ITER + 1)); continue
  fi
  ```
  This detects no-op iterations before sinking eval time into them.

## [LOW] Bug #7: Trial DBs leak when WFO workers crash or are killed

- **Location:** `src/analytics/optuna_harness.py:760-831` (trial DB lifecycle)
- **Hypothesis:** Each Optuna trial writes a sqlite trial DB at `.agents/tmp/optuna_dbs/trial_{pid}_{ms}.db`. The cleanup `for suffix in ("", "-journal", "-wal", "-shm"): os.remove(p)` runs in the worker's `finally` block. SIGKILL or process crash skips `finally`. Eval timeouts via `timeout(1)` at the driver level kill children that may not unwind cleanly.
- **Evidence (filesystem state):**
  ```
  trial DB count:  436 files
  total size:      458 MB
  oldest:          Apr 15 00:52 (2 weeks of accumulated debris)
  ```
- **Reproduction:** Run a driver iteration. SIGKILL it (or hit the 10800s timeout). Inspect `.agents/tmp/optuna_dbs/` — surviving DBs from the killed run.
- **Impact:** Disk space leak (~1 GB/month at current cadence). Not a correctness bug, but a "research cycle waste" if disk fills and writes start failing silently. Also slows down `os.listdir` in the optuna db dir over time.
- **Root cause:** Cleanup is per-trial inside the worker process. When the worker dies non-gracefully, no cleanup runs. There's no orphan-DB sweeper at driver startup.
- **Fix:** Add a startup sweeper to the driver (or the harness itself):
  ```bash
  # At driver init, after BASELINE_COMMIT setup:
  find .agents/tmp/optuna_dbs/ -name 'trial_*.db*' -mmin +60 -delete 2>/dev/null
  ```
  Files older than 60 minutes can't belong to any in-flight eval (eval timeout is 180m max but trial DBs are short-lived per trial). Sweep them at startup. Apply in commit alongside other driver hardening.
