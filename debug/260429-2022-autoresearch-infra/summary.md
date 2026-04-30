# Autoresearch Infrastructure Debug — Summary

**Session:** `debug/260429-2022-autoresearch-infra/`
**Symptom:** wasted research cycles, false-positive scores, silent corruption
**Scope:** `scripts/cerberus_autoresearch.py`, `scripts/autoresearch_driver.sh`, `scripts/extract_wfo_insights.py`, `program_cerberus.md`, `src/analytics/optuna_harness.py`, `src/backtest/runner.py`
**Hypotheses tested:** 9 (7 confirmed, 1 disproven, 1 partial)
**Iterations:** 13

## Findings — ranked by impact on wasted research cycles

| # | Severity | Title | Where | Effect |
|---|----------|-------|-------|--------|
| 1 | **CRITICAL** | `trades` list dropped in `BacktestReportCard.to_dict()` | `backtest_report.py:582`, `optuna_harness.py:808` | Regime-diversity multiplier always = 1.0 (anti-overfit guard inert). `extract_wfo_insights` always sees zero trade detail. Entire run produced 18,087 trades with 0 in `oos_metrics["trades"]`. |
| 2 | **HIGH** | SPY benchmark span includes gaps when `--target-regime` filter is active | `cerberus_autoresearch.py:384-390` | Specialists scored against 8.5-yr SPY span while only trading 6.5 yrs of it. ratio_vs_spy systematically biased — under-scores bull specialists, over-scores bear specialists. |
| 3 | **HIGH** | Bootstrap KEEP path enshrines gate-failed iterations | `autoresearch_driver.sh:302-308` | First strategy with >30 trades is kept even if `composite_score=-2.0` (gates failed). Lineage roots in a known-broken strategy. |
| 4 | **MEDIUM** | Stale `<strategy>_latest.json` survives crashed evals | `cerberus_autoresearch.py:446` | If eval crashes mid-WFO, next iter's `extract_wfo_insights` reads previous successful eval's JSON, feeding agent phantom data. |
| 5 | **MEDIUM** | `loc_penalty` added to `ratio_vs_spy` mixes units | `cerberus_autoresearch.py:414` | A barely-positive ratio (0.01) inflated to 0.51 by +0.5 LOC bonus. Composite no longer means "how many SPYs we beat". |
| 6 | **MEDIUM** | Driver doesn't verify agent edited the target strategy file | `autoresearch_driver.sh:217+` | Agent commits no-op or wrong-file edits → 30-75min eval against unchanged strategy. |
| 7 | **LOW** | Trial DBs leak from killed/crashed workers | `optuna_harness.py:825` | 436 files / 458MB accumulated since Apr 15. Disk leak, no startup sweeper. |

## Disproven hypotheses

- **Heredoc command injection via agent commit message** — bash heredoc expansion does not re-parse variable values for `$()` or backticks. `tr -d '\n\r'` strips newlines so multiline injection of an EOFRESULT delimiter is impossible.

## Cause-of-waste reasoning

The user's symptom — "wasted research cycles" — is dominated by bugs **#1**, **#3**, and **#4**:

- **#1 (trades dropped)** means the agent navigates iterations *blind*. The advertised "regime diversity scoring" is a no-op multiplier of 1.0. The agent's `extract_wfo_insights` output is the empty-trades fallback for every iteration. **Anti-overfit guards are advertised but absent.**

- **#3 (bootstrap keep)** means the very first iteration with any trades is enshrined as the "best" even if the harness flagged `composite_score=-2.0` (gates failed). All subsequent iterations build on a known-broken foundation. Combined with the simplicity rule's pressure to delete, the *useful* parts of the bootstrap strategy can be pruned away leaving only its broken core.

- **#4 (stale JSON)** turns a single crash into multiple wasted iterations: the next agent reads a JSON from a different commit, draws hypotheses from foreign trade data, and produces irrelevant edits.

Bugs **#2**, **#5**, **#6** add noise — the score doesn't mean what the agent (or user) thinks, and ~1 in N iterations runs eval on no-op commits.

## Recommended fix order

1. **Bug #1** — single-line fix in `BacktestReportCard.to_dict()`. Unlocks the entire trade-level analysis surface and reactivates the regime-diversity multiplier. **Highest leverage.**
2. **Bug #3** — gate the bootstrap on `composite > GATE_FLOOR`. 5-line change in driver.
3. **Bug #4** — `rm -f` the latest.json before each eval (driver-side, simplest); optionally stamp commit_sha in the JSON header (harness-side, robust).
4. **Bug #6** — add the `git diff --name-only` check before launching the eval. ~10 lines in driver.
5. **Bug #2** — switch from span-based SPY to compounded per-window SPY. ~10 lines in harness.
6. **Bug #5** — make `loc_penalty` multiplicative (or drop from composite, expose separately).
7. **Bug #7** — add a startup `find -mmin +60 -delete` sweeper.

Bugs #1, #3, #4, #6 together should eliminate the bulk of the wasted-cycle pattern. Bugs #2 and #5 are correctness-of-score, not directly wasted cycles, but mislead the user reading results.

## Files
- `findings.md` — full bug detail with reproducer + suggested fix per bug
- `eliminated.md` — the disproven heredoc-injection hypothesis
- `debug-results.tsv` — iteration log
