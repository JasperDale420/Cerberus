# Autoresearch Loop Debug — Executive Summary

**Session:** debug/260430-1728-autoresearch-loop
**Date:** 2026-04-30
**Scope:** Full autoresearch loop infrastructure
**Iterations:** 30 (bounded mode)
**Result:** 17 bugs confirmed, 3 disproven

## TL;DR

The autoresearch loop is producing iteration after iteration of `composite_score = -2.0` (the gate floor). I found **17 bugs**, including **2 CRITICAL** that explain the headline metric. The loop's headline goal — "beat 2x SPY" — is mathematically broken in two independent ways that compound.

## The two CRITICAL bugs that explain the live behavior

| # | Bug | Where | Effect |
|---|-----|-------|--------|
| C1 | `ratio_vs_spy = strat/spy` flips sign when SPY < 0 | `cerberus_autoresearch.py:401` | A strategy losing 20% in a -10% SPY window scores `ratio=2.0` (looks like 2x SPY); a strategy +5% in -10% SPY scores `-0.5` (fails gate) |
| C2 | Strategy uses additive return (`sum_pnl/$100k`); SPY uses compounded return | `cerberus_autoresearch.py:373` vs `:388-396` | 18 windows of identical 5% returns → strategy reports 90%, SPY reports 140.7%, ratio=0.64. Strategy systematically penalized vs SPY. |

**Combined impact:** The headline scoring metric is wrong in opposite directions in bull vs bear markets:
- In a bull WFO (SPY positive): strategy returns are diluted by additive accounting → ratio looks worse
- In a bear WFO (SPY negative): strategy losses look like SPY-beating wins → ratio looks better

The `ratio_vs_spy >= 2.0` headline goal is therefore not measuring what the playbook says it measures. **The 2026-04-25 honest-scoring fix (`2ef480a4`) is compromised.**

## The 3 HIGH bugs that explain why iterations don't progress

| # | Bug | Where | Effect |
|---|-----|-------|--------|
| H1 | `$EVAL_STRATEGY` unbound in protected-files violation TSV row | `autoresearch_driver.sh:220` | When agent edits a protected file, driver crashes with `set -u` exit instead of skipping iteration |
| H2 | Phase pivot doesn't reset BEST_SCORE/BEST_COMMIT; bootstrap branch dead code | `autoresearch_driver.sh:107,137-149,325-333` | After baseline sets BEST_SCORE=-2.0, bootstrap (requires `<= -999`) can never trigger. New-phase strategies are compared against old-phase scores. |
| H3 | Baseline runs `$STRATEGY` arg, iters target `REGIME_PHASES[CURRENT_PHASE]` — strategy mismatch | `autoresearch_driver.sh:10,89,125` | Live log shows `Strategy: regime_adaptive` baseline but iter 1+ target `regime_trend_up`. Two different strategies under one comparison. |

**Combined impact:** Even if scoring were correct, the loop's bookkeeping ensures the agent is comparing apples to oranges across phases. The "Best score: -2.0" the agent sees is meaningless across pivots.

## MEDIUM bugs (loop friction)

- **M1:** `--target-regime` falls open silently when no windows match → FLAT+NORMAL phase guaranteed to mis-score
- **M2:** No timeout on `claude -p` agent calls → hung agent stalls loop indefinitely
- **M3:** 4 legacy `autoresearch_score*.py` scripts still use deprecated weighted-average scoring; runnable footguns
- **M4:** Driver's auto-config-appender uses restrictive activation (missing premarket/close/shock vol)
- **M5:** Advertised "3-month holdout" anti-overfit guard is inert — date range is excluded from training but no validation runs on it
- **M6:** Near-zero SPY (|±1%|) makes FLAT specialists unscoreable (forces gate floor)
- **M7:** `tail -4 | head -3` HISTORY block drops the most recent iteration from agent visibility

## LOW bugs (polish)

- **L1:** PROTECTED_FILES doesn't cover v3 playbook + launcher
- **L2:** Tab characters in commit messages corrupt the TSV
- **L3:** LOC penalty counts docstrings (encourages stripping documentation)
- **L4:** 2,569 eval log files / 61 MB accumulated, no cleanup
- **L5:** Driver hardcodes `--n-trials 5`; harness default is 8 (inconsistent direct-vs-driver)

## Recommended fix order

1. **C1 + C2 first.** These are silent — the loop will keep "succeeding" by its own broken metric. Fix scoring before anything else.
2. **H1, H2, H3** — restore loop bookkeeping. Without these, phase pivots are incoherent.
3. **M1 + M5 + M6** — restore the advertised guards.
4. **M2 + M4 + M7** — operational hygiene.
5. **L1–L5** — polish, can be batched.

## Estimated fix complexity

| Severity | Total LOC ~ | Complexity |
|----------|-------------|------------|
| CRITICAL (×2) | ~30 LOC | Medium — requires unit tests for the new ratio math |
| HIGH (×3) | ~20 LOC | Easy — bash variable + control flow |
| MEDIUM (×7) | ~80 LOC | Easy-Medium — mostly small driver/harness tweaks; M5 is the most involved (actual holdout backtest impl) |
| LOW (×5) | ~30 LOC | Trivial |
| **Total** | **~160 LOC** | Most fixes are localized; no architectural rewrites needed |

## Next step

Auto-chaining to `/autoresearch:fix --from-debug` to land fixes against this findings set. CRITICAL + HIGH bugs are infrastructure-only (no live trading code). All fixes commit on `main` per autoresearch playbook (no branches per memory rule).
