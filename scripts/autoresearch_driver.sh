#!/usr/bin/env bash
# Cerberus Autoresearch Driver v2
#
# Karpathy-inspired: edit → eval → keep/discard → repeat
# Agent sees compact summary + trade insights. Never sees WFO output.

set -uo pipefail
cd "$(dirname "$0")/.."

# H3 fix: $1 was ambiguous — formerly used as the baseline strategy but ignored
# during iterations (which always used REGIME_PHASES[CURRENT_PHASE]). Now $1
# selects the *starting phase* by strategy name; baseline runs against that
# same phase. Defaults to the first phase if $1 is unrecognised.
START_PHASE_ARG="${1:-}"
MAX_ITER="${2:-50}"
# L5 fix: was hardcoded --n-trials 5; harness default is now 15. n_trials=5 produced
# noisy WFO param surfaces (1 trial per ~3.4 windows) — CV-based stability rejection
# became a coin flip. 15 is the new minimum that keeps each iteration trustworthy.
N_TRIALS="${N_TRIALS:-15}"
TSV="autoresearch/results.tsv"
BEST_SCORE_FILE="autoresearch/.best_score"
LAST_RESULT_FILE="autoresearch/.last_result"
PREV_RESULT_FILE="autoresearch/.prev_result"
CONSECUTIVE_DISCARDS=0
MAX_CONSECUTIVE_DISCARDS=5

# Regime phases — advance when stuck
REGIME_PHASES=("regime_trend_up" "regime_bear" "regime_flat" "regime_adaptive")
REGIME_DESCS=("UP+NORMAL (43% of data)" "DOWN+HIGH (18%)" "FLAT+NORMAL (5%)" "Cross-regime generalist")
REGIME_TARGETS=("UP+NORMAL" "DOWN+HIGH" "FLAT+NORMAL" "")
REGIME_HINTS=("Trend-following: buy pullbacks in uptrends. BUY-only." "Bear specialist: short breakdowns or fade oversold bounces." "Mean reversion: RSI bounce, BB fade. Both directions." "Adaptive: check regime labels, switch behavior.")
CURRENT_PHASE=0
PHASE_ITER=0

# Resolve $1 to a phase index if it matches a phase strategy name.
if [ -n "$START_PHASE_ARG" ]; then
    for _i in "${!REGIME_PHASES[@]}"; do
        if [ "${REGIME_PHASES[$_i]}" = "$START_PHASE_ARG" ]; then
            CURRENT_PHASE="$_i"
            break
        fi
    done
fi
# STRATEGY is now always the current phase's strategy file (was: $1 with default rsi_bounce).
STRATEGY="${REGIME_PHASES[$CURRENT_PHASE]}"

# ── Setup ──────────────────────────────────────────────────────────
mkdir -p autoresearch artifacts/autoresearch/logs

# Sweep stale trial DBs from prior crashed/killed workers.
# Files older than 60 minutes can't belong to any in-flight eval (eval timeout = 180m max
# but trial DBs are short-lived per-trial). Reclaims disk and keeps the dir performant.
find .agents/tmp/optuna_dbs/ -name 'trial_*.db*' -mmin +60 -delete 2>/dev/null || true

# L4 fix: sweep stale eval logs >24h old. Without this, artifacts/autoresearch/logs/
# grows linearly with iterations (was 2569 files / 61MB at debug session start).
find artifacts/autoresearch/logs/ -name '*.log' -mmin +1440 -delete 2>/dev/null || true

if [ ! -f "$TSV" ]; then
    printf "iteration\tcommit\tstrategy\tcomposite_score\tstatus\twindows_profitable\ttotal_trades\tavg_sortino\tregime_breakdown\tdescription\n" > "$TSV"
fi

if [ ! -f "$BEST_SCORE_FILE" ]; then
    echo "-999.0" > "$BEST_SCORE_FILE"
fi

BEST_SCORE=$(cat "$BEST_SCORE_FILE")
ITER=$(tail -n +2 "$TSV" | wc -l | tr -d ' ')

# ── Baseline commit + best commit pointers ────────────────────────
# CRITICAL: discards reset to BEST_COMMIT (last known-good), not HEAD~1.
# This prevents the driver from walking back through infrastructure commits
# (harness fixes, baseline reverts, etc.) when the agent makes multiple
# commits per iteration or when discards accumulate.
BASELINE_COMMIT_FILE="autoresearch/.baseline_commit"
BEST_COMMIT_FILE="autoresearch/.best_commit"
if [ ! -f "$BASELINE_COMMIT_FILE" ]; then
    git rev-parse HEAD > "$BASELINE_COMMIT_FILE"
fi
BASELINE_COMMIT=$(cat "$BASELINE_COMMIT_FILE")
if [ ! -f "$BEST_COMMIT_FILE" ]; then
    echo "$BASELINE_COMMIT" > "$BEST_COMMIT_FILE"
fi
BEST_COMMIT=$(cat "$BEST_COMMIT_FILE")

# Files the agent must never modify; restored to BASELINE before every iteration.
# L1 fix: include v3 playbook + v3 launcher (harmless under v2 driver, but
# protects them if the v3 path ever runs in the same workspace).
PROTECTED_FILES=(
    "scripts/cerberus_autoresearch.py"
    "scripts/extract_wfo_insights.py"
    "scripts/autoresearch_driver.sh"
    "scripts/autoresearch_loop.sh"
    "scripts/run_holdout.py"
    "program_cerberus.md"
    "program_cerberus_v3.md"
)

restore_protected() {
    for f in "${PROTECTED_FILES[@]}"; do
        git checkout "$BASELINE_COMMIT" -- "$f" 2>/dev/null || true
    done
}

echo "============================================================"
echo "  Cerberus Autoresearch v2"
echo "  Strategy: $STRATEGY | Best: $BEST_SCORE | Iter: $ITER"
echo "  Baseline: ${BASELINE_COMMIT:0:8} | Best commit: ${BEST_COMMIT:0:8}"
echo "============================================================"

# ── Baseline ──────────────────────────────────────────────────────
if [ "$ITER" -eq 0 ]; then
    echo "[iter 0] Running baseline..."
    COMMIT=$(git rev-parse --short HEAD)
    # Remove any stale latest.json from a prior crashed eval — prevents
    # extract_wfo_insights from feeding phantom data into the next iteration.
    rm -f "artifacts/autoresearch/${STRATEGY}_latest.json"
    EVAL_OUTPUT=$(timeout 10800 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" --n-trials "$N_TRIALS" 2>&1 || true)
    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    BENCHMARK_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_BENCHMARK" || echo "")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter 0] ERROR: No result line"; echo "$EVAL_OUTPUT" | tail -20; exit 1
    fi

    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    WIN_PROF=$(echo "$RESULT_LINE" | grep -o 'windows_profitable=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    SORTINO=$(echo "$RESULT_LINE" | grep -o 'avg_sortino=[^ ]*' | cut -d= -f2)
    REGIMES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" | tr '\n' '|' | sed 's/|$//')
    AGGREGATES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_AGGREGATE" || echo "")

    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        0 "$COMMIT" "$STRATEGY" "$SCORE" "baseline" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "baseline" >> "$TSV"
    echo "$SCORE" > "$BEST_SCORE_FILE"
    BEST_SCORE="$SCORE"

    cat > "$LAST_RESULT_FILE" <<EOF
baseline: $STRATEGY score=$SCORE trades=$TRADES windows=$WIN_PROF sortino=$SORTINO

SPY benchmark (goal: strategy_return ≥ 2x SPY over the OOS span):
  ${BENCHMARK_LINE:-(missing)}

Regimes: $(echo "$REGIMES" | tr '|' '\n' | head -8)
Aggregates: $(echo "$AGGREGATES" | head -5)
EOF

    echo "[iter 0] Baseline: score=$SCORE trades=$TRADES"
    ITER=1
fi

# ── Main Loop ─────────────────────────────────────────────────────
while [ "$ITER" -le "$MAX_ITER" ]; do
    STRAT_FILE="${REGIME_PHASES[$CURRENT_PHASE]}"
    REGIME_DESC="${REGIME_DESCS[$CURRENT_PHASE]}"
    TARGET_REGIME="${REGIME_TARGETS[$CURRENT_PHASE]}"
    REGIME_HINT="${REGIME_HINTS[$CURRENT_PHASE]}"

    echo ""
    echo "============================================================"
    echo "  Iter $ITER/$MAX_ITER | Phase: $REGIME_DESC | Best: $BEST_SCORE | Discards: $CONSECUTIVE_DISCARDS"
    echo "============================================================"

    # ── Phase advancement: switch regime when stuck ────────────────
    MAX_PHASE_IDX=$(( ${#REGIME_PHASES[@]} - 1 ))
    if [ "$CONSECUTIVE_DISCARDS" -ge "$MAX_CONSECUTIVE_DISCARDS" ] && [ "$CURRENT_PHASE" -lt "$MAX_PHASE_IDX" ]; then
        CURRENT_PHASE=$((CURRENT_PHASE + 1))
        STRAT_FILE="${REGIME_PHASES[$CURRENT_PHASE]}"
        REGIME_DESC="${REGIME_DESCS[$CURRENT_PHASE]}"
        TARGET_REGIME="${REGIME_TARGETS[$CURRENT_PHASE]}"
        REGIME_HINT="${REGIME_HINTS[$CURRENT_PHASE]}"
        CONSECUTIVE_DISCARDS=0
        PHASE_ITER=0
        # H2 fix: Reset BEST_SCORE and BEST_COMMIT on pivot. Without this, the
        # new-phase strategy is compared against the previous phase's score
        # (incoherent across-strategy comparison) and discards reset to the
        # previous phase's BEST_COMMIT (wrong code base).
        BEST_SCORE="-999.0"
        echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
        BEST_COMMIT="$BASELINE_COMMIT"
        echo "$BEST_COMMIT" > "$BEST_COMMIT_FILE"
        echo "[iter $ITER] PIVOT → Phase $CURRENT_PHASE: $REGIME_DESC (BEST_SCORE=-999, BEST_COMMIT=baseline)"
    elif [ "$CONSECUTIVE_DISCARDS" -ge "$MAX_CONSECUTIVE_DISCARDS" ] && [ "$CURRENT_PHASE" -ge "$MAX_PHASE_IDX" ]; then
        echo "[iter $ITER] Stuck on final phase ($REGIME_DESC) — no more phases to try"
        # Don't reset counter — let it accumulate so the agent sees the full discard streak
    fi

    # ── Step 1: Spawn agent ───────────────────────────────────────
    # Defense in depth: restore harness/playbook to BASELINE before each iteration.
    # If a previous agent or interrupted process left those files modified, the
    # eval will run against the baseline harness, not a contaminated one.
    restore_protected
    if ! git diff-index --quiet HEAD -- "${PROTECTED_FILES[@]}" 2>/dev/null; then
        # Stage and commit the baseline restore so HEAD matches working tree
        git add "${PROTECTED_FILES[@]}" 2>/dev/null && git commit -m "driver: restore protected files to baseline" --no-verify 2>/dev/null || true
        BEST_COMMIT=$(git rev-parse HEAD)
        echo "$BEST_COMMIT" > "$BEST_COMMIT_FILE"
    fi

    echo "[iter $ITER] Spawning agent..."
    LAST_RESULT=$(cat "$LAST_RESULT_FILE" 2>/dev/null || echo "(no previous result)")
    # M7 fix: was `tail -4 | head -3` which dropped the most recent iteration
    # (rows N-4..N-2 instead of N-3..N-1). Agent's HISTORY block missed iter N-1,
    # making it possible to repeat iter N-1's failed approach.
    HISTORY=$(tail -3 "$TSV" | awk -F'\t' '{printf "  iter%s: score=%s status=%s trades=%s — %s\n", $1, $4, $5, $7, substr($10,1,60)}' 2>/dev/null || echo "  (none)")

    TASK_INSTRUCTION=""
    if [ ! -f "src/strategies/${STRAT_FILE}.py" ]; then
        TASK_INSTRUCTION="CREATE src/strategies/${STRAT_FILE}.py for ${REGIME_DESC}. Hint: ${REGIME_HINT}
Read program_cerberus.md and src/strategies/base.py for the interface.
Add config to config/strategies.yaml with PERMISSIVE activation (all sessions/trends/vols).
Keep it SIMPLE — under 80 lines. 3 factors max."
    else
        TASK_INSTRUCTION="ITERATE on src/strategies/${STRAT_FILE}.py to beat score ${BEST_SCORE}.
Read the code, then make ONE change. Prefer removing code over adding.
$(if [ "$CONSECUTIVE_DISCARDS" -ge 3 ]; then echo "WARNING: ${CONSECUTIVE_DISCARDS} discards. Try something RADICALLY different. Re-read the trade analysis."; fi)"
    fi

    AGENT_PROMPT="Score to beat: ${BEST_SCORE} | Strategy: ${STRAT_FILE} | Regime: ${REGIME_DESC}

## Last Result
${LAST_RESULT}

## History (don't repeat failed approaches)
${HISTORY}

## Task
${TASK_INSTRUCTION}

## Simplicity Rule
A 0.1 improvement adding 20 lines? Not worth it. Equal score from deleting code? Keep.
Under 50 LOC gets a score bonus. Over 100 LOC gets penalized.

Commit as: experiment(${STRAT_FILE}): iter${ITER} — <description>
Then STOP."

    # M2 fix: bound the agent call. Without timeout, a hung Claude session
    # (network stall, infinite tool loop) would block the loop indefinitely.
    # 1800s (30min) is generous for a single edit-and-commit task.
    AGENT_RESULT=$(timeout 1800 claude -p "$AGENT_PROMPT" --model sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode bypassPermissions 2>&1 || true)

    NEW_COMMIT=$(git rev-parse --short HEAD)
    PREV_COMMIT=$(tail -1 "$TSV" | cut -f2)

    # ── Protected-file violation check ────────────────────────────
    # If the agent modified harness, playbook, or driver in any way (committed
    # or uncommitted), reset to BEST_COMMIT and skip this iteration. The score
    # would otherwise be measured against a contaminated harness.
    VIOLATIONS=""
    for pf in "${PROTECTED_FILES[@]}"; do
        if ! git diff "$BASELINE_COMMIT" HEAD -- "$pf" | head -1 | grep -q "."; then
            : # no diff vs baseline — clean
        else
            VIOLATIONS="$VIOLATIONS $pf"
        fi
        if ! git diff-index --quiet HEAD -- "$pf" 2>/dev/null; then
            VIOLATIONS="$VIOLATIONS ${pf}(uncommitted)"
        fi
    done
    if [ -n "$VIOLATIONS" ]; then
        echo "[iter $ITER] AGENT VIOLATED PROTECTED FILES:$VIOLATIONS — resetting to ${BEST_COMMIT:0:8}"
        git reset --hard "$BEST_COMMIT"
        # Use STRAT_FILE (in scope from L125) — EVAL_STRATEGY isn't assigned until Step 2 (L272).
        # Without this fix, set -u crashes the entire driver session here. See debug 260430-1728 H1.
        printf "%d\t%s\t%s\t-2.0\tprotected_violation\t0/0\t0\t0.0\t\tagent_modified:%s\n" "$ITER" "$NEW_COMMIT" "$STRAT_FILE" "$VIOLATIONS" >> "$TSV"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1)); ITER=$((ITER + 1)); continue
    fi

    # ── Import/config verification for new strategies ─────────────
    if [ "$NEW_COMMIT" != "$PREV_COMMIT" ] && [ -f "src/strategies/${STRAT_FILE}.py" ]; then
        IMPORT_CHECK=$(uv run python -c "
import sys; sys.path.insert(0, '.')
from src.backtest.runner import _dynamic_import_strategy_class
cls = _dynamic_import_strategy_class('${STRAT_FILE}')
if cls is None: print('IMPORT_FAIL'); sys.exit(1)
print(f'IMPORT_OK: {cls.__name__}')
" 2>&1 || echo "IMPORT_FAIL")

        if echo "$IMPORT_CHECK" | grep -q "IMPORT_FAIL"; then
            echo "[iter $ITER] Import FAILED — spawning fix agent"
            # M2 fix: bound the import-fix agent. 600s (10min) is enough for a
            # single targeted fix; longer hangs indicate a deeper issue.
            timeout 600 claude -p "Fix import error in src/strategies/${STRAT_FILE}.py: $IMPORT_CHECK. Read file, fix, ruff check, commit." \
                --model sonnet --allowedTools "Read,Edit,Write,Bash" --permission-mode bypassPermissions 2>&1 || true
            NEW_COMMIT=$(git rev-parse --short HEAD)
        fi

        if ! grep -q "^  ${STRAT_FILE}:" config/strategies.yaml 2>/dev/null; then
            echo "[iter $ITER] Adding config entry for ${STRAT_FILE}"
            # M4 fix: use the FULL activation set (premarket+close session, shock vol)
            # to match existing strategies. Previous restrictive set silently locked
            # new strategies out of premarket/close/shock conditions despite the
            # agent prompt's own "keep activation permissive" instruction.
            printf "\n  %s:\n    enabled: true\n    activation:\n      session: [premarket, opening, midday, power_hour, close]\n      trend: [up, down, flat]\n      vol: [low, normal, high, shock]\n      liquidity: [good, thin]\n      risk: [risk_on, neutral, risk_off]\n      min_confidence: 0.0\n" "$STRAT_FILE" >> config/strategies.yaml
            git add config/strategies.yaml && git commit -m "fix: add ${STRAT_FILE} config" --no-verify 2>/dev/null || true
            NEW_COMMIT=$(git rev-parse --short HEAD)
        fi
    fi

    # ── Quota/no-commit handling ──────────────────────────────────
    if [ "$NEW_COMMIT" = "$PREV_COMMIT" ]; then
        if echo "$AGENT_RESULT" | grep -qi "rate.limit\|quota\|overloaded\|429\|capacity"; then
            echo "[iter $ITER] Quota hit — sleeping 5 min..."; sleep 300; continue
        fi
        echo "[iter $ITER] No commit — skipping eval"
        ITER=$((ITER + 1)); CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1)); continue
    fi

    # Sanitize: strip newlines AND tabs (L2 fix). TSV is tab-delimited, so a tab
    # in the agent's commit message would corrupt the description column and
    # break HISTORY parsing in subsequent iterations.
    COMMIT_MSG=$(git log -1 --format='%s' | tr -d '\n\r\t')
    echo "[iter $ITER] Committed: $NEW_COMMIT — $COMMIT_MSG"

    # Verify agent's commit actually touched the target strategy file.
    # Without this guard, a no-op or wrong-file commit triggers a wasted ~60min eval.
    if ! git diff "$BEST_COMMIT" HEAD --name-only | grep -q "^src/strategies/${STRAT_FILE}\.py$"; then
        echo "[iter $ITER] Agent did not modify src/strategies/${STRAT_FILE}.py — skipping eval"
        git reset --hard "$BEST_COMMIT"
        printf "%d\t%s\t%s\t-2.0\tno_strategy_change\t0/0\t0\t0.0\t\tagent_no_change\n" "$ITER" "$NEW_COMMIT" "$STRAT_FILE" >> "$TSV"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1)); ITER=$((ITER + 1)); continue
    fi

    # ── Step 2: Evaluate ──────────────────────────────────────────
    EVAL_STRATEGY="$STRAT_FILE"
    REGIME_FLAG=""
    [ -n "$TARGET_REGIME" ] && REGIME_FLAG="--target-regime $TARGET_REGIME"

    echo "[iter $ITER] Evaluating $EVAL_STRATEGY..."
    EVAL_START=$(date +%s)
    # Remove any stale latest.json from a prior crashed eval — prevents
    # extract_wfo_insights from feeding phantom data into this iteration's prompt.
    rm -f "artifacts/autoresearch/${EVAL_STRATEGY}_latest.json"
    EVAL_OUTPUT=$(timeout 10800 uv run python scripts/cerberus_autoresearch.py "$EVAL_STRATEGY" --n-trials "$N_TRIALS" $REGIME_FLAG 2>&1 || true)
    EVAL_END=$(date +%s)
    EVAL_DURATION=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "[iter $ITER] Eval completed in ${EVAL_DURATION}m"

    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    BENCHMARK_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_BENCHMARK" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")
    AGGREGATE_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_AGGREGATE" || echo "")
    INSIGHTS=$(uv run python scripts/extract_wfo_insights.py "$EVAL_STRATEGY" 2>/dev/null || echo "NO_INSIGHTS")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter $ITER] ERROR: eval failed"
        printf "%d\t%s\t%s\t-999.0\terror\t0/0\t0\t0.0\t\t%s\n" "$ITER" "$NEW_COMMIT" "$EVAL_STRATEGY" "$COMMIT_MSG" >> "$TSV"
        git reset --hard "$BEST_COMMIT"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1)); ITER=$((ITER + 1)); continue
    fi

    # ── Step 3: Parse ─────────────────────────────────────────────
    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    WIN_PROF=$(echo "$RESULT_LINE" | grep -o 'windows_profitable=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    SORTINO=$(echo "$RESULT_LINE" | grep -o 'avg_sortino=[^ ]*' | cut -d= -f2)
    REGIMES=$(echo "$REGIME_LINES" | tr '\n' '|' | sed 's/|$//')

    echo "[iter $ITER] Result: score=$SCORE trades=$TRADES windows=$WIN_PROF"

    # ── Step 4: Keep or discard ───────────────────────────────────
    # KEEP gate: must (a) have trades, (b) beat current best, AND (c) clear the
    # gate floor. Previously the "improved" branch only checked (a)+(b) — after
    # a phase pivot reset BEST_SCORE to -999, a gate-failed score=-2.0 was
    # silently kept as the new baseline (last 20-iter run iter20 reproduction).
    # Bootstrap branch removed: the unified gate covers it (score > -999 AND
    # score > -2.0 is the same condition).
    KEEP=false
    KEEP_REASON=""
    HAS_TRADES=false
    GATE_FLOOR_LOCAL=-2.0
    [ "$TRADES" -gt 30 ] 2>/dev/null && HAS_TRADES=true

    if [ "$HAS_TRADES" = "true" ] \
           && echo "$SCORE $BEST_SCORE" | awk '{exit ($1 > $2) ? 0 : 1}' \
           && echo "$SCORE $GATE_FLOOR_LOCAL" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="improved: $SCORE > best=$BEST_SCORE > floor=$GATE_FLOOR_LOCAL"
    fi

    PREV_BEST_COMMIT="$BEST_COMMIT"
    PREV_BEST_SCORE="$BEST_SCORE"
    HOLDOUT_LINE=""

    if [ "$KEEP" = "true" ]; then
        STATUS="keep"
        CONSECUTIVE_DISCARDS=0
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
        BEST_COMMIT=$(git rev-parse HEAD)
        echo "$BEST_COMMIT" > "$BEST_COMMIT_FILE"
        echo "[iter $ITER] KEEP (provisional) — $KEEP_REASON (best_commit=${BEST_COMMIT:0:8})"

        # ── Holdout validation ────────────────────────────────────
        # Re-run the kept iteration on the reserved 2026-Q1 window using
        # WFO-selected consensus params from the eval JSON. If the holdout
        # ratio_vs_spy collapses (< 50% of OOS or sign-flips negative), the
        # iteration is suspected of overfitting — downgrade to discard.
        HOLDOUT_PARAMS_PATH="artifacts/autoresearch/${EVAL_STRATEGY}_latest.json"
        rm -f "artifacts/autoresearch/holdout/${EVAL_STRATEGY}_holdout_latest.json"
        echo "[iter $ITER] Holdout: 2026-01-01..2026-03-19"
        HOLDOUT_OUTPUT=$(timeout 1800 uv run python scripts/run_holdout.py \
            --strategy "$EVAL_STRATEGY" \
            --start "2026-01-01" --end "2026-03-19" \
            --params-from "$HOLDOUT_PARAMS_PATH" \
            --quiet 2>&1 || true)
        HOLDOUT_LINE=$(echo "$HOLDOUT_OUTPUT" | grep "^HOLDOUT_RESULT" || echo "")
        HOLDOUT_RATIO=$(echo "$HOLDOUT_LINE" | grep -o 'ratio_vs_spy=[^ ]*' | cut -d= -f2 || echo "")
        HOLDOUT_TRADES=$(echo "$HOLDOUT_LINE" | grep -o 'n_trades=[^ ]*' | cut -d= -f2 || echo "0")
        OOS_RATIO=$(echo "$BENCHMARK_LINE" | grep -o 'ratio_vs_spy=[^ ]*' | cut -d= -f2 || echo "")

        if [ -n "$HOLDOUT_RATIO" ] && [ -n "$OOS_RATIO" ]; then
            HOLDOUT_VERDICT=$(echo "$OOS_RATIO $HOLDOUT_RATIO" | awk '{
                oos=$1; hold=$2;
                if (oos > 0 && (hold < 0 || hold < 0.5*oos)) print "fail";
                else print "ok";
            }')
            if [ "$HOLDOUT_VERDICT" = "fail" ]; then
                echo "[iter $ITER] HOLDOUT FAIL — OOS=$OOS_RATIO holdout=$HOLDOUT_RATIO (sign-flip or <50%×OOS) — downgrading to discard"
                STATUS="holdout_fail"
                KEEP=false
                CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
                BEST_COMMIT="$PREV_BEST_COMMIT"
                BEST_SCORE="$PREV_BEST_SCORE"
                echo "$BEST_COMMIT" > "$BEST_COMMIT_FILE"
                echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
                git reset --hard "$BEST_COMMIT"
            else
                echo "[iter $ITER] HOLDOUT OK — ratio=$HOLDOUT_RATIO trades=$HOLDOUT_TRADES (OOS=$OOS_RATIO)"
            fi
        else
            echo "[iter $ITER] HOLDOUT no-result — keeping iter without validation (run_holdout.py may have errored)"
        fi
    else
        STATUS="discard"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        # Reset to last known-good commit (NOT HEAD~1 — agent may have made
        # multiple commits, and walking back blindly can corrupt infra commits).
        git reset --hard "$BEST_COMMIT"
        echo "[iter $ITER] DISCARD — score=$SCORE vs best=$BEST_SCORE trades=$TRADES ($COMMIT_MSG); reset to ${BEST_COMMIT:0:8}"
    fi

    # ── Step 5: Record ────────────────────────────────────────────
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ITER" "$NEW_COMMIT" "$EVAL_STRATEGY" "$SCORE" "$STATUS" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "$COMMIT_MSG" >> "$TSV"

    # Save compact last result for next agent
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iter=$ITER status=$STATUS score=$SCORE best=$BEST_SCORE trades=$TRADES windows=$WIN_PROF
change: $COMMIT_MSG

SPY benchmark (goal: composite ≥ 3.0 — pretax 3× clears short-term cap-gains tax friction):
  ${BENCHMARK_LINE:-(missing)}

Holdout (2026-01-01..2026-03-19):
  ${HOLDOUT_LINE:-(skipped — only runs after KEEP)}

Regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /' | head -8)

Aggregates:
$(echo "$AGGREGATE_LINES" | sed 's/^/  /' | head -5)

Trade analysis:
$(echo "$INSIGHTS" | head -20)

discards=$CONSECUTIVE_DISCARDS
EOFRESULT

    ITER=$((ITER + 1))
    PHASE_ITER=$((PHASE_ITER + 1))
done

echo ""
echo "============================================================"
echo "  Complete: $MAX_ITER iterations | Best: $BEST_SCORE"
echo "============================================================"
