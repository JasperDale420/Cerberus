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
PROTECTED_FILES=(
    "scripts/cerberus_autoresearch.py"
    "scripts/extract_wfo_insights.py"
    "scripts/autoresearch_driver.sh"
    "program_cerberus.md"
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
    EVAL_OUTPUT=$(timeout 10800 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" --n-trials 5 2>&1 || true)
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
    HISTORY=$(tail -4 "$TSV" | head -3 | awk -F'\t' '{printf "  iter%s: score=%s status=%s trades=%s — %s\n", $1, $4, $5, $7, substr($10,1,60)}' 2>/dev/null || echo "  (none)")

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

    AGENT_RESULT=$(claude -p "$AGENT_PROMPT" --model sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode bypassPermissions 2>&1 || true)

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
            claude -p "Fix import error in src/strategies/${STRAT_FILE}.py: $IMPORT_CHECK. Read file, fix, ruff check, commit." \
                --model sonnet --allowedTools "Read,Edit,Write,Bash" --permission-mode bypassPermissions 2>&1 || true
            NEW_COMMIT=$(git rev-parse --short HEAD)
        fi

        if ! grep -q "^  ${STRAT_FILE}:" config/strategies.yaml 2>/dev/null; then
            echo "[iter $ITER] Adding config entry for ${STRAT_FILE}"
            printf "\n  %s:\n    enabled: true\n    activation:\n      session: [opening, midday, power_hour]\n      trend: [up, down, flat]\n      vol: [low, normal, high]\n      liquidity: [good, thin]\n      risk: [risk_on, neutral, risk_off]\n      min_confidence: 0.0\n" "$STRAT_FILE" >> config/strategies.yaml
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

    # Sanitize: strip newlines to prevent heredoc injection from agent-authored commit messages
    COMMIT_MSG=$(git log -1 --format='%s' | tr -d '\n\r')
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
    EVAL_OUTPUT=$(timeout 10800 uv run python scripts/cerberus_autoresearch.py "$EVAL_STRATEGY" --n-trials 5 $REGIME_FLAG 2>&1 || true)
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
    KEEP=false
    KEEP_REASON=""
    HAS_TRADES=false
    [ "$TRADES" -gt 30 ] 2>/dev/null && HAS_TRADES=true

    # Keep if improved (with trades)
    if [ "$HAS_TRADES" = "true" ] && echo "$SCORE $BEST_SCORE" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="improved: $SCORE > $BEST_SCORE"
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    # Bootstrap: first strategy with trades AND a non-gate-failed score escapes -999.
    # Gate-failed scores (composite=-2.0) must NOT become the bootstrap baseline,
    # otherwise the agent's lineage roots in a known-broken strategy.
    GATE_FLOOR_LOCAL=-2.0
    if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] \
           && echo "$BEST_SCORE" | awk '{exit ($1 <= -999) ? 0 : 1}' \
           && echo "$SCORE $GATE_FLOOR_LOCAL" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="bootstrap: first $TRADES trades, score $SCORE > floor"
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    if [ "$KEEP" = "true" ]; then
        STATUS="keep"
        CONSECUTIVE_DISCARDS=0
        BEST_COMMIT=$(git rev-parse HEAD)
        echo "$BEST_COMMIT" > "$BEST_COMMIT_FILE"
        echo "[iter $ITER] KEEP — $KEEP_REASON (best_commit=${BEST_COMMIT:0:8})"
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

SPY benchmark (goal: strategy_return ≥ 2x SPY over the OOS span):
  ${BENCHMARK_LINE:-(missing)}

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
