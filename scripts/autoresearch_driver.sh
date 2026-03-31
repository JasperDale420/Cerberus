#!/usr/bin/env bash
# Cerberus Autoresearch Driver v2
#
# Karpathy-inspired: edit → eval → keep/discard → repeat
# Agent sees compact summary + trade insights. Never sees WFO output.

set -uo pipefail
cd "$(dirname "$0")/.."

STRATEGY="${1:-rsi_bounce}"
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
PHASE_BEST=-999.0

# ── Setup ──────────────────────────────────────────────────────────
mkdir -p autoresearch artifacts/autoresearch/logs

if [ ! -f "$TSV" ]; then
    printf "iteration\tcommit\tstrategy\tcomposite_score\tstatus\twindows_profitable\ttotal_trades\tavg_sortino\tregime_breakdown\tdescription\n" > "$TSV"
fi

if [ ! -f "$BEST_SCORE_FILE" ]; then
    echo "-999.0" > "$BEST_SCORE_FILE"
fi

BEST_SCORE=$(cat "$BEST_SCORE_FILE")
ITER=$(tail -n +2 "$TSV" | wc -l | tr -d ' ')

echo "============================================================"
echo "  Cerberus Autoresearch v2"
echo "  Strategy: $STRATEGY | Best: $BEST_SCORE | Iter: $ITER"
echo "============================================================"

# ── Baseline ──────────────────────────────────────────────────────
if [ "$ITER" -eq 0 ]; then
    echo "[iter 0] Running baseline..."
    COMMIT=$(git rev-parse --short HEAD)
    EVAL_OUTPUT=$(timeout 4800 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" --n-trials 5 2>&1 || true)
    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")

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
    if [ "$CONSECUTIVE_DISCARDS" -ge "$MAX_CONSECUTIVE_DISCARDS" ] && [ "$CURRENT_PHASE" -lt 3 ]; then
        CURRENT_PHASE=$((CURRENT_PHASE + 1))
        STRAT_FILE="${REGIME_PHASES[$CURRENT_PHASE]}"
        REGIME_DESC="${REGIME_DESCS[$CURRENT_PHASE]}"
        TARGET_REGIME="${REGIME_TARGETS[$CURRENT_PHASE]}"
        REGIME_HINT="${REGIME_HINTS[$CURRENT_PHASE]}"
        CONSECUTIVE_DISCARDS=0
        PHASE_ITER=0
        echo "[iter $ITER] PIVOT → Phase $CURRENT_PHASE: $REGIME_DESC"
    fi

    # ── Step 1: Spawn agent ───────────────────────────────────────
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

    COMMIT_MSG=$(git log -1 --format='%s')
    echo "[iter $ITER] Committed: $NEW_COMMIT — $COMMIT_MSG"

    # ── Step 2: Evaluate ──────────────────────────────────────────
    EVAL_STRATEGY="$STRAT_FILE"
    REGIME_FLAG=""
    [ -n "$TARGET_REGIME" ] && REGIME_FLAG="--target-regime $TARGET_REGIME"

    echo "[iter $ITER] Evaluating $EVAL_STRATEGY..."
    EVAL_START=$(date +%s)
    EVAL_OUTPUT=$(timeout 4800 uv run python scripts/cerberus_autoresearch.py "$EVAL_STRATEGY" --n-trials 5 $REGIME_FLAG 2>&1 || true)
    EVAL_END=$(date +%s)
    EVAL_DURATION=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "[iter $ITER] Eval completed in ${EVAL_DURATION}m"

    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")
    AGGREGATE_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_AGGREGATE" || echo "")
    INSIGHTS=$(uv run python scripts/extract_wfo_insights.py "$EVAL_STRATEGY" 2>/dev/null || echo "NO_INSIGHTS")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter $ITER] ERROR: eval failed"
        printf "%d\t%s\t%s\t-999.0\terror\t0/0\t0\t0.0\t\t%s\n" "$ITER" "$NEW_COMMIT" "$EVAL_STRATEGY" "$COMMIT_MSG" >> "$TSV"
        git reset --hard HEAD~1
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

    # Bootstrap: first strategy with trades escapes -999
    if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] && echo "$BEST_SCORE" | awk '{exit ($1 <= -999) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="bootstrap: first $TRADES trades"
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    if [ "$KEEP" = "true" ]; then
        STATUS="keep"
        CONSECUTIVE_DISCARDS=0
        echo "[iter $ITER] KEEP — $KEEP_REASON"
    else
        STATUS="discard"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        git reset --hard HEAD~1
        echo "[iter $ITER] DISCARD — score=$SCORE vs best=$BEST_SCORE trades=$TRADES ($COMMIT_MSG)"
    fi

    # ── Step 5: Record ────────────────────────────────────────────
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ITER" "$NEW_COMMIT" "$EVAL_STRATEGY" "$SCORE" "$STATUS" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "$COMMIT_MSG" >> "$TSV"

    # Save compact last result for next agent
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iter=$ITER status=$STATUS score=$SCORE best=$BEST_SCORE trades=$TRADES windows=$WIN_PROF
change: $COMMIT_MSG

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
