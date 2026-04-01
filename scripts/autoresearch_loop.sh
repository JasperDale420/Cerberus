#!/usr/bin/env bash
# Cerberus Autoresearch Loop v3 — Hybrid Architecture
#
# Combines Karpathy single-agent (full context for edits) with
# decoupled WFO (survives agent death).
#
# Flow:
#   1. Bash runs WFO (nohup, survives everything)
#   2. Bash spawns agent for edit step only (~30 sec, minimal context)
#   3. Agent reads results, edits strategy, commits, exits
#   4. Bash runs next WFO
#   5. Repeat

set -uo pipefail
cd "$(dirname "$0")/.."

STRATEGY="${1:-autoresearch_strategy}"
MAX_ITER="${2:-100}"
TSV="autoresearch/results.tsv"
BEST_SCORE_FILE="autoresearch/.best_score"
LAST_RESULT_FILE="autoresearch/.last_result"
CONSECUTIVE_DISCARDS=0

mkdir -p autoresearch artifacts/autoresearch/logs

[ ! -f "$TSV" ] && printf "iteration\tcommit\tcomposite_score\ttrades\tstatus\tdescription\n" > "$TSV"
[ ! -f "$BEST_SCORE_FILE" ] && echo "-999.0" > "$BEST_SCORE_FILE"

BEST_SCORE=$(cat "$BEST_SCORE_FILE")
ITER=$(tail -n +2 "$TSV" | wc -l | tr -d ' ')

echo "============================================================"
echo "  Cerberus Autoresearch v3 — Hybrid Loop"
echo "  Strategy: $STRATEGY | Best: $BEST_SCORE | Iter: $ITER"
echo "============================================================"

# ── Run WFO and wait for completion ───────────────────────────────
run_wfo() {
    local strategy="$1"
    echo "[wfo] Running for $strategy..."
    ./scripts/run_eval_with_logging.sh "$strategy" --n-trials 5
}

# ── Baseline ──────────────────────────────────────────────────────
if [ "$ITER" -eq 0 ]; then
    echo "[iter 0] Running baseline..."
    run_wfo "$STRATEGY"
    RESULT_LINE=$(grep "^AUTORESEARCH_RESULT" run.log 2>/dev/null || echo "")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter 0] ERROR: baseline failed"
        cat autoresearch/.eval_status 2>/dev/null
        exit 1
    fi

    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    COMMIT=$(git rev-parse --short HEAD)

    printf "%d\t%s\t%s\t%s\tbaseline\tbaseline\n" 0 "$COMMIT" "$SCORE" "$TRADES" >> "$TSV"
    echo "$SCORE" > "$BEST_SCORE_FILE"
    BEST_SCORE="$SCORE"

    REGIME_LINES=$(grep "^REGIME_BREAKDOWN" run.log 2>/dev/null || echo "")
    AGGREGATE_LINES=$(grep "^REGIME_AGGREGATE" run.log 2>/dev/null || echo "")
    INSIGHTS=$(uv run python scripts/extract_wfo_insights.py "$STRATEGY" 2>/dev/null || echo "NO_INSIGHTS")

    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iter=0 status=baseline score=$SCORE best=$SCORE trades=$TRADES

Regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /' | head -10)

Aggregates:
$(echo "$AGGREGATE_LINES" | sed 's/^/  /' | head -5)

Trade analysis:
$(echo "$INSIGHTS" | head -25)
EOFRESULT

    echo "[iter 0] Baseline: score=$SCORE trades=$TRADES"
    ITER=1
fi

# ── Main Loop ─────────────────────────────────────────────────────
while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "============================================================"
    echo "  Iter $ITER/$MAX_ITER | Best: $BEST_SCORE | Discards: $CONSECUTIVE_DISCARDS"
    echo "============================================================"

    # ── Step 1: Spawn Sonnet agent for edit only ──────────────────
    echo "[iter $ITER] Spawning edit agent..."
    LAST_RESULT=$(cat "$LAST_RESULT_FILE" 2>/dev/null || echo "(no previous result)")
    HISTORY=$(tail -4 "$TSV" | head -3 | awk -F'\t' '{printf "  iter%s: score=%s status=%s trades=%s — %s\n", $1, $3, $5, $4, substr($6,1,60)}' 2>/dev/null || echo "  (none)")

    AGENT_PROMPT="Score to beat: ${BEST_SCORE} | Strategy: ${STRATEGY}

## Last Result
${LAST_RESULT}

## History (don't repeat)
${HISTORY}

## Task
Read src/strategies/${STRATEGY}.py, then make ONE change to beat ${BEST_SCORE}.
$(if [ "$CONSECUTIVE_DISCARDS" -ge 5 ]; then echo "WARNING: ${CONSECUTIVE_DISCARDS} consecutive discards. Try something RADICALLY different."; fi)

Rules:
- Read program_cerberus_v3.md for the BaseStrategy interface if needed
- Prefer removing code over adding. Simpler = better.
- ruff check src/strategies/${STRATEGY}.py before committing
- Commit as: experiment(autoresearch): iter${ITER} — <description>
- Then STOP. Do NOT run any WFO."

    claude -p "$AGENT_PROMPT" --model sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode bypassPermissions 2>&1 || true

    # ── Verify commit ─────────────────────────────────────────────
    NEW_COMMIT=$(git rev-parse --short HEAD)
    PREV_COMMIT=$(tail -1 "$TSV" | cut -f2)

    # Auto-commit if agent edited but forgot to commit
    if [ "$NEW_COMMIT" = "$PREV_COMMIT" ] && [ -f "src/strategies/${STRATEGY}.py" ]; then
        if ! git diff --quiet src/strategies/${STRATEGY}.py 2>/dev/null; then
            git add src/strategies/${STRATEGY}.py config/strategies.yaml 2>/dev/null
            git commit -m "experiment(autoresearch): iter${ITER} — agent edit (auto-committed)" --no-verify 2>/dev/null || true
            NEW_COMMIT=$(git rev-parse --short HEAD)
        fi
    fi

    if [ "$NEW_COMMIT" = "$PREV_COMMIT" ]; then
        # Quota check
        echo "[iter $ITER] No changes — skipping"
        ITER=$((ITER + 1))
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        continue
    fi

    COMMIT_MSG=$(git log -1 --format='%s' | tr -d '\n\r')
    echo "[iter $ITER] Committed: $NEW_COMMIT — $COMMIT_MSG"

    # ── Step 2: Run WFO (decoupled from agent) ────────────────────
    echo "[iter $ITER] Running WFO..."
    run_wfo "$STRATEGY"

    RESULT_LINE=$(grep "^AUTORESEARCH_RESULT" run.log 2>/dev/null || echo "")
    if [ -z "$RESULT_LINE" ]; then
        echo "[iter $ITER] ERROR: WFO failed"
        cat autoresearch/.eval_status 2>/dev/null
        printf "%d\t%s\t-999.0\t0\terror\t%s\n" "$ITER" "$NEW_COMMIT" "$COMMIT_MSG" >> "$TSV"
        git reset --hard HEAD~1
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        ITER=$((ITER + 1))
        continue
    fi

    # ── Step 3: Parse + keep/discard ──────────────────────────────
    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    REGIME_LINES=$(grep "^REGIME_BREAKDOWN" run.log 2>/dev/null || echo "")
    AGGREGATE_LINES=$(grep "^REGIME_AGGREGATE" run.log 2>/dev/null || echo "")
    INSIGHTS=$(uv run python scripts/extract_wfo_insights.py "$STRATEGY" 2>/dev/null || echo "NO_INSIGHTS")

    echo "[iter $ITER] Result: score=$SCORE trades=$TRADES"

    KEEP=false
    HAS_TRADES=false
    [ "$TRADES" -gt 30 ] 2>/dev/null && HAS_TRADES=true

    # Keep if improved
    if [ "$HAS_TRADES" = "true" ] && echo "$SCORE $BEST_SCORE" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    # Bootstrap from -999
    if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] && echo "$BEST_SCORE" | awk '{exit ($1 <= -999) ? 0 : 1}'; then
        KEEP=true
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    if [ "$KEEP" = "true" ]; then
        STATUS="keep"
        CONSECUTIVE_DISCARDS=0
        echo "[iter $ITER] KEEP — score=$SCORE"
    else
        STATUS="discard"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        git reset --hard HEAD~1
        echo "[iter $ITER] DISCARD — score=$SCORE vs best=$BEST_SCORE"
    fi

    # ── Step 4: Record ────────────────────────────────────────────
    printf "%d\t%s\t%s\t%s\t%s\t%s\n" "$ITER" "$NEW_COMMIT" "$SCORE" "$TRADES" "$STATUS" "$COMMIT_MSG" >> "$TSV"

    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iter=$ITER status=$STATUS score=$SCORE best=$BEST_SCORE trades=$TRADES
change: $COMMIT_MSG

Regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /' | head -10)

Aggregates:
$(echo "$AGGREGATE_LINES" | sed 's/^/  /' | head -5)

Trade analysis:
$(echo "$INSIGHTS" | head -25)

discards=$CONSECUTIVE_DISCARDS
EOFRESULT

    ITER=$((ITER + 1))
done

echo ""
echo "============================================================"
echo "  Complete: $MAX_ITER iterations | Best: $BEST_SCORE"
echo "============================================================"
