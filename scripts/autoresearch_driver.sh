#!/usr/bin/env bash
# Cerberus Autoresearch Driver — context-efficient loop
#
# Architecture:
#   1. Short-lived Claude agent: reads last result, edits strategy code, commits
#   2. This script: runs WFO evaluation (heavy compute, no agent context cost)
#   3. This script: parses result, decides keep/discard, appends to TSV
#   4. Repeat
#
# The agent NEVER sees WFO output — only the ~10-line summary from last iteration.
# This keeps agent context at ~15-20K tokens per iteration instead of 200K+.
#
# Usage:
#   ./scripts/autoresearch_driver.sh [strategy_name] [max_iterations]
#
# Defaults: rsi_bounce, 50 iterations

set -euo pipefail
cd "$(dirname "$0")/.."

STRATEGY="${1:-rsi_bounce}"
MAX_ITER="${2:-50}"
TSV="autoresearch/results.tsv"
PROMPT_TEMPLATE="autoresearch/agent_prompt.md"
BEST_SCORE_FILE="autoresearch/.best_score"
LAST_RESULT_FILE="autoresearch/.last_result"
CONSECUTIVE_DISCARDS=0
MAX_CONSECUTIVE_DISCARDS=5

# ── Setup ──────────────────────────────────────────────────────────
mkdir -p autoresearch artifacts/autoresearch/logs

# Initialize TSV if needed
if [ ! -f "$TSV" ]; then
    printf "iteration\tcommit\tstrategy\tcomposite_score\tstatus\twindows_profitable\ttotal_trades\tavg_sortino\tregime_breakdown\tdescription\n" > "$TSV"
fi

# Initialize best score
if [ ! -f "$BEST_SCORE_FILE" ]; then
    echo "-999.0" > "$BEST_SCORE_FILE"
fi

BEST_SCORE=$(cat "$BEST_SCORE_FILE")

# Count existing iterations
ITER=$(tail -n +2 "$TSV" | wc -l | tr -d ' ')

echo "============================================================"
echo "  Cerberus Autoresearch Driver"
echo "  Strategy: $STRATEGY"
echo "  Starting at iteration: $ITER"
echo "  Best score so far: $BEST_SCORE"
echo "  Max iterations: $MAX_ITER"
echo "============================================================"

# ── Run baseline if iteration 0 ───────────────────────────────────
if [ "$ITER" -eq 0 ]; then
    echo ""
    echo "[iter 0] Running baseline evaluation..."
    COMMIT=$(git rev-parse --short HEAD)

    EVAL_OUTPUT=$(uv run python scripts/cerberus_autoresearch.py "$STRATEGY" 2>&1 || true)
    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter 0] ERROR: No AUTORESEARCH_RESULT line in output"
        echo "  Raw output (last 20 lines):"
        echo "$EVAL_OUTPUT" | tail -20
        exit 1
    fi

    # Parse fields
    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    WIN_PROF=$(echo "$RESULT_LINE" | grep -o 'windows_profitable=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    SORTINO=$(echo "$RESULT_LINE" | grep -o 'avg_sortino=[^ ]*' | cut -d= -f2)
    REGIMES=$(echo "$REGIME_LINES" | tr '\n' '|' | sed 's/|$//')

    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ITER" "$COMMIT" "$STRATEGY" "$SCORE" "baseline" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "baseline" >> "$TSV"

    echo "$SCORE" > "$BEST_SCORE_FILE"
    BEST_SCORE="$SCORE"

    # Save last result for agent context
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iteration: 0
status: baseline
strategy: $STRATEGY
composite_score: $SCORE
windows_profitable: $WIN_PROF
total_trades: $TRADES
avg_sortino: $SORTINO

Regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /')

Best score: $BEST_SCORE
Consecutive discards: 0
EOFRESULT

    echo "[iter 0] Baseline: score=$SCORE windows=$WIN_PROF trades=$TRADES sortino=$SORTINO"
    ITER=1
fi

# ── Main loop ──────────────────────────────────────────────────────
while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "============================================================"
    echo "  Iteration $ITER / $MAX_ITER"
    echo "  Best score: $BEST_SCORE | Consecutive discards: $CONSECUTIVE_DISCARDS"
    echo "============================================================"

    # ── Step 1: Spawn agent to modify strategy ─────────────────────
    echo "[iter $ITER] Spawning agent for strategy modification..."

    # Build the agent prompt with last result context
    LAST_RESULT=$(cat "$LAST_RESULT_FILE")

    AGENT_PROMPT=$(cat <<EOFPROMPT
You are a quant researcher iterating on the Cerberus "$STRATEGY" strategy.

## Last Result
$LAST_RESULT

## Your Task
Make ONE focused change to improve the strategy. Read the current code, understand what's happening, then make a single modification.

$(if [ "$CONSECUTIVE_DISCARDS" -ge "$MAX_CONSECUTIVE_DISCARDS" ]; then
echo "WARNING: $CONSECUTIVE_DISCARDS consecutive discards. Your recent approach is NOT working."
echo "Try something FUNDAMENTALLY different — new signal logic, different indicator, or a new strategy entirely."
echo ""
fi)

## Rules
1. Read src/strategies/${STRATEGY}.py first
2. Read program_cerberus.md for framework reference (BaseStrategy, Signal, ConfluenceScorer interfaces)
3. Make ONE change — don't rewrite the whole strategy
4. Run: ruff check src/strategies/${STRATEGY}.py (fix any errors)
5. Commit with a descriptive message explaining your hypothesis
6. Then STOP. Do not run any evaluation. The driver handles that.

## What to focus on based regime breakdown
- Windows with negative scores need the most help
- Windows with PF < 1.0 are losing money — reduce trade frequency or tighten entry criteria
- Windows with 0 trades — the gates are too strict, loosen them
- High trade count + negative score = the strategy is over-trading, add selectivity

Write your commit message in this format:
  experiment(<strategy>): iter$ITER — <one line description of change>
EOFPROMPT
)

    # Run the agent (short-lived, just edits code)
    AGENT_RESULT=$(claude -p "$AGENT_PROMPT" -m sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --max-turns 20 2>&1 || true)

    # Verify the agent committed something
    NEW_COMMIT=$(git rev-parse --short HEAD)
    PREV_COMMIT=$(tail -1 "$TSV" | cut -f2)

    if [ "$NEW_COMMIT" = "$PREV_COMMIT" ]; then
        echo "[iter $ITER] Agent did not commit. Skipping evaluation."
        ITER=$((ITER + 1))
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        continue
    fi

    COMMIT_MSG=$(git log -1 --format='%s')
    echo "[iter $ITER] Agent committed: $NEW_COMMIT — $COMMIT_MSG"

    # ── Step 2: Run WFO evaluation (outside agent context) ─────────
    echo "[iter $ITER] Running WFO evaluation (~15 min)..."
    EVAL_START=$(date +%s)

    EVAL_OUTPUT=$(timeout 2400 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" 2>&1 || true)

    EVAL_END=$(date +%s)
    EVAL_DURATION=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "[iter $ITER] Evaluation completed in ${EVAL_DURATION}m"

    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")

    if [ -z "$RESULT_LINE" ]; then
        echo "[iter $ITER] ERROR: Evaluation failed (no result line)"
        printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$ITER" "$NEW_COMMIT" "$STRATEGY" "-999.0" "error" "0/0" "0" "0.0" "" "$COMMIT_MSG" >> "$TSV"
        git reset --hard HEAD~1
        echo "[iter $ITER] Reverted commit (evaluation error)"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        ITER=$((ITER + 1))
        continue
    fi

    # ── Step 3: Parse results ──────────────────────────────────────
    SCORE=$(echo "$RESULT_LINE" | grep -o 'composite_score=[^ ]*' | cut -d= -f2)
    WIN_PROF=$(echo "$RESULT_LINE" | grep -o 'windows_profitable=[^ ]*' | cut -d= -f2)
    TRADES=$(echo "$RESULT_LINE" | grep -o 'total_oos_trades=[^ ]*' | cut -d= -f2)
    SORTINO=$(echo "$RESULT_LINE" | grep -o 'avg_sortino=[^ ]*' | cut -d= -f2)
    REGIMES=$(echo "$REGIME_LINES" | tr '\n' '|' | sed 's/|$//')

    echo "[iter $ITER] Result: score=$SCORE windows=$WIN_PROF trades=$TRADES sortino=$SORTINO"

    # ── Step 4: Keep or discard ────────────────────────────────────
    # Keep if: composite improved, OR a regime window scored > 3.0 (specialist potential)
    KEEP=false
    REGIME_SPECIALIST=""

    # Check if composite improved
    if echo "$SCORE $BEST_SCORE" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        echo "[iter $ITER] KEEP — composite improved: $SCORE > $BEST_SCORE"
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    # Check for regime specialist windows (score > 3.0 in any window)
    if [ "$KEEP" = "false" ]; then
        for regime_line in $(echo "$REGIME_LINES" | tr '|' '\n'); do
            window_score=$(echo "$regime_line" | grep -o 'oos_score=[^ ]*' | cut -d= -f2)
            window_regime=$(echo "$regime_line" | grep -o 'regime=[^ ]*' | cut -d= -f2)
            if echo "$window_score" | awk '{exit ($1 > 3.0) ? 0 : 1}'; then
                KEEP=true
                REGIME_SPECIALIST="$window_regime:$window_score"
                echo "[iter $ITER] KEEP — regime specialist: $window_regime scored $window_score"
                break
            fi
        done
    fi

    if [ "$KEEP" = "true" ]; then
        STATUS="keep"
        CONSECUTIVE_DISCARDS=0
    else
        STATUS="discard"
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        git reset --hard HEAD~1
        echo "[iter $ITER] DISCARD — score=$SCORE did not beat best=$BEST_SCORE (revert $NEW_COMMIT)"
    fi

    # ── Step 5: Record result ──────────────────────────────────────
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ITER" "$NEW_COMMIT" "$STRATEGY" "$SCORE" "$STATUS" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "$COMMIT_MSG" >> "$TSV"

    # Update last result for next agent iteration
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iteration: $ITER
status: $STATUS
strategy: $STRATEGY
composite_score: $SCORE
windows_profitable: $WIN_PROF
total_trades: $TRADES
avg_sortino: $SORTINO

Regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /')

Best score: $BEST_SCORE
Consecutive discards: $CONSECUTIVE_DISCARDS
Previous change: $COMMIT_MSG
EOFRESULT

    ITER=$((ITER + 1))
done

echo ""
echo "============================================================"
echo "  Autoresearch complete: $MAX_ITER iterations"
echo "  Final best score: $BEST_SCORE"
echo "  Results: $TSV"
echo "============================================================"
