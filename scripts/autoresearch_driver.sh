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

set -uo pipefail
# Note: -e disabled intentionally — we handle errors manually per-step
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

    EVAL_OUTPUT=$(timeout 4800 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" --n-trials 5 2>&1 || true)
    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")
    AGGREGATE_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_AGGREGATE" || echo "")

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

    # Save last result with full analytics for agent context
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iteration: 0
status: baseline (rsi_bounce — this strategy is broken, you will create NEW specialists)
strategy: $STRATEGY
composite_score: $SCORE
windows_profitable: $WIN_PROF
total_trades: $TRADES
avg_sortino: $SORTINO

Per-window regime breakdown:
$(echo "$REGIME_LINES" | sed 's/^/  /')

Per-regime aggregate:
$(echo "$AGGREGATE_LINES" | sed 's/^/  /')

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

    # Determine which regime phase we're in based on iteration count
    if [ "$ITER" -le 15 ]; then
        REGIME_PHASE="UP_NORMAL"
        REGIME_DESC="UP+NORMAL (trending bull, low vol — 43% of data)"
        STRAT_FILE="regime_trend_up"
        STRAT_HINT="Trend-following: buy pullbacks in uptrends. EMA crossover, VWAP reclaim, ATR trailing stop. BUY-only. The market is going up — ride it."
        TARGET_REGIME="UP+NORMAL"
    elif [ "$ITER" -le 30 ]; then
        REGIME_PHASE="DOWN_HIGH"
        REGIME_DESC="DOWN+HIGH (bear market, high vol — 18% of data)"
        STRAT_FILE="regime_bear"
        STRAT_HINT="Bear market specialist: short momentum breakdowns, VIX spike fading, or defensive mean-reversion on oversold bounces. SELL-biased or fade extremes."
        TARGET_REGIME="DOWN+HIGH"
    elif [ "$ITER" -le 45 ]; then
        REGIME_PHASE="FLAT_NORMAL"
        REGIME_DESC="FLAT+NORMAL (range-bound, normal vol — 5% of data)"
        STRAT_FILE="regime_flat"
        STRAT_HINT="Mean reversion: RSI bounce, Bollinger band fading, VWAP reversion. This is where mean-reversion actually works. Trade both directions."
        TARGET_REGIME="FLAT+NORMAL"
    else
        REGIME_PHASE="GENERALIST"
        REGIME_DESC="Cross-regime generalist"
        STRAT_FILE="regime_adaptive"
        STRAT_HINT="Adaptive strategy that checks the current regime and switches behavior. Use regime labels from symbol_state.meta['regime_labels'] to condition entry logic."
        TARGET_REGIME=""
    fi

    AGENT_PROMPT=$(cat <<EOFPROMPT
You are a quant researcher building a REGIME-SPECIALIST ENSEMBLE for the Cerberus trading system.

## Current Phase: $REGIME_DESC (iterations $ITER)
You are building a specialist strategy for the **$REGIME_PHASE** regime.

## Strategy File: src/strategies/${STRAT_FILE}.py

## Last Result
$LAST_RESULT

## Recent Iteration History (do NOT repeat failed approaches)
$(tail -6 "$TSV" | head -5 | awk -F'\t' '{printf "  iter %s: score=%s status=%s trades=%s — %s\n", $1, $4, $5, $7, $10}' 2>/dev/null || echo "  (no history yet)")

## Your Task
$(if [ ! -f "src/strategies/${STRAT_FILE}.py" ]; then
echo "CREATE a new strategy file: src/strategies/${STRAT_FILE}.py"
echo "This strategy should ONLY work well in ${REGIME_DESC}."
echo ""
echo "Strategy hint: ${STRAT_HINT}"
echo ""
echo "1. Read program_cerberus.md for the BaseStrategy interface, Signal dataclass, and available indicators."
echo "   Read src/strategies/base.py to understand the on_bar() → Signal flow."
echo "3. Create src/strategies/${STRAT_FILE}.py extending BaseStrategy with name = '${STRAT_FILE}'"
echo "4. Add a config block in config/strategies.yaml:"
echo "   ${STRAT_FILE}:"
echo "     enabled: true"
echo "     # your params here"
echo "     activation:"
echo "       session: [opening, midday, power_hour]"
echo "       trend: [up, down, flat]"
echo "       vol: [low, normal, high]"
echo "       liquidity: [good, thin]"
echo "       risk: [risk_on, neutral, risk_off]"
echo "       min_confidence: 0.0"
echo ""
echo "   IMPORTANT: The activation policy MUST be permissive (all sessions, trends, vols)."
echo "   If activation is too restrictive, the strategy won't fire in any WFO window."
echo "5. Keep it SIMPLE — 3-5 factors max. Simpler strategies generalize better."
echo "6. Run: ruff check src/strategies/${STRAT_FILE}.py"
echo "7. Commit and STOP."
else
echo "ITERATE on the existing ${STRAT_FILE} strategy to improve its performance in ${REGIME_DESC}."
echo ""
echo "Read src/strategies/${STRAT_FILE}.py first, then make ONE focused change."
echo ""
if [ "$CONSECUTIVE_DISCARDS" -ge "$MAX_CONSECUTIVE_DISCARDS" ]; then
echo "WARNING: $CONSECUTIVE_DISCARDS consecutive discards. Try something FUNDAMENTALLY different."
echo "Don't keep tweaking the same approach — change the core signal logic."
echo ""
fi
echo "Focus on the REGIME_BREAKDOWN lines — which regime windows scored well vs poorly?"
echo "- Windows with 0 trades → gates too strict, loosen"
echo "- High trade count + PF < 1.0 → over-trading, add selectivity"
echo "- PF > 1.0 windows → understand WHY they work, amplify that"
fi)

## Rules
1. Read program_cerberus.md for framework reference (BaseStrategy, Signal, ConfluenceScorer)
2. Run: ruff check src/strategies/${STRAT_FILE}.py (fix any errors)
3. Commit with message: experiment(${STRAT_FILE}): iter$ITER — <description>
4. Then STOP. Do not run evaluation. The driver handles that.

## Available Data in symbol_state
- symbol_state.bars_1m — deque of recent 1-minute bars
- symbol_state.indicators — dict of precomputed EMA, RSI, ATR, BB
- symbol_state.meta["regime_labels"] — dict with regime_trend, regime_vol, liquidity_regime, correlation_regime, near_earnings, near_fomc, opex_week
- symbol_state.meta["session_phase"] — opening_15m, morning, midday, power_hour, close_15m
- symbol_state.position — current position (None if flat)
EOFPROMPT
)

    # Run the agent (short-lived, just edits code)
    AGENT_RESULT=$(claude -p "$AGENT_PROMPT" --model sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode bypassPermissions 2>&1 || true)

    # Verify the agent committed something
    NEW_COMMIT=$(git rev-parse --short HEAD)
    PREV_COMMIT=$(tail -1 "$TSV" | cut -f2)

    # If this is a NEW strategy creation, verify it imports correctly before WFO
    if [ "$NEW_COMMIT" != "$PREV_COMMIT" ] && [ -f "src/strategies/${STRAT_FILE}.py" ]; then
        IMPORT_CHECK=$(uv run python -c "
import sys; sys.path.insert(0, '.')
from src.strategies.base import BaseStrategy
from src.backtest.runner import _dynamic_import_strategy_class
cls = _dynamic_import_strategy_class('${STRAT_FILE}')
if cls is None:
    print('IMPORT_FAIL: no BaseStrategy subclass found')
    sys.exit(1)
print(f'IMPORT_OK: {cls.__name__} (name={getattr(cls, \"name\", \"?\")})')
" 2>&1 || echo "IMPORT_FAIL: exception")

        if echo "$IMPORT_CHECK" | grep -q "IMPORT_FAIL"; then
            echo "[iter $ITER] Strategy import check FAILED: $IMPORT_CHECK"
            echo "[iter $ITER] Spawning fix agent..."
            FIX_PROMPT="The strategy file src/strategies/${STRAT_FILE}.py failed to import:
$IMPORT_CHECK

Fix the import error. Common issues:
- Missing 'name' class attribute
- Syntax errors
- Wrong base class
- Missing imports

Read the file, fix it, run 'ruff check src/strategies/${STRAT_FILE}.py', and commit."
            FIX_RESULT=$(claude -p "$FIX_PROMPT" --model sonnet --allowedTools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode bypassPermissions 2>&1 || true)
            NEW_COMMIT=$(git rev-parse --short HEAD)
        else
            echo "[iter $ITER] Strategy import check: $IMPORT_CHECK"
        fi

        # Also verify config entry exists
        if ! grep -q "^  ${STRAT_FILE}:" config/strategies.yaml 2>/dev/null; then
            if ! grep -q "^  ${STRAT_FILE}:" config/backtest_v2.yaml 2>/dev/null; then
                echo "[iter $ITER] WARNING: No config entry for ${STRAT_FILE} — adding minimal entry"
                echo "
  ${STRAT_FILE}:
    enabled: true
    activation:
      session: [opening, midday, power_hour]
      trend: [up, down, flat]
      vol: [low, normal, high]
      liquidity: [good, thin]
      risk: [risk_on, neutral, risk_off]
      min_confidence: 0.0" >> config/strategies.yaml
                git add config/strategies.yaml && git commit -m "fix: add ${STRAT_FILE} config entry for WFO" --no-verify 2>/dev/null || true
                NEW_COMMIT=$(git rev-parse --short HEAD)
            fi
        fi
    fi

    if [ "$NEW_COMMIT" = "$PREV_COMMIT" ]; then
        # Check if it was a quota/rate limit issue vs genuine no-op
        if echo "$AGENT_RESULT" | grep -qi "rate.limit\|quota\|overloaded\|429\|capacity"; then
            echo "[iter $ITER] Agent hit rate limit/quota. Sleeping 5 minutes before retry..."
            sleep 300
            # Don't increment iteration — retry the same one
            continue
        fi
        echo "[iter $ITER] Agent did not commit. Skipping evaluation."
        ITER=$((ITER + 1))
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        continue
    fi

    COMMIT_MSG=$(git log -1 --format='%s')
    echo "[iter $ITER] Agent committed: $NEW_COMMIT — $COMMIT_MSG"

    # Evaluate the regime specialist for the current phase
    EVAL_STRATEGY="$STRAT_FILE"
    echo "[iter $ITER] Phase=$REGIME_PHASE evaluating strategy=$EVAL_STRATEGY"

    # ── Step 2: Run WFO evaluation (outside agent context) ─────────
    echo "[iter $ITER] Running WFO evaluation (~15 min)..."
    EVAL_START=$(date +%s)

    REGIME_FLAG=""
    if [ -n "$TARGET_REGIME" ]; then
        REGIME_FLAG="--target-regime $TARGET_REGIME"
    fi
    EVAL_OUTPUT=$(timeout 4800 uv run python scripts/cerberus_autoresearch.py "$EVAL_STRATEGY" --n-trials 5 $REGIME_FLAG 2>&1 || true)

    EVAL_END=$(date +%s)
    EVAL_DURATION=$(( (EVAL_END - EVAL_START) / 60 ))
    echo "[iter $ITER] Evaluation completed in ${EVAL_DURATION}m"

    RESULT_LINE=$(echo "$EVAL_OUTPUT" | grep "^AUTORESEARCH_RESULT" || echo "")
    REGIME_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_BREAKDOWN" || echo "")
    AGGREGATE_LINES=$(echo "$EVAL_OUTPUT" | grep "^REGIME_AGGREGATE" || echo "")

    # Extract detailed trade-level insights from the results JSON
    INSIGHTS=$(uv run python scripts/extract_wfo_insights.py "$EVAL_STRATEGY" 2>/dev/null || echo "NO_INSIGHTS")

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
    # Strict improvement-only logic:
    #   1. GATE: must have >30 trades to even be considered (otherwise meaningless)
    #   2. KEEP (improved): composite score improved over best
    #   3. KEEP (first trades): first iteration that generates >30 trades (bootstrap)
    #   4. DISCARD: everything else — revert to preserve the best-scoring code
    #
    # The agent sees the detailed trade analysis in .last_result either way,
    # so it learns from discarded iterations without the code being kept.
    KEEP=false
    KEEP_REASON=""

    # Gate: must have trades to be meaningful
    HAS_TRADES=false
    if [ "$TRADES" -gt 30 ] 2>/dev/null; then
        HAS_TRADES=true
    fi

    # Check if composite improved (must also have trades)
    if [ "$HAS_TRADES" = "true" ] && echo "$SCORE $BEST_SCORE" | awk '{exit ($1 > $2) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="composite improved: $SCORE > $BEST_SCORE"
        echo "$SCORE" > "$BEST_SCORE_FILE"
        BEST_SCORE="$SCORE"
    fi

    # Bootstrap: if best_score is still -999 and we got trades for the first time, keep
    if [ "$KEEP" = "false" ] && [ "$HAS_TRADES" = "true" ] && echo "$BEST_SCORE" | awk '{exit ($1 <= -999) ? 0 : 1}'; then
        KEEP=true
        KEEP_REASON="first working strategy: $TRADES trades (bootstrap from -999)"
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
        echo "[iter $ITER] DISCARD — 0 trades or evaluation error (revert $NEW_COMMIT)"
    fi

    # ── Step 5: Record result ──────────────────────────────────────
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ITER" "$NEW_COMMIT" "$STRATEGY" "$SCORE" "$STATUS" "$WIN_PROF" "$TRADES" "$SORTINO" "$REGIMES" "$COMMIT_MSG" >> "$TSV"

    # Update last result for next agent iteration — include full analytics
    cat > "$LAST_RESULT_FILE" <<EOFRESULT
iteration: $ITER
status: $STATUS
strategy: $EVAL_STRATEGY
composite_score: $SCORE
windows_profitable: $WIN_PROF
total_trades: $TRADES
avg_sortino: $SORTINO

Per-window regime breakdown (each window is a separate OOS test period):
$(echo "$REGIME_LINES" | sed 's/^/  /')

Per-regime aggregate performance (grouped across all windows with same regime):
$(echo "$AGGREGATE_LINES" | sed 's/^/  /')

Analysis guidance:
- PF > 1.0 = profitable. PF < 1.0 = losing money. Target PF > 1.3.
- Sharpe > 0 = positive risk-adjusted return. Target > 1.0.
- trades=0 means gates are too strict — loosen entry criteria.
- High trades + low PF = over-trading — add selectivity (higher confluence, tighter time window).
- Check which REGIME windows are profitable vs losing — your strategy should excel in $REGIME_PHASE.
- best_regime/worst_regime in AUTORESEARCH_RESULT shows where the strategy shines and struggles.

Detailed trade analysis (from WFO results JSON):
$INSIGHTS

Best score so far: $BEST_SCORE
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
