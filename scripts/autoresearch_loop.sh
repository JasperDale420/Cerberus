#!/usr/bin/env bash
# Cerberus Autoresearch v3 — Single persistent agent with session resume
#
# Faithful to Karpathy: one long-lived agent that accumulates context.
# The bash wrapper ONLY handles session resurrection — if the agent dies
# (quota, timeout, context limit), we resume the same session.
#
# The agent runs the eval in background, waits for completion, reads results,
# and iterates. All reasoning and memory lives in the conversation history.
#
# Usage: ./scripts/autoresearch_loop.sh [session_name]

set -uo pipefail
cd "$(dirname "$0")/.."

SESSION_NAME="${1:-autoresearch-v3}"
STRATEGY="autoresearch_strategy"
MAX_DEATHS=20  # Max times we'll resurrect before giving up
DEATHS=0

mkdir -p autoresearch artifacts/autoresearch/logs

echo "============================================================"
echo "  Cerberus Autoresearch v3 — Persistent Agent"
echo "  Session: $SESSION_NAME"
echo "  Strategy: $STRATEGY"
echo "  Will resurrect up to $MAX_DEATHS times"
echo "============================================================"

# Initial prompt for first launch
INITIAL_PROMPT="You are an autonomous quant researcher for the Cerberus trading system.

Working directory: /Users/jacobmcmillan/Empire/Cerberus

Read program_cerberus_v3.md for full instructions. Key rules:

1. You modify ONE file: src/strategies/autoresearch_strategy.py
2. Run WFO in background: use Bash with run_in_background:true for:
   ./scripts/run_eval_with_logging.sh autoresearch_strategy --n-trials 5
3. After background task completes, read ONLY:
   - cat autoresearch/.eval_status (5 lines)
   - grep '^AUTORESEARCH_RESULT' run.log (1 line)
   - grep '^REGIME_BREAKDOWN' run.log (8 lines)
   - uv run python scripts/extract_wfo_insights.py autoresearch_strategy (~20 lines)
4. NEVER cat run.log — it will kill your context
5. Keep if score improves, git reset --hard HEAD~1 if not
6. Record in autoresearch/results.tsv
7. Loop FOREVER. Never stop. Never ask.

Check autoresearch/results.tsv for previous results. Score to beat: $(cat autoresearch/.best_score 2>/dev/null || echo '-999.0')

Start by reading program_cerberus_v3.md and the current strategy file. Then begin the loop."

# Resume prompt when resurrecting after death
RESUME_PROMPT="You are an autonomous quant researcher. Your session was interrupted (quota/timeout/context limit).

RESUME your autoresearch loop:
1. Read autoresearch/results.tsv to see what's been done
2. Read autoresearch/.last_result for the latest WFO results
3. Read autoresearch/.eval_status to check if an eval was running when you died
4. Read src/strategies/autoresearch_strategy.py for current code
5. Continue the loop: edit → eval (background) → parse → keep/discard → repeat

Score to beat: $(cat autoresearch/.best_score 2>/dev/null || echo '-999.0')

Rules: run WFO with run_in_background:true, NEVER cat run.log, read program_cerberus_v3.md if needed.

Pick up where you left off. NEVER STOP."

# ── Main resurrection loop ────────────────────────────────────────
while [ "$DEATHS" -lt "$MAX_DEATHS" ]; do
    if [ "$DEATHS" -eq 0 ]; then
        echo "[launcher] Starting fresh session: $SESSION_NAME"
        claude -p "$INITIAL_PROMPT" \
            --model opus \
            --name "$SESSION_NAME" \
            --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
            --permission-mode bypassPermissions \
            2>&1 || true
    else
        echo "[launcher] Resurrecting session (death #$DEATHS)..."
        # Try to continue the named session
        claude -p "$RESUME_PROMPT" \
            --model opus \
            --continue \
            --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
            --permission-mode bypassPermissions \
            2>&1 || true
    fi

    EXIT_CODE=$?
    DEATHS=$((DEATHS + 1))
    echo ""
    echo "[launcher] Agent exited (code=$EXIT_CODE, death #$DEATHS/$MAX_DEATHS)"
    echo "[launcher] Sleeping 30s before resurrection..."
    sleep 30

    # Update the resume prompt with latest score
    RESUME_PROMPT="You are an autonomous quant researcher. Your session was interrupted (death #$DEATHS).

RESUME your autoresearch loop:
1. Read autoresearch/results.tsv to see what's been done
2. Read autoresearch/.last_result for the latest WFO results
3. Read autoresearch/.eval_status to check if an eval was running
4. Read src/strategies/autoresearch_strategy.py for current code
5. Continue: edit → eval (background) → parse → keep/discard → repeat

Score to beat: $(cat autoresearch/.best_score 2>/dev/null || echo '-999.0')

Rules: run WFO with run_in_background:true, NEVER cat run.log, read program_cerberus_v3.md if needed.

NEVER STOP."
done

echo "[launcher] Max deaths ($MAX_DEATHS) reached. Stopping."
echo "  Final results: autoresearch/results.tsv"
echo "  Best score: $(cat autoresearch/.best_score 2>/dev/null || echo 'N/A')"
