#!/usr/bin/env bash
# Cerberus Autoresearch v5 — Beat SPY 2x. FROZEN scripts.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="/opt/homebrew/bin:/opt/homebrew/Caskroom/miniforge/base/bin:$HOME/.local/bin:$PATH"

STRATEGY="daily_research_strategy"
SESSION="autoresearch-daily-${1:-s1}"
MAX_DEATHS=20
DEATHS=0
TSV="autoresearch/results_daily.tsv"
BEST_FILE="autoresearch/.best_score_daily"

mkdir -p autoresearch artifacts/autoresearch/logs
command -v claude &>/dev/null || { echo "ERROR: claude not found"; exit 1; }
[ ! -f "$TSV" ] && printf "iteration\tcommit\tnet_pnl\tspy_ratio\tsharpe\ttrades\tstatus\tdescription\n" > "$TSV"
[ ! -f "$BEST_FILE" ] && echo "-999.0" > "$BEST_FILE"
BEST=$(cat "$BEST_FILE")

echo "============================================================"
echo "  Autoresearch v5 — Beat SPY 2x | Strategy: $STRATEGY | Best: $BEST"
echo "============================================================"

INITIAL="You are an autonomous quant researcher. Goal: BEAT SPY BUY-AND-HOLD BY 2x.
Working directory: /Users/jacobmcmillan/Empire/Cerberus

Read autoresearch/frozen/program_cerberus_v5.md for FULL instructions. This is CRITICAL.

SPY returned \$47,323 on \$100k over the eval period. Your target: \$94,646+ (2x SPY).

Files you can edit:
- src/strategies/${STRATEGY}.py — primary strategy
- config/strategies.yaml — the ${STRATEGY}: block
- src/core/indicators.py — add new indicators if needed
- src/analytics/param_spaces.py — add optimization spaces

Eval (MANDATORY background):
autoresearch/frozen/run_eval_with_logging.sh ${STRATEGY} --n-trials 5 --n-workers 4 --daily
NEVER cat run.log. Read only: .eval_status, grep AUTORESEARCH_RESULT, grep REGIME_BREAKDOWN
Insights: uv run python autoresearch/frozen/extract_wfo_insights.py ${STRATEGY}

Result shows: net_pnl=\$X spy_ratio=Yx beats_2x_spy=YES/NO
Keep if score > ${BEST}. Revert: git checkout HEAD~1 -- files && git commit -m revert
NEVER git reset --hard. Results: ${TSV}. Use WebSearch for research. NEVER STOP."

while [ "$DEATHS" -lt "$MAX_DEATHS" ]; do
    CURRENT=$(cat "$BEST_FILE" 2>/dev/null || echo '-999.0')
    if [ "$DEATHS" -eq 0 ]; then
        echo "[v5] Starting: $SESSION"
        claude -p "$INITIAL" --model opus --name "$SESSION" \
            --allowedTools "Read,Edit,Write,Bash,Glob,Grep,WebSearch,WebFetch" \
            --permission-mode bypassPermissions 2>&1 || true
    else
        echo "[v5] Resuming (death #$DEATHS)..."
        claude -p "Session interrupted. Resume. Goal: beat SPY 2x (\$94,646+). Current best: \${CURRENT}. Strategy: ${STRATEGY}. Read autoresearch/frozen/program_cerberus_v5.md. Eval: autoresearch/frozen/run_eval_with_logging.sh ${STRATEGY} --n-trials 5 --n-workers 4 --daily (background). NEVER cat run.log. NEVER git reset --hard. NEVER STOP." \
            --model opus --continue \
            --allowedTools "Read,Edit,Write,Bash,Glob,Grep,WebSearch,WebFetch" \
            --permission-mode bypassPermissions 2>&1 || true
    fi
    DEATHS=$((DEATHS + 1))
    echo "[v5] Agent exited (death #$DEATHS/$MAX_DEATHS). Sleeping 30s..."
    sleep 30
done
echo "[v5] Done. Best: $(cat "$BEST_FILE" 2>/dev/null || echo N/A)"
