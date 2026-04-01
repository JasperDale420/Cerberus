#!/usr/bin/env bash
# Wrapper for cerberus_autoresearch.py that captures exit codes, signals, and timing.
# Writes a status file that the agent can check after the eval.
#
# Usage: ./scripts/run_eval_with_logging.sh <strategy_name> [extra args...]
# Output: run.log (eval output), autoresearch/.eval_status (exit info)

set -uo pipefail
cd "$(dirname "$0")/.."

STRATEGY="${1:?Usage: run_eval_with_logging.sh <strategy_name> [args...]}"
shift
STATUS_FILE="autoresearch/.eval_status"
LOG_FILE="run.log"
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "EVAL_START strategy=$STRATEGY time=$START_ISO pid=$$" > "$STATUS_FILE"

# Run with timeout (80 min max) and capture exit code
timeout 4800 uv run python scripts/cerberus_autoresearch.py "$STRATEGY" "$@" > "$LOG_FILE" 2>&1
EXIT_CODE=$?

END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# Interpret exit code
case $EXIT_CODE in
    0)   EXIT_REASON="success" ;;
    1)   EXIT_REASON="python_error" ;;
    124) EXIT_REASON="timeout (exceeded 80 min)" ;;
    137) EXIT_REASON="killed (SIGKILL — likely OOM)" ;;
    139) EXIT_REASON="segfault (SIGSEGV)" ;;
    143) EXIT_REASON="terminated (SIGTERM — manual kill or system)" ;;
    *)   EXIT_REASON="unknown (exit code $EXIT_CODE)" ;;
esac

# Check if result line exists
HAS_RESULT="false"
if grep -q "^AUTORESEARCH_RESULT" "$LOG_FILE" 2>/dev/null; then
    HAS_RESULT="true"
fi

# Write status
cat > "$STATUS_FILE" <<EOF
EVAL_COMPLETE
strategy=$STRATEGY
exit_code=$EXIT_CODE
exit_reason=$EXIT_REASON
has_result=$HAS_RESULT
duration_minutes=$DURATION
start=$START_ISO
end=$END_ISO
log_file=$LOG_FILE
EOF

# Also append to a persistent death log for debugging
echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") strategy=$STRATEGY exit=$EXIT_CODE reason=$EXIT_REASON duration=${DURATION}m has_result=$HAS_RESULT" >> autoresearch/eval_history.log

# Print summary for the agent
echo ""
echo "EVAL_STATUS exit=$EXIT_CODE reason=$EXIT_REASON duration=${DURATION}m has_result=$HAS_RESULT"

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "EVAL_FAILED — last 30 lines of run.log:"
    tail -30 "$LOG_FILE"
fi
