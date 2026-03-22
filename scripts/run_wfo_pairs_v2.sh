#!/usr/bin/env bash
# Run Pair Trading V2 WFO with nohup so it survives terminal disconnects.
#
# Usage:
#   bash scripts/run_wfo_pairs_v2.sh
#
# Logs are written to artifacts/optimization/runs/pair_trading_v2/wfo_<timestamp>.log
# Monitor with: tail -f artifacts/optimization/runs/pair_trading_v2/wfo_*.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Create output directory
LOG_DIR="artifacts/optimization/runs/pair_trading_v2"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/wfo_${TIMESTAMP}.log"

echo "Starting Pair Trading V2 WFO..."
echo "  Log file: $LOG_FILE"
echo "  Monitor:  tail -f $LOG_FILE"
echo ""

nohup uv run python scripts/wfo_pairs_runner.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "  PID: $PID"
echo "  Process running in background. Safe to close terminal."
echo ""
echo "  To stop: kill $PID"
echo "  To check: ps -p $PID"
