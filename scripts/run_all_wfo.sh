#!/usr/bin/env bash
# Sequential WFO optimization for all 5 V2 strategies with param spaces.
# Runs strategies one at a time, each with --workers 3 to limit CPU usage.
# Uses 1m bars (required — strategies depend on MTF pipeline that assumes 1m input).
#
# Usage: nohup bash scripts/run_all_wfo.sh > artifacts/optimization/wfo_all.log 2>&1 &
#   or:  tmux new -d -s wfo 'bash scripts/run_all_wfo.sh'

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/optimization

WORKERS=3
TRIALS=40
START="2024-01-02"
END="2024-12-31"
DATA_DIR="data/bars_2024"
CONFIG="config/backtest_v2.yaml"

STRATEGIES=(
    "trend_rider_pro"
    "mean_reversion_pro"
    "orb_v2"
    "rsi_bounce"
    "momentum_fade"
)

echo "=========================================="
echo "  Cerberus WFO — ${#STRATEGIES[@]} V2 Strategies"
echo "  Started: $(date)"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Workers: $WORKERS"
echo "  Trials per window: $TRIALS"
echo "  Mode: rolling WFO"
echo "=========================================="

for i in "${!STRATEGIES[@]}"; do
    STRAT="${STRATEGIES[$i]}"
    NUM=$((i + 1))
    echo ""
    echo ">>> [$NUM/${#STRATEGIES[@]}] $STRAT"
    echo "    Started: $(date)"
    PYTHONPATH=. uv run python scripts/optimize_strategy.py \
        --strategy "$STRAT" \
        --wfo --wfo-mode rolling \
        --start "$START" --end "$END" \
        --trials "$TRIALS" \
        --workers "$WORKERS" \
        --data-dir "$DATA_DIR" \
        --config "$CONFIG"
    echo "    Finished: $(date)"
done

echo ""
echo "=========================================="
echo "  All WFO complete: $(date)"
echo "=========================================="
