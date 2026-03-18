#!/bin/bash
# Run WFO parameter optimization for all strategies (except flow_alpha)
# after the quant upgrade. Full 2024 data, rolling mode.
set -euo pipefail
cd "$(dirname "$0")/.."

STRATEGIES=(mean_reversion_pro trend_rider_pro orb_v2 rsi_bounce momentum_fade)
START="2024-01-02"
END="2024-12-31"
CONFIG="config/backtest_v2.yaml"
DATA_DIR="data/bars_2024"
TRIALS=50
RUN_TAG="quant_v1"

echo "=============================================="
echo "  CERBERUS QUANT WFO PARAMETER OPTIMIZATION"
echo "  Period: $START to $END"
echo "  Trials per window: $TRIALS"
echo "  Run tag: $RUN_TAG"
echo "=============================================="
echo ""

for strat in "${STRATEGIES[@]}"; do
    echo "=== Optimizing: $strat ==="
    echo "Started at: $(date)"
    PYTHONPATH=. uv run python scripts/optimize_strategy.py \
        --strategy "$strat" \
        --wfo \
        --wfo-mode rolling \
        --wfo-train-months 3 \
        --wfo-test-months 1 \
        --start "$START" \
        --end "$END" \
        --trials "$TRIALS" \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --run-tag "$RUN_TAG" \
        2>&1 | tee "artifacts/optimization/${strat}_wfo_quant_v1.log"
    echo ""
    echo "=== $strat COMPLETE at $(date) ==="
    echo ""
done

echo "=============================================="
echo "  ALL WFO OPTIMIZATIONS COMPLETE"
echo "  Results in: artifacts/optimization/"
echo "=============================================="
