#!/bin/bash
# Run all 5-year strategy isolation backtests sequentially
# Each takes ~8-10 minutes with performance optimizations

set -e

RESULTS_DIR="results"
DATA_DIR="data/offline_bars_5yr"
CONFIG_DIR="config/backtest_5yr"
START_DATE="2020-01-02"
END_DATE="2024-12-31"

mkdir -p "$RESULTS_DIR"

echo "=== 5-Year Strategy Isolation Backtests ==="
echo "Start: $(date)"
echo ""

# Array of strategies to test
STRATEGIES=(
    "orb"
    "vwap_reversion"
    "trend_pullback"
    "vwap_trend_rider"
    "failed_breakout"
    "index_mean_reversion"
    "gap_fill"
)

for strategy in "${STRATEGIES[@]}"; do
    if [ "$strategy" = "orb" ]; then
        CONFIG="$CONFIG_DIR/config.yaml"
    else
        CONFIG="$CONFIG_DIR/config_${strategy}.yaml"
    fi

    OUTPUT="$RESULTS_DIR/backtest_5yr_${strategy}.json"
    LOG="$RESULTS_DIR/backtest_5yr_${strategy}.log"

    if [ -f "$OUTPUT" ]; then
        echo "[$strategy] Skipping - already exists: $OUTPUT"
        continue
    fi

    echo "[$strategy] Starting backtest..."
    echo "  Config: $CONFIG"
    echo "  Output: $OUTPUT"

    time python scripts/run_backtest.py \
        --config "$CONFIG" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --offline-bars "$DATA_DIR" \
        --output "$OUTPUT" 2>&1 | tee "$LOG"

    echo "[$strategy] Complete"
    echo ""
done

echo "=== All Backtests Complete ==="
echo "End: $(date)"
echo ""
echo "Results:"
ls -la "$RESULTS_DIR"/backtest_5yr_*.json 2>/dev/null || echo "No results found"
