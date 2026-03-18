#!/bin/bash
# Run quant upgrade baseline backtests: all strategies together, then each individually.
# Usage: bash scripts/run_quant_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPORT_DIR="artifacts/backtest_reports"
CONFIG_DIR="config"
DATA_DIR="data/bars_2024"
START="2024-01-02"
END="2024-12-31"
BASE_CONFIG="config/backtest_v2.yaml"

mkdir -p "$REPORT_DIR"

STRATEGIES=(mean_reversion_pro trend_rider_pro flow_alpha orb_v2 rsi_bounce momentum_fade)

# Helper: create a temp config enabling only specific strategies
make_config() {
    local name="$1"
    shift
    local enabled_strategies=("$@")
    local out="$CONFIG_DIR/backtest_${name}.yaml"

    # Start with base config
    cp "$BASE_CONFIG" "$out"

    # Enable/disable strategies via yq-style sed replacements
    for strat in "${STRATEGIES[@]}"; do
        local is_enabled="false"
        for es in "${enabled_strategies[@]}"; do
            if [[ "$strat" == "$es" ]]; then
                is_enabled="true"
                break
            fi
        done
        # Replace 'enabled: true/false' under the strategy block
        python3 -c "
import yaml, sys
with open('$out') as f:
    cfg = yaml.safe_load(f)
cfg['strategies']['$strat']['enabled'] = ('$is_enabled' == 'true')
cfg['database_url'] = 'sqlite:///backtest_${name}.db'
with open('$out', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    done
    echo "$out"
}

echo "=============================================="
echo "  CERBERUS QUANT UPGRADE BASELINE BACKTESTS"
echo "  Period: $START to $END"
echo "=============================================="
echo ""

# --- Run 1: All strategies together ---
echo "[1/7] Running ALL strategies together..."
config_all=$(make_config "all_strategies" "${STRATEGIES[@]}")
rm -f backtest_all_strategies.db
PYTHONPATH=. uv run python src/backtest/runner.py \
    --start "$START" --end "$END" \
    --config "$config_all" \
    --data-dir "$DATA_DIR" 2>&1 | tee "$REPORT_DIR/baseline_all_log.txt"
echo ""
echo "--- ALL STRATEGIES COMPLETE ---"
echo ""

# --- Run 2-7: Each strategy individually ---
i=2
for strat in "${STRATEGIES[@]}"; do
    echo "[$i/7] Running $strat individually..."
    config_single=$(make_config "$strat" "$strat")
    rm -f "backtest_${strat}.db"
    PYTHONPATH=. uv run python src/backtest/runner.py \
        --start "$START" --end "$END" \
        --config "$config_single" \
        --data-dir "$DATA_DIR" 2>&1 | tee "$REPORT_DIR/baseline_${strat}_log.txt"
    echo ""
    echo "--- $strat COMPLETE ---"
    echo ""
    i=$((i + 1))
done

echo "=============================================="
echo "  ALL BACKTESTS COMPLETE"
echo "  Reports in: $REPORT_DIR/"
echo "=============================================="

# Cleanup temp configs
rm -f "$CONFIG_DIR"/backtest_all_strategies.yaml
for strat in "${STRATEGIES[@]}"; do
    rm -f "$CONFIG_DIR/backtest_${strat}.yaml"
done
