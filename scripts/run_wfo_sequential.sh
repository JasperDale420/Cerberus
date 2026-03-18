#!/bin/bash
# Run WFO for trend_rider_pro and mean_reversion_pro sequentially
cd /Users/jacobmcmillan/Empire/Cerberus

echo "=== Starting trend_rider_pro WFO at $(date) ==="
uv run python scripts/run_wfo.py trend_rider_pro 2>&1
echo "=== Finished trend_rider_pro WFO at $(date) ==="

echo ""
echo "=== Starting mean_reversion_pro WFO at $(date) ==="
uv run python scripts/run_wfo.py mean_reversion_pro 2>&1
echo "=== Finished mean_reversion_pro WFO at $(date) ==="

echo ""
echo "=== Both WFO runs complete at $(date) ==="
