#!/bin/bash
# Run WFO for both strategies sequentially
cd /Users/jacobmcmillan/Empire/Cerberus

echo "=== Starting trend_rider_pro WFO at $(date) ==="
uv run python scripts/run_wfo.py trend_rider_pro 2>&1 | grep -v '"level": "INFO"\|"level": "WARNING"\|"level": "ERROR"\|"level": "DEBUG"' | tee /tmp/trp_wfo_clean.log
echo "=== Finished trend_rider_pro WFO at $(date) ==="

echo ""
echo "=== Starting mean_reversion_pro WFO at $(date) ==="
uv run python scripts/run_wfo.py mean_reversion_pro 2>&1 | grep -v '"level": "INFO"\|"level": "WARNING"\|"level": "ERROR"\|"level": "DEBUG"' | tee /tmp/mrp_wfo_clean.log
echo "=== Finished mean_reversion_pro WFO at $(date) ==="
