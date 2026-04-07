#!/usr/bin/env bash
set -euo pipefail
cd /Users/jacobmcmillan/Empire/Cerberus
echo "$(date): Starting Cerberus live session"
exec uv run python -m src.main --mode paper --order-executor gateway
