#!/usr/bin/env bash
set -euo pipefail

# Runs the real bot entrypoint (streaming Alpaca feed) continuously.
# Intended to be used by launchd or a process manager for auto-restart.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/config.yaml}"
MODE="${MODE:-paper}"                  # paper | live
ORDER_EXECUTOR="${ORDER_EXECUTOR:-alpaca}"  # alpaca | noop

# Optional toggles
RUN_AGENT="${RUN_AGENT:-false}" # true to run Stage 1 at startup
RUN_ONCE="${RUN_ONCE:-false}"   # true to scan once and exit

ARGS=(-m src.main --mode "$MODE" --order-executor "$ORDER_EXECUTOR" --config "$CONFIG_PATH")
if [[ "$RUN_AGENT" == "true" ]]; then
  ARGS+=(--run-agent)
fi
if [[ "$RUN_ONCE" == "true" ]]; then
  ARGS+=(--run-once)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
