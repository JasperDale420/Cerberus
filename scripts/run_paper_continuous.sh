#!/usr/bin/env bash
set -euo pipefail

# Runs the paper-live harness continuously (long soak).
# Intended to be used by launchd or a process manager.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/artifacts/shakedown_config/config.yaml}"
SCENARIO="${SCENARIO:-happy}"

# Duration is in minutes; use a large value for "continuous" without code changes.
DURATION_MINUTES="${DURATION_MINUTES:-1440}" # 24h

INJECT_SIGNAL="${INJECT_SIGNAL:-false}"

ARGS=(paper_live_harness.py --config "$CONFIG_PATH" --scenario "$SCENARIO" --duration "$DURATION_MINUTES")
if [[ "$INJECT_SIGNAL" == "true" ]]; then
  ARGS+=(--inject-signal)
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
