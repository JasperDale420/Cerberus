# Environment Variables

This document is the source of truth for environment variables used by Cerberus.

## Runtime Credentials

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `ALPACA_API_KEY` | Yes (or `APCA_API_KEY_ID`) | none | `src/data/alpaca.py`, `src/core/settings.py` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes (or `APCA_API_SECRET_KEY`) | none | `src/data/alpaca.py`, `src/core/settings.py` | Alpaca API secret |
| `ALPACA_BASE_URL` | No | none | `src/core/settings.py` | Alpaca REST base URL alias |
| `ALPACA_PAPER` | No | `false` | `src/data/alpaca.py`, `src/core/settings.py` | Paper/live flag |
| `APCA_API_KEY_ID` | Yes (alias) | none | `src/core/settings.py` | Alpaca key alias |
| `APCA_API_SECRET_KEY` | Yes (alias) | none | `src/core/settings.py` | Alpaca secret alias |
| `APCA_API_BASE_URL` | No (alias) | none | `src/core/settings.py` | Alpaca base URL alias |

## Unusual Whales

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `UW_API_TOKEN` | No | empty | `src/data/unusual_whales.py` | Token for UW API requests |
| `UW_BASE_URL` | No | `https://api.unusualwhales.com` | `src/data/unusual_whales.py` | UW API base URL |
| `UNUSUAL_WHALES_FLOW_URL_TEMPLATE` | No | empty | `src/data/unusual_whales.py` | Optional custom flow endpoint template |

## Agent and Approval Flags

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `CERBERUS_STAGE3_APPROVED` | Conditional | empty | `src/agent/core.py`, `src/agent/stage3.py` | Approval gate for Stage 3 proposal/report operations |

## Config Override Pattern

Cerberus supports environment-driven config overrides through `APP_*` keys in `src/core/config.py`.

Pattern:
- `APP_<NESTED>_<KEY>=value`

Examples:
- `APP_LOG_LEVEL=DEBUG`
- `APP_RISK_MAX_DAILY_LOSS=1000`
- `APP_SCANNER_INTERVAL_MINUTES=1`

The loader attempts type coercion for booleans and numeric values.

## Test/Script Variables (Non-Core Runtime)

| Variable | Used By | Notes |
|---|---|---|
| `KILL_SWITCH` | `scripts/paper_live_test.py` | Script-level stop trigger |
| `PAPER_LIVE` | `scripts/paper_live_test.py` | Script-level mode flag |
| `DATA_INGESTION_URL` | tests | Used in tests only |

## Related Files

- Template: [`../.env.example`](../.env.example)
- Config merge behavior: [`../src/core/config.py`](../src/core/config.py)
- Runtime settings aliases: [`../src/core/settings.py`](../src/core/settings.py)
