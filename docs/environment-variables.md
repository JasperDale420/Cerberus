# Environment Variables

This document is the source of truth for environment variables used by Cerberus.

## Runtime Credentials

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `ALPACA_API_KEY` | Yes (or `APCA_API_KEY_ID`) | none | `src/data/alpaca.py`, `src/core/settings.py` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes (or `APCA_API_SECRET_KEY`) | none | `src/data/alpaca.py`, `src/core/settings.py` | Alpaca API secret |
| `ALPACA_BASE_URL` | No | none | `src/core/settings.py` | Alpaca REST base URL alias |
| `ALPACA_PAPER` | No | `true` | `src/data/alpaca.py`, `src/core/settings.py` | Paper/live flag |
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

## Data-Gateway and Heber Integration

| Variable | Required | Default | Used By | Description |
|---|---|---|---|---|
| `CERBERUS_DATA_BACKEND` | No | `gateway` | `src/core/settings.py`, `src/core/health.py` | Data read mode: `legacy`, `gateway`, `dual` |
| `CERBERUS_STORAGE_BACKEND` | No | `sqlite` | `src/core/settings.py` | Storage mode: `sqlite`, `heber`, `dual` |
| `CERBERUS_GATEWAY_URL` | No | `http://localhost:8080` | `src/core/settings.py`, `src/data/api_client.py`, `src/core/health.py` | Base URL for Data-Gateway |
| `CERBERUS_GATEWAY_KEY` | No | empty | `src/core/settings.py`, `src/data/api_client.py`, `src/core/health.py` | API key sent as `X-Gateway-Key` |
| `CERBERUS_GATEWAY_TIMEOUT_SECONDS` | No | `30` | `src/core/settings.py`, `src/data/api_client.py`, `src/core/health.py` | Gateway request timeout |
| `CERBERUS_HEBER_CATALOG_URL` | No | empty | `src/core/settings.py`, `src/core/health.py` | Heber catalog API base URL |
| `CERBERUS_HEBER_DATA_ROOT` | No | empty | `src/core/settings.py` | Optional local data root for Heber reads |
| `CERBERUS_DUAL_READ_COMPARE` | No | `false` | `src/core/settings.py` | Enable side-by-side read comparison in dual mode |
| `CERBERUS_FAILOVER_TO_LEGACY` | No | `true` | `src/core/settings.py` | Allow fallback to legacy data path on errors |

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
| `DATA_INGESTION_URL` | `src/data/api_client.py`, tests | Backward-compatible alias for gateway URL |
| `CENTRAL_LLM_API_URL` | `src/data/api_client.py` | Optional override for LLM `/v1/chat/completions` base URL |

## Related Files

- Template: [`../.env.example`](../.env.example)
- Config merge behavior: [`../src/core/config.py`](../src/core/config.py)
- Runtime settings aliases: [`../src/core/settings.py`](../src/core/settings.py)
