# Unified Data Client for Forward Testing

**Date:** 2026-03-13
**Status:** Approved

## Problem

Cerberus has three separate data paths (direct Alpaca, Data-Gateway REST, Data-Gateway WebSocket) controlled by a `CERBERUS_DATA_BACKEND` setting with `legacy`/`gateway`/`dual` modes. This creates branching complexity in `main.py`, makes forward testing brittle, and means strategies implicitly depend on data that may or may not have been fetched depending on which mode is active.

## Solution

Replace all three data paths with a single `UnifiedDataClient` that talks exclusively to Data-Gateway. WebSocket for real-time streaming, REST for on-demand fetches. Strategies explicitly declare their data requirements.

## Architecture

### UnifiedDataClient (`src/data/client.py`)

Single class with two responsibilities:

**Real-time streaming (WebSocket):**
- Connect to `ws://{gateway_url}/ws`, authenticate, subscribe to feeds
- Dispatch incoming messages to registered callbacks as clean domain objects
- `update_subscriptions(symbols)` — diff-based subscribe/unsubscribe after scanner passes
- Auto-reconnect with exponential backoff (1s → 30s cap), fail loud after 5 consecutive failures
- Heartbeat pong on server heartbeats
- Auth failure = immediate startup crash

**On-demand fetches (REST):**
- `get_historical_bars(symbol, timeframe, start, end)` — via `/api/v1/alpaca/stocks/{symbol}/bars`
- `get_snapshot(symbol)` — via `/api/v1/alpaca/stocks/{symbol}/snapshot`
- `get_flow(symbol)` — via `/api/v1/uw/stock/{symbol}/flow-recent`
- `get_gex(symbol)` — via `/api/v1/uw/gex/{symbol}`
- `get_prior_day_stats(symbol)` — via `/api/v1/alpaca/stocks/{symbol}/bars` (1Day, limit=2)
- `get_avg_daily_volume(symbol, days)` — via bars endpoint aggregation
- Uses `empire_core.http_client` patterns (retry, structured errors)

### Strategy Data Requirements (`src/data/requirements.py`)

```python
@dataclass
class DataRequirements:
    streams: list[str]           # ["bars", "quotes", "trades"]
    on_scan: list[str]           # ["flow", "prior_day", "gex"]
    indicators: list[str]        # ["vwap", "bbands", "rsi", "atr"]
```

Each strategy declares `data_requirements` as a class attribute. The engine aggregates (unions) all requirements at startup to determine what feeds to subscribe and what to fetch on each scan.

### Simplified main.py Flow

```
create UnifiedDataClient
  → build engine + register strategies
  → aggregate data requirements
  → client.connect() + auth
  → initial scan → get watchlist
  → client.subscribe(feeds, symbols)
  → run session loop (WS dispatch + periodic rescan + flatten at 15:45)
  → client.disconnect()
```

No branching on data backend mode. One client, one path.

## Files Changed

### Removed
- `src/data/alpaca.py` — direct Alpaca client
- `src/data/gateway_stream.py` — old Gateway WS client
- `src/data/central_api.py` — old Gateway REST client

### New
- `src/data/client.py` — UnifiedDataClient
- `src/data/requirements.py` — DataRequirements dataclass + aggregation

### Modified
- `src/strategies/base.py` — add `data_requirements` class attribute with sensible defaults
- `src/strategies/*.py` — each strategy adds its `data_requirements`
- `src/main.py` — simplify startup (remove all backend branching)
- `src/core/settings.py` — remove `CERBERUS_DATA_BACKEND`, keep only `CERBERUS_GATEWAY_URL` + `CERBERUS_GATEWAY_KEY`
- `src/data/pipeline.py` — rewire FeaturePipeline to use UnifiedDataClient
- `src/scanner/universe.py` — rewire UniverseBuilder to use UnifiedDataClient

### Untouched
- `src/engine/execution.py` — event handlers unchanged (`on_bar`, `on_quote`, etc.)
- `src/engine/orders.py` — order executors unchanged
- `src/engine/position_manager.py` — position tracking unchanged
- `src/engine/risk.py` — risk checks unchanged
- `src/scanner/core.py` — scanner orchestration unchanged
- `src/backtest/` — backtest runner uses its own offline data
- All strategy logic (signals, indicators) — unchanged

## WebSocket Protocol

```
1. Connect: ws://{gateway_url}/ws
2. Auth:    {"action": "auth", "key": "gw_..."}
3. Ack:     {"type": "auth_result", "status": "ok"}
4. Sub:     {"action": "subscribe", "feeds": ["stock_bars", ...], "symbols": ["AAPL", ...]}
5. Data:    {"type": "data", "feed": "bars", "symbol": "AAPL", "envelope": {...}}
6. Heart:   {"type": "heartbeat"} → respond {"action": "heartbeat"}
7. Resub:   {"action": "subscribe"/"unsubscribe"} on watchlist changes
```

Feed names map: `bars` → `stock_bars`, `quotes` → `stock_quotes`, `trades` → `stock_trades`.

## Configuration

```bash
CERBERUS_GATEWAY_URL=http://localhost:8080   # required
CERBERUS_GATEWAY_KEY=gw_cerberus_key         # required
```

That's it. No more `CERBERUS_DATA_BACKEND`, `ALPACA_API_KEY` (for data — still needed if using Alpaca order executor), or dual-mode settings.
