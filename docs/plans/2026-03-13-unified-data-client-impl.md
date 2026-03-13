# Unified Data Client Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the legacy/gateway/dual data backend with a single `UnifiedDataClient` that talks exclusively to Data-Gateway, enabling reliable forward testing.

**Architecture:** A single `UnifiedDataClient` class owns both the WebSocket connection (real-time bars/quotes/trades) and REST calls (historical bars, flow, GEX, snapshots) to Data-Gateway. Strategies declare their data needs via a `DataRequirements` dataclass. The engine aggregates requirements and manages subscriptions. All Alpaca direct-data and dual-mode code is removed.

**Tech Stack:** Python 3.11+, websockets, httpx, asyncio, pydantic-settings, pytest

---

### Task 1: Create DataRequirements dataclass

**Files:**
- Create: `src/data/requirements.py`
- Test: `tests/data/test_requirements.py`

**Step 1: Write the failing test**

```python
# tests/data/test_requirements.py
from src.data.requirements import DataRequirements, aggregate_requirements


def test_default_requirements():
    r = DataRequirements()
    assert r.streams == ["bars"]
    assert r.on_scan == []
    assert r.indicators == []


def test_aggregate_unions_streams():
    r1 = DataRequirements(streams=["bars"], on_scan=["flow"])
    r2 = DataRequirements(streams=["bars", "quotes"], on_scan=["gex"])
    agg = aggregate_requirements([r1, r2])
    assert set(agg.streams) == {"bars", "quotes"}
    assert set(agg.on_scan) == {"flow", "gex"}


def test_aggregate_empty():
    agg = aggregate_requirements([])
    assert agg.streams == []
    assert agg.on_scan == []
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_requirements.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# src/data/requirements.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataRequirements:
    """Declares what data a strategy needs from the unified data client."""

    streams: list[str] = field(default_factory=lambda: ["bars"])
    on_scan: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)


def aggregate_requirements(reqs: list[DataRequirements]) -> DataRequirements:
    """Union all requirements from registered strategies."""
    streams: set[str] = set()
    on_scan: set[str] = set()
    indicators: set[str] = set()
    for r in reqs:
        streams.update(r.streams)
        on_scan.update(r.on_scan)
        indicators.update(r.indicators)
    return DataRequirements(
        streams=sorted(streams),
        on_scan=sorted(on_scan),
        indicators=sorted(indicators),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_requirements.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(data): add DataRequirements dataclass and aggregation
```

---

### Task 2: Create UnifiedDataClient — REST methods

**Files:**
- Create: `src/data/client.py`
- Test: `tests/data/test_unified_client_rest.py`

**Step 1: Write the failing test**

```python
# tests/data/test_unified_client_rest.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.data.client import UnifiedDataClient


@pytest.fixture
def client():
    return UnifiedDataClient(
        gateway_url="http://localhost:8080",
        gateway_key="gw_test_key",
    )


def test_client_initializes():
    c = UnifiedDataClient(gateway_url="http://test:8080", gateway_key="key")
    assert c.gateway_url == "http://test:8080"
    assert c.gateway_key == "key"


def test_ws_url_construction():
    c = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="k")
    assert c._ws_url == "ws://localhost:8080/ws"

    c2 = UnifiedDataClient(gateway_url="https://gw.example.com", gateway_key="k")
    assert c2._ws_url == "wss://gw.example.com/ws"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_unified_client_rest.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

Create `src/data/client.py` with:
- Class `UnifiedDataClient` with `__init__(self, gateway_url, gateway_key, timeout=30.0)`
- Internal `httpx.Client` with `X-Gateway-Key` header and retry logic (reuse pattern from `CentralApiClient._request_with_retry`)
- `_ws_url` property that converts http→ws, https→wss, appends `/ws`
- REST methods that wrap Data-Gateway endpoints:
  - `get_historical_bars(symbol, start, end, timeframe)` → `GET /api/v1/alpaca/stocks/{symbol}/bars`
  - `get_snapshot(symbol)` → `GET /api/v1/alpaca/stocks/{symbol}/snapshot`
  - `get_flow(symbol, date_str)` → `GET /api/v1/uw/flow/{symbol}`
  - `get_gex(symbol)` → `GET /api/v1/uw/gex/{symbol}`
  - `get_prior_day_stats(symbol)` → calls `get_historical_bars` with 1Day timeframe, extracts last complete day
  - `get_avg_daily_volume(symbol, days)` → calls `get_historical_bars` with 1Day, computes mean volume
  - `get_most_actives(top)` → `GET /api/v1/alpaca/screener/most-actives`
  - `get_movers(top)` → `GET /api/v1/alpaca/screener/movers`
  - `submit_order(...)` → `POST /api/v1/alpaca/orders` (forward to existing gateway order API)
  - `get_orders(...)` → `GET /api/v1/alpaca/orders`
  - `cancel_order(order_id)` → `DELETE /api/v1/alpaca/orders/{order_id}`
- `close()` method to close httpx client
- Response normalization: bars responses normalized to `{"bars": [{"t":..,"o":..,"h":..,"l":..,"c":..,"v":..}]}`

Port the retry logic from `CentralApiClient._request_with_retry` (retry on 429/5xx, exponential backoff, respect Retry-After header). Port normalization from `CentralApiClient._normalize_bars_response` and `_normalize_trades_response`.

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_unified_client_rest.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(data): add UnifiedDataClient REST methods
```

---

### Task 3: Create UnifiedDataClient — WebSocket streaming

**Files:**
- Modify: `src/data/client.py`
- Test: `tests/data/test_unified_client_ws.py`

**Step 1: Write the failing test**

```python
# tests/data/test_unified_client_ws.py
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.client import UnifiedDataClient
from src.core.domain import Bar


@pytest.mark.asyncio
async def test_connect_authenticates():
    """Test that connect sends auth and waits for ack."""
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="gw_test")

    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({"type": "auth_result", "status": "ok"}))

    with patch("websockets.connect", return_value=_async_ctx(mock_ws)):
        await client.connect()
        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["action"] == "auth"
        assert sent["key"] == "gw_test"


@pytest.mark.asyncio
async def test_subscribe_sends_correct_message():
    """Test subscription message format."""
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="gw_test")
    client._ws = AsyncMock()

    await client.subscribe(feeds=["bars", "quotes"], symbols=["AAPL", "MSFT"])
    client._ws.send.assert_called_once()
    sent = json.loads(client._ws.send.call_args[0][0])
    assert sent["action"] == "subscribe"
    assert set(sent["feeds"]) == {"stock_bars", "stock_quotes"}
    assert set(sent["symbols"]) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_update_subscriptions_sends_delta():
    """Test that update_subscriptions sends only add/remove deltas."""
    client = UnifiedDataClient(gateway_url="http://localhost:8080", gateway_key="gw_test")
    client._ws = AsyncMock()
    client._subscribed_symbols = {"AAPL", "MSFT"}
    client._subscribed_feeds = {"stock_bars"}

    await client.update_subscriptions(["MSFT", "TSLA"])
    # Should unsubscribe AAPL and subscribe TSLA
    calls = client._ws.send.call_args_list
    assert len(calls) == 2  # one unsub, one sub


def _async_ctx(mock_ws):
    """Helper to create async context manager from mock."""
    class _Ctx:
        async def __aenter__(self):
            return mock_ws
        async def __aexit__(self, *args):
            pass
    return _Ctx()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_unified_client_ws.py -v`
Expected: FAIL

**Step 3: Add WebSocket methods to UnifiedDataClient**

Add to `src/data/client.py`:
- `async connect()` — open WS, send auth, verify ack. Raise on failure.
- `async disconnect()` — close WS gracefully
- `async subscribe(feeds, symbols)` — send subscribe with feed name mapping (`bars`→`stock_bars`, `quotes`→`stock_quotes`, `trades`→`stock_trades`)
- `async unsubscribe(feeds, symbols)` — send unsubscribe
- `async update_subscriptions(new_symbols)` — diff `_subscribed_symbols` vs new set, send sub/unsub deltas
- `async start_stream(on_bar, on_quote=None, on_trade=None, on_reconnect=None)` — main recv loop:
  - Parse messages, route by feed type
  - Normalize payloads to domain objects (`Bar`, `StreamQuote`, `StreamTrade`) using existing normalizers from `gateway_stream.py`
  - Auto-reconnect with exponential backoff (1s→30s), max 5 consecutive failures before raising
  - Heartbeat pong on `{"type": "heartbeat"}`
  - Call `on_reconnect` after successful reconnection
- Internal state: `_ws`, `_running`, `_subscribed_symbols: set[str]`, `_subscribed_feeds: set[str]`
- Move `StreamQuote` and `StreamTrade` dataclasses from `gateway_stream.py` into `client.py`

Port normalization logic from `GatewayStreamClient._normalize_bar_from_data`, `_normalize_quote_from_data`, `_normalize_trade_from_data`, `_extract_data_payload`, `_parse_timestamp`.

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_unified_client_ws.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(data): add UnifiedDataClient WebSocket streaming
```

---

### Task 4: Add data_requirements to BaseStrategy and all strategies

**Files:**
- Modify: `src/strategies/base.py`
- Modify: `src/strategies/mean_reversion_pro.py`
- Modify: `src/strategies/trend_rider_pro.py`
- Modify: `src/strategies/flow_alpha.py`
- Modify: `src/strategies/orb_v2.py`
- Modify: `src/strategies/pair_trading_v2.py`
- Modify: All other strategy files in `src/strategies/`
- Test: `tests/strategies/test_data_requirements.py`

**Step 1: Write the failing test**

```python
# tests/strategies/test_data_requirements.py
from src.data.requirements import DataRequirements
from src.strategies.base import BaseStrategy


def test_base_strategy_has_default_requirements():
    """All strategies should declare data_requirements."""
    assert hasattr(BaseStrategy, "data_requirements")
    assert isinstance(BaseStrategy.data_requirements, DataRequirements)


def test_flow_alpha_needs_flow():
    from src.strategies.flow_alpha import FlowAlphaStrategy
    assert "flow" in FlowAlphaStrategy.data_requirements.on_scan


def test_all_registered_strategies_have_requirements():
    """Every strategy in the registry declares data_requirements."""
    from src.main import _build_strategy_registry
    registry = _build_strategy_registry()
    for name, cls in registry.items():
        assert hasattr(cls, "data_requirements"), f"{name} missing data_requirements"
        assert isinstance(cls.data_requirements, DataRequirements), f"{name} has wrong type"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/strategies/test_data_requirements.py -v`
Expected: FAIL

**Step 3: Add data_requirements**

In `src/strategies/base.py`, add to `BaseStrategy`:
```python
from src.data.requirements import DataRequirements

class BaseStrategy(ABC):
    name: str = "base"
    data_requirements: DataRequirements = DataRequirements()  # default: bars stream only
```

Then add overrides per strategy:

| Strategy | streams | on_scan |
|----------|---------|---------|
| `mean_reversion_pro` | `["bars", "quotes"]` | `["flow", "prior_day"]` |
| `trend_rider_pro` | `["bars", "quotes"]` | `["prior_day"]` |
| `flow_alpha` | `["bars", "quotes"]` | `["flow", "gex"]` |
| `orb_v2` | `["bars"]` | `["prior_day"]` |
| `pair_trading_v2` | `["bars"]` | `[]` |
| `vwap_reversion` | `["bars", "quotes"]` | `["flow"]` |
| `orb` | `["bars"]` | `["prior_day"]` |
| `vwap_trend_rider` | `["bars"]` | `["flow"]` |
| `index_mean_reversion` | `["bars"]` | `[]` |
| `flow_momentum` | `["bars"]` | `["flow", "gex"]` |
| `gap_fill` | `["bars"]` | `["prior_day"]` |
| `vix_spike_fade` | `["bars"]` | `[]` |
| `momentum_continuation` | `["bars"]` | `[]` |
| `fusion_v1` | `["bars", "quotes"]` | `["flow", "gex"]` |
| `pair_trading` | `["bars"]` | `[]` |
| `trend_pullback` | `["bars"]` | `[]` |
| `failed_breakout` | `["bars"]` | `["prior_day"]` |
| `order_flow_imbalance` | `["bars", "quotes", "trades"]` | `["flow"]` |
| `intraday_momentum` | `["bars"]` | `[]` |

For each strategy file, add a single line at the class level:
```python
data_requirements = DataRequirements(streams=["bars", "quotes"], on_scan=["flow"])
```

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/strategies/test_data_requirements.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(strategies): declare data_requirements on all strategies
```

---

### Task 5: Rewire DataFetcher to use UnifiedDataClient

**Files:**
- Modify: `src/data/fetcher.py`
- Test: `tests/data/test_fetcher_unified.py`

**Step 1: Write the failing test**

```python
# tests/data/test_fetcher_unified.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from src.data.fetcher import DataFetcher
from src.data.client import UnifiedDataClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=UnifiedDataClient)
    client.get_historical_bars = MagicMock(return_value={"bars": [
        {"t": "2026-03-13T10:00:00Z", "o": 150.0, "h": 151.0, "l": 149.0, "c": 150.5, "v": 1000}
    ]})
    return client


def test_fetcher_uses_unified_client(mock_client):
    logger = MagicMock()
    fetcher = DataFetcher(unified_client=mock_client, logger=logger)
    assert fetcher.unified_client is mock_client
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_fetcher_unified.py -v`
Expected: FAIL

**Step 3: Rewrite DataFetcher**

Gut `src/data/fetcher.py`:
- Remove: `AlpacaClient`, `CentralApiClient`, `HeberReadClient` imports and all dual/legacy/failover logic
- Remove: `_compare_bars_with_legacy`, `_compare_trades_with_legacy`, `_compare_quotes_with_legacy`, `_compare_flow_with_legacy`, `_compare_gex_with_legacy`, `_compare_bar_values`
- Remove: `use_gateway_data`, `use_heber_storage`, `allow_legacy_failover`, `enable_dual_compare`, `heber_client` fields
- New `__init__`: `def __init__(self, unified_client: UnifiedDataClient, unusual_whales_client, logger, config=None, clock=None)`
- `_get_historical_bars_sync` → delegates to `self.unified_client.get_historical_bars()`
- `fetch_bars` → same cache logic, but calls `unified_client.get_historical_bars()`
- `fetch_trades` → calls `unified_client.get_trades()` (add to UnifiedDataClient if missing)
- `fetch_flow` → calls `unified_client.get_flow()`
- `fetch_gex` → calls `unified_client.get_gex()`
- `fetch_avg_daily_volume` → calls `unified_client.get_avg_daily_volume()`
- `fetch_prior_day_stats` → calls `unified_client.get_prior_day_stats()`
- Keep: LRU cache, `_resolve_fetch_start`, `_extract_volume`, `_parse_bar_time`, `_get_bar_field`

**Step 4: Run tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/data/test_fetcher_unified.py -v`
Expected: PASS

**Step 5: Commit**

```
refactor(data): rewire DataFetcher to use UnifiedDataClient
```

---

### Task 6: Rewire FeaturePipeline and UniverseBuilder

**Files:**
- Modify: `src/data/pipeline.py`
- Modify: `src/scanner/universe.py`

**Step 1: Update FeaturePipeline**

Change `__init__` signature:
```python
def __init__(
    self,
    unified_client: UnifiedDataClient,
    unusual_whales_client: UnusualWhalesClient,
    logger: StructuredLogger,
    config=None,
    clock=None,
    ...
):
```
- Remove `alpaca_client` and `central_api_client` params
- Create `DataFetcher` with `unified_client` instead
- Remove all imports of `AlpacaClient`, `CentralApiClient`

**Step 2: Update UniverseBuilder**

Change `__init__` signature:
```python
def __init__(
    self,
    unified_client: UnifiedDataClient,
    logger: StructuredLogger,
    config=None,
    ...
):
```
- Remove `alpaca_client`, `central_api_client`, `config_loader` params
- Remove `use_gateway_data`, `allow_legacy_failover` fields and all backend switching
- `_get_historical_bars` → `self.unified_client.get_historical_bars()`
- `_get_screener_most_actives` → `self.unified_client.get_most_actives()`
- `_get_screener_movers` → `self.unified_client.get_movers()`

**Step 3: Run existing tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/scanner/ tests/data/test_pipeline*.py -v --timeout=30`
Fix any failures from changed signatures (update test fixtures to pass `unified_client` mock instead of `alpaca_client` + `central_api_client`).

**Step 4: Commit**

```
refactor(data): rewire FeaturePipeline and UniverseBuilder to UnifiedDataClient
```

---

### Task 7: Simplify main.py

**Files:**
- Modify: `src/main.py`

**Step 1: Rewrite startup**

Replace the entire component initialization section (lines ~338-522) with:

```python
# 2. Components — single unified data client
from src.data.client import UnifiedDataClient
from src.data.requirements import aggregate_requirements

unified_client = UnifiedDataClient(
    gateway_url=runtime_settings.cerberus_gateway_url,
    gateway_key=runtime_settings.cerberus_gateway_key,
    timeout=runtime_settings.cerberus_gateway_timeout_seconds,
)

# Unusual Whales Client (still separate for UW-specific flow enrichment)
uw_client = UnusualWhalesClient(config_loader, logger, config=config)

feature_pipeline = FeaturePipeline(
    unified_client,
    uw_client,
    logger,
    config=config,
    clock=clock,
)

universe_builder = UniverseBuilder(
    unified_client,
    logger,
    config=config,
    config_path_or_dir=args.config,
    clock=clock,
)
scanner = Scanner(universe_builder, feature_pipeline, logger, config=config)

# Database
db = DatabaseDatabase(config_loader, logger, config=config, config_path_or_dir=args.config)
db.init_db()

# Engine
engine = ExecutionEngine(config, logger, db, clock=clock)
engine.scanner = scanner

# Order executor
if args.order_executor == "gateway":
    from src.engine.orders import GatewayOrderExecutor
    engine.order_executor = GatewayOrderExecutor(unified_client, logger, db=db, clock=clock)
elif args.order_executor == "noop":
    from src.engine.orders import NoopOrderExecutor
    engine.order_executor = NoopOrderExecutor(logger, db=db, clock=clock)

# Register strategies
strategy_registry = _build_strategy_registry()
strategies_cfg = config.get("strategies", {})
for name in sorted(strategies_cfg.keys()):
    strat_cfg = strategies_cfg.get(name)
    if not isinstance(strat_cfg, dict) or not bool(strat_cfg.get("enabled", True)):
        continue
    cls = strategy_registry.get(str(name))
    if cls is None:
        continue
    engine.register_strategy(cls(strat_cfg, logger))

# Aggregate data requirements from all registered strategies
required = aggregate_requirements([s.data_requirements for s in engine.strategies])

# Start streams
await unified_client.connect()
stream_task = asyncio.create_task(
    unified_client.start_stream(
        on_bar=engine.on_bar,
        on_quote=engine.on_quote,
        on_trade=engine.on_trade_data,
        on_reconnect=engine.reconcile_broker_state,
    )
)

# Initial scan + subscribe
await engine.run_scan()
watchlist_symbols = [w.symbol for w in (engine.last_scan_result.watchlist if engine.last_scan_result else [])]
watchlist_symbols.append(config.get("index_symbol", "SPY"))
await unified_client.subscribe(feeds=required.streams, symbols=watchlist_symbols)

reconcile_task = asyncio.create_task(engine.reconcile_loop())
```

**Step 2: Remove dead functions and imports**

Remove from `main.py`:
- `_should_initialize_alpaca_client()`
- `_should_start_alpaca_stream()`
- `_capture_screener_snapshot()` (was Alpaca-only; replace with Gateway REST call if needed)
- `_start_gateway_stream_task()` and `_restart_gateway_stream()` (replaced by `unified_client.start_stream`)
- Imports: `AlpacaClient`, `CentralApiClient`, `GatewayStreamClient`
- All variables: `alpaca_client`, `central_api_client`, `gateway_stream_client`, `gateway_stream_task`, `alpaca_stream_task`, `trade_stream_task`

**Step 3: Update main loop scanner integration**

After `engine.run_scan()` in the main loop, add subscription update:
```python
await engine.run_scan()
# Update subscriptions to match new watchlist
if engine.last_scan_result:
    new_symbols = [w.symbol for w in engine.last_scan_result.watchlist]
    new_symbols.append(config.get("index_symbol", "SPY"))
    await unified_client.update_subscriptions(new_symbols)
```

**Step 4: Update finally block**

```python
finally:
    if stream_task is not None and not stream_task.done():
        stream_task.cancel()
    if not reconcile_task.done():
        reconcile_task.cancel()
    await unified_client.disconnect()
    await uw_client.close()
```

**Step 5: Commit**

```
refactor(main): simplify startup to use UnifiedDataClient exclusively
```

---

### Task 8: Simplify Settings

**Files:**
- Modify: `src/core/settings.py`

**Step 1: Remove dead settings**

Remove from `Settings`:
- `cerberus_data_backend` field
- `cerberus_storage_backend` field
- `cerberus_dual_read_compare` field
- `cerberus_failover_to_legacy` field
- `use_gateway_data` property
- `use_heber_storage` property
- The legacy/dual validation logic from `validate_startup_mode()`

Keep:
- `cerberus_gateway_url` (required)
- `cerberus_gateway_key` (required)
- `cerberus_gateway_timeout_seconds`
- `cerberus_asset_class`
- `cerberus_heber_catalog_url` and `cerberus_heber_data_root` (for future Heber integration)
- All Alpaca credential fields (still needed for Alpaca order executor)
- `validate_runtime_execution_requirements()` (still validates Alpaca creds for alpaca executor)

Update `validate_startup_mode()`:
```python
def validate_startup_mode(self) -> list[str]:
    errors: list[str] = []
    if not self.cerberus_gateway_key:
        errors.append("CERBERUS_GATEWAY_KEY is required")
    return errors
```

**Step 2: Run all settings tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/ -k "settings" -v`
Update any tests that reference removed fields.

**Step 3: Commit**

```
refactor(settings): remove legacy/dual/gateway backend mode settings
```

---

### Task 9: Delete old data clients

**Files:**
- Delete: `src/data/alpaca.py`
- Delete: `src/data/gateway_stream.py`
- Delete: `src/data/api_client.py`

**Step 1: Search for remaining imports**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && grep -rn "from src.data.alpaca import\|from src.data.gateway_stream import\|from src.data.api_client import" src/ tests/`

Fix any remaining references:
- If in test files: update to use `UnifiedDataClient` mock
- If in `src/engine/orders.py` (`GatewayOrderExecutor` uses `CentralApiClient`): rewire to use `UnifiedDataClient`
- If in `src/engine/execution.py`: remove `gateway_client` param, engine doesn't need direct stream access anymore

**Step 2: Delete the files**

```bash
rm src/data/alpaca.py src/data/gateway_stream.py src/data/api_client.py
```

**Step 3: Run full test suite**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/ -x --timeout=30`
Fix any remaining import errors.

**Step 4: Commit**

```
refactor(data): remove AlpacaClient, GatewayStreamClient, CentralApiClient
```

---

### Task 10: Update GatewayOrderExecutor

**Files:**
- Modify: `src/engine/orders.py`

**Step 1: Check current GatewayOrderExecutor**

It currently takes `CentralApiClient`. Change it to take `UnifiedDataClient`:
- `submit_order()` → calls `unified_client.submit_order()`
- `get_orders()` → calls `unified_client.get_orders()`
- `cancel_order()` → calls `unified_client.cancel_order()`

**Step 2: Update and test**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/engine/ -v --timeout=30`

**Step 3: Commit**

```
refactor(orders): wire GatewayOrderExecutor to UnifiedDataClient
```

---

### Task 11: Update ExecutionEngine

**Files:**
- Modify: `src/engine/execution.py`

**Step 1: Remove gateway_client param**

The engine no longer needs a direct reference to the stream client. Remove `gateway_client` from `__init__`. The stream dispatches events via callbacks — the engine just receives `on_bar(symbol, bar)` etc.

If the engine calls `gateway_client.subscribe(symbol)` or `gateway_client.unsubscribe(symbol)` anywhere, replace with a callback pattern: the engine emits a "subscription changed" event and `main.py` handles calling `unified_client.update_subscriptions()`.

**Step 2: Run engine tests**

Run: `cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/engine/ -v --timeout=30`

**Step 3: Commit**

```
refactor(engine): remove gateway_client dependency from ExecutionEngine
```

---

### Task 12: Full integration test + lint

**Files:**
- Test: Run full suite

**Step 1: Run full test suite**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus && uv run pytest tests/ --timeout=60 -x
```

**Step 2: Run linter**

```bash
cd /Users/jacobmcmillan/Empire/Cerberus && ruff check src/ tests/ && ruff format --check src/ tests/
```

**Step 3: Fix any failures**

**Step 4: Commit**

```
chore: fix remaining lint and test issues after data client unification
```

---

### Task 13: Update CHANGELOG.md

**Files:**
- Modify: `CHANGELOG.md`

Add under `## [Unreleased]`:

```markdown
### Changed
- Replaced legacy/gateway/dual data backend modes with unified `UnifiedDataClient` that talks exclusively to Data-Gateway
- Simplified `main.py` startup — single data client, no backend branching
- Strategies now declare `data_requirements` for explicit data dependency tracking
- `DataFetcher` uses `UnifiedDataClient` instead of direct Alpaca/Gateway clients

### Removed
- `AlpacaClient` (`src/data/alpaca.py`) — all data now flows through Data-Gateway
- `GatewayStreamClient` (`src/data/gateway_stream.py`) — replaced by `UnifiedDataClient` WebSocket
- `CentralApiClient` (`src/data/api_client.py`) — replaced by `UnifiedDataClient` REST
- `CERBERUS_DATA_BACKEND` setting and all dual/legacy/failover logic
- Dual-read parity comparison code
```

**Commit:**

```
docs: update CHANGELOG for unified data client migration
```

---

## Execution Strategy

**Parallelizable tasks (can run as concurrent agents):**
- Task 1 (DataRequirements) — independent
- Task 2 (REST client) — independent
- Task 3 (WS client) — depends on Task 2
- Task 4 (strategy requirements) — depends on Task 1

**Sequential tasks (must run after parallel wave):**
- Task 5 (DataFetcher rewire) — depends on Task 2
- Task 6 (Pipeline + Universe rewire) — depends on Task 5
- Task 7 (main.py simplify) — depends on Tasks 2, 3, 4, 6
- Task 8 (settings) — depends on Task 7
- Task 9 (delete old files) — depends on Tasks 7, 10, 11
- Task 10 (orders rewire) — depends on Task 2
- Task 11 (engine cleanup) — depends on Task 7
- Task 12 (integration test) — depends on all
- Task 13 (changelog) — depends on all

**Recommended agent swarm waves:**

| Wave | Tasks | Agents |
|------|-------|--------|
| 1 | 1, 2, 4 | 3 parallel |
| 2 | 3, 5 | 2 parallel |
| 3 | 6, 10 | 2 parallel |
| 4 | 7, 8, 11 | 3 parallel |
| 5 | 9, 12, 13 | 1 sequential |
