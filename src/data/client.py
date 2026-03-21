"""Unified data client for Cerberus — REST + WebSocket streaming.

Replaces CentralApiClient with a cleaner interface that talks to Data-Gateway.
REST methods provide historical/snapshot data; WebSocket methods provide real-time streaming.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

import httpx

from src.core.domain import Bar
from src.core.http_client import create_http_client
from src.core.logger import StructuredLogger


@dataclass
class StreamQuote:
    """Normalized quote from the Data-Gateway WebSocket stream."""

    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: datetime


@dataclass
class StreamTrade:
    """Normalized trade from the Data-Gateway WebSocket stream."""

    symbol: str
    price: float
    size: float
    timestamp: datetime
    conditions: list[str] = field(default_factory=list)


FEED_NAME_MAP = {
    "bars": "stock_bars",
    "quotes": "stock_quotes",
    "trades": "stock_trades",
}
REVERSE_FEED_MAP = {v: k for k, v in FEED_NAME_MAP.items()}

logger = StructuredLogger("unified_data_client")


class UnifiedDataClient:
    """REST client for Data-Gateway endpoints with retry and normalization."""

    def __init__(self, gateway_url: str, gateway_key: str, timeout: float = 30.0) -> None:
        self.gateway_url = gateway_url
        self.gateway_key = gateway_key
        self._timeout = timeout

        # Compute WebSocket URL (http→ws, https→wss, append /ws)
        ws_base = gateway_url.rstrip("/")
        if ws_base.startswith("https://"):
            ws_base = "wss://" + ws_base[len("https://") :]
        elif ws_base.startswith("http://"):
            ws_base = "ws://" + ws_base[len("http://") :]
        self._ws_url = ws_base + "/ws"

        headers: Dict[str, str] = {}
        if gateway_key:
            headers["X-Gateway-Key"] = gateway_key

        self.client = create_http_client(
            base_url=gateway_url,
            timeout=timeout,
            headers=headers,
        )

        self._max_retries = 3
        self._retry_backoff_base = 1.0

        # WebSocket state
        self._ws = None
        self._running = False
        self._subscribed_symbols: set[str] = set()
        self._subscribed_feeds: set[str] = set()

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """Return True when an HTTP status should be retried."""
        return status_code == 429 or status_code >= 500

    def _get_retry_delay_seconds(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None,
    ) -> float:
        """Calculate delay before next retry attempt."""
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return self._retry_backoff_base * (2 ** max(0, attempt - 1))

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry on transient failures."""
        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.request(method, path, params=params, json=json)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= max_attempts:
                    raise
                delay = self._get_retry_delay_seconds(attempt)
                logger.warning(
                    "Gateway request retrying after transport error",
                    method=method,
                    path=path,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=delay,
                    error=str(exc),
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            if response.status_code < 400:
                return response

            if response.status_code in {401, 403}:
                response.raise_for_status()

            if self._is_retryable_status(response.status_code) and attempt < max_attempts:
                delay = self._get_retry_delay_seconds(attempt, response=response)
                logger.warning(
                    "Gateway request retrying after HTTP error",
                    method=method,
                    path=path,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status_code=response.status_code,
                    delay_seconds=delay,
                )
                if delay > 0:
                    time.sleep(delay)
                continue

            response.raise_for_status()

        raise RuntimeError("Request retry loop exhausted unexpectedly")

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_bars_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Data-Gateway envelope into {bars: [...]} format."""
        if "bars" in payload:
            return payload
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("bars"), list):
            return {"bars": [self._normalize_bar(item) for item in data["bars"]]}
        return payload

    def _normalize_bar(self, item: Any) -> Dict[str, Any]:
        """Normalize a bar item into short-key format."""
        if not isinstance(item, dict):
            return {}
        return {
            "t": item.get("t") or item.get("timestamp"),
            "o": item.get("o") if item.get("o") is not None else item.get("open"),
            "h": item.get("h") if item.get("h") is not None else item.get("high"),
            "l": item.get("l") if item.get("l") is not None else item.get("low"),
            "c": item.get("c") if item.get("c") is not None else item.get("close"),
            "v": item.get("v") if item.get("v") is not None else item.get("volume"),
        }

    def _normalize_trades_response(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Data-Gateway trade response into list of trade dicts."""
        trades: List[Dict[str, Any]] = []
        raw = payload
        data = payload.get("data")
        if isinstance(data, dict):
            raw = cast(Dict[str, Any], data)

        raw_trades = raw.get("trades")
        if not isinstance(raw_trades, list):
            return trades

        for item in raw_trades:
            if not isinstance(item, dict):
                continue
            trades.append(
                {
                    "t": item.get("t") or item.get("timestamp"),
                    "p": item.get("p") if item.get("p") is not None else item.get("price", 0.0),
                    "s": item.get("s") if item.get("s") is not None else item.get("size", 0.0),
                    "c": item.get("c") if item.get("c") is not None else item.get("conditions", []),
                    "x": item.get("x") if item.get("x") is not None else item.get("exchange", ""),
                    "z": item.get("z") if item.get("z") is not None else item.get("tape", ""),
                }
            )
        return trades

    # ------------------------------------------------------------------
    # REST methods
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe: str = "1Min",
    ) -> Dict[str, Any]:
        """Fetch historical bars via Data-Gateway."""
        params: Dict[str, Any] = {"timeframe": timeframe}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/bars",
            params=params,
        )
        payload = cast(Dict[str, Any], response.json())
        return self._normalize_bars_response(payload)

    def get_trades(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Fetch historical trades via Data-Gateway."""
        params: Dict[str, Any] = {"limit": int(limit)}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/trades",
            params=params,
        )
        payload = cast(Dict[str, Any], response.json())
        return self._normalize_trades_response(payload)

    def get_quotes(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fetch historical quotes via Data-Gateway."""
        params: Dict[str, Any] = {}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/quotes",
            params=params or None,
        )
        return cast(Dict[str, Any], response.json())

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest snapshot for a symbol via Data-Gateway."""
        response = self._request_with_retry(
            "GET",
            f"/api/v1/alpaca/stocks/{symbol.upper()}/snapshot",
        )
        return cast(Dict[str, Any], response.json())

    def get_flow(self, symbol: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Fetch options flow via Data-Gateway UW endpoint."""
        params: Dict[str, Any] = {}
        if date_str:
            params["date"] = date_str

        response = self._request_with_retry(
            "GET",
            f"/api/v1/uw/flow/{symbol.upper()}",
            params=params or None,
        )
        return cast(Dict[str, Any], response.json())

    def get_gex(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch GEX data via Data-Gateway UW endpoint."""
        response = self._request_with_retry("GET", f"/api/v1/uw/gex/{symbol.upper()}")
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        if isinstance(data, list):
            return cast(List[Dict[str, Any]], data)
        if isinstance(data, dict):
            return cast(List[Dict[str, Any]], data.get("rows", []))
        return []

    def get_most_actives(self, top: int = 20) -> List[str]:
        """Fetch most active stock symbols via Data-Gateway screener."""
        response = self._request_with_retry(
            "GET",
            "/api/v1/alpaca/screener/most-actives",
            params={"by": "volume", "top": int(top)},
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data", {})
        rows = data.get("most_actives", []) if isinstance(data, dict) else []
        symbols: List[str] = []
        for row in rows:
            if isinstance(row, dict):
                sym = row.get("symbol")
                if sym:
                    symbols.append(str(sym).upper())
        return symbols

    def get_movers(self, top: int = 10) -> Dict[str, List[str]]:
        """Fetch top gainers/losers via Data-Gateway screener."""
        response = self._request_with_retry(
            "GET",
            "/api/v1/alpaca/screener/movers",
            params={"market_type": "stocks", "top": int(top)},
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return {"gainers": [], "losers": []}
        out: Dict[str, List[str]] = {"gainers": [], "losers": []}
        for key in ("gainers", "losers"):
            rows = data.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    sym = row.get("symbol")
                    if sym:
                        out[key].append(str(sym).upper())
        return out

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an order via Data-Gateway trading endpoint."""
        body: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force,
        }
        if qty is not None:
            body["qty"] = float(qty)
        if notional is not None:
            body["notional"] = float(notional)
        if limit_price is not None:
            body["limit_price"] = float(limit_price)
        if stop_price is not None:
            body["stop_price"] = float(stop_price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        response = self._request_with_retry(
            "POST",
            "/api/v1/alpaca/orders",
            json=body,
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        return cast(Dict[str, Any], data) if isinstance(data, dict) else payload

    def get_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch orders via Data-Gateway trading endpoint."""
        response = self._request_with_retry(
            "GET",
            "/api/v1/alpaca/orders",
            params={"status": status, "limit": int(limit)},
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        return cast(List[Dict[str, Any]], data) if isinstance(data, list) else []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via Data-Gateway trading endpoint."""
        response = self._request_with_retry(
            "DELETE",
            f"/api/v1/alpaca/orders/{order_id}",
        )
        payload = cast(Dict[str, Any], response.json())
        data = payload.get("data")
        if isinstance(data, dict) and "cancelled" in data:
            return bool(data.get("cancelled"))
        return bool(payload.get("success", True))

    def get_prior_day_stats(self, symbol: str, current_time: datetime) -> tuple[float, float, float]:
        """Fetch 1Day bars for last 7 days and return (high, low, close) of last complete day."""
        start = current_time - timedelta(days=7)
        result = self.get_historical_bars(symbol, start, current_time, timeframe="1Day")
        bars = result.get("bars", [])
        if not bars:
            return (0.0, 0.0, 0.0)

        # Determine whether the last bar belongs to today (potentially incomplete)
        today = current_time.date()
        last_bar = bars[-1]
        last_bar_ts = last_bar.get("t") or last_bar.get("timestamp")
        last_bar_date = None
        if last_bar_ts:
            if isinstance(last_bar_ts, str):
                from datetime import datetime as _dt

                try:
                    last_bar_date = _dt.fromisoformat(last_bar_ts.replace("Z", "+00:00")).date()
                except (ValueError, TypeError):
                    pass
            elif hasattr(last_bar_ts, "date"):
                last_bar_date = last_bar_ts.date()

        if last_bar_date == today and len(bars) >= 2:
            # Last bar is today (incomplete) — prior day is second-to-last
            prior = bars[-2]
        else:
            # Last bar is a prior complete day (called before market open or no today bar)
            prior = bars[-1]
        return (float(prior.get("h", 0)), float(prior.get("l", 0)), float(prior.get("c", 0)))

    def get_avg_daily_volume(self, symbol: str, end: datetime, lookback_days: int = 20) -> float:
        """Fetch 1Day bars and compute mean volume over lookback window."""
        start = end - timedelta(days=lookback_days + 5)  # buffer for weekends/holidays
        result = self.get_historical_bars(symbol, start, end, timeframe="1Day")
        bars = result.get("bars", [])
        if not bars:
            return 0.0
        volumes = [float(b.get("v", 0)) for b in bars[-lookback_days:] if isinstance(b, dict)]
        return sum(volumes) / len(volumes) if volumes else 0.0

    def close(self) -> None:
        """Close underlying HTTP client."""
        self.client.close()

    async def close_async(self) -> None:
        """Close both REST and WebSocket connections."""
        await self.disconnect()
        self.close()

    # ------------------------------------------------------------------
    # WebSocket: timestamp parsing & payload normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(value) -> datetime:
        """Parse a timestamp from various formats into a timezone-aware datetime."""
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise ValueError(f"Invalid timestamp: {value}")

    def _normalize_stream_bar(self, payload: dict) -> Bar:
        """Normalize a WebSocket bar payload into a Bar domain object."""
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in bar payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        return Bar(
            symbol=symbol,
            time=ts,
            open=float(payload.get("o") if payload.get("o") is not None else payload.get("open", 0.0)),
            high=float(payload.get("h") if payload.get("h") is not None else payload.get("high", 0.0)),
            low=float(payload.get("l") if payload.get("l") is not None else payload.get("low", 0.0)),
            close=float(payload.get("c") if payload.get("c") is not None else payload.get("close", 0.0)),
            volume=float(payload.get("v") if payload.get("v") is not None else payload.get("volume", 0.0)),
            vwap=payload.get("vw") if payload.get("vw") is not None else payload.get("vwap"),
            trade_count=payload.get("n") if payload.get("n") is not None else payload.get("trade_count"),
        )

    def _normalize_stream_quote(self, payload: dict) -> StreamQuote:
        """Normalize a WebSocket quote payload into a StreamQuote."""
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in quote payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        return StreamQuote(
            symbol=symbol,
            timestamp=ts,
            bid_price=float(payload.get("bp") if payload.get("bp") is not None else payload.get("bid_price", 0.0)),
            bid_size=float(payload.get("bs") if payload.get("bs") is not None else payload.get("bid_size", 0.0)),
            ask_price=float(payload.get("ap") if payload.get("ap") is not None else payload.get("ask_price", 0.0)),
            ask_size=float(payload.get("as") if payload.get("as") is not None else payload.get("ask_size", 0.0)),
        )

    def _normalize_stream_trade(self, payload: dict) -> StreamTrade:
        """Normalize a WebSocket trade payload into a StreamTrade."""
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in trade payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        raw_conditions = payload.get("c") or payload.get("conditions") or []
        conditions = list(raw_conditions) if isinstance(raw_conditions, (list, tuple)) else []
        return StreamTrade(
            symbol=symbol,
            timestamp=ts,
            price=float(payload.get("p") if payload.get("p") is not None else payload.get("price", 0.0)),
            size=float(payload.get("s") if payload.get("s") is not None else payload.get("size", 0.0)),
            conditions=conditions,
        )

    @staticmethod
    def _extract_data_payload(message: dict) -> tuple[str, dict | None]:
        """Extract feed name and data payload from a WebSocket message."""
        if message.get("type") != "data":
            return "", None
        feed = str(message.get("feed") or "")
        data = message.get("data")
        if isinstance(data, dict):
            return feed, data
        envelope = message.get("envelope")
        if isinstance(envelope, dict):
            payload = envelope.get("payload")
            if isinstance(payload, dict):
                return feed, payload
        return feed, None

    # ------------------------------------------------------------------
    # WebSocket: connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to gateway WebSocket and authenticate."""
        import websockets

        self._ws = await websockets.connect(self._ws_url)
        # Auth
        await self._ws.send(json.dumps({"action": "auth", "key": self.gateway_key}))
        raw = await self._ws.recv()
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("type") != "auth_result" or msg.get("status") != "ok":
            await self._ws.close()
            self._ws = None
            raise RuntimeError(f"Gateway WebSocket auth failed: {msg}")

    async def disconnect(self) -> None:
        """Gracefully close WebSocket connection."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._subscribed_symbols.clear()
        self._subscribed_feeds.clear()

    async def subscribe(self, feeds: list[str], symbols: list[str]) -> None:
        """Subscribe to feeds for symbols. Maps short feed names to gateway feed names."""
        if not self._ws or not feeds or not symbols:
            return
        mapped_feeds = [FEED_NAME_MAP.get(f, f) for f in feeds]
        upper_symbols = [s.upper() for s in symbols]
        await self._ws.send(
            json.dumps(
                {
                    "action": "subscribe",
                    "feeds": mapped_feeds,
                    "symbols": upper_symbols,
                }
            )
        )
        self._subscribed_feeds.update(mapped_feeds)
        self._subscribed_symbols.update(upper_symbols)

    async def unsubscribe(self, feeds: list[str], symbols: list[str]) -> None:
        """Unsubscribe from feeds for symbols."""
        if not self._ws or not feeds or not symbols:
            return
        mapped_feeds = [FEED_NAME_MAP.get(f, f) for f in feeds]
        upper_symbols = [s.upper() for s in symbols]
        await self._ws.send(
            json.dumps(
                {
                    "action": "unsubscribe",
                    "feeds": mapped_feeds,
                    "symbols": upper_symbols,
                }
            )
        )
        self._subscribed_symbols -= set(upper_symbols)

    async def update_subscriptions(self, new_symbols: list[str]) -> None:
        """Diff current vs new symbols and send subscribe/unsubscribe deltas."""
        if not self._ws:
            return
        new_set = {s.upper() for s in new_symbols}
        current = self._subscribed_symbols
        to_add = new_set - current
        to_remove = current - new_set
        feeds_list = sorted(self._subscribed_feeds) if self._subscribed_feeds else ["stock_bars"]
        if to_remove:
            await self._ws.send(
                json.dumps(
                    {
                        "action": "unsubscribe",
                        "feeds": feeds_list,
                        "symbols": sorted(to_remove),
                    }
                )
            )
        if to_add:
            await self._ws.send(
                json.dumps(
                    {
                        "action": "subscribe",
                        "feeds": feeds_list,
                        "symbols": sorted(to_add),
                    }
                )
            )
        self._subscribed_symbols = new_set

    # ------------------------------------------------------------------
    # WebSocket: streaming loop
    # ------------------------------------------------------------------

    async def start_stream(
        self,
        on_bar=None,
        on_quote=None,
        on_trade=None,
        on_reconnect=None,
    ) -> None:
        """Main WebSocket recv loop with auto-reconnect."""

        self._running = True
        backoff = 1.0
        consecutive_failures = 0
        max_failures = 5

        while self._running:
            try:
                if self._ws is None:
                    await self.connect()
                    # Re-subscribe after reconnect
                    if self._subscribed_feeds and self._subscribed_symbols:
                        await self._ws.send(
                            json.dumps(
                                {
                                    "action": "subscribe",
                                    "feeds": sorted(self._subscribed_feeds),
                                    "symbols": sorted(self._subscribed_symbols),
                                }
                            )
                        )
                    if on_reconnect and consecutive_failures > 0:
                        if asyncio.iscoroutinefunction(on_reconnect):
                            await on_reconnect()
                        else:
                            on_reconnect()

                backoff = 1.0
                consecutive_failures = 0

                while self._running:
                    raw = await self._ws.recv()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    messages = parsed if isinstance(parsed, list) else [parsed]
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        # Heartbeat pong
                        if msg.get("type") == "heartbeat":
                            await self._ws.send(json.dumps({"action": "heartbeat"}))
                            continue
                        # Data dispatch
                        feed, payload = self._extract_data_payload(msg)
                        if payload is None:
                            continue
                        await self._dispatch_event(feed, payload, on_bar, on_quote, on_trade)

            except asyncio.CancelledError:
                self._running = False
                raise
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    raise RuntimeError(f"WebSocket stream failed after {max_failures} consecutive failures") from exc
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _dispatch_event(self, feed, payload, on_bar, on_quote, on_trade) -> None:
        """Dispatch a parsed event to the appropriate callback."""
        # Reverse-map gateway feed names (e.g. "stock_bars") to short names ("bars")
        feed = REVERSE_FEED_MAP.get(feed, feed)
        try:
            if feed == "bars" and on_bar:
                bar = self._normalize_stream_bar(payload)
                if asyncio.iscoroutinefunction(on_bar):
                    await on_bar(bar.symbol, bar)
                else:
                    on_bar(bar.symbol, bar)
            elif feed == "quotes" and on_quote:
                quote = self._normalize_stream_quote(payload)
                if asyncio.iscoroutinefunction(on_quote):
                    await on_quote(quote.symbol, quote)
                else:
                    on_quote(quote.symbol, quote)
            elif feed == "trades" and on_trade:
                trade = self._normalize_stream_trade(payload)
                if asyncio.iscoroutinefunction(on_trade):
                    await on_trade(trade.symbol, trade)
                else:
                    on_trade(trade.symbol, trade)
        except Exception:
            logger.error("dispatch_event_callback_failed", feed=feed, exc_info=True)
