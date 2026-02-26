from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import websockets

from src.core.config import ConfigLoader
from src.core.domain import Bar
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


class GatewayStreamClient:
    """WebSocket client for Data-Gateway live streaming (bars, quotes, trades)."""

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        base_url = config_loader.get_env(
            "CERBERUS_GATEWAY_URL",
            config_loader.get_env("DATA_INGESTION_URL", "http://localhost:8080"),
        )
        self.gateway_key = config_loader.get_env("CERBERUS_GATEWAY_KEY", "")
        self.ws_url = self._build_ws_url(base_url)

        # Per-feed symbol tracking
        self._desired_bar_symbols: set[str] = set()
        self._desired_quote_symbols: set[str] = set()
        self._desired_trade_symbols: set[str] = set()

        self._ws: Any = None
        self._running = False

    # ── Backward-compatible property for existing code that reads _desired_symbols ──
    @property
    def _desired_symbols(self) -> set[str]:
        return self._desired_bar_symbols

    @staticmethod
    def _build_ws_url(base_url: str) -> str:
        parsed = urlparse(str(base_url).strip())
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        if path.endswith("/ws"):
            ws_path = path
        elif not path:
            ws_path = "/ws"
        else:
            ws_path = f"{path}/ws"
        return urlunparse((scheme, parsed.netloc, ws_path, "", "", ""))

    # ── Timestamp Parsing ──

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise ValueError(f"Invalid timestamp: {value}")

    # ── Payload Normalization ──

    def _normalize_bar_from_data(self, payload: dict[str, Any]) -> Bar:
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in gateway bar payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        return Bar(
            symbol=symbol,
            time=ts,
            open=float(payload.get("o") if payload.get("o") is not None else payload.get("open", 0.0)),
            high=float(payload.get("h") if payload.get("h") is not None else payload.get("high", 0.0)),
            low=float(payload.get("l") if payload.get("l") is not None else payload.get("low", 0.0)),
            close=float(payload.get("c") if payload.get("c") is not None else payload.get("close", 0.0)),
            volume=float(payload.get("v") if payload.get("v") is not None else payload.get("volume", 0.0)),
            vwap=payload.get("vw") or payload.get("vwap"),
            trade_count=payload.get("n") or payload.get("trade_count"),
        )

    def _normalize_quote_from_data(self, payload: dict[str, Any]) -> StreamQuote:
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in gateway quote payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        return StreamQuote(
            symbol=symbol,
            bid_price=float(payload.get("bp") if payload.get("bp") is not None else payload.get("bid_price", 0.0)),
            bid_size=float(payload.get("bs") if payload.get("bs") is not None else payload.get("bid_size", 0.0)),
            ask_price=float(payload.get("ap") if payload.get("ap") is not None else payload.get("ask_price", 0.0)),
            ask_size=float(payload.get("as") if payload.get("as") is not None else payload.get("ask_size", 0.0)),
            timestamp=ts,
        )

    def _normalize_trade_from_data(self, payload: dict[str, Any]) -> StreamTrade:
        symbol = str(payload.get("S") or payload.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Missing symbol in gateway trade payload")
        ts = self._parse_timestamp(payload.get("t") or payload.get("timestamp"))
        raw_conditions = payload.get("c") or payload.get("conditions") or []
        conditions = list(raw_conditions) if isinstance(raw_conditions, (list, tuple)) else []
        return StreamTrade(
            symbol=symbol,
            price=float(payload.get("p") if payload.get("p") is not None else payload.get("price", 0.0)),
            size=float(payload.get("s") if payload.get("s") is not None else payload.get("size", 0.0)),
            timestamp=ts,
            conditions=conditions,
        )

    # ── Message Extraction ──

    @staticmethod
    def _extract_data_payload(message: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Extract the feed type and payload from a gateway data message.

        Returns (feed_name, payload_dict) or ("", None) if not a data message.
        """
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

    # ── WebSocket Wire Protocol ──

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        await ws.send(json.dumps(payload))

    async def _send_feed_subscribe(self, feeds: list[str], symbols: list[str]) -> None:
        if not symbols or not feeds:
            return
        await self._send_json(
            {
                "action": "subscribe",
                "feeds": feeds,
                "symbols": symbols,
            }
        )

    async def _send_feed_unsubscribe(self, feeds: list[str], symbols: list[str]) -> None:
        if not symbols or not feeds:
            return
        await self._send_json(
            {
                "action": "unsubscribe",
                "feeds": feeds,
                "symbols": symbols,
            }
        )

    # ── Backward-compatible bar subscribe (used widely) ──

    async def _send_subscribe(self, symbols: list[str]) -> None:
        await self._send_feed_subscribe(["stock_bars"], symbols)

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        await self._send_feed_unsubscribe(["stock_bars"], symbols)

    # ── Public Subscribe/Unsubscribe API ──

    def _schedule_task(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(coro)

    def subscribe(self, symbol: str) -> None:
        """Subscribe to bar data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_bar_symbols.add(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_subscribe(["stock_bars"], [sym]))

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from bar data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_bar_symbols.discard(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_unsubscribe(["stock_bars"], [sym]))

    def subscribe_quotes(self, symbol: str) -> None:
        """Subscribe to quote data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_quote_symbols.add(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_subscribe(["stock_quotes"], [sym]))

    def unsubscribe_quotes(self, symbol: str) -> None:
        """Unsubscribe from quote data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_quote_symbols.discard(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_unsubscribe(["stock_quotes"], [sym]))

    def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_trade_symbols.add(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_subscribe(["stock_trades"], [sym]))

    def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trade data for a symbol."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        self._desired_trade_symbols.discard(sym)
        if self._ws is not None:
            self._schedule_task(self._send_feed_unsubscribe(["stock_trades"], [sym]))

    # ── Auth ──

    async def _auth(self) -> None:
        if not self.gateway_key:
            raise ValueError("CERBERUS_GATEWAY_KEY is required for Data-Gateway websocket streaming")
        await self._send_json({"action": "auth", "key": self.gateway_key})
        raw = await self._ws.recv()
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("type") != "auth_result" or msg.get("status") != "ok":
            raise RuntimeError(f"Gateway websocket auth failed: {msg}")

    # ── Callback Invocation ──

    @staticmethod
    async def _invoke_callback(callback: Callable[..., Any], *args: Any) -> None:
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)

    def _inspect_arity(self, callback: Callable[..., Any]) -> int:
        import inspect

        try:
            sig = inspect.signature(callback)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            return len([p for p in params if p.default is inspect.Parameter.empty])
        except Exception:
            return -1

    async def _invoke_bar_callback(self, callback: Callable[..., Any], bar: Bar, symbol: str, arity: int) -> None:
        if arity == 1:
            await self._invoke_callback(callback, bar)
        elif arity >= 2:
            await self._invoke_callback(callback, symbol, bar)
        else:
            try:
                await self._invoke_callback(callback, bar)
            except TypeError:
                await self._invoke_callback(callback, symbol, bar)

    # ── Message Routing ──

    async def _handle_message(
        self,
        bar_callback: Callable[..., Any],
        bar_arity: int,
        msg: dict[str, Any],
        on_quote: Optional[Callable[..., Any]] = None,
        on_trade: Optional[Callable[..., Any]] = None,
    ) -> None:
        feed, payload = self._extract_data_payload(msg)
        if payload is None:
            return

        try:
            if feed == "bars":
                bar = self._normalize_bar_from_data(payload)
                await self._invoke_bar_callback(bar_callback, bar, bar.symbol, bar_arity)
            elif feed == "quotes" and on_quote is not None:
                quote = self._normalize_quote_from_data(payload)
                await self._invoke_callback(on_quote, quote.symbol, quote)
            elif feed == "trades" and on_trade is not None:
                trade = self._normalize_trade_from_data(payload)
                await self._invoke_callback(on_trade, trade.symbol, trade)
        except Exception as e:
            self.logger.warning("Failed to process gateway stream message", feed=feed, error=str(e))

    # ── Send All Subscriptions ──

    async def _subscribe_all_feeds(self) -> None:
        if self._desired_bar_symbols:
            await self._send_feed_subscribe(["stock_bars"], sorted(self._desired_bar_symbols))
        if self._desired_quote_symbols:
            await self._send_feed_subscribe(["stock_quotes"], sorted(self._desired_quote_symbols))
        if self._desired_trade_symbols:
            await self._send_feed_subscribe(["stock_trades"], sorted(self._desired_trade_symbols))

    async def _handle_reconnect(self, on_reconnect: Optional[Callable[..., Any]]) -> None:
        if on_reconnect is None:
            return
        if asyncio.iscoroutinefunction(on_reconnect):
            await on_reconnect()
        else:
            on_reconnect()

    async def _process_raw_message(
        self,
        raw: str,
        bar_callback: Callable[..., Any],
        bar_arity: int,
        on_quote: Optional[Callable[..., Any]],
        on_trade: Optional[Callable[..., Any]],
    ) -> None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return
        messages = parsed if isinstance(parsed, list) else [parsed]
        for item in messages:
            if isinstance(item, dict):
                await self._handle_message(bar_callback, bar_arity, item, on_quote, on_trade)

    async def start_stream(
        self,
        callback: Callable[..., Any],
        on_reconnect: Optional[Callable[..., Any]] = None,
        on_quote: Optional[Callable[..., Any]] = None,
        on_trade: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Start the WebSocket stream loop.

        Args:
            callback: Bar data callback (existing behavior).
            on_reconnect: Called after reconnection.
            on_quote: Optional callback for quote data: on_quote(symbol, StreamQuote).
            on_trade: Optional callback for trade data: on_trade(symbol, StreamTrade).
        """
        bar_arity = self._inspect_arity(callback)
        self._running = True
        backoff = 1.0
        had_failure = False

        while self._running:
            try:
                if had_failure:
                    await self._handle_reconnect(on_reconnect)

                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    await self._auth()
                    await self._subscribe_all_feeds()
                    self.logger.info(
                        "Gateway stream connected",
                        ws_url=self.ws_url,
                        bar_symbols=len(self._desired_bar_symbols),
                        quote_symbols=len(self._desired_quote_symbols),
                        trade_symbols=len(self._desired_trade_symbols),
                    )
                    backoff = 1.0
                    while self._running:
                        raw = await ws.recv()
                        await self._process_raw_message(raw, callback, bar_arity, on_quote, on_trade)
            except asyncio.CancelledError:
                self._running = False
                raise
            except Exception as e:
                had_failure = True
                self.logger.warning(
                    "Gateway stream disconnected",
                    error=str(e),
                    reconnect_in_seconds=round(backoff, 2),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                self._ws = None
