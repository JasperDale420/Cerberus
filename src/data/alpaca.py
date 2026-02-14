import random
from datetime import datetime
from typing import Any, Callable, Optional, Set

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (
    MarketMoversRequest,
    MostActivesRequest,
    StockBarsRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

from src.core.config import ConfigLoader
from src.core.domain import Bar
from src.core.logger import StructuredLogger

# WebSocket resilience constants (from KI: engineering/alpaca_resilience.md)
HEARTBEAT_TIMEOUT_SEC = 120.0  # Force reconnect if no data for 2 minutes
FIRST_BAR_TIMEOUT_SEC = 10.0  # Fast fallback on cold start
BACKOFF_MAX_SEC = 30.0
BACKOFF_JITTER_MAX = 0.5
MAX_RETRIES = 5


class AlpacaClient:
    """
    Wrapper for Alpaca API clients (Trading, Historical Data, Live Data).
    """

    def __init__(self, config_loader: ConfigLoader, logger: StructuredLogger):
        self.logger = logger
        self._config_loader = config_loader  # Store for later use (e.g., feed config)

        try:
            self.api_key = config_loader.get_env("ALPACA_API_KEY")
            self.secret_key = config_loader.get_env("ALPACA_SECRET_KEY")
            self.paper = config_loader.get_env("ALPACA_PAPER", "True").lower() == "true"
        except ValueError as e:
            self.logger.critical("Failed to load Alpaca credentials", error=str(e))
            raise

        try:
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            self.screener_client = ScreenerClient(self.api_key, self.secret_key)
            self.trading_stream_client: Optional[TradingStream] = None
            # Stream is initialized on demand or separately as it blocks/runs in a loop
            self.stream_client: Optional[StockDataStream] = None
            self._pending_symbols: Set[str] = set()
            self._subscribed_symbols: Set[str] = set()
        except Exception as e:
            self.logger.critical("Failed to initialize Alpaca clients", error=str(e))
            raise

    def _to_bar(self, data: Any, symbol_override: Optional[str] = None) -> Bar:
        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        symbol = symbol_override
        if not symbol:
            if isinstance(data, dict):
                symbol = data.get("symbol") or "UNKNOWN"
            else:
                symbol = getattr(data, "symbol", None) or "UNKNOWN"

        if isinstance(data, dict):
            t = data.get("t") or data.get("timestamp")
            # P2.2 fix: Validate timestamp before using
            if not isinstance(t, datetime):
                raise ValueError(f"Invalid timestamp in bar data: {t}")
            o = data.get("o") or data.get("open")
            h = data.get("h") or data.get("high")
            low = data.get("l") or data.get("low")
            c = data.get("c") or data.get("close")
            v = data.get("v") or data.get("volume")
            return Bar(
                symbol=symbol,
                time=t,
                open=_f(o),
                high=_f(h),
                low=_f(low),
                close=_f(c),
                volume=_f(v),
                vwap=data.get("vwap"),
                trade_count=data.get("trade_count"),
            )

        # Try 'time' first, then 'timestamp' - validate result is datetime
        ts = getattr(data, "time", None)
        if not isinstance(ts, datetime):
            ts = getattr(data, "timestamp", None)
        # P2.2 fix: Validate timestamp before using
        if not isinstance(ts, datetime):
            raise ValueError(f"Invalid timestamp in bar data: {ts}")
        return Bar(
            symbol=symbol,
            time=ts,
            open=float(data.open),
            high=float(data.high),
            low=float(data.low),
            close=float(data.close),
            volume=float(data.volume),
            vwap=getattr(data, "vwap", None),
            trade_count=getattr(data, "trade_count", None),
        )

    def get_account(self):
        """
        Fetches account information.
        """
        try:
            return self.trading_client.get_account()
        except Exception as e:
            from src.core.errors import ErrorCode

            self.logger.error(
                "Failed to fetch account info",
                error_code=ErrorCode.ALPACA_ACCOUNT_FETCH_FAILED.value,
                error=str(e),
            )
            raise

    def get_historical_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"):
        """
        Fetches historical bars for a symbol via Alpaca historical data client.
        """
        try:
            tf = str(timeframe).strip().lower()
            if tf in ("1min", "1m", "minute"):
                tf_obj = TimeFrame.Minute
            elif tf in ("1day", "day", "d"):
                tf_obj = TimeFrame.Day
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=tf_obj,
            )
            resp = self.historical_client.get_stock_bars(req)
            if isinstance(resp, dict):
                bars = (resp.get("data") or {}).get(symbol) or []
            else:
                bars = (getattr(resp, "data", None) or {}).get(symbol) or []

            # Return a stable dict format aligned with existing FeaturePipeline parsing.
            out = []
            for b in bars:
                out.append(
                    {
                        "t": getattr(b, "timestamp", None) or getattr(b, "t", None),
                        "o": float(getattr(b, "open", 0.0)),
                        "h": float(getattr(b, "high", 0.0)),
                        "l": float(getattr(b, "low", 0.0)),
                        "c": float(getattr(b, "close", 0.0)),
                        "v": float(getattr(b, "volume", 0.0)),
                    }
                )
            return {"bars": out}
        except Exception as e:
            from src.core.errors import ErrorCode

            self.logger.error(
                "Failed to fetch historical bars",
                error_code=ErrorCode.ALPACA_BARS_FETCH_FAILED.value,
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            raise

    def get_historical_trades(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        """
        Fetches historical trades for a symbol via Alpaca historical data client.
        Required for Trade Flow Imbalance (TFI) calculation.
        """
        try:
            req = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
            )
            resp = self.historical_client.get_stock_trades(req)
            if isinstance(resp, dict):
                trades = (resp.get("data") or {}).get(symbol) or []
            else:
                trades = (getattr(resp, "data", None) or {}).get(symbol) or []

            out = []
            for t in trades:
                out.append(
                    {
                        "t": getattr(t, "timestamp", None),
                        "p": float(getattr(t, "price", 0.0)),
                        "s": float(getattr(t, "size", 0.0)),
                        "c": getattr(t, "conditions", []),
                        "x": getattr(t, "exchange", ""),
                        "z": getattr(t, "tape", ""),
                    }
                )
            return out
        except Exception as e:
            from src.core.errors import ErrorCode

            self.logger.error(
                "Failed to fetch historical trades",
                error_code=ErrorCode.ALPACA_TRADES_FETCH_FAILED.value,
                symbol=symbol,
                error=str(e),
            )
            raise

    def get_most_actives(self, top: int = 20) -> list[str]:
        """
        Fetches most active stocks by volume using Alpaca Screener API.
        Returns list of symbols sorted by volume.
        """
        try:
            req = MostActivesRequest(top=top, by="volume")
            resp = self.screener_client.get_most_actives(req)
            symbols = []
            for item in getattr(resp, "most_actives", []) or []:
                sym = getattr(item, "symbol", None)
                if sym:
                    symbols.append(str(sym).upper())
            self.logger.info(
                "Fetched most active stocks",
                count=len(symbols),
                top=top,
            )
            return symbols
        except Exception as e:
            self.logger.error(
                "Failed to fetch most actives",
                error=str(e),
            )
            return []

    def get_movers(self, top: int = 10) -> dict[str, list[str]]:
        """
        Fetches top gainers and losers using Alpaca Screener API.
        Returns {"gainers": [...], "losers": [...]}.
        """
        try:
            req = MarketMoversRequest(top=top)
            resp = self.screener_client.get_market_movers(req)
            gainers = []
            losers = []
            for item in getattr(resp, "gainers", []) or []:
                sym = getattr(item, "symbol", None)
                if sym:
                    gainers.append(str(sym).upper())
            for item in getattr(resp, "losers", []) or []:
                sym = getattr(item, "symbol", None)
                if sym:
                    losers.append(str(sym).upper())
            self.logger.info(
                "Fetched market movers",
                gainers_count=len(gainers),
                losers_count=len(losers),
            )
            return {"gainers": gainers, "losers": losers}
        except Exception as e:
            self.logger.error(
                "Failed to fetch movers",
                error=str(e),
            )
            return {"gainers": [], "losers": []}

    def get_stream_client(self) -> StockDataStream:
        """
        Returns a configured StockDataStream client.
        Always pass explicit feed parameter to avoid IEX fallback on premium accounts.
        """
        if not self.stream_client:
            # Determine feed from env; default to SIP for premium, IEX for free
            feed_str = self._config_loader.get_env("ALPACA_DATA_FEED", "sip").lower()
            feed = DataFeed.SIP if feed_str == "sip" else DataFeed.IEX
            self.stream_client = StockDataStream(self.api_key, self.secret_key, feed=feed)
            self.logger.info("Stock data stream initialized", feed=feed_str)
        return self.stream_client

    def get_trading_stream_client(self) -> TradingStream:
        if not self.trading_stream_client:
            self.trading_stream_client = TradingStream(self.api_key, self.secret_key, paper=self.paper)
        return self.trading_stream_client

    def _inspect_arity(self, callback: Callable) -> int:
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

    async def _invoke_bar_callback(self, callback: Callable, bar: Bar, symbol: str, arity: int) -> None:
        import asyncio

        # Helper for common invocation pattern
        async def _call(cb, *args):
            if asyncio.iscoroutinefunction(cb):
                await cb(*args)
            else:
                cb(*args)

        if arity == 1:
            await _call(callback, bar)
        elif arity >= 2:
            await _call(callback, symbol, bar)
        else:
            # Fallback: try (bar) then (symbol, bar)
            try:
                await _call(callback, bar)
            except TypeError:
                await _call(callback, symbol, bar)

    def _make_bar_handler(self, callback):
        """
        Creates a bar handler that normalizes data and safely invokes the callback.
        """
        arity = self._inspect_arity(callback)

        async def on_bar_wrapper(data):
            symbol = getattr(data, "symbol", "UNKNOWN")
            try:
                bar = self._to_bar(data, symbol_override=symbol)
            except Exception as e:
                self.logger.error("Failed to normalize bar", symbol=symbol, error=str(e))
                return

            await self._invoke_bar_callback(callback, bar, symbol, arity)

        return on_bar_wrapper

    async def _invoke_callback_safely(self, callback: Any) -> None:
        import asyncio

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
        except Exception as e:
            self.logger.error("Callback execution failed", error=str(e))

    def _stop_stream_safely(self, stream: Any) -> None:
        try:
            stream.stop()
        except Exception as e:
            # P2.1 fix: Log instead of silent pass
            self.logger.debug("Stream stop failed", error=str(e))

    def _flush_subscriptions(self, stream: Any) -> None:
        handler = getattr(self, "_bar_handler", None)
        if not handler:
            return

        for sym in sorted(self._pending_symbols):
            try:
                stream.subscribe_bars(handler, sym)
                self._subscribed_symbols.add(sym)
            except Exception as e:
                self.logger.error("Failed to subscribe queued symbol", symbol=sym, error=str(e))
        self._pending_symbols.clear()

    async def _run_stream_with_backoff(
        self,
        stream: Any,
        on_reconnect: Any,
        pre_run_hook: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """
        Runs stream with exponential backoff + jitter.
        Detects terminal errors (connection limit exceeded) and raises for fallback.
        """
        import asyncio

        backoff = 1.0
        had_failure = False
        retries = 0

        while retries < MAX_RETRIES:
            if had_failure and on_reconnect:
                await self._invoke_callback_safely(on_reconnect)

            if pre_run_hook:
                try:
                    pre_run_hook(stream)
                except Exception as e:
                    self.logger.error("Pre-run hook failed", error=str(e))

            try:
                await asyncio.to_thread(stream.run)
                return
            except asyncio.CancelledError:
                self._stop_stream_safely(stream)
                raise
            except Exception as e:
                error_str = str(e).lower()
                # Terminal error: connection limit exceeded - don't retry
                if "connection limit exceeded" in error_str:
                    self.logger.error(
                        "Terminal error: connection limit exceeded, falling back",
                        error=str(e),
                    )
                    raise  # Bubble up for REST fallback

                # Add jitter to backoff
                jitter = random.uniform(0, BACKOFF_JITTER_MAX)
                delay = backoff + jitter
                self.logger.error(
                    "Stream failed; retrying",
                    error=str(e),
                    backoff_sec=delay,
                    retry=retries + 1,
                    max_retries=MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                backoff = min(backoff * 2.0, BACKOFF_MAX_SEC)
                had_failure = True
                retries += 1

        # Exhausted retries
        self.logger.error("Stream retries exhausted", retries=retries)
        raise RuntimeError(f"WebSocket stream failed after {MAX_RETRIES} retries")

    async def start_stream(self, callback, on_reconnect=None):
        """
        Starts the WebSocket stream and registers the callback.
        Blocking call (runs loop).
        """
        stream = self.get_stream_client()
        self._bar_handler = self._make_bar_handler(callback)
        await self._run_stream_with_backoff(stream, on_reconnect, pre_run_hook=self._flush_subscriptions)

    async def start_trade_stream(self, callback, on_reconnect=None) -> None:
        """
        Starts Alpaca trading updates stream (fills/order lifecycle) with backoff.
        """
        import asyncio

        stream = self.get_trading_stream_client()

        # Async handler wrapper required by Alpaca SDK
        async def on_trade_update(data: Any) -> None:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error("Trade update callback failed", error=str(e))

        stream.subscribe_trade_updates(on_trade_update)
        await self._run_stream_with_backoff(stream, on_reconnect)

    def subscribe(self, symbol: str):
        """
        Subscribes to bar data for a symbol.
        """
        self._pending_symbols.add(symbol)
        stream = self.get_stream_client()
        if not hasattr(self, "_bar_handler"):
            self.logger.warning("Subscribe called before start_stream; queued symbol", symbol=symbol)
            return

        if symbol not in self._subscribed_symbols:
            stream.subscribe_bars(self._bar_handler, symbol)
            self._subscribed_symbols.add(symbol)
            self._pending_symbols.discard(symbol)

    def unsubscribe(self, symbol: str):
        """
        Unsubscribes from bar data.
        """
        stream = self.get_stream_client()
        try:
            stream.unsubscribe_bars(symbol)
        except KeyError:
            # Alpaca SDK can raise KeyError if handler map doesn't contain symbol
            # (e.g., queued subscribes, reconnect races, or churn during warmup).
            self.logger.warning("Unsubscribe no-op; symbol not subscribed", symbol=symbol)
        self._subscribed_symbols.discard(symbol)
        self._pending_symbols.discard(symbol)
