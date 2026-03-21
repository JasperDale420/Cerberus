import asyncio
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.errors import ErrorCode
from src.core.logger import StructuredLogger
from src.data.client import UnifiedDataClient
from src.data.unusual_whales import UnusualWhalesClient


class DataFetcher:
    """
    Handles data retrieval for the FeaturePipeline (I/O only).
    Manages caching and concurrency limits.
    All data flows through UnifiedDataClient (Data-Gateway).
    Flow alerts are read from Heber Silver (populated by Gateway poller).
    """

    def __init__(
        self,
        unified_client: UnifiedDataClient,
        unusual_whales_client: UnusualWhalesClient,
        logger: StructuredLogger,
        config: dict[str, Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        heber_client: Any | None = None,
    ):
        self.unified_client = unified_client
        self.unusual_whales_client = unusual_whales_client
        self.heber_client = heber_client
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(UTC))

        # LRU cache with maxsize for bars and trades
        self._bars_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._trades_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        try:
            self._cache_maxsize = int(self.config.get("bars_cache_maxsize", 500))
        except (TypeError, ValueError):
            self._cache_maxsize = 500

    def _resolve_fetch_start(self, symbol: str, start: datetime) -> tuple[datetime, list[dict[str, Any]]]:
        cached = self._bars_cache.get(symbol)
        if not cached or cached.get("start") != start:
            return start, []

        existing = cached.get("bars", [])
        if not existing:
            return start, []

        last_bar = existing[-1]
        raw_ts = last_bar.get("t") or last_bar.get("timestamp")
        if not raw_ts:
            return start, existing

        try:
            last_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            return last_ts + timedelta(seconds=1), existing
        except Exception as e:
            self.logger.debug(
                "Timestamp parsing failed",
                operation="parse_bar_timestamp",
                symbol=symbol,
                error=str(e),
            )
            return start, existing

    def _get_historical_bars_sync(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Any:
        """Fetch bars from UnifiedDataClient."""
        return self.unified_client.get_historical_bars(symbol.upper(), start, end, timeframe)

    async def _fetch_bars_internal(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> list[dict[str, Any]]:
        new_bars = await asyncio.to_thread(
            self.unified_client.get_historical_bars,
            symbol,
            start,
            end,
            timeframe,
        )
        if isinstance(new_bars, dict) and "bars" in new_bars:
            return list(new_bars["bars"])
        return list(new_bars or [])

    async def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min"
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """
        Fetches bars for a symbol, utilizing local cache.
        Returns (bars, metrics_delta).
        """
        metrics = {
            "alpaca_fetch_fail": 0,
            "alpaca_no_bars": 0,
            "cache_hits": 0,
            "incremental_fetches": 0,
        }
        sym = str(symbol).strip().upper()

        fetch_start, existing_bars = self._resolve_fetch_start(sym, start)
        new_bars = []

        if fetch_start < end:
            try:
                new_bars = await self._fetch_bars_internal(sym, fetch_start, end, timeframe)
                if new_bars and existing_bars:
                    metrics["incremental_fetches"] += 1
            except Exception as e:
                metrics["alpaca_fetch_fail"] += 1
                self.logger.warning(
                    "Bars fetch failed",
                    error_code=ErrorCode.ALPACA_BARS_FETCH_FAILED.value,
                    symbol=sym,
                    error=str(e),
                )
                if not existing_bars:
                    return [], metrics

        if existing_bars:
            metrics["cache_hits"] += 1

        final_bars = existing_bars + new_bars
        self._bars_cache[sym] = {"start": start, "bars": final_bars}
        self._bars_cache.move_to_end(sym)

        # Evict oldest entries if over maxsize
        while len(self._bars_cache) > self._cache_maxsize:
            evicted_sym, _ = self._bars_cache.popitem(last=False)
            self.logger.debug("Evicted bars cache entry", symbol=evicted_sym)

        if not final_bars:
            metrics["alpaca_no_bars"] += 1

        return final_bars, metrics

    async def fetch_trades(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fetches historical trades for a symbol."""
        metrics = {"trades_fetch_fail": 0, "no_trades": 0}
        sym = str(symbol).strip().upper()
        try:
            trades = await asyncio.to_thread(self.unified_client.get_trades, sym, start, end)
            if not trades:
                metrics["no_trades"] += 1
            return trades, metrics
        except Exception as e:
            metrics["trades_fetch_fail"] += 1
            self.logger.warning("Trades fetch failed", symbol=sym, error=str(e))
            return [], metrics

    async def fetch_quotes(
        self, symbol: str, start: datetime, end: datetime
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fetches historical quotes for a symbol."""
        metrics = {"quotes_fetch_fail": 0, "no_quotes": 0}
        sym = str(symbol).strip().upper()
        try:
            quotes = await asyncio.to_thread(self.unified_client.get_quotes, sym, start, end)
            if not quotes:
                metrics["no_quotes"] += 1
            return quotes, metrics
        except Exception as e:
            metrics["quotes_fetch_fail"] += 1
            self.logger.warning("Quotes fetch failed", symbol=sym, error=str(e))
            return [], metrics

    async def fetch_flow(self, symbol: str, date_str: str) -> list[Any]:
        """Fetches option flow data from Heber Silver (populated by Gateway poller).

        Falls back to Gateway REST proxy if Heber is not configured.
        """
        sym = str(symbol).strip().upper()

        # Primary: read from Heber Silver (no extra API calls — poller already fetched it)
        if self.heber_client is not None:
            try:
                dt = datetime.fromisoformat(date_str)
                start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)
                end = start + timedelta(hours=23, minutes=59, seconds=59)
                rows = await asyncio.to_thread(self.heber_client.get_flow_alerts, sym, start, end)
                if rows:
                    return rows
                self.logger.debug("No flow alerts in Heber for symbol", symbol=sym, date=date_str)
            except Exception as e:
                self.logger.warning("Heber flow read failed; falling back to gateway", symbol=sym, error=str(e))

        # Fallback: Gateway REST proxy (makes a live UW API call)
        try:
            response = await asyncio.to_thread(self.unified_client.get_flow, sym, date_str)
            data = response.get("data") if isinstance(response, dict) else None
            return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.warning("Flow fetch failed", symbol=sym, error=str(e))
            return []

    async def fetch_gex(self, symbol: str) -> list[dict[str, Any]]:
        """Fetches GEX data for a symbol."""
        try:
            return await asyncio.to_thread(self.unified_client.get_gex, symbol)
        except Exception as e:
            self.logger.warning("GEX fetch failed", symbol=symbol, error=str(e))
            return []

    def _extract_volume(self, bar: Any) -> float | None:
        try:
            if isinstance(bar, dict):
                v = bar.get("v") if bar.get("v") is not None else bar.get("volume")
            else:
                v = getattr(bar, "v", None)
                if v is None:
                    v = getattr(bar, "volume", None)
            return float(v) if v is not None else None
        except Exception as e:
            self.logger.debug(
                "Volume extraction failed",
                operation="extract_volume",
                error=str(e),
            )
            return None

    def fetch_avg_daily_volume(self, symbol: str, end: datetime, lookback_days: int) -> float | None:
        """Fetches daily bars to calculate average volume."""
        if lookback_days <= 0:
            return None
        try:
            return self.unified_client.get_avg_daily_volume(symbol, end, lookback_days)
        except Exception as e:
            self.logger.warning("Failed to fetch avg volume", symbol=symbol, error=str(e))
            return None

    def _parse_bar_time(self, bar: Any) -> datetime | None:
        try:
            bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
            raw_t = bd.get("t") or bd.get("timestamp") or getattr(bar, "t", None)
            if not raw_t:
                return None
            if isinstance(raw_t, datetime):
                dt = raw_t
            else:
                dt = datetime.fromisoformat(str(raw_t).replace("Z", "+00:00"))
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
        except Exception as e:
            self.logger.debug(
                "Bar time parsing failed",
                operation="parse_bar_time",
                error=str(e),
            )
            return None

    def _get_bar_field(self, bar: Any, keys: list[str]) -> float:
        bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
        for k in keys:
            val = bd.get(k)
            if val is None:
                val = getattr(bar, k, None)
            if val is not None:
                return float(val)
        return 0.0

    def fetch_prior_day_stats(self, symbol: str, current_time: datetime) -> tuple[float, float, float]:
        """Returns (High, Low, Close) from the prior complete day."""
        try:
            return self.unified_client.get_prior_day_stats(symbol, current_time)
        except Exception as e:
            self.logger.warning("Failed to fetch prior day stats", symbol=symbol, error=str(e))
            return (0.0, 0.0, 0.0)
