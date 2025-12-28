from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytz

from src.core.errors import ErrorCode
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.unusual_whales import UnusualWhalesClient


class DataFetcher:
    """
    Handles data retrieval for the FeaturePipeline (I/O only).
    Manages caching and concurrency limits.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        unusual_whales_client: UnusualWhalesClient,
        logger: StructuredLogger,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.logger = logger
        self.config = config or {}
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        # Cache structure: {symbol: {"start": datetime, "bars": List[dict]}}
        self._bars_cache: Dict[str, Dict[str, Any]] = {}

    def _resolve_fetch_start(
        self, symbol: str, start: datetime
    ) -> Tuple[datetime, List[Dict[str, Any]]]:
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
        except Exception:
            return start, existing

    async def _fetch_alpaca_bars_internal(
        self, symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> List[Dict[str, Any]]:
        import asyncio

        new_bars = await asyncio.to_thread(
            self.alpaca_client.get_historical_bars,
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
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
                new_bars = await self._fetch_alpaca_bars_internal(
                    sym, fetch_start, end, timeframe
                )
                if new_bars and existing_bars:
                    metrics["incremental_fetches"] += 1
            except Exception as e:
                metrics["alpaca_fetch_fail"] += 1
                self.logger.warning(
                    "Alpaca bars fetch failed",
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

        if not final_bars:
            metrics["alpaca_no_bars"] += 1

        return final_bars, metrics

    def _extract_volume(self, bar: Any) -> Optional[float]:
        try:
            if isinstance(bar, dict):
                v = bar.get("v") if bar.get("v") is not None else bar.get("volume")
            else:
                v = getattr(bar, "v", None) or getattr(bar, "volume", None)
            return float(v) if v is not None else None
        except Exception:
            return None

    def fetch_avg_daily_volume(
        self, symbol: str, end: datetime, lookback_days: int
    ) -> Optional[float]:
        """
        Fetches daily bars to calculate average volume.
        """
        if lookback_days <= 0:
            return None

        start = end - timedelta(days=int(max(lookback_days * 3, 10)))
        try:
            daily = self.alpaca_client.get_historical_bars(
                symbol, start, end, timeframe="1Day"
            )
        except Exception as e:
            self.logger.warning(
                "Failed to fetch avg volume", symbol=symbol, error=str(e)
            )
            return None

        if not daily:
            return None

        bars_list = (
            daily.get("bars") if isinstance(daily, dict) and "bars" in daily else daily
        )
        if not isinstance(bars_list, list):
            return None

        vols = [v for b in bars_list if (v := self._extract_volume(b)) is not None]

        if not vols:
            return None

        window = vols[-lookback_days:]
        return float(sum(window) / len(window)) if window else None

    def _parse_bar_time(self, bar: Any) -> Optional[datetime]:
        try:
            bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
            raw_t = bd.get("t") or bd.get("timestamp") or getattr(bar, "t", None)
            if not raw_t:
                return None
            if isinstance(raw_t, datetime):
                dt = raw_t
            else:
                dt = datetime.fromisoformat(str(raw_t).replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except Exception:
            return None

    def _get_bar_field(self, bar: Any, keys: List[str]) -> float:
        bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
        for k in keys:
            val = bd.get(k) or getattr(bar, k, None)
            if val is not None:
                return float(val)
        return 0.0

    def fetch_prior_day_stats(
        self, symbol: str, current_time: datetime
    ) -> Tuple[float, float, float]:
        """
        Returns (High, Low, Close) from daily bars for the prior complete day.
        """
        try:
            start = current_time - timedelta(days=7)
            bars = self.alpaca_client.get_historical_bars(
                symbol, start, current_time, timeframe="1Day"
            )
            if not bars:
                return (0.0, 0.0, 0.0)

            bars_list = (
                bars.get("bars") if isinstance(bars, dict) and "bars" in bars else bars
            )
            if not isinstance(bars_list, list):
                return (0.0, 0.0, 0.0)

            et_tz = pytz.timezone("US/Eastern")
            cutoff_date_et = current_time.astimezone(et_tz).date()

            valid_bars = []
            for b in bars_list:
                t_dt = self._parse_bar_time(b)
                if t_dt and t_dt.astimezone(et_tz).date() < cutoff_date_et:
                    valid_bars.append(b)

            if not valid_bars:
                return (0.0, 0.0, 0.0)

            last = valid_bars[-1]
            h = self._get_bar_field(last, ["h", "high"])
            low_px = self._get_bar_field(last, ["l", "low"])
            c = self._get_bar_field(last, ["c", "close"])
            return (h, low_px, c)

        except Exception as e:
            self.logger.warning(
                "Failed to fetch prior day stats", symbol=symbol, error=str(e)
            )
            return (0.0, 0.0, 0.0)

    async def fetch_flow(self, symbol: str, date_str: str) -> List[Any]:
        """
        Fetches Unusual Whales option flow for a specific date.
        """
        try:
            # mypy: ignore
            return await self.unusual_whales_client.get_option_flow(symbol, date_str)  # type: ignore
        except Exception as e:
            self.logger.warning(
                "Unusual Whales flow fetch failed; using neutral flow",
                error_code=ErrorCode.UW_FLOW_FETCH_FAILED.value,
                symbol=symbol,
                error=str(e),
            )
            return []
