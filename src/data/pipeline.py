from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pytz  # type: ignore

from src.core.domain import SymbolFeatures
from src.core.errors import ErrorCode
from src.core.indicators import RollingEMA, RollingStd
from src.core.logger import StructuredLogger
from src.data.alpaca import AlpacaClient
from src.data.unusual_whales import UnusualWhalesClient


class FeaturePipeline:
    """
    Fetches data and computes features for symbols.
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

        fp_cfg = (
            (self.config.get("feature_pipeline") or {})
            if isinstance(self.config, dict)
            else {}
        )
        self.daily_volume_lookback_days = int(
            fp_cfg.get("daily_volume_lookback_days", 20)
        )
        self.max_concurrency = int(fp_cfg.get("max_concurrency", 6))
        # Cache structure: {symbol: {"start": datetime, "bars": List[dict]}}
        self._bars_cache: Dict[str, Dict[str, Any]] = {}

    async def compute_features(
        self, symbols: List[str], as_of: Optional[datetime] = None
    ) -> Dict[str, SymbolFeatures]:
        """
        Computes features for a list of symbols.
        """
        features: Dict[str, SymbolFeatures] = {}
        if as_of is None:
            raise ValueError(
                "FeaturePipeline.compute_features requires as_of for deterministic behavior"
            )

        import time as _time

        start_wall = _time.perf_counter()

        # PRD 11.3: summarize failures per data source deterministically.
        metrics: Dict[str, int] = {
            "symbols_total": int(len(symbols)),
            "features_ok": 0,
            "alpaca_fetch_fail": 0,
            "alpaca_no_bars": 0,
            "technicals_fail": 0,
            "uw_fetch_fail": 0,
            "cache_hits": 0,
            "incremental_fetches": 0,
        }

        # Bounded concurrency per symbol for deterministic scan behavior.
        # Alpaca historical fetch runs in a worker thread; UW flow fetch is async.

        import asyncio

        sem = asyncio.Semaphore(max(1, int(self.max_concurrency)))

        async def _compute_one(
            symbol: str,
        ) -> tuple[str, Optional[SymbolFeatures], Dict[str, int]]:
            local: Dict[str, int] = {
                "features_ok": 0,
                "alpaca_fetch_fail": 0,
                "alpaca_no_bars": 0,
                "technicals_fail": 0,
                "uw_fetch_fail": 0,
            }
            sym = str(symbol).strip().upper()

            async with sem:
                try:
                    end = as_of
                    if isinstance(end, datetime) and end.tzinfo is None:
                        end = end.replace(tzinfo=timezone.utc)

                    # PRD 4.3/7.2: features like premarket_volume and true gap_pct
                    # require premarket + session open context; use an ET-based window.
                    et_tz = pytz.timezone("US/Eastern")
                    end_et = end.astimezone(et_tz)
                    start_day = end_et.date()
                    if end_et.time() < time(4, 0):
                        start_day = start_day - timedelta(days=1)
                    start_et = et_tz.localize(datetime.combine(start_day, time(4, 0)))
                    start = start_et.astimezone(timezone.utc)

                    # Caching Logic
                    cached = self._bars_cache.get(sym)
                    fetch_start = start
                    existing_bars = []

                    if cached and cached.get("start") == start:
                        existing_bars = cached.get("bars", [])
                        if existing_bars:
                            # Resume fetch from last bar timestamp
                            last_bar = existing_bars[-1]
                            raw_ts = last_bar.get("t") or last_bar.get("timestamp")
                            if raw_ts:
                                try:
                                    last_ts = datetime.fromisoformat(
                                        raw_ts.replace("Z", "+00:00")
                                    )
                                    fetch_start = last_ts + timedelta(
                                        seconds=1
                                    )  # Avoid overlap
                                except Exception:
                                    pass

                    bars_data = []
                    # Only fetch if there is a gap to cover
                    if fetch_start < end:
                        try:
                            new_bars = await asyncio.to_thread(
                                self.alpaca_client.get_historical_bars,
                                sym,
                                fetch_start,
                                end,
                                "1Min",
                            )
                            if isinstance(new_bars, dict) and "bars" in new_bars:
                                new_bars = new_bars["bars"]

                            if new_bars:
                                bars_data = new_bars
                                if existing_bars:
                                    metrics["incremental_fetches"] += 1
                        except Exception as e:
                            local["alpaca_fetch_fail"] += 1
                            self.logger.warning(
                                "Alpaca bars fetch failed",
                                error_code=ErrorCode.ALPACA_BARS_FETCH_FAILED.value,
                                symbol=sym,
                                error=str(e),
                            )
                            # Fallback to existing bars if we have them?
                            # For robustness, yes, but logging error.
                            if not existing_bars:
                                return sym, None, local

                    # Merge and Update Cache
                    if existing_bars:
                        metrics["cache_hits"] += 1
                        # Simple append; assuming time-ordered and no overlap due to fetch_start logic
                        final_bars = existing_bars + bars_data
                    else:
                        final_bars = bars_data

                    # Update cache
                    self._bars_cache[sym] = {"start": start, "bars": final_bars}
                    bars_data = final_bars

                    if not bars_data:
                        self.logger.warning("No bars found for symbol", symbol=sym)
                        local["alpaca_no_bars"] += 1
                        return sym, None, local

                    def _session_open_price(
                        bars: List[Any], now_utc: datetime
                    ) -> float:
                        """
                        Best-effort first regular-session (>=09:30 ET) bar open price for the ET trading day.
                        """
                        open_px = 0.0
                        et = pytz.timezone("US/Eastern")
                        now_et = now_utc.astimezone(et)
                        day = now_et.date()
                        market_open = time(9, 30)

                        best_ts: Optional[datetime] = None
                        for b in bars:
                            bd = (
                                b
                                if isinstance(b, dict)
                                else getattr(b, "__dict__", {}) or {}
                            )
                            ts = (
                                bd.get("t")
                                or bd.get("timestamp")
                                or getattr(b, "t", None)
                            )
                            if isinstance(ts, str):
                                try:
                                    ts = datetime.fromisoformat(
                                        ts.replace("Z", "+00:00")
                                    )
                                except Exception:
                                    continue
                            if not isinstance(ts, datetime):
                                continue
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            ts_et = ts.astimezone(et)
                            if ts_et.date() != day:
                                continue
                            if ts_et.time() < market_open:
                                continue

                            o = (
                                bd.get("o")
                                if bd.get("o") is not None
                                else bd.get("open")
                            )
                            try:
                                o_f = float(o) if o is not None else 0.0
                            except Exception:
                                continue
                            if best_ts is None or ts < best_ts:
                                best_ts = ts
                                open_px = o_f
                        return float(open_px)

                    try:
                        tech_result = self._compute_technicals(bars_data)
                        if not tech_result:
                            local["technicals_fail"] += 1
                            return sym, None, local

                        (
                            price,
                            volume,
                            timestamp,
                            atr_pct,
                            intraday_range_pct,
                            gap_pct,
                            ema20_slope,
                            distance_from_vwap,
                            adx,
                            distance_from_ema20,
                            prior_day_high,
                            prior_day_low,
                            bb_upper,
                            bb_lower,
                            price_zscore,
                            premarket_vol,
                        ) = tech_result

                        avg_daily_volume = await asyncio.to_thread(
                            self._fetch_avg_daily_volume,
                            sym,
                            end,
                            self.daily_volume_lookback_days,
                        )

                        # PRD 4.3/7.2: compute true gap_pct as (session open - prior close) / prior close.
                        p_high, p_low, p_close = await asyncio.to_thread(
                            self._fetch_prior_day_stats,
                            sym,
                            end,
                        )
                        if prior_day_high == 0.0 or prior_day_low == 0.0:
                            if p_high > 0:
                                prior_day_high = p_high
                                prior_day_low = p_low

                        session_open = _session_open_price(
                            bars_data if isinstance(bars_data, list) else [],
                            end,
                        )
                        if p_close > 0 and session_open > 0:
                            gap_pct = (session_open - p_close) / p_close

                    except Exception as e:
                        local["technicals_fail"] += 1
                        self.logger.error(
                            "Failed to compute technicals", symbol=sym, error=str(e)
                        )
                        return sym, None, local

                    try:
                        flow_data = await self.unusual_whales_client.get_option_flow(
                            sym, end.strftime("%Y-%m-%d")
                        )
                    except Exception as e:
                        local["uw_fetch_fail"] += 1
                        self.logger.warning(
                            "Options flow unavailable; using neutral flow features",
                            error_code=ErrorCode.UW_FLOW_FETCH_FAILED.value,
                            symbol=sym,
                            error=str(e),
                        )
                        flow_data = []

                    (
                        call_put_ratio,
                        flow_zscore,
                        sweep_count,
                        aggressive_flow_share,
                        flow_bias,
                    ) = self._compute_flow_metrics(flow_data)

                    feat = SymbolFeatures(
                        symbol=sym,
                        last_updated=(
                            timestamp
                            if isinstance(timestamp, datetime)
                            else datetime.fromisoformat(
                                str(timestamp).replace("Z", "+00:00")
                            )
                        ),
                        price=price,
                        avg_volume=(
                            float(avg_daily_volume)
                            if avg_daily_volume is not None
                            else float(volume)
                        ),
                        atr_pct=atr_pct,
                        intraday_range_pct=intraday_range_pct,
                        gap_pct=gap_pct,
                        ema20_slope=ema20_slope,
                        ema_trend_strength=abs(ema20_slope),  # Proxy
                        distance_from_vwap=distance_from_vwap,
                        premarket_volume=premarket_vol,
                        adx=adx,
                        distance_from_ema20=distance_from_ema20,
                        prior_day_high=prior_day_high,
                        prior_day_low=prior_day_low,
                        bb_upper=bb_upper,
                        bb_lower=bb_lower,
                        price_zscore=price_zscore,
                        flow_zscore=flow_zscore,
                        call_put_ratio=call_put_ratio,
                        large_sweeps_count=sweep_count,
                        aggressive_flow_share=aggressive_flow_share,
                        extra={
                            "flow_raw_count": len(flow_data) if flow_data else 0,
                            # Preserve the previous normalized volume imbalance for analytics/backward compatibility.
                            "flow_bias": float(flow_bias),
                            "volatility": atr_pct,
                            "last_bar_volume": float(volume),
                            "avg_daily_volume_days": int(
                                self.daily_volume_lookback_days
                            ),
                        },
                    )

                    local["features_ok"] += 1
                    return sym, feat, local
                except Exception as e:
                    self.logger.error(
                        "Failed to compute features", symbol=sym, error=str(e)
                    )
                    return sym, None, local

        results = await asyncio.gather(*[_compute_one(s) for s in list(symbols)])
        for sym, feat, local in results:
            for k, v in local.items():
                metrics[k] = int(metrics.get(k, 0)) + int(v)
            if feat is not None:
                features[sym] = feat

        # Emit one summary log per scan for PRD 11.3.
        try:
            metrics["max_concurrency"] = int(self.max_concurrency)
            metrics["duration_ms"] = int((_time.perf_counter() - start_wall) * 1000)
            self.last_run_metrics = dict(metrics)
            self.logger.info("FeaturePipeline summary", **self.last_run_metrics)
        except Exception:
            pass

        return features

    def _fetch_avg_daily_volume(
        self, symbol: str, end: datetime, lookback_days: int
    ) -> Optional[float]:
        if lookback_days <= 0:
            return None

        # Give some slack for weekends/holidays by fetching a wider window.
        start = end - timedelta(days=int(max(lookback_days * 3, 10)))
        try:
            daily = self.alpaca_client.get_historical_bars(
                symbol, start, end, timeframe="1Day"
            )
        except Exception as e:
            self.logger.warning(
                "Failed to fetch daily bars for avg volume; falling back",
                symbol=symbol,
                error=str(e),
            )
            return None

        if not daily:
            return None
        if isinstance(daily, dict) and "bars" in daily:
            daily = daily["bars"]
        if not isinstance(daily, list):
            return None

        vols: List[float] = []
        for b in daily:
            try:
                if isinstance(b, dict):
                    v = b.get("v") if b.get("v") is not None else b.get("volume")
                else:
                    v = getattr(b, "v", None) or getattr(b, "volume", None)
                if v is None:
                    continue
                vols.append(float(v))
            except Exception:
                continue

        if not vols:
            return None

        # Use the last `lookback_days` bars deterministically.
        window = vols[-lookback_days:]
        return float(sum(window) / len(window)) if window else None

    def _compute_technicals(self, bars_data: List[Any]) -> Optional[tuple]:
        if not bars_data:
            return None

        def _parse_ts(v: Any) -> Optional[datetime]:
            if v is None:
                return None
            if isinstance(v, datetime):
                dt = v
            else:
                s = str(v)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                except Exception:
                    return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        rows: List[
            tuple[datetime, float, float, float, float, float, Optional[float]]
        ] = []
        for b in bars_data:
            bd = b if isinstance(b, dict) else getattr(b, "__dict__", {})
            ts = _parse_ts(bd.get("t") or bd.get("timestamp") or getattr(b, "t", None))
            if ts is None:
                continue
            o = _to_float(bd.get("o") or bd.get("open") or getattr(b, "o", None))
            h = _to_float(bd.get("h") or bd.get("high") or getattr(b, "h", None))
            low_val = _to_float(bd.get("l") or bd.get("low") or getattr(b, "l", None))
            c = _to_float(bd.get("c") or bd.get("close") or getattr(b, "c", None))
            v = _to_float(bd.get("v") or bd.get("volume") or getattr(b, "v", None))
            vwap = _to_float(bd.get("vwap") or getattr(b, "vwap", None))
            if o is None or h is None or low_val is None or c is None or v is None:
                continue
            rows.append((ts, o, h, low_val, c, v, vwap))

        if not rows:
            return None

        rows.sort(key=lambda r: r[0])
        timestamps = [r[0] for r in rows]
        opens = [r[1] for r in rows]
        highs = [r[2] for r in rows]
        lows = [r[3] for r in rows]
        closes = [r[4] for r in rows]
        volumes = [r[5] for r in rows]
        vwaps = [r[6] for r in rows]

        price = float(closes[-1])
        volume = float(volumes[-1])
        timestamp = timestamps[-1]

        high = float(highs[-1])
        low = float(lows[-1])
        open_ = float(opens[-1])
        intraday_range_pct = ((high - low) / open_) if open_ > 0 else 0.0

        gap_pct = 0.0
        if len(closes) >= 2:
            prev_close = float(closes[-2])
            gap_pct = ((open_ - prev_close) / prev_close) if prev_close > 0 else 0.0

        # ATR(14) via Wilder smoothing (gracefully returns 0.0 for short series).
        atr_value = 0.0
        if len(closes) >= 2:
            trs: List[float] = []
            prev_close = float(closes[0])
            for i in range(1, len(closes)):
                tr = max(
                    float(highs[i] - lows[i]),
                    abs(float(highs[i] - prev_close)),
                    abs(float(lows[i] - prev_close)),
                )
                trs.append(float(tr))
                prev_close = float(closes[i])
            if trs:
                period = 14
                p = min(period, len(trs))
                atr = float(sum(trs[:p]) / max(1, p))
                for tr in trs[p:]:
                    atr = ((atr * (period - 1)) + float(tr)) / float(period)
                atr_value = float(atr)

        atr_pct = (atr_value / price) if price > 0 else 0.0

        # EMA20 + slope (delta) and distance.
        ema20 = RollingEMA.from_period(20)
        for c in closes:
            ema20.update(float(c))
        ema20_val = float(ema20.value) if ema20.value is not None else 0.0
        ema20_prev = float(ema20.prev_value) if ema20.prev_value is not None else None

        ema20_slope = (ema20_val - ema20_prev) if ema20_prev is not None else 0.0
        distance_from_ema20 = (price - ema20_val) / ema20_val if ema20_val > 0 else 0.0

        # VWAP (day-anchored, best-effort).
        vwap_val = float(vwaps[-1]) if vwaps[-1] is not None else 0.0
        et_tz = pytz.timezone("US/Eastern")
        latest_date_et = timestamp.astimezone(et_tz).date()
        if vwap_val <= 0.0:
            num = 0.0
            den = 0.0
            for ts, h, low_val, c, vol in zip(
                timestamps, highs, lows, closes, volumes, strict=True
            ):
                if ts.astimezone(et_tz).date() != latest_date_et:
                    continue
                tp = (float(h) + float(low_val) + float(c)) / 3.0
                v = float(vol)
                if v <= 0:
                    continue
                num += tp * v
                den += v
            vwap_val = float(num / den) if den > 0 else float(price)

        distance_from_vwap = ((price - vwap_val) / vwap_val) if vwap_val > 0 else 0.0

        # ADX(14) (Wilder), returns 0.0 until enough data is present.
        adx_val = 0.0
        if len(closes) >= 3:
            period = 14
            tr_list: List[float] = []
            plus_dm_list: List[float] = []
            minus_dm_list: List[float] = []
            for i in range(1, len(closes)):
                up_move = float(highs[i] - highs[i - 1])
                down_move = float(lows[i - 1] - lows[i])
                plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
                minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

                prev_close = float(closes[i - 1])
                tr = max(
                    float(highs[i] - lows[i]),
                    abs(float(highs[i] - prev_close)),
                    abs(float(lows[i] - prev_close)),
                )
                tr_list.append(float(tr))
                plus_dm_list.append(float(plus_dm))
                minus_dm_list.append(float(minus_dm))

            def _wilder_smooth(values: List[float], p: int) -> List[float]:
                if p <= 0 or len(values) < p:
                    return []
                sm = float(sum(values[:p]))
                out = [sm]
                for x in values[p:]:
                    sm = sm - (sm / float(p)) + float(x)
                    out.append(float(sm))
                return out

            sm_tr = _wilder_smooth(tr_list, period)
            sm_pdm = _wilder_smooth(plus_dm_list, period)
            sm_mdm = _wilder_smooth(minus_dm_list, period)

            if sm_tr and len(sm_tr) == len(sm_pdm) == len(sm_mdm):
                dx_list: List[float] = []
                for tr, pdm, mdm in zip(sm_tr, sm_pdm, sm_mdm, strict=True):
                    if tr <= 0:
                        dx_list.append(0.0)
                        continue
                    di_plus = 100.0 * (float(pdm) / float(tr))
                    di_minus = 100.0 * (float(mdm) / float(tr))
                    denom = di_plus + di_minus
                    dx = 100.0 * abs(di_plus - di_minus) / denom if denom > 0 else 0.0
                    dx_list.append(float(dx))

                if len(dx_list) >= period:
                    adx = float(sum(dx_list[:period]) / float(period))
                    for dx in dx_list[period:]:
                        adx = ((adx * (period - 1)) + float(dx)) / float(period)
                    adx_val = float(adx)

        # Prior day H/L from intraday bars (ET dates).
        prior_day_high = 0.0
        prior_day_low = 0.0
        by_date: Dict[Any, List[float]] = {}
        for ts, h, low_val in zip(timestamps, highs, lows, strict=True):
            d = ts.astimezone(et_tz).date()
            if d not in by_date:
                by_date[d] = [float(h), float(low_val)]
            else:
                by_date[d][0] = max(by_date[d][0], float(h))
                by_date[d][1] = min(by_date[d][1], float(low_val))

        prior_dates = [d for d in by_date.keys() if d < latest_date_et]
        if prior_dates:
            last_full = max(prior_dates)
            prior_day_high = float(by_date[last_full][0])
            prior_day_low = float(by_date[last_full][1])

        # Bollinger (20, 2) using deterministic population std.
        bb_upper = 0.0
        bb_lower = 0.0
        price_zscore = 0.0
        std20 = RollingStd.create(20)
        mean = 0.0
        stdev = 0.0
        for c in closes:
            mean, stdev = std20.update(float(c))
        if stdev > 0:
            bb_upper = float(mean + (2.0 * stdev))
            bb_lower = float(mean - (2.0 * stdev))
            price_zscore = float((price - mean) / stdev)

        # Premarket volume (ET): sum volume for latest ET day before 09:30.
        premarket_vol = 0.0
        market_open = time(9, 30)
        for ts, vol in zip(timestamps, volumes, strict=True):
            ts_et = ts.astimezone(et_tz)
            if ts_et.date() != latest_date_et:
                continue
            if ts_et.time() < market_open:
                premarket_vol += float(vol)

        return (
            price,
            volume,
            timestamp,
            atr_pct,
            intraday_range_pct,
            gap_pct,
            ema20_slope,
            distance_from_vwap,
            adx_val,
            distance_from_ema20,
            prior_day_high,
            prior_day_low,
            bb_upper,
            bb_lower,
            price_zscore,
            premarket_vol,
        )

    def _compute_flow_metrics(self, flow_data: List[Any]) -> tuple:
        """
        PRD 4.3: options flow features.

        Notes on semantics:
        - `flow_zscore` is a deterministic, z-score-like statistic based on CALL vs PUT trade-count
          imbalance within the provided flow sample.
        - `flow_bias` preserves the prior normalized volume imbalance ((call_vol-put_vol)/total_qty).
        """
        import math

        call_vol = 0.0
        put_vol = 0.0
        call_n = 0
        put_n = 0
        sweep_count = 0
        aggressive_qty = 0.0
        total_qty = 0.0

        if flow_data and isinstance(flow_data, list):
            for trade in flow_data:
                t = trade if isinstance(trade, dict) else trade.__dict__

                size = float(t.get("size", 0))
                pc = t.get("put_call", "UNKNOWN")

                total_qty += size

                if pc == "CALL":
                    call_vol += size
                    call_n += 1
                elif pc == "PUT":
                    put_vol += size
                    put_n += 1

                tags = t.get("tags", [])
                if "sweep" in tags or t.get("sentiment") == "BULLISH":
                    sweep_count += 1

                if t.get("ask_side") or t.get("sentiment") in [
                    "BULLISH",
                    "BEARISH",
                ]:
                    aggressive_qty += size

        call_put_ratio = (
            (call_vol / put_vol) if put_vol > 0 else (call_vol if call_vol > 0 else 0.0)
        )
        aggressive_flow_share = (aggressive_qty / total_qty) if total_qty > 0 else 0.0
        n = int(call_n + put_n)
        flow_zscore = ((call_n - put_n) / math.sqrt(n)) if n > 0 else 0.0
        flow_bias = ((call_vol - put_vol) / total_qty) if total_qty > 0 else 0.0
        return (
            call_put_ratio,
            flow_zscore,
            sweep_count,
            aggressive_flow_share,
            flow_bias,
        )

    def _fetch_prior_day_stats(self, symbol: str, current_time: datetime) -> tuple:
        """
        Fetches daily bars to find prior day High/Low/Close.
        Returns (High, Low, Close) or (0,0,0) if failed.
        """
        try:
            # Fetch last 5 days
            start = current_time - timedelta(days=7)
            bars = self.alpaca_client.get_historical_bars(
                symbol, start, current_time, timeframe="1Day"
            )

            if not bars:
                return (0.0, 0.0, 0.0)

            # Handle response format (Dict or List) from AlpacaClient.get_historical_bars
            # Filter for completed days < current_time.date()
            # Note: current_time is UTC. We should compare dates carefully.

            # Flatten if dict
            if isinstance(bars, dict) and "bars" in bars:
                bars = bars["bars"]

            et_tz = pytz.timezone("US/Eastern")
            now = current_time
            if isinstance(now, datetime) and now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            cutoff_date_et = now.astimezone(et_tz).date()

            valid_bars = []
            for b in bars:
                # Normalize to dict
                bd = b if isinstance(b, dict) else b.__dict__
                t = bd.get("t") or bd.get("timestamp")
                if isinstance(t, str):
                    t = datetime.fromisoformat(t.replace("Z", "+00:00"))

                if not isinstance(t, datetime):
                    continue
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t.astimezone(et_tz).date() < cutoff_date_et:
                    valid_bars.append(bd)

            if not valid_bars:
                return (0.0, 0.0, 0.0)

            last_bar = valid_bars[-1]
            h = float(last_bar.get("h") or last_bar.get("high") or 0.0)
            low_px = float(last_bar.get("l") or last_bar.get("low") or 0.0)
            c = float(last_bar.get("c") or last_bar.get("close") or 0.0)

            return (h, low_px, c)

        except Exception as e:
            self.logger.warning(
                "Failed to fetch prior day stats", symbol=symbol, error=str(e)
            )
            return (0.0, 0.0, 0.0)
