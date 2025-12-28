import math
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz

from src.core.indicators import RollingEMA, RollingStd


class FeatureCalculator:
    """
    Pure logic component for FeaturePipeline.
    Computes technical indicators and flow metrics from raw data.
    """

    def compute_technicals(
        self, bars_data: List[Any], current_time: Optional[datetime] = None
    ) -> Optional[tuple]:
        """
        Computes technical indicators from a list of bar data.
        Returns a tuple of features or None if data is insufficient.
        """
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

        # ATR(14) via Wilder smoothing
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

        # EMA20 + slope
        ema20 = RollingEMA.from_period(20)
        for c in closes:
            ema20.update(float(c))
        ema20_val = float(ema20.value) if ema20.value is not None else 0.0
        ema20_prev = float(ema20.prev_value) if ema20.prev_value is not None else 0.0

        ema20_slope = (ema20_val - ema20_prev) if ema20_prev > 0 else 0.0
        distance_from_ema20 = (price - ema20_val) / ema20_val if ema20_val > 0 else 0.0

        # VWAP (intraday)
        vwap_val = float(vwaps[-1]) if vwaps[-1] is not None else 0.0
        et_tz = pytz.timezone("US/Eastern")
        latest_date_et = timestamp.astimezone(et_tz).date()

        # Recalculate VWAP if missing or stale logic (inherited from old pipeline)
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

        # ADX(14)
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

        # Prior day H/L from intraday bars (approximation if daily not fetched)
        # This is a fallback calculation; normally Fetcher gets daily stats.
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

        # Bollinger (20, 2)
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

        # Premarket volume
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

    def compute_flow_metrics(
        self, flow_data: List[Any]
    ) -> Tuple[float, float, int, float, float]:
        """
        Computes options flow metrics.
        Returns (call_put_ratio, flow_zscore, sweep_count, aggressive_flow_share, flow_bias)
        """
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

                if t.get("ask_side") or t.get("sentiment") in ["BULLISH", "BEARISH"]:
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

    def calculate_session_open_price(self, bars: List[Any], now_utc: datetime) -> float:
        """Helper to find session open price"""
        open_px = 0.0
        et = pytz.timezone("US/Eastern")
        now_et = now_utc.astimezone(et)
        day = now_et.date()
        market_open = time(9, 30)
        best_ts: Optional[datetime] = None

        for b in bars:
            bd = b if isinstance(b, dict) else getattr(b, "__dict__", {})
            ts = bd.get("t") or bd.get("timestamp") or getattr(b, "t", None)

            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
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

            o = bd.get("o") if bd.get("o") is not None else bd.get("open")
            try:
                o_f = float(o) if o is not None else 0.0
            except Exception:
                continue

            if best_ts is None or ts < best_ts:
                best_ts = ts
                open_px = o_f

        return float(open_px)
