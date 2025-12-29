import math
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz

from src.core.domain import TechnicalFeatures
from src.core.indicators import RollingEMA, RollingStd

# Cache timezone object at module level for performance
_ET_TZ = pytz.timezone("US/Eastern")


class FeatureCalculator:
    """
    Pure logic component for FeaturePipeline.
    Computes technical indicators and flow metrics from raw data.
    """

    DEFAULT_TIMEZONE = "US/Eastern"
    UTC_OFFSET_STR = "+00:00"

    @staticmethod
    def calculate_ema(values: List[float], period: int) -> Optional[float]:
        """
        Calculate Exponential Moving Average for a list of values.

        Uses alpha = 2 / (period + 1) formula.
        Returns the final EMA value or None if insufficient data.
        """
        if not values or period <= 0:
            return None
        p = max(1, int(period))
        alpha = 2.0 / (p + 1.0)
        ema: Optional[float] = None
        for x in values:
            ema = x if ema is None else (alpha * x) + ((1.0 - alpha) * ema)
        return ema

    @staticmethod
    def calculate_rsi(closes: List[float], period: int) -> Optional[float]:
        """
        Calculate Relative Strength Index using Wilder smoothing.

        Args:
            closes: List of closing prices
            period: RSI period (typically 2, 14, etc.)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(closes) < period + 1 or period <= 0:
            return None

        p = max(1, int(period))

        # Calculate price changes
        diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in diffs]
        losses = [max(-d, 0.0) for d in diffs]

        # Initial average (SMA for first period)
        avg_gain = sum(gains[:p]) / p
        avg_loss = sum(losses[:p]) / p

        # Wilder smoothing for remaining values
        for i in range(p, len(gains)):
            avg_gain = (avg_gain * (p - 1) + gains[i]) / p
            avg_loss = (avg_loss * (p - 1) + losses[i]) / p

        # Calculate RSI
        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute_technicals(self, bars_data: List[Any]) -> Optional[TechnicalFeatures]:
        """
        Computes technical indicators from a list of bar data.
        Returns a TechnicalFeatures object or None if data is insufficient.
        """
        if not bars_data:
            return None

        rows = self._parse_bars(bars_data)
        if not rows:
            return None

        # Single-pass extraction using zip instead of 7 list comprehensions
        timestamps, opens, highs, lows, closes, volumes, vwaps = (
            list(x) for x in zip(*rows, strict=True)
        )

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

        _, atr_pct = self._compute_atr(closes, highs, lows, price)
        ema20_slope, distance_from_ema20 = self._compute_ema_metrics(closes, price)
        _, distance_from_vwap = self._compute_vwap(
            vwaps, timestamps, highs, lows, closes, volumes, price
        )
        adx_val = self._compute_adx(highs, lows, closes)
        prior_day_high, prior_day_low = self._compute_prior_day_high_low(
            timestamps, highs, lows
        )
        bb_upper, bb_lower, price_zscore = self._compute_bollinger(closes, price)
        premarket_vol = self._compute_premarket_volume(timestamps, volumes)

        return TechnicalFeatures(
            price=price,
            volume=volume,
            timestamp=timestamp,
            atr_pct=atr_pct,
            intraday_range_pct=intraday_range_pct,
            gap_pct=gap_pct,
            ema20_slope=ema20_slope,
            distance_from_vwap=distance_from_vwap,
            adx=adx_val,
            distance_from_ema20=distance_from_ema20,
            prior_day_high=prior_day_high,
            prior_day_low=prior_day_low,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            price_zscore=price_zscore,
            premarket_volume=premarket_vol,
        )

    def _parse_ts(self, v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            dt = v
        else:
            s = str(v)
            # Fail fast if format is invalid
            dt = datetime.fromisoformat(s.replace("Z", self.UTC_OFFSET_STR))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _to_float(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        return float(v)

    def _parse_bars(self, bars_data: List[Any]) -> List[tuple]:
        rows = []
        for b in bars_data:
            bd = b if isinstance(b, dict) else getattr(b, "__dict__", {})
            ts = self._parse_ts(
                bd.get("t") or bd.get("timestamp") or getattr(b, "t", None)
            )
            if ts is None:
                continue
            o = self._to_float(bd.get("o") or bd.get("open") or getattr(b, "o", None))
            h = self._to_float(bd.get("h") or bd.get("high") or getattr(b, "h", None))
            low_val = self._to_float(
                bd.get("l") or bd.get("low") or getattr(b, "l", None)
            )
            c = self._to_float(bd.get("c") or bd.get("close") or getattr(b, "c", None))
            v = self._to_float(bd.get("v") or bd.get("volume") or getattr(b, "v", None))
            vwap = self._to_float(bd.get("vwap") or getattr(b, "vwap", None))
            if o is None or h is None or low_val is None or c is None or v is None:
                continue
            rows.append((ts, o, h, low_val, c, v, vwap))
        rows.sort(key=lambda r: r[0])
        return rows

    def _compute_atr(
        self, closes: List[float], highs: List[float], lows: List[float], price: float
    ) -> Tuple[float, float]:
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
        return atr_value, atr_pct

    def _compute_ema_metrics(
        self, closes: List[float], price: float
    ) -> Tuple[float, float]:
        ema20 = RollingEMA.from_period(20)
        for c in closes:
            ema20.update(float(c))
        ema20_val = float(ema20.value) if ema20.value is not None else 0.0
        ema20_prev = float(ema20.prev_value) if ema20.prev_value is not None else 0.0
        ema20_slope = (ema20_val - ema20_prev) if ema20_prev > 0 else 0.0
        distance_from_ema20 = (price - ema20_val) / ema20_val if ema20_val > 0 else 0.0
        return ema20_slope, distance_from_ema20

    def _compute_vwap(
        self,
        vwaps: List[Optional[float]],
        timestamps: List[datetime],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        price: float,
    ) -> Tuple[float, float]:
        vwap_val = float(vwaps[-1]) if vwaps[-1] is not None else 0.0
        latest_date_et = timestamps[-1].astimezone(_ET_TZ).date()

        if vwap_val <= 0.0:
            num = 0.0
            den = 0.0
            for ts, h, low_val, c, vol in zip(
                timestamps, highs, lows, closes, volumes, strict=True
            ):
                if ts.astimezone(_ET_TZ).date() != latest_date_et:
                    continue
                v = float(vol)
                if v <= 0:
                    continue
                tp = (float(h) + float(low_val) + float(c)) / 3.0
                num += tp * v
                den += v
            vwap_val = float(num / den) if den > 0 else float(price)

        distance_from_vwap = ((price - vwap_val) / vwap_val) if vwap_val > 0 else 0.0
        return vwap_val, distance_from_vwap

    def _wilder_smooth(self, values: List[float], p: int) -> List[float]:
        if p <= 0 or len(values) < p:
            return []
        sm = float(sum(values[:p]))
        out = [sm]
        for x in values[p:]:
            sm = sm - (sm / float(p)) + float(x)
            out.append(float(sm))
        return out

    def _compute_directional_movements(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
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

        return tr_list, plus_dm_list, minus_dm_list

    def _compute_adx(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> float:
        if len(closes) < 3:
            return 0.0

        period = 14
        tr_list, plus_dm_list, minus_dm_list = self._compute_directional_movements(
            highs, lows, closes
        )

        sm_tr = self._wilder_smooth(tr_list, period)
        sm_pdm = self._wilder_smooth(plus_dm_list, period)
        sm_mdm = self._wilder_smooth(minus_dm_list, period)

        if not sm_tr or len(sm_tr) != len(sm_pdm) or len(sm_pdm) != len(sm_mdm):
            return 0.0

        dx_list = []
        for tr, pdm, mdm in zip(sm_tr, sm_pdm, sm_mdm, strict=True):
            if tr <= 0:
                dx_list.append(0.0)
                continue
            di_plus = 100.0 * (float(pdm) / float(tr))
            di_minus = 100.0 * (float(mdm) / float(tr))
            denom = di_plus + di_minus
            dx = 100.0 * abs(di_plus - di_minus) / denom if denom > 0 else 0.0
            dx_list.append(float(dx))

        if len(dx_list) < period:
            return 0.0

        adx = float(sum(dx_list[:period]) / float(period))
        for dx in dx_list[period:]:
            adx = ((adx * (period - 1)) + float(dx)) / float(period)

        return float(adx)

    def _compute_prior_day_high_low(
        self, timestamps: List[datetime], highs: List[float], lows: List[float]
    ) -> Tuple[float, float]:
        prior_day_high = 0.0
        prior_day_low = 0.0
        by_date: Dict[Any, List[float]] = {}
        latest_date_et = timestamps[-1].astimezone(_ET_TZ).date()

        for ts, h, low_val in zip(timestamps, highs, lows, strict=True):
            d = ts.astimezone(_ET_TZ).date()
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
        return prior_day_high, prior_day_low

    def _compute_bollinger(
        self, closes: List[float], price: float
    ) -> Tuple[float, float, float]:
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
        return bb_upper, bb_lower, price_zscore

    def _compute_premarket_volume(
        self, timestamps: List[datetime], volumes: List[float]
    ) -> float:
        premarket_vol = 0.0
        market_open = time(9, 30)
        latest_date_et = timestamps[-1].astimezone(_ET_TZ).date()

        for ts, vol in zip(timestamps, volumes, strict=True):
            ts_et = ts.astimezone(_ET_TZ)
            if ts_et.date() != latest_date_et:
                continue
            if ts_et.time() < market_open:
                premarket_vol += float(vol)
        return premarket_vol

    def _process_single_flow_trade(
        self, trade: Any
    ) -> Tuple[float, float, int, int, int, float, float]:
        t = trade if isinstance(trade, dict) else trade.__dict__

        size = float(t.get("size", 0))
        pc = t.get("put_call", "UNKNOWN")

        call_vol = 0.0
        put_vol = 0.0
        call_n = 0
        put_n = 0
        sweep_count = 0
        aggressive_qty = 0.0

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

        return call_vol, put_vol, call_n, put_n, sweep_count, aggressive_qty, size

    def compute_flow_metrics(
        self, flow_data: List[Any]
    ) -> Tuple[float, float, int, float, float]:
        """
        Computes options flow metrics.
        Returns (call_put_ratio, flow_zscore, sweep_count, aggressive_flow_share, flow_bias)
        """
        call_vol_total = 0.0
        put_vol_total = 0.0
        call_n_total = 0
        put_n_total = 0
        sweep_count_total = 0
        aggressive_qty_total = 0.0
        total_qty_total = 0.0

        if flow_data and isinstance(flow_data, list):
            for trade in flow_data:
                cv, pv, cn, pn, sc, aq, size = self._process_single_flow_trade(trade)
                call_vol_total += cv
                put_vol_total += pv
                call_n_total += cn
                put_n_total += pn
                sweep_count_total += sc
                aggressive_qty_total += aq
                total_qty_total += size

        if put_vol_total > 0:
            call_put_ratio = call_vol_total / put_vol_total
        elif call_vol_total > 0:
            call_put_ratio = float(call_vol_total)
        else:
            call_put_ratio = 0.0

        aggressive_flow_share = (
            (aggressive_qty_total / total_qty_total) if total_qty_total > 0 else 0.0
        )
        n_total = int(call_n_total + put_n_total)
        flow_zscore = (
            ((call_n_total - put_n_total) / math.sqrt(n_total)) if n_total > 0 else 0.0
        )
        flow_bias = (
            ((call_vol_total - put_vol_total) / total_qty_total)
            if total_qty_total > 0
            else 0.0
        )

        return (
            call_put_ratio,
            flow_zscore,
            sweep_count_total,
            aggressive_flow_share,
            flow_bias,
        )

    def _parse_and_validate_ts(
        self, bar: Any, et_tz: Any, day: Any, market_open: time
    ) -> Optional[Tuple[datetime, float]]:
        bd = bar if isinstance(bar, dict) else getattr(bar, "__dict__", {})
        ts_raw = bd.get("t") or bd.get("timestamp") or getattr(bar, "t", None)

        ts = None
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", self.UTC_OFFSET_STR))
            except Exception:
                return None
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            return None

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        ts_et = ts.astimezone(et_tz)
        if ts_et.date() != day or ts_et.time() < market_open:
            return None

        o = bd.get("o") if bd.get("o") is not None else bd.get("open")
        if o is None:
            return None
        try:
            return ts, float(o)
        except Exception:
            return None

    def calculate_session_open_price(self, bars: List[Any], now_utc: datetime) -> float:
        """Helper to find session open price"""
        open_px = 0.0
        now_et = now_utc.astimezone(_ET_TZ)
        day = now_et.date()
        market_open = time(9, 30)
        best_ts: Optional[datetime] = None

        for b in bars:
            res = self._parse_and_validate_ts(b, _ET_TZ, day, market_open)
            if res is None:
                continue

            ts, o_f = res
            if best_ts is None or ts < best_ts:
                best_ts = ts
                open_px = o_f

        return float(open_px)
