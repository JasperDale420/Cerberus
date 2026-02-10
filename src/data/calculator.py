import math
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz

from src.core.domain import TechnicalFeatures
from src.core.indicators import RollingADX, RollingATR, RollingEMA, RollingStd

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

    @staticmethod
    def calculate_relative_strength(
        symbol_closes: List[float], benchmark_closes: List[float]
    ) -> float:
        """
        Calculate Relative Strength (Price Performance) vs Benchmark.
        Returns the difference in returns over the provided window.
        """
        if not symbol_closes or not benchmark_closes:
            return 0.0

        sym_ret = (symbol_closes[-1] / symbol_closes[0]) - 1.0
        bench_ret = (benchmark_closes[-1] / benchmark_closes[0]) - 1.0

        return sym_ret - bench_ret

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

        atr, atr_pct = self._compute_atr(closes, highs, lows, price)
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
        orb_high, orb_low = self._compute_opening_range(timestamps, highs, lows)

        ma_dist_50 = self._compute_ma_dist(closes, price, 50)
        ma_dist_200 = self._compute_ma_dist(closes, price, 200)

        return TechnicalFeatures(
            price=price,
            volume=volume,
            timestamp=timestamp,
            atr=atr,
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
            last_updated=timestamp,  # Use the bar timestamp as last_updated
            orb_high=orb_high,
            orb_low=orb_low,
            ma_dist_50=ma_dist_50,
            ma_dist_200=ma_dist_200,
            tfi=0.0,
            net_gex=0.0,
            gex_flip_dist=0.0,
            frac_diff_close=0.0,
            hurst_exponent=0.0,
        )

    def _compute_opening_range(
        self, timestamps: List[datetime], highs: List[float], lows: List[float]
    ) -> Tuple[float, float]:
        """
        Calculates the Opening Range High and Low for the current day.
        Assumes a 30-minute opening range from market open (9:30 AM ET).
        """
        orb_high = 0.0
        orb_low = float("inf")
        market_open_time = time(9, 30)
        orb_duration_minutes = 30

        if not timestamps:
            return 0.0, 0.0

        latest_date_et = timestamps[-1].astimezone(_ET_TZ).date()

        for ts, h, low_val in zip(timestamps, highs, lows, strict=True):
            ts_et = ts.astimezone(_ET_TZ)

            if ts_et.date() == latest_date_et:
                # Check if within the opening range
                market_open_dt = _ET_TZ.localize(
                    datetime.combine(latest_date_et, market_open_time)
                )
                orb_end_dt = market_open_dt + timedelta(minutes=orb_duration_minutes)

                if market_open_dt <= ts_et <= orb_end_dt:
                    orb_high = max(orb_high, float(h))
                    orb_low = min(orb_low, float(low_val))

        if orb_low == float("inf"):  # No data found in ORB
            orb_low = 0.0

        return orb_high, orb_low

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

            # 1. Parse Timestamp (handle 'time', 't', or 'timestamp')
            ts_val = (
                bd.get("time")
                or bd.get("t")
                or bd.get("timestamp")
                or getattr(b, "time", None)
                or getattr(b, "t", None)
            )
            ts = self._parse_ts(ts_val)
            if ts is None:
                continue

            # 2. Parse OHLCV (handle full names or shorthand)
            o = self._to_float(
                bd.get("open")
                or bd.get("o")
                or getattr(b, "open", None)
                or getattr(b, "o", None)
            )
            h = self._to_float(
                bd.get("high")
                or bd.get("h")
                or getattr(b, "high", None)
                or getattr(b, "h", None)
            )
            low_val = self._to_float(
                bd.get("low")
                or bd.get("l")
                or getattr(b, "low", None)
                or getattr(b, "l", None)
            )
            c = self._to_float(
                bd.get("close")
                or bd.get("c")
                or getattr(b, "close", None)
                or getattr(b, "c", None)
            )
            v = self._to_float(
                bd.get("volume")
                or bd.get("v")
                or getattr(b, "volume", None)
                or getattr(b, "v", None)
            )
            vwap = self._to_float(bd.get("vwap") or getattr(b, "vwap", None))

            if o is None or h is None or low_val is None or c is None or v is None:
                continue
            rows.append((ts, o, h, low_val, c, v, vwap))

        rows.sort(key=lambda r: r[0])
        return rows

    def _compute_atr(
        self, closes: List[float], highs: List[float], lows: List[float], price: float
    ) -> Tuple[float, float]:
        """Compute ATR using RollingATR incremental indicator."""
        if len(closes) < 2:
            return 0.0, 0.0

        atr = RollingATR(period=14)
        for i in range(len(closes)):
            atr.update(float(highs[i]), float(lows[i]), float(closes[i]))

        atr_value = float(atr.value) if atr.value is not None else 0.0
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

    def _compute_ma_dist(self, closes: List[float], price: float, period: int) -> float:
        """Simple Moving Average distance calculation."""
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        sma = sum(window) / period
        return (price - sma) / sma if sma > 0 else 0.0

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

    def _compute_adx(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> float:
        """Compute ADX using RollingADX incremental indicator."""
        if len(closes) < 3:
            return 0.0

        adx = RollingADX(period=14)
        for i in range(len(closes)):
            adx.update(float(highs[i]), float(lows[i]), float(closes[i]))

        return float(adx.value) if adx.value is not None else 0.0

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

        # Flow bias: normalized directional pressure in [-1, 1]
        denom = call_vol_total + put_vol_total
        flow_bias = ((call_vol_total - put_vol_total) / denom) if denom > 0 else 0.0

        aggressive_flow_share = (
            (aggressive_qty_total / total_qty_total) if total_qty_total > 0 else 0.0
        )
        n_total = int(call_n_total + put_n_total)
        flow_zscore = (
            ((call_n_total - put_n_total) / math.sqrt(n_total)) if n_total > 0 else 0.0
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

    def calculate_tfi(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculates Trade Flow Imbalance (TFI) using the Tick Test (Lee-Ready).
        TFI = (BuyVolume - SellVolume) / TotalVolume
        Result range: [-1.0, 1.0]
        """
        if not trades:
            return 0.0

        buy_vol = 0.0
        sell_vol = 0.0
        total_vol = 0.0
        last_price: Optional[float] = None
        last_dir = 0  # 1 for buy, -1 for sell

        for t in trades:
            px = float(t.get("p", 0))
            sz = float(t.get("s", 0))
            total_vol += sz

            if last_price is not None:
                if px > last_price:
                    # Uptick -> Buy
                    buy_vol += sz
                    last_dir = 1
                elif px < last_price:
                    # Downtick -> Sell
                    sell_vol += sz
                    last_dir = -1
                else:
                    # No change -> Use last direction (Tick Test)
                    if last_dir == 1:
                        buy_vol += sz
                    elif last_dir == -1:
                        sell_vol += sz

            last_price = px

        if total_vol <= 0:
            return 0.0

        return (buy_vol - sell_vol) / total_vol

    def calculate_gex_metrics(
        self, exposure_data: List[Dict[str, Any]], spot_price: float
    ) -> Tuple[float, float]:
        """
        Calculates Net Gamma Exposure (GEX) and distance to GEX Flip.
        Net GEX = (Sum of Call Gamma - Sum of Put Gamma) * Multiplier
        Note: Multiplier is assumed to be 0.1% or 1% of spot as per UW definition.

        Returns: (net_gex, gex_flip_dist)
        """
        if not exposure_data or spot_price <= 0:
            return 0.0, 0.0

        total_net_gex = 0.0
        # Stratify by strike to find flip
        by_strike: Dict[float, float] = {}

        for item in exposure_data:
            try:
                strike = float(item.get("strike", 0))
                cg = float(item.get("call_gamma", 0))
                pg = float(item.get("put_gamma", 0))

                # Standard convention: Net GEX at strike = CallGamma - PutGamma
                # This approximates MMs being short calls/puts
                net_at_strike = cg - pg
                total_net_gex += net_at_strike

                by_strike[strike] = net_at_strike
            except (TypeError, ValueError):
                continue

        # Find "Flip" strike (where net GEX is min / closest to zero)
        # Or ideally where it crosses zero. For simplicity, we find the absolute minimum net exposure strike
        # as a proxy for the 'Zero GEX' pin or flip point.
        if not by_strike:
            return total_net_gex, 0.0

        # Sort strikes
        sorted_strikes = sorted(by_strike.keys())

        # Simple heuristic: find the strike closest to zero net exposure
        flip_strike = min(by_strike.keys(), key=lambda s: abs(by_strike[s]))

        # Alternatively: Find where it crosses zero
        for i in range(len(sorted_strikes) - 1):
            s1, s2 = sorted_strikes[i], sorted_strikes[i + 1]
            v1, v2 = by_strike[s1], by_strike[s2]
            if (v1 < 0 and v2 > 0) or (v1 > 0 and v2 < 0):
                # Zero cross found!
                flip_strike = s1 if abs(v1) < abs(v2) else s2
                break

        dist_to_flip = (
            (flip_strike - spot_price) / spot_price if spot_price > 0 else 0.0
        )

        return total_net_gex, dist_to_flip

    @staticmethod
    def get_frac_diff_weights(d: float, size: int) -> List[float]:
        """
        Compute weights for fractional differentiation using iterative formula.
        w_k = -w_{k-1} * (d - k + 1) / k
        """
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1.0) / k
            w.append(w_k)
        return w

    def apply_frac_diff(
        self, series: List[float], d: float, threshold: float = 1e-5
    ) -> float:
        """
        Applies Fractional Differentiation (Fixed-width window) to a series.
        Preserves memory while achieving stationarity.
        Returns the last fractionally differentiated value.
        """
        if len(series) < 2:
            return series[-1] if series else 0.0

        # Determine weight window (ignore weights below threshold)
        weights = self.get_frac_diff_weights(d, len(series))

        # Only use weights above threshold for computational efficiency
        # We start from the end and apply weights backwards
        res = 0.0
        for i, w in enumerate(weights):
            if abs(w) < threshold and i > 0:
                break
            # Current value is at index -1, previous at -2, etc.
            res += w * series[-(i + 1)]

        return res

    def calculate_hurst_exponent(self, series: List[float]) -> float:
        """
        Calculates the Hurst exponent using Rescaled Range (R/S) analysis.
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(series) < 50:  # Need sufficient data for R/S analysis
            return 0.5

        # Log returns
        rets = [math.log(series[i] / series[i - 1]) for i in range(1, len(series))]
        n = len(rets)

        # We use a single window for simplicity in this implementation,
        # but full R/S would use multiple window sizes.
        # This is the "Simplified Hurst" or "Single Window Hurst".

        mean_ret = sum(rets) / n
        centered = [r - mean_ret for r in rets]

        # Cumulative deviation
        cum_dev = []
        curr = 0.0
        for c in centered:
            curr += c
            cum_dev.append(curr)

        # Range
        r = max(cum_dev) - min(cum_dev)

        # Standard Deviation
        variance = sum((r_i - mean_ret) ** 2 for r_i in rets) / n
        s = math.sqrt(variance)

        if s == 0:
            return 0.5

        # R/S = (Max - Min) / StdDev
        rs = r / s

        # Hurst: rs = (n/2)^H  => H = log(rs) / log(n/2)
        # Using n/2 as a standard normalization for short-window Hurst estimations
        try:
            h = math.log(rs) / math.log(n)
            return max(0.0, min(1.0, h))
        except (ValueError, ZeroDivisionError):
            return 0.5
