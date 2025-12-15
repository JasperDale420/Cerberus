from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta
import pytz  # type: ignore

from src.core.domain import SymbolFeatures
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
    ):
        self.alpaca_client = alpaca_client
        self.unusual_whales_client = unusual_whales_client
        self.logger = logger

    async def compute_features(self, symbols: List[str]) -> Dict[str, SymbolFeatures]:
        """
        Computes features for a list of symbols.
        """
        features = {}

        # In a real implementation, we would batch requests or use async gather
        # For this slice, we'll iterate (or use a simple gather if clients support async)

        # Alpaca historical client is synchronous in the wrapper I wrote,
        # but we can wrap it or just call it.
        # UW client is async.

        for symbol in symbols:
            try:
                # 1. Fetch Price Data (Snapshot)
                # Using get_stock_snapshot or similar if available, or just latest bar
                # For now, let's assume we want the latest minute bar
                from datetime import timezone

                end = datetime.now(timezone.utc)
                start = end - timedelta(minutes=5)

                # This is a synchronous call in my wrapper
                bars_data = self.alpaca_client.get_historical_bars(symbol, start, end)

                if not bars_data:
                    self.logger.warning("No bars found for symbol", symbol=symbol)
                    continue

                # Assuming bars_data is a list of dicts or BarResponse
                if isinstance(bars_data, dict) and "bars" in bars_data:
                    bars_data = bars_data["bars"]

                # 1. Compute Technicals
                try:
                    tech_result = self._compute_technicals(bars_data)
                    if not tech_result:
                        continue

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

                    # Fallback for Prior Day Levels if missing (e.g., first run of day)
                    if prior_day_high == 0.0 or prior_day_low == 0.0:
                        p_high, p_low, p_close = self._fetch_prior_day_stats(
                            symbol, end
                        )
                        if p_high > 0:
                            prior_day_high = p_high
                            prior_day_low = p_low
                            # Recalculate gap if we have prior close now
                            # Gap = (Open - PriorClose) / PriorClose
                            # We need today's Open. We can get it from bars_data[0] if available?
                            # Or use 'price' as approx if very early? No.
                            # Let's try to get open from technicals or bars_data
                            # _compute_technicals doesn't return open of day easily unless added.
                            # But we have bars_data.
                            if bars_data and len(bars_data) > 0:
                                # Assuming bars_data is sorted by time
                                first_bar = bars_data[0]
                                open_val = (
                                    first_bar["o"]
                                    if isinstance(first_bar, dict)
                                    else first_bar.o
                                )
                                gap_pct = (
                                    ((open_val - p_close) / p_close)
                                    if p_close > 0
                                    else 0.0
                                )

                except Exception as e:
                    self.logger.error(
                        "Failed to compute technicals", symbol=symbol, error=str(e)
                    )
                    continue

                # 2. Fetch & Compute Options Flow
                # Async call
                flow_data = await self.unusual_whales_client.get_option_flow(
                    symbol, end.strftime("%Y-%m-%d")
                )

                call_put_ratio, flow_zscore, sweep_count, aggressive_flow_share = (
                    self._compute_flow_metrics(flow_data)
                )

                feat = SymbolFeatures(
                    symbol=symbol,
                    last_updated=(
                        timestamp
                        if isinstance(timestamp, datetime)
                        else datetime.fromisoformat(
                            str(timestamp).replace("Z", "+00:00")
                        )
                    ),
                    price=price,
                    avg_volume=float(volume),
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
                        "volatility": atr_pct,
                    },
                )

                features[symbol] = feat

            except Exception as e:
                self.logger.error(
                    "Failed to compute features", symbol=symbol, error=str(e)
                )

        return features

    def _compute_technicals(self, bars_data: List[Any]) -> Optional[tuple]:
        if not bars_data:
            return None

        data_list = []
        for b in bars_data:
            if isinstance(b, dict):
                data_list.append(
                    {
                        "timestamp": b.get("t") or b.get("timestamp"),
                        "open": b.get("o") or b.get("open"),
                        "high": b.get("h") or b.get("high"),
                        "low": b.get("l") or b.get("low"),
                        "close": b.get("c") or b.get("close"),
                        "volume": b.get("v") or b.get("volume"),
                        "vwap": b.get("vwap"),
                    }
                )
            else:
                data_list.append(
                    {
                        "timestamp": b.t,
                        "open": b.o,
                        "high": b.h,
                        "low": b.l,
                        "close": b.c,
                        "volume": b.v,
                        "vwap": getattr(b, "vwap", None),
                    }
                )

        df = pd.DataFrame(data_list)
        if df.empty:
            return None

        # Ensure index for pandas-ta
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True, drop=False)

        # Calculate Technicals
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema20"] = ta.ema(df["close"], length=20)

        df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

        # VWAP
        if "vwap" not in df.columns or df["vwap"].isnull().all():
            # Use explicit args to ensure correct computation
            df.ta.vwap(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                append=True,
            )

        latest = df.iloc[-1]
        price = float(latest["close"])
        volume = float(latest["volume"])
        timestamp = latest["timestamp"]

        atr_value = float(latest["atr"]) if pd.notnull(latest["atr"]) else 0.0
        atr_pct = (atr_value / price) if price > 0 else 0.0

        high = float(latest["high"])
        low = float(latest["low"])
        open_ = float(latest["open"])
        intraday_range_pct = ((high - low) / open_) if open_ > 0 else 0.0

        gap_pct = 0.0
        if len(df) > 1:
            prev_close = float(df.iloc[-2]["close"])
            gap_pct = ((open_ - prev_close) / prev_close) if prev_close > 0 else 0.0

        ema20_slope = 0.0
        distance_from_ema20 = 0.0

        if pd.notnull(latest["ema20"]):
            ema20_val = float(latest["ema20"])
            if len(df) >= 2 and pd.notnull(df.iloc[-2]["ema20"]):
                ema20_slope = ema20_val - float(df.iloc[-2]["ema20"])

            if ema20_val > 0:
                distance_from_ema20 = (price - ema20_val) / ema20_val

        vwap_val = float(latest.get("VWAP_D", latest.get("vwap", price)))
        distance_from_vwap = ((price - vwap_val) / vwap_val) if vwap_val > 0 else 0.0

        adx_val = float(latest["adx"]) if pd.notnull(latest.get("adx")) else 0.0

        # Prior Day Levels
        # If we have enough data (at least 2 days of intraday bars), we can approximate.
        # Ideally, we'd query daily bars separately.
        # For this slice, assuming 'df' covers enough history (e.g. 5 days of minute bars).
        # We resample to daily to find prior H/L.

        prior_day_high = 0.0
        prior_day_low = 0.0

        try:
            # Resample to Daily
            daily_df = (
                df.resample("D")
                .agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )

            # We want the LAST COMPLETED day, not today.
            # If 'latest' is today, we want the row before it.

            latest_date = latest["timestamp"].date()
            prior_days = daily_df[daily_df.index.date < latest_date]

            if not prior_days.empty:
                last_full_day = prior_days.iloc[-1]
                prior_day_high = float(last_full_day["high"])
                prior_day_low = float(last_full_day["low"])
        except Exception:
            # Fallback or strict error?
            # Safe to proceed with 0.0, strategy filters will just fail.
            pass

        # Ensure timestamp is timezone-aware for comparison
        if df["timestamp"].dt.tz is None:
            # Assume UTC if naive, or ET? Alpaca usually UTC in API, but let's check config.
            # Best effort: localize to UTC then convert.
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        # Convert to ET for session logic
        et_tz = pytz.timezone("US/Eastern")
        df["timestamp_et"] = df["timestamp"].dt.tz_convert(et_tz)

        # Bollinger Bands (20, 2)
        bb_upper = 0.0
        bb_lower = 0.0
        price_zscore = 0.0

        try:
            # pandas_ta bbands returns a dataframe with columns BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
            bb = ta.bbands(df["close"], length=20, std=2.0)
            if bb is not None:
                # Append to df to use latest easily
                df = pd.concat([df, bb], axis=1)
                latest_bb = df.iloc[-1]

                # Check for correct column names, pandas_ta format is typically BBL_length_std
                bbl = latest_bb.get("BBL_20_2.0", 0.0)
                bbu = latest_bb.get("BBU_20_2.0", 0.0)
                bbm = latest_bb.get("BBM_20_2.0", 0.0)  # SMA

                bb_upper = float(bbu)
                bb_lower = float(bbl)

                # Z-Score approximation: (Price - Mean) / StdDev
                # StdDev approx = (Upper - Mean) / 2.0?
                # Or use ta.zscore if available?
                # Let's trust pandas_ta zscore if we want exact, or derive from bands.
                # Derive: BandWidth = 4 * std. std = (Upper - Lower) / 4.
                if (bb_upper - bb_lower) > 0:
                    std_dev = (bb_upper - bb_lower) / 4.0
                    if std_dev > 0:
                        price_zscore = (price - float(bbm)) / std_dev
                if (bb_upper - bb_lower) > 0:
                    std_dev = (bb_upper - bb_lower) / 4.0
                    if std_dev > 0:
                        price_zscore = (price - float(bbm)) / std_dev
        except Exception:
            pass

        # Premarket Volume
        # Sum volume where time < 09:30 of current day
        premarket_vol = 0.0
        try:
            # Use ET timestamps
            latest_et = df["timestamp_et"].iloc[-1]
            current_date_et = latest_et.date()

            # Filter for today (ET)
            today_mask = df["timestamp_et"].dt.date == current_date_et
            today_df = df[today_mask]

            # Market Open 09:30 ET
            # Create timestamp in ET
            market_open_et = (
                pd.Timestamp(current_date_et)
                .tz_localize("US/Eastern")
                .replace(hour=9, minute=30)
            )

            # Simple check
            premarket_mask = today_df["timestamp_et"] < market_open_et
            premarket_df = today_df[premarket_mask]

            if not premarket_df.empty:
                premarket_vol = float(premarket_df["volume"].sum())

        except Exception:
            pass

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
        call_vol = 0.0
        put_vol = 0.0
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
                elif pc == "PUT":
                    put_vol += size

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
        flow_zscore = ((call_vol - put_vol) / total_qty) if total_qty > 0 else 0.0
        return (call_put_ratio, flow_zscore, sweep_count, aggressive_flow_share)

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

            # Handle response format (Dict or List)
            # Assuming list of dicts based on get_historical_bars placeholder
            # Filter for completed days < current_time.date()
            # Note: current_time is UTC. We should compare dates carefully.

            # Flatten if dict
            if isinstance(bars, dict) and "bars" in bars:
                bars = bars["bars"]

            valid_bars = []
            for b in bars:
                # Normalize to dict
                bd = b if isinstance(b, dict) else b.__dict__
                t = bd.get("t") or bd.get("timestamp")
                if isinstance(t, str):
                    t = datetime.fromisoformat(t.replace("Z", "+00:00"))

                # Check for t being valid
                if t and t.date() < current_time.date():
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
