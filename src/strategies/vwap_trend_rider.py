from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta as ta

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class VWAPTrendRiderStrategy(BaseStrategy):
    """
    VWAP Trend Rider.
    Enters when price pulls back to VWAP and then RECLAIMS it in the direction of the dominant trend.
    Requires:
    1. Trend (EMA20 vs EMA50)
    2. Pullback (Price touches/crosses VWAP)
    3. Reclaim (Price crosses back above/below VWAP)
    4. Volume Confirmation (Volume > Avg Volume * mult)
    """

    name = "vwap_trend_rider"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.ema_fast_len = config.get("ema_fast", 20)
        self.ema_slow_len = config.get("ema_slow", 50)
        self.vol_mult = config.get("vol_mult", 1.2)  # Volume must be 1.2x average
        self.risk_reward = config.get("risk_reward", 2.0)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Need history
        if not symbol_state.bars or len(symbol_state.bars) < self.ema_slow_len + 5:
            return None

        # 1. Calculate Indicators
        bars = list(symbol_state.bars)
        df = pd.DataFrame([vars(b) for b in bars])
        close = df["close"].astype(float)

        # EMAs
        ema_fast = ta.ema(close, length=self.ema_fast_len)
        ema_slow = ta.ema(close, length=self.ema_slow_len)

        # VWAP (if not in bar, calc it)
        # Assuming bar has 'vwap' or we calc using pandas_ta
        # Pipeline usually attaches it, but we re-calc to be safe or verify
        if "vwap" not in df.columns or df["vwap"].isnull().all():
            df.ta.vwap(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                append=True,
            )

        # Average Volume (Simple Moving Average 20)
        avg_vol_series = ta.sma(df["volume"], length=20)

        if ema_fast is None or ema_slow is None or avg_vol_series is None:
            return None

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]

        # Use latest VWAP (Check column names, pandas_ta produces VWAP_D usually)
        vwap_col = "VWAP_D" if "VWAP_D" in df.columns else "vwap"
        if vwap_col not in df.columns:
            return None  # Can't trade without VWAP

        current_vwap = df[vwap_col].iloc[-1]
        prev_vwap = df[vwap_col].iloc[-2]

        current_vol = bar.volume
        avg_vol = avg_vol_series.iloc[
            -2
        ]  # Use prev bar avg to not bias with current? Or current bar SMA?
        # Current bar is closed (we assume on_bar is called after close usually, or on new bar)
        # If 'bar' is the *just closed* bar, we use its volume.

        # 2. Determine Trend
        is_uptrend = current_fast > current_slow
        is_downtrend = current_fast < current_slow

        # 3. Check for Reclaim + Volume

        # BULLISH RIDER
        if is_uptrend and market_state.regime in [Regime.BULL, Regime.CHOP]:
            # Trigger: Cross OVER VWAP
            # Condition: Previous Close < Prev VWAP (or Low < VWAP)
            #           Current Close > Current VWAP

            # More lenient pullback: Low touches VWAP area, Close strong.
            # Strict Reclaim: Close crosses from below to above.

            prev_close = df["close"].iloc[-2]

            # Did we cross UP?
            cross_up = (prev_close < prev_vwap) and (bar.close > current_vwap)

            if cross_up:
                # Check Volume
                if current_vol > (avg_vol * self.vol_mult):
                    # ENTRY LONG
                    stop_loss = min(
                        [b.low for b in bars[-3:]]
                    )  # Recent swing low match
                    if stop_loss >= bar.close:
                        stop_loss = bar.close * 0.995  # fallback

                    risk = bar.close - stop_loss
                    target_price = bar.close + (risk * self.risk_reward)

                    return Signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        size_hint=0,
                        entry_price=bar.close,
                        stop_price=stop_loss,
                        target_price=target_price,
                        strategy=self.name,
                        regime=market_state.regime,
                        generated_at=bar.time,
                        meta={"vwap": current_vwap, "vol_mult": current_vol / avg_vol},
                        correlation_id=f"{self.name}-long-{symbol}-{bar.time.timestamp()}",
                    )

        # BEARISH RIDER
        if is_downtrend and market_state.regime in [Regime.BEAR, Regime.CHOP]:
            prev_close = df["close"].iloc[-2]

            # Did we cross DOWN?
            cross_down = (prev_close > prev_vwap) and (bar.close < current_vwap)

            if cross_down:
                # Check Volume
                if current_vol > (avg_vol * self.vol_mult):
                    # ENTRY SHORT
                    stop_loss = max([b.high for b in bars[-3:]])
                    if stop_loss <= bar.close:
                        stop_loss = bar.close * 1.005

                    risk = stop_loss - bar.close
                    target_price = bar.close - (risk * self.risk_reward)

                    return Signal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        size_hint=0,
                        entry_price=bar.close,
                        stop_price=stop_loss,
                        target_price=target_price,
                        strategy=self.name,
                        regime=market_state.regime,
                        generated_at=bar.time,
                        meta={"vwap": current_vwap, "vol_mult": current_vol / avg_vol},
                        correlation_id=f"{self.name}-short-{symbol}-{bar.time.timestamp()}",
                    )

        return None
