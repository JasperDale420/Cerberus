from typing import Any, Dict, Optional

import pandas as pd
import pandas_ta as ta

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


class TrendPullbackStrategy(BaseStrategy):
    """
    Trend Pullback Strategy.
    Enters on pullbacks to EMA20 in the direction of the trend (EMA20 vs EMA50).
    """

    name = "trend_pullback"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self.ema_fast_len = config.get("ema_fast", 20)
        self.ema_slow_len = config.get("ema_slow", 50)
        self.rsi_len = config.get("rsi_len", 2)  # Fast RSI for trigger
        self.risk_reward = config.get("risk_reward", 2.0)

        # Thresholds
        self.rsi_oversold = config.get("rsi_oversold", 10)
        self.rsi_overbought = config.get("rsi_overbought", 90)

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Need history
        if not symbol_state.bars or len(symbol_state.bars) < self.ema_slow_len + 10:
            return None

        # 1. Calculate Indicators
        bars = list(symbol_state.bars)
        df = pd.DataFrame([vars(b) for b in bars])

        # Ensure 'close' is float
        close = df["close"].astype(float)

        # Compute EMAs
        ema_fast = ta.ema(close, length=self.ema_fast_len)
        ema_slow = ta.ema(close, length=self.ema_slow_len)
        rsi = ta.rsi(close, length=self.rsi_len)

        if ema_fast is None or ema_slow is None or rsi is None:
            return None

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_price = bar.close

        # 2. Determine Trend Direction
        is_bullish = current_fast > current_slow
        is_bearish = current_fast < current_slow

        # 3. Check Signal
        signal = None

        if is_bullish and market_state.regime in [Regime.BULL, Regime.CHOP]:
            # Bullish Pullback
            # Condition 1: RSI was oversold recently (pullback depth)
            # OR logic: Price touched EMA20? (Implicit in RSI oversold usually)

            # Simple RSI trigger: Crossing back UP over threshold
            if prev_rsi < self.rsi_oversold and current_rsi > self.rsi_oversold:
                # Trigger ENTRY LONG
                stop_loss = min([b.low for b in bars[-3:]])  # Low of Swing
                # If stop is too tight, use ATR (omitted for now)

                risk = current_price - stop_loss
                if risk <= 0:
                    return None

                target_price = current_price + (risk * self.risk_reward)

                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    size_hint=0,
                    entry_price=current_price,
                    stop_price=stop_loss,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={
                        "ema_fast": current_fast,
                        "ema_slow": current_slow,
                        "rsi": current_rsi,
                    },
                    correlation_id=f"{self.name}-{symbol}-{bar.time.timestamp()}",
                )

        elif is_bearish and market_state.regime in [Regime.BEAR, Regime.CHOP]:
            # Bearish Pullback
            if prev_rsi > self.rsi_overbought and current_rsi < self.rsi_overbought:
                # Trigger ENTRY SHORT
                stop_loss = max([b.high for b in bars[-3:]])

                risk = stop_loss - current_price
                if risk <= 0:
                    return None

                target_price = current_price - (risk * self.risk_reward)

                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    size_hint=0,
                    entry_price=current_price,
                    stop_price=stop_loss,
                    target_price=target_price,
                    strategy=self.name,
                    regime=market_state.regime,
                    generated_at=bar.time,
                    meta={
                        "ema_fast": current_fast,
                        "ema_slow": current_slow,
                        "rsi": current_rsi,
                    },
                    correlation_id=f"{self.name}-{symbol}-{bar.time.timestamp()}",
                )

        return signal
