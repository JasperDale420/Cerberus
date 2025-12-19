from typing import Any, Dict, Optional

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
        self.pullback_depth_pct = float(config.get("pullback_depth_pct", 0.0) or 0.0)
        self.entry_confirmation = str(
            config.get("entry_confirmation", "rsi") or "rsi"
        ).lower()

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

        bars = list(symbol_state.bars)
        current_price = float(bar.close)

        # Prefer cached indicators from engine; fall back to deterministic local computation.
        current_fast = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_fast_len)}"
        )
        current_slow = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_slow_len)}"
        )
        if current_fast is None or current_slow is None:
            # Local EMA fallback
            def _ema_last(values: list[float], period: int) -> float | None:
                p = max(1, int(period))
                alpha = 2.0 / (p + 1.0)
                ema: float | None = None
                for x in values:
                    ema = x if ema is None else (alpha * x) + ((1.0 - alpha) * ema)
                return ema

            closes = [float(b.close) for b in bars]
            current_fast = _ema_last(closes, int(self.ema_fast_len))
            current_slow = _ema_last(closes, int(self.ema_slow_len))

        if current_fast is None or current_slow is None:
            return None

        current_rsi = None
        prev_rsi = None
        if self.entry_confirmation == "rsi":
            current_rsi = symbol_state.indicators.get(f"rsi:{int(self.rsi_len)}")
            prev_rsi = symbol_state.indicators.get(f"rsi:{int(self.rsi_len)}:prev")
            if current_rsi is None or prev_rsi is None:
                # Local RSI fallback (Wilder smoothing), return last two values.
                closes = [float(b.close) for b in bars]
                p = max(1, int(self.rsi_len))
                if len(closes) < p + 2:
                    return None
                avg_gain = None
                avg_loss = None
                prev_close = closes[0]
                series: list[float] = []
                for c in closes[1:]:
                    change = c - prev_close
                    gain = max(0.0, change)
                    loss = max(0.0, -change)
                    if avg_gain is None or avg_loss is None:
                        avg_gain = gain
                        avg_loss = loss
                    else:
                        avg_gain = ((avg_gain * (p - 1)) + gain) / p
                        avg_loss = ((avg_loss * (p - 1)) + loss) / p
                    prev_close = c
                    if (avg_loss or 0.0) == 0.0:
                        series.append(100.0)
                    else:
                        rs = float(avg_gain or 0.0) / float(avg_loss)
                        series.append(100.0 - (100.0 / (1.0 + rs)))
                if len(series) < 2:
                    return None
                prev_rsi = series[-2]
                current_rsi = series[-1]

        # 2. Determine Trend Direction
        is_bullish = float(current_fast) > float(current_slow)
        is_bearish = float(current_fast) < float(current_slow)

        # 3. Check Signal
        signal = None

        if is_bullish and market_state.regime == Regime.BULL:
            if self.pullback_depth_pct > 0 and float(current_fast):
                dist = abs((current_price - float(current_fast)) / float(current_fast))
                if dist > self.pullback_depth_pct:
                    return None
            # Bullish Pullback
            # Condition 1: RSI was oversold recently (pullback depth)
            # OR logic: Price touched EMA20? (Implicit in RSI oversold usually)

            # Simple RSI trigger: Crossing back UP over threshold
            if self.entry_confirmation == "none":
                confirmed = True
            else:
                confirmed = bool(
                    prev_rsi is not None
                    and current_rsi is not None
                    and float(prev_rsi) < self.rsi_oversold
                    and float(current_rsi) > self.rsi_oversold
                )
            if confirmed:
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
                        "ema_fast": float(current_fast),
                        "ema_slow": float(current_slow),
                        "rsi": float(current_rsi) if current_rsi is not None else None,
                    },
                )

        elif is_bearish and market_state.regime == Regime.BEAR:
            if self.pullback_depth_pct > 0 and float(current_fast):
                dist = abs((current_price - float(current_fast)) / float(current_fast))
                if dist > self.pullback_depth_pct:
                    return None
            # Bearish Pullback
            if self.entry_confirmation == "none":
                confirmed = True
            else:
                confirmed = bool(
                    prev_rsi is not None
                    and current_rsi is not None
                    and float(prev_rsi) > self.rsi_overbought
                    and float(current_rsi) < self.rsi_overbought
                )
            if confirmed:
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
                        "ema_fast": float(current_fast),
                        "ema_slow": float(current_slow),
                        "rsi": float(current_rsi) if current_rsi is not None else None,
                    },
                )

        return signal
