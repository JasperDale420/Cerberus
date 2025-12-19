from typing import Any, Dict, Optional

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
        # PRD 7.2: only in strong trend (trend_score high).
        self.min_trend_score = float(config.get("min_trend_score", 1.5) or 1.5)

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

        # PRD 7.2: VWAP Trend Rider is BULL/BEAR only and requires strong trend_score.
        if market_state.regime not in (Regime.BULL, Regime.BEAR):
            return None
        trend_score = None
        if isinstance(getattr(market_state, "meta", None), dict):
            trend_score = market_state.meta.get("trend_score")
        try:
            trend_score_f = float(trend_score) if trend_score is not None else 0.0
        except Exception:
            trend_score_f = 0.0
        if trend_score_f < self.min_trend_score:
            return None

        bars = list(symbol_state.bars)

        # Prefer cached EMAs and volume SMA from engine; fall back to deterministic local computation.
        current_fast = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_fast_len)}"
        )
        current_slow = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_slow_len)}"
        )
        if current_fast is None or current_slow is None:

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

        # VWAP is injected by engine as session VWAP (preferred).
        current_vwap = getattr(bar, "vwap", None)
        prev_vwap = None
        if len(bars) >= 2:
            prev_vwap = getattr(bars[-2], "vwap", None)
        try:
            current_vwap_f = float(current_vwap) if current_vwap is not None else None
            prev_vwap_f = float(prev_vwap) if prev_vwap is not None else None
        except Exception:
            current_vwap_f = None
            prev_vwap_f = None
        if current_vwap_f is None or prev_vwap_f is None:
            return None

        current_vol = float(getattr(bar, "volume", 0.0) or 0.0)
        avg_vol = symbol_state.indicators.get("sma_vol:20:prev")
        if avg_vol is None:
            # Local SMA fallback
            vols = [float(b.volume) for b in bars[-21:-1]]  # prior 20 bars
            if not vols:
                return None
            avg_vol = sum(vols) / len(vols)
        try:
            avg_vol_f = float(avg_vol)
        except Exception:
            return None

        # 2. Determine Trend
        is_uptrend = float(current_fast) > float(current_slow)
        is_downtrend = float(current_fast) < float(current_slow)

        # 3. Check for Reclaim + Volume

        # BULLISH RIDER
        if is_uptrend and market_state.regime == Regime.BULL:
            # Trigger: Cross OVER VWAP
            # Condition: Previous Close < Prev VWAP (or Low < VWAP)
            #           Current Close > Current VWAP

            # More lenient pullback: Low touches VWAP area, Close strong.
            # Strict Reclaim: Close crosses from below to above.

            prev_close = float(bars[-2].close) if len(bars) >= 2 else float(bar.close)

            # Did we cross UP?
            cross_up = (prev_close < prev_vwap_f) and (
                float(bar.close) > current_vwap_f
            )

            if cross_up:
                # Check Volume
                if current_vol > (avg_vol_f * float(self.vol_mult)):
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
                        meta={
                            "vwap": float(current_vwap_f),
                            "vol_mult": (
                                (current_vol / avg_vol_f) if avg_vol_f else None
                            ),
                            "trend_score": float(trend_score_f),
                        },
                    )

        # BEARISH RIDER
        if is_downtrend and market_state.regime == Regime.BEAR:
            prev_close = float(bars[-2].close) if len(bars) >= 2 else float(bar.close)

            # Did we cross DOWN?
            cross_down = (prev_close > prev_vwap_f) and (
                float(bar.close) < current_vwap_f
            )

            if cross_down:
                # Check Volume
                if current_vol > (avg_vol_f * float(self.vol_mult)):
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
                        meta={
                            "vwap": float(current_vwap_f),
                            "vol_mult": (
                                (current_vol / avg_vol_f) if avg_vol_f else None
                            ),
                            "trend_score": float(trend_score_f),
                        },
                    )

        return None
