from datetime import time as time_type
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.data.calculator import FeatureCalculator
from src.strategies.base import BaseStrategy
from src.strategies.config_models import VWAPTrendRiderConfig


class VWAPTrendRiderStrategy(BaseStrategy):
    """
    VWAP Trend Rider Strategy.
    Rides trends that align with VWAP slope and EMA crossover.
    """

    name = "vwap_trend_rider"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Override to parse strategy-specific parameters."""
        super()._set_params(config)
        cfg = VWAPTrendRiderConfig(**config)
        self.ema_fast_len = cfg.ema_fast
        self.ema_slow_len = cfg.ema_slow
        self.vol_mult = cfg.vol_mult
        self.risk_reward = cfg.risk_reward
        self.min_trend_score = cfg.min_trend_score
        # P2.2 fix: Add time window to avoid early/late session false signals
        self.entry_start = time_type(9, 45)  # After dust settles
        self.entry_end = time_type(15, 30)  # Before EOD volatility

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # P2.2 fix: Add time window check
        current_time = time_utils.get_eastern_time_of_day(bar.time)
        if not (self.entry_start <= current_time <= self.entry_end):
            return None

        # Need enough data for slow EMA
        if not self._require_min_bars(symbol_state, self.ema_slow_len + 5):
            return None

        # Regime gating removed - handled by engine. Keep trend_score check for signal quality.
        trend_score = None
        if isinstance(getattr(market_state, "meta", None), dict):
            trend_score = market_state.meta.get("trend_score")
        try:
            trend_score_f = float(trend_score) if trend_score is not None else 0.0
        except Exception:
            trend_score_f = 0.0
        if trend_score_f < self.min_trend_score:
            # Debug: Log when trend score is too low
            if len(symbol_state.bars) % 200 == 0:
                self.logger.debug(
                    "VWAPTrendRider: Trend score too low",
                    symbol=symbol,
                    trend_score=trend_score_f,
                    min_required=self.min_trend_score,
                )
            return None

        bars = list(symbol_state.bars)
        if not bars or bars[-1].time != bar.time:
            bars.append(bar)

        # Prefer cached EMAs and volume SMA from engine; fall back to deterministic local computation.
        current_fast = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_fast_len)}"
        )
        current_slow = symbol_state.indicators.get(
            f"ema_close:{int(self.ema_slow_len)}"
        )
        if current_fast is None or current_slow is None:
            # Fallback to centralized calculator
            closes = [float(b.close) for b in bars]
            current_fast = FeatureCalculator.calculate_ema(
                closes, int(self.ema_fast_len)
            )
            current_slow = FeatureCalculator.calculate_ema(
                closes, int(self.ema_slow_len)
            )

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
            # Debug: Log when VWAP is missing (likely cause of no signals)
            if len(symbol_state.bars) % 200 == 0:
                self.logger.debug(
                    "VWAPTrendRider: VWAP missing from bar",
                    symbol=symbol,
                    has_current_vwap=current_vwap is not None,
                    has_prev_vwap=prev_vwap is not None,
                )
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

        # BULLISH RIDER - regime gating removed
        if is_uptrend:
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
                    target_price = bar.close + (risk * float(self.risk_reward))

                    return self._create_signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        bar=bar,
                        market_state=market_state,
                        stop_price=stop_loss,
                        target_price=target_price,
                        meta={
                            "ema_fast": float(current_fast),
                            "ema_slow": float(current_slow),
                            "vwap": float(current_vwap_f),
                            "trigger": "pullback_bounce",
                            "vol_mult": (
                                (current_vol / avg_vol_f) if avg_vol_f else None
                            ),
                            "trend_score": float(trend_score_f),
                        },
                    )

        # BEARISH RIDER - regime gating removed
        if is_downtrend:
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
                    target_price = bar.close - (risk * float(self.risk_reward))

                    return self._create_signal(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        bar=bar,
                        market_state=market_state,
                        stop_price=stop_loss,
                        target_price=target_price,
                        meta={
                            "ema_fast": float(current_fast),
                            "ema_slow": float(current_slow),
                            "vwap": float(current_vwap_f),
                            "trigger": "pullback_bounce",
                            "vol_mult": (
                                (current_vol / avg_vol_f) if avg_vol_f else None
                            ),
                            "trend_score": float(trend_score_f),
                        },
                    )

        return None
