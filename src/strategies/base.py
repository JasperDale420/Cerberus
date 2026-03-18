from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState, TrendRegime
from src.core.logger import StructuredLogger
from src.data.requirements import DataRequirements


class BaseStrategy(ABC):
    name: str = "base"
    data_requirements: DataRequirements = DataRequirements()  # default: bars stream only

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self._set_params(config)
        self.last_signal_time: Dict[str, datetime] = {}

    def _set_params(self, config: Dict[str, Any]) -> None:
        """Internal helper to set parameters from config."""
        self.cooldown_bars = int(config.get("cooldown_bars", 5))
        # M5 fix: Configurable bar duration for accurate cooldown across timeframes
        self.bar_duration_minutes = float(config.get("bar_duration_minutes", 1.0))
        # Multi-timeframe trend alignment gate
        self.higher_tf_alignment = bool(config.get("higher_tf_alignment", True))
        self.tf_alignment_mode = str(config.get("tf_alignment_mode", "trend"))  # "trend" or "mean_reversion"
        # Signal timeframe gate: only evaluate on N-minute candle closes (1=every bar, 5=5m, 15=15m)
        self.signal_timeframe = int(config.get("signal_timeframe", 1))

    def update_params(self, config: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.
        Preserves transient state (cooldowns, etc.) while applying new configuration.
        """
        self.config.update(config)
        self._set_params(self.config)
        self.logger.info("Strategy parameters updated", strategy=self.name)

    def _check_cooldown(self, symbol: str, current_time: Any) -> bool:
        """
        Returns True if cooldown has passed and it's safe to signal.
        """
        if self.cooldown_bars <= 0:
            return True
        last = self.last_signal_time.get(symbol)
        if last is None:
            return True
        # M5 fix: Use configured bar duration instead of hardcoded 1 minute
        delta = timedelta(minutes=self.cooldown_bars * self.bar_duration_minutes)
        if current_time - last < delta:
            return False
        return True

    def is_past_hard_stop(self, current_time: datetime) -> bool:
        """
        Returns True if the current time is at or past the configured hard stop.
        """
        hard_stop = self.config.get("hard_stop_time")
        if not hard_stop:
            return False

        try:
            stop_h, stop_m = map(int, hard_stop.split(":"))
            # Compare only hour and minute
            if current_time.hour > stop_h:
                return True
            if current_time.hour == stop_h and current_time.minute >= stop_m:
                return True
        except (ValueError, AttributeError):
            pass

        return False

    def _is_evaluation_bar(self, bar: Bar) -> bool:
        """Return True if this bar falls on a signal_timeframe boundary.

        When signal_timeframe=1 (default), every bar is an evaluation bar.
        When signal_timeframe=5, only bars at :00, :05, :10, ... are evaluation bars.
        This prevents strategies from reacting to intra-candle noise on lower timeframes.

        Market open (9:30) is always an evaluation bar regardless of timeframe.
        """
        if self.signal_timeframe <= 1:
            return True
        minute = bar.time.minute
        return minute % self.signal_timeframe == 0

    def _require_min_bars(self, symbol_state: SymbolState, min_count: int, log: bool = True) -> bool:
        """
        Check if symbol_state has minimum required bars for analysis.

        This helper eliminates duplicate bar count validation across strategies.

        Args:
            symbol_state: Symbol state containing bars
            min_count: Minimum number of bars required
            log: Whether to log when insufficient bars (default True)

        Returns:
            True if sufficient bars available, False otherwise

        Example:
            if not self._require_min_bars(symbol_state, 20):
                return None
        """
        bars = symbol_state.bars
        if not bars or len(bars) < min_count:
            if log:
                self.logger.debug(
                    f"{self.name}: insufficient bars",
                    min_required=min_count,
                    available=len(bars) if bars else 0,
                )
            return False
        return True

    def _check_higher_tf_alignment(
        self,
        symbol_state: SymbolState,
        side: "OrderSide",
    ) -> bool:
        """Check if the 5-minute EMA trend aligns with the proposed signal direction.

        Uses the pre-computed indicator cache populated by ExecutionEngine._resample_bars().
        Two modes:
          - "trend": requires 5m EMA fast > slow for BUY, fast < slow for SELL
          - "mean_reversion": requires EMAs to be within 0.15% of each other (flat/choppy)

        Returns True if aligned (or no data / disabled), False to reject the signal.
        """
        if not self.higher_tf_alignment:
            return True

        ema_fast = symbol_state.indicators.get("ema_close_5m:20")
        ema_slow = symbol_state.indicators.get("ema_close_5m:50")

        # Not enough data yet — allow signal through
        if ema_fast is None or ema_slow is None:
            return True

        ema_fast = float(ema_fast)
        ema_slow = float(ema_slow)

        if abs(ema_slow) < 1e-9:
            return True

        spread_pct = abs(ema_fast - ema_slow) / ema_slow

        if self.tf_alignment_mode == "mean_reversion":
            # Mean reversion: trade when higher TF is not strongly trending
            # 0.5% spread allows mild trends where RSI/BB mean-reversion still works
            return spread_pct < 0.005  # 0.5% spread

        # Trend mode: higher TF must trend in signal direction
        if side == OrderSide.BUY:
            return ema_fast > ema_slow
        else:
            return ema_fast < ema_slow

    def _create_signal(
        self,
        symbol: str,
        side: OrderSide,
        bar: Bar,
        market_state: MarketState,
        stop_price: float,
        target_price: float,
        entry_price: Optional[float] = None,
        size_hint: float = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Create a Signal with standardized parameters.

        This helper eliminates duplicate Signal construction code across strategies.

        Args:
            symbol: Symbol to trade
            side: OrderSide.BUY or OrderSide.SELL
            bar: Current bar being processed
            market_state: Current market state
            stop_price: Stop loss price
            target_price: Take profit price
            entry_price: Entry price (defaults to bar.close)
            size_hint: Size hint for position sizing (default 0)
            meta: Optional metadata dictionary

        Returns:
            Signal object ready to be processed by execution engine
        """
        # Extract multi-axis regime data from snapshot
        snapshot = market_state.regime_snapshot
        regime_tags = snapshot.regime_tags if snapshot else {}
        regime_confidence = snapshot.confidence if snapshot else {}

        return Signal(
            symbol=symbol,
            side=side,
            size_hint=size_hint,
            entry_price=entry_price if entry_price is not None else bar.close,
            stop_price=stop_price,
            target_price=target_price,
            strategy=self.name,
            generated_at=bar.time,
            meta=meta or {},
            regime_tags=regime_tags,
            regime_confidence=regime_confidence,
        )

    def _apply_regime_volatility_multiplier(self, base_distance: float, market_state: MarketState) -> float:
        """
        Dynamically adjusts a stop-loss or take-profit distance based on the current Volatility Regime.
        Higher volatility regimes lead to wider stops, which inherently reduces position sizing
        through the central Risk Manager's constant-risk math.
        """
        snapshot = market_state.regime_snapshot
        if not snapshot or not snapshot.vol:
            return base_distance

        vol_name = str(snapshot.vol.value).lower()

        # In highly volatile environments, we need wider stops to avoid getting chopped out.
        # This naturally decreases the execution size (risk_per_share goes up).
        # In low volatility environments, we can run tighter stops to increase capital efficiency.
        multipliers = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.2,
            "shock": 1.5,
        }

        mult = multipliers.get(vol_name, 1.0)
        return base_distance * mult

    def _get_dynamic_risk_reward(self, base_rr: float, market_state: MarketState) -> float:
        """
        Adjust the risk-reward ratio based on the current trend regime.

        In strongly trending markets a higher RR is achievable; in flat/choppy
        markets a lower RR reflects reduced follow-through probability.

        Returns the base_rr unchanged when no regime snapshot is available.
        """
        snapshot = market_state.regime_snapshot
        if not snapshot or not snapshot.trend:
            return base_rr

        trend_name = str(snapshot.trend.value).lower()

        multipliers = {
            TrendRegime.UP.value: 1.1,
            TrendRegime.DOWN.value: 1.1,
            TrendRegime.FLAT.value: 0.8,
        }

        mult = multipliers.get(trend_name, 1.0)
        return base_rr * mult

    @abstractmethod
    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Process a new bar and potentially return a Signal.
        """
        pass
