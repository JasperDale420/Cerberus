from __future__ import annotations

from collections import Counter, deque
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from src.core.domain import Bar, Regime
from src.core.logger import StructuredLogger

if TYPE_CHECKING:
    from src.core.domain import (
        LiquidityRegime,
        MarketRegimeSnapshot,
        RiskRegime,
        SessionRegime,
        TrendRegime,
        VolRegime,
    )


# Backwards compatibility: legacy single-label regime detector (archived).


# --- PRD Addendum: Multi-Axis Regime Classification ---


class MarketContextService:
    """
    Multi-axis market regime classification per PRD addendum.

    Replaces simple BULL/BEAR/CHOP with orthogonal axes:
    - trend: UP/DOWN/FLAT
    - vol: LOW/NORMAL/HIGH/SHOCK
    - liquidity: GOOD/THIN/STRESSED
    - risk: RISK_ON/NEUTRAL/RISK_OFF
    - session: PREMARKET/OPENING/MIDDAY/POWER_HOUR/CLOSE
    """

    # Session time boundaries (ET)
    SESSION_BOUNDARIES = {
        "premarket": (4, 0, 9, 30),  # 04:00-09:30
        "opening": (9, 30, 10, 30),  # 09:30-10:30
        "midday": (10, 30, 15, 0),  # 10:30-15:00
        "power_hour": (15, 0, 16, 0),  # 15:00-16:00
        "close": (16, 0, 20, 0),  # 16:00-20:00
    }

    def __init__(
        self,
        window: int = 60,
        min_bars: int = 20,
        vol_baseline_window: int = 120,
        smooth_k: int = 5,
        logger: Optional[StructuredLogger] = None,
        tz: str = "America/New_York",
        index_symbol: str = "SPY",
        vol_symbol: Optional[str] = None,
    ):
        from zoneinfo import ZoneInfo

        from src.core.domain import (
            LiquidityRegime,
            MarketRegimeSnapshot,
            RiskRegime,
            SessionRegime,
            TrendRegime,
            VolRegime,
        )

        self.window = window
        self.min_bars = min_bars
        self.vol_baseline_window = vol_baseline_window
        self.smooth_k = smooth_k
        self.logger = logger
        self.tz = ZoneInfo(tz)
        self.index_symbol = index_symbol
        self.vol_symbol = vol_symbol

        # Price/return history
        self.prices: deque[float] = deque(maxlen=window)
        self.vol_history: deque[float] = deque(maxlen=vol_baseline_window)

        # VXX price history for risk axis
        self.vol_prices: deque[float] = deque(maxlen=window)
        self.last_vol_ret: Optional[float] = None  # VXX cumulative return

        # Per-axis hysteresis buffers
        self.trend_history: deque[TrendRegime] = deque(maxlen=smooth_k)
        self.vol_regime_history: deque[VolRegime] = deque(maxlen=smooth_k)

        # Current snapshot
        self.current_snapshot: Optional[MarketRegimeSnapshot] = None

        # Expose for legacy compatibility
        self.last_cum_ret: Optional[float] = None
        self.last_trend_score: Optional[float] = None
        self.last_vol: Optional[float] = None

        # Store classes for type hints
        self._TrendRegime = TrendRegime
        self._VolRegime = VolRegime
        self._LiquidityRegime = LiquidityRegime
        self._RiskRegime = RiskRegime
        self._SessionRegime = SessionRegime
        self._MarketRegimeSnapshot = MarketRegimeSnapshot

    def update_vol(self, bar: Bar) -> None:
        """
        Update VXX price history for risk axis calculation.
        Call this with VXX bars when available.
        """
        self.vol_prices.append(float(bar.close))

        # Compute VXX cumulative return if we have enough data
        if len(self.vol_prices) >= self.min_bars:
            first_price = self.vol_prices[0]
            last_price = self.vol_prices[-1]
            if first_price > 0:
                self.last_vol_ret = (last_price - first_price) / first_price
            else:
                self.last_vol_ret = 0.0
        else:
            self.last_vol_ret = None

    def update(self, bar: Bar) -> "MarketRegimeSnapshot":
        """
        Compute all regime axes and return a MarketRegimeSnapshot.
        """
        from datetime import timezone as tz_module

        self.prices.append(float(bar.close))

        # Compute time in ET
        bar_time = bar.time
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=tz_module.utc)
        local_time = bar_time.astimezone(self.tz)

        # Default snapshot if not enough data
        if len(self.prices) < self.min_bars:
            return self._default_snapshot(bar_time, local_time)

        try:
            # Compute continuous features
            cum_ret, trend_strength, realized_vol = self._compute_metrics()
            self.last_cum_ret = cum_ret
            self.last_trend_score = trend_strength
            self.last_vol = realized_vol

            # Track vol for baseline computation
            self.vol_history.append(realized_vol)

            # Classify each axis
            trend = self._classify_trend(cum_ret, trend_strength)
            vol = self._classify_vol(realized_vol)
            liquidity = self._classify_liquidity(bar)
            risk = self._classify_risk()
            session = self._classify_session(local_time)

            # Apply per-axis hysteresis
            self.trend_history.append(trend)
            self.vol_regime_history.append(vol)

            smoothed_trend = self._smooth_axis(
                self.trend_history, self._TrendRegime.FLAT
            )
            smoothed_vol = self._smooth_axis(
                self.vol_regime_history, self._VolRegime.NORMAL
            )

            # Compute confidence (simple: agreement ratio)
            confidence = {
                "trend": self._axis_confidence(self.trend_history, smoothed_trend),
                "vol": self._axis_confidence(self.vol_regime_history, smoothed_vol),
                "liquidity": 1.0,  # No hysteresis yet
                "risk": 1.0,
                "session": 1.0,
            }

            snapshot = self._MarketRegimeSnapshot(
                time=bar_time,
                index_symbol=self.index_symbol,
                vol_symbol=self.vol_symbol,
                trend=smoothed_trend,
                vol=smoothed_vol,
                liquidity=liquidity,
                risk=risk,
                session=session,
                cum_ret=cum_ret,
                trend_strength=trend_strength,
                realized_vol=realized_vol,
                vol_of_vol=self._compute_vol_of_vol(),
                liquidity_score=self._compute_liquidity_score(bar),
                risk_score=0.0,  # Requires VXX data
                confidence=confidence,
            )

            self.current_snapshot = snapshot

            if self.logger:
                self.logger.debug(
                    "Regime snapshot updated",
                    regime_tags=snapshot.regime_tags,
                    cum_ret=cum_ret,
                    trend_strength=trend_strength,
                    realized_vol=realized_vol,
                )

            return snapshot

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "MarketContextService update failed",
                    error=str(e),
                )
            return self._default_snapshot(bar_time, local_time)

    def _default_snapshot(self, bar_time, local_time) -> "MarketRegimeSnapshot":
        """Returns a neutral snapshot when insufficient data."""
        return self._MarketRegimeSnapshot(
            time=bar_time,
            index_symbol=self.index_symbol,
            vol_symbol=self.vol_symbol,
            trend=self._TrendRegime.FLAT,
            vol=self._VolRegime.NORMAL,
            liquidity=self._LiquidityRegime.GOOD,
            risk=self._RiskRegime.NEUTRAL,
            session=self._classify_session(local_time),
            confidence={
                "trend": 0.0,
                "vol": 0.0,
                "liquidity": 0.0,
                "risk": 0.0,
                "session": 1.0,
            },
        )

    def _compute_metrics(self) -> Tuple[float, float, float]:
        """Compute cumulative return and trend score."""
        prices = np.array(self.prices)
        returns = np.diff(np.log(prices))
        cum_ret = float(np.sum(returns))
        vol = float(np.std(returns)) + 1e-9
        trend_strength = abs(cum_ret) / vol
        return cum_ret, trend_strength, vol

    def _classify_trend(
        self, cum_ret: float, trend_strength: float, flat_thresh: float = 1.0
    ) -> "TrendRegime":
        """UP/DOWN/FLAT based on trend_strength threshold."""
        if trend_strength < flat_thresh:
            return self._TrendRegime.FLAT
        return self._TrendRegime.UP if cum_ret > 0 else self._TrendRegime.DOWN

    def _classify_vol(
        self,
        realized_vol: float,
        low_z: float = 0.7,
        high_z: float = 1.5,
        shock_z: float = 3.0,
    ) -> "VolRegime":
        """LOW/NORMAL/HIGH/SHOCK based on z-score vs baseline."""
        if len(self.vol_history) < 10:
            return self._VolRegime.NORMAL

        baseline = float(np.median(self.vol_history))
        if baseline < 1e-9:
            return self._VolRegime.NORMAL

        z = realized_vol / baseline

        if z >= shock_z:
            return self._VolRegime.SHOCK
        if z >= high_z:
            return self._VolRegime.HIGH
        if z <= low_z:
            return self._VolRegime.LOW
        return self._VolRegime.NORMAL

    def _classify_liquidity(self, bar: Bar) -> "LiquidityRegime":
        """GOOD/THIN/STRESSED based on dollar volume / range."""
        score = self._compute_liquidity_score(bar)
        # Simple thresholds (can be tuned)
        if score < 1e6:
            return self._LiquidityRegime.STRESSED
        if score < 1e7:
            return self._LiquidityRegime.THIN
        return self._LiquidityRegime.GOOD

    def _compute_liquidity_score(self, bar: Bar) -> float:
        """Dollar volume / (range_pct + eps)."""
        dollar_vol = bar.close * bar.volume
        range_pct = (bar.high - bar.low) / (bar.close + 1e-9)
        return dollar_vol / (range_pct + 1e-6)

    def _classify_risk(self) -> "RiskRegime":
        """
        RISK_ON/NEUTRAL/RISK_OFF based on VXX momentum.
        Rising VXX = fear increasing = RISK_OFF
        Falling VXX = fear decreasing = RISK_ON
        Falls back to SPY cumulative return if VXX data unavailable.
        """
        # Use VXX momentum if available (preferred)
        if self.last_vol_ret is not None:
            if self.last_vol_ret > 0.005:  # VXX rising >0.5% = fear up
                return self._RiskRegime.RISK_OFF
            if self.last_vol_ret < -0.005:  # VXX falling <-0.5% = fear down
                return self._RiskRegime.RISK_ON
            return self._RiskRegime.NEUTRAL

        # Fallback: use SPY trend direction as proxy
        if self.last_cum_ret is not None:
            if self.last_cum_ret > 0.005:
                return self._RiskRegime.RISK_ON
            if self.last_cum_ret < -0.005:
                return self._RiskRegime.RISK_OFF
        return self._RiskRegime.NEUTRAL

    def _classify_session(self, local_time) -> "SessionRegime":
        """Classify session based on wall clock time."""
        hour, minute = local_time.hour, local_time.minute
        time_mins = hour * 60 + minute

        for session, (sh, sm, eh, em) in self.SESSION_BOUNDARIES.items():
            start = sh * 60 + sm
            end = eh * 60 + em
            if start <= time_mins < end:
                return self._SessionRegime(session)

        # Outside defined sessions
        if time_mins < 4 * 60:
            return self._SessionRegime.CLOSE
        return self._SessionRegime.CLOSE

    def _compute_vol_of_vol(self) -> float:
        """Rolling std of realized vol."""
        if len(self.vol_history) < 5:
            return 0.0
        return float(np.std(self.vol_history))

    def _smooth_axis(self, history: deque, default):
        """Majority vote for an axis."""
        if not history:
            return default
        counts = Counter(history)
        top, _ = counts.most_common(1)[0]
        return top

    def _axis_confidence(self, history: deque, value) -> float:
        """Confidence as agreement ratio."""
        if not history:
            return 0.0
        return sum(1 for x in history if x == value) / len(history)

    def get_snapshot(self) -> Optional["MarketRegimeSnapshot"]:
        """Returns current snapshot or None if not yet computed."""
        return self.current_snapshot

    def get_legacy_regime(self) -> Regime:
        """Returns legacy BULL/BEAR/CHOP for backwards compatibility."""
        if self.current_snapshot:
            return self.current_snapshot.legacy_regime
        return Regime.CHOP
