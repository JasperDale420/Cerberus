from __future__ import annotations

from collections import Counter, deque
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from src.analysis.bocpd import BOCPDDetector
from src.analysis.entropy import EntropyAnalyzer
from src.analysis.vrp import VRPCalculator
from src.core.domain import Bar, ComplexityRegime, Regime
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

    # Liquidity multipliers per session (relative to RTH midday)
    SESSION_LIQUIDITY_MULTIPLIERS = {
        "premarket": 0.1,  # 10% of RTH volume
        "opening": 1.0,
        "midday": 1.0,
        "power_hour": 1.2,
        "close": 0.1,
    }

    def __init__(
        self,
        window: int = 60,
        min_bars: int = 20,
        vol_baseline_window: int = 120,
        smooth_k: int = 5,
        liquidity_window: int = 390,
        hurst_lag_max: int = 20,
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
        # We track squared returns directly for EWMA variance
        self.sq_returns: deque[float] = deque(maxlen=vol_baseline_window)
        # Persistent EWMA variance state (proper recursive EWMA)
        self._short_ewma_var: float | None = None
        self._long_ewma_var: float | None = None
        self._alpha_short = 2.0 / (10 + 1)  # ~10-bar span
        self._alpha_long = 2.0 / (vol_baseline_window + 1)  # ~120-bar span

        # Track liquidity scores for dynamic percentile thresholds
        self.liquidity_window = liquidity_window
        self.liquidity_history: deque[float] = deque(maxlen=liquidity_window)

        self.hurst_lag_max = hurst_lag_max

        # VXX price history for risk axis
        self.vol_prices: deque[float] = deque(maxlen=window)
        self.last_vol_ret: Optional[float] = None  # VXX cumulative return

        # Per-axis hysteresis buffers
        self.trend_history: deque[TrendRegime] = deque(maxlen=smooth_k)
        self.vol_regime_history: deque[VolRegime] = deque(maxlen=smooth_k)

        # BOCPD changepoint detector
        self._bocpd = BOCPDDetector(logger=logger)

        # Permutation entropy analyzer
        self._entropy = EntropyAnalyzer(logger=logger)

        # Last BOCPD/entropy results (for snapshot construction)
        self._last_cp_prob: float = 0.0
        self._last_bars_since_cp: int = 0
        self._last_complexity: Optional[ComplexityRegime] = None
        self._last_entropy_score: float = 0.0

        # VRP calculator for risk axis
        self._vrp = VRPCalculator(logger=logger)
        self._last_vrp_score: float = 0.0
        self._vrp_enabled: bool = True  # On by default since it enhances existing VXX data

        # GEX analyzer for dealer positioning regime
        from src.analysis.gex import GEXAnalyzer

        self._gex = GEXAnalyzer(logger=logger)
        self._last_gex_regime: Optional[str] = None
        self._last_net_gex: float = 0.0
        self._last_gamma_flip_dist: float = 0.0

        # IV Surface Dynamics analyzer
        from src.analysis.iv_surface import IVSurfaceAnalyzer

        self._iv_surface = IVSurfaceAnalyzer(logger=logger)
        self._last_iv_skew_zscore: float = 0.0
        self._last_iv_term_inverted: bool = False
        self._last_iv_signal_strength: float = 0.0

        # ETF Flow Propagation analyzer
        from src.analysis.etf_flow import ETFFlowAnalyzer

        self._etf_flow = ETFFlowAnalyzer(logger=logger)
        self._last_etf_divergence: float = 0.0
        self._last_etf_rotation: bool = False
        self._last_etf_lead_signal: float = 0.0

        # Current snapshot
        self.current_snapshot: Optional[MarketRegimeSnapshot] = None

        # Expose for legacy compatibility
        self.last_cum_ret: Optional[float] = None
        self.last_trend_score: Optional[float] = None
        self.last_vol: Optional[float] = None

        # Store classes for type hints
        self._trend_regime = TrendRegime
        self._vol_regime = VolRegime
        self._liquidity_regime = LiquidityRegime
        self._risk_regime = RiskRegime
        self._session_regime = SessionRegime
        self._market_regime_snapshot = MarketRegimeSnapshot

    def update_gex(self, current_price: float, chain_data: list) -> None:
        """Update GEX from options chain data. Call when chain data refreshes (typically EOD)."""
        result = self._gex.update(current_price, chain_data)
        if result is not None:
            self._last_gex_regime = result.gex_regime
            self._last_net_gex = result.net_gex
            self._last_gamma_flip_dist = result.gamma_flip_distance

    def update_gex_normalized(self, current_price: float, gex_data: dict) -> None:
        """Update GEX from Data-Gateway NormalizedGreekExposure format."""
        result = self._gex.update_from_normalized(current_price, gex_data)
        if result is not None:
            self._last_gex_regime = result.gex_regime
            self._last_net_gex = result.net_gex
            self._last_gamma_flip_dist = result.gamma_flip_distance

    def update_iv_surface(
        self,
        chain_data: list,
        current_price: float = 0.0,
        risk_free_rate: float = 0.05,
        time_to_expiry: float = 30 / 365,
        term_data: list | None = None,
    ) -> None:
        """Update IV surface from options chain data. Call when chain refreshes (typically EOD)."""
        # Extract strikes and call prices from chain for RND computation
        strike_price_pairs: list[tuple[float, float]] = []
        for row in chain_data:
            delta = float(row.get("delta", 0))
            if delta > 0:  # Calls have positive delta
                strike = float(row.get("strike", 0))
                mid = float(row.get("mid", 0) or row.get("mark", 0) or row.get("last", 0))
                if strike > 0 and mid > 0:
                    strike_price_pairs.append((strike, mid))
        # Sort by strike — RND finite difference requires monotonically increasing strikes
        strike_price_pairs.sort()
        strikes = [s for s, _ in strike_price_pairs]
        call_prices = [p for _, p in strike_price_pairs]
        result = self._iv_surface.analyze(
            chain=chain_data,
            term_data=term_data or [],
            strikes=strikes,
            call_prices=call_prices,
            rate=risk_free_rate,
            tte=time_to_expiry,
        )
        if result is not None:
            self._last_iv_skew_zscore = result.skew_zscore
            self._last_iv_term_inverted = result.is_inverted
            self._last_iv_signal_strength = result.signal_strength

    def update_etf_flow(
        self,
        etf_symbol: str,
        etf_data: dict,
        stock_symbol: str,
        stock_data: dict,
    ) -> None:
        """Update ETF flow propagation from UW-format flow dicts."""
        self._etf_flow.update_etf_flow(etf_symbol, etf_data)
        self._etf_flow.update_stock_flow(stock_symbol, stock_data)
        result = self._etf_flow.analyze(stock_symbol, etf_symbol)
        if result is not None:
            self._last_etf_divergence = result.divergence_score
            self._last_etf_rotation = result.rotation_detected
            self._last_etf_lead_signal = result.lead_signal

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
            returns_array = np.diff(np.log(np.array(self.prices))) if len(self.prices) > 1 else np.array([0.0])
            cum_ret = float(np.sum(returns_array))

            # Use EWMA for volatility rather than simple std dev
            current_var, baseline_var, realized_vol = self._compute_ewma_vol(returns_array)
            hurst_exp = self._compute_hurst(np.array(self.prices))

            # Store for metrics mapping
            self.last_cum_ret = cum_ret
            self.last_trend_score = hurst_exp  # Using Hurst as trend score
            self.last_vol = realized_vol

            # Track squared returns for baseline computation
            if len(returns_array) > 0:
                self.sq_returns.append(returns_array[-1] ** 2)

            # BOCPD: feed log return to changepoint detector
            if len(returns_array) > 0:
                bocpd_result = self._bocpd.update(float(returns_array[-1]))
                self._last_cp_prob = bocpd_result.changepoint_probability
                self._last_bars_since_cp = bocpd_result.map_run_length

            # Entropy: feed log return to entropy analyzer
            if len(returns_array) > 0:
                entropy_result = self._entropy.update(float(returns_array[-1]))
                if entropy_result is not None:
                    self._last_entropy_score = entropy_result.pe_normalized
                    self._last_complexity = entropy_result.regime

            # VRP: update if we have VXX price and realized vol
            if len(self.vol_prices) > 0 and self.last_vol is not None:
                vxx_price = self.vol_prices[-1]
                vrp_result = self._vrp.update(vxx_price, self.last_vol)
                if vrp_result is not None:
                    self._last_vrp_score = vrp_result.vrp_zscore

            # Classify each axis
            trend = self._classify_trend(cum_ret, hurst_exp)
            vol = self._classify_vol(current_var, baseline_var)
            liquidity = self._classify_liquidity(bar)
            risk = self._classify_risk()
            session = self._classify_session(local_time)

            # Apply per-axis hysteresis
            self.trend_history.append(trend)
            self.vol_regime_history.append(vol)

            smoothed_trend = self._smooth_axis(self.trend_history, self._trend_regime.FLAT)
            smoothed_vol = self._smooth_axis(self.vol_regime_history, self._vol_regime.NORMAL)

            # Compute confidence (simple: agreement ratio)
            trend_conf = self._axis_confidence(self.trend_history, smoothed_trend)
            vol_conf = self._axis_confidence(self.vol_regime_history, smoothed_vol)

            # BOCPD: reduce regime confidence when changepoint probability is high
            if self._last_cp_prob > 0.7:
                cp_reduction = min(self._last_cp_prob, 0.5)
                trend_conf = max(0.0, trend_conf - cp_reduction)
                vol_conf = max(0.0, vol_conf - cp_reduction)

            confidence = {
                "trend": trend_conf,
                "vol": vol_conf,
                "liquidity": 1.0,  # No hysteresis yet
                "risk": 1.0,
                "session": 1.0,
            }

            snapshot = self._market_regime_snapshot(
                time=bar_time,
                index_symbol=self.index_symbol,
                vol_symbol=self.vol_symbol,
                trend=smoothed_trend,
                vol=smoothed_vol,
                liquidity=liquidity,
                risk=risk,
                session=session,
                cum_ret=cum_ret,
                trend_strength=hurst_exp,
                realized_vol=realized_vol,
                vol_of_vol=self._compute_vol_of_vol(),
                liquidity_score=self._compute_liquidity_score(bar),
                risk_score=0.0,  # Requires VXX data
                changepoint_probability=self._last_cp_prob,
                bars_since_changepoint=self._last_bars_since_cp,
                complexity=self._last_complexity,
                entropy_score=self._last_entropy_score,
                vrp_score=self._last_vrp_score,
                gex_regime=self._last_gex_regime,
                net_gex=self._last_net_gex,
                gamma_flip_distance=self._last_gamma_flip_dist,
                iv_skew_zscore=self._last_iv_skew_zscore,
                iv_term_inverted=self._last_iv_term_inverted,
                iv_signal_strength=self._last_iv_signal_strength,
                etf_flow_divergence=self._last_etf_divergence,
                etf_rotation_detected=self._last_etf_rotation,
                etf_lead_signal=self._last_etf_lead_signal,
                confidence=confidence,
            )

            self.current_snapshot = snapshot

            if self.logger:
                self.logger.debug(
                    "Regime snapshot updated",
                    regime_tags=snapshot.regime_tags,
                    cum_ret=cum_ret,
                    trend_strength=hurst_exp,
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
        return self._market_regime_snapshot(
            time=bar_time,
            index_symbol=self.index_symbol,
            vol_symbol=self.vol_symbol,
            trend=self._trend_regime.FLAT,
            vol=self._vol_regime.NORMAL,
            liquidity=self._liquidity_regime.GOOD,
            risk=self._risk_regime.NEUTRAL,
            session=self._classify_session(local_time),
            confidence={
                "trend": 0.0,
                "vol": 0.0,
                "liquidity": 0.0,
                "risk": 0.0,
                "session": 1.0,
            },
        )

    def _compute_hurst(self, prices: np.ndarray) -> float:
        """Compute the Hurst exponent of the price series using Rescaled Range (R/S)."""
        if len(prices) < 5:
            return 0.5  # Random walk default

        # Use log-returns for scale invariance across price levels
        log_prices = np.log(prices[prices > 0]) if np.all(prices > 0) else np.log(np.maximum(prices, 1e-10))
        returns = np.diff(log_prices)
        if len(returns) < 3:
            return 0.5

        std_dev = np.std(returns)

        # If standard deviation is exactly zero, price is a straight line
        if std_dev < 1e-8:
            if np.abs(np.mean(returns)) > 1e-8:
                return 1.0  # Perfectly trending
            return 0.5  # Perfectly flat

        mean_ret = np.mean(returns)

        # Mean centered deviations
        deviations = returns - mean_ret
        cum_deviations = np.cumsum(deviations)

        # Range of the cumulative deviations
        r_range = max(np.max(cum_deviations) - np.min(cum_deviations), 1e-8)

        # R/S Hurst: H = log(R/S) / log(n) where n = number of returns
        h_exponent = np.log(r_range / std_dev) / np.log(len(returns))

        return float(np.clip(h_exponent, 0.0, 1.0))

    def _compute_ewma_vol(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """Compute short and long-term EWMA of squared returns for volatility clustering."""
        if len(returns) == 0:
            return 1e-9, 1e-9, 0.0

        current_sq_ret = returns[-1] ** 2

        # Recursive EWMA with persistent state and fixed alpha
        if self._long_ewma_var is None:
            self._long_ewma_var = current_sq_ret
            self._short_ewma_var = current_sq_ret
        else:
            self._long_ewma_var = self._alpha_long * current_sq_ret + (1 - self._alpha_long) * self._long_ewma_var
            self._short_ewma_var = self._alpha_short * current_sq_ret + (1 - self._alpha_short) * self._short_ewma_var

        realized_vol = float(np.sqrt(self._long_ewma_var))
        return float(self._short_ewma_var), float(self._long_ewma_var), realized_vol

    def _classify_trend(self, cum_ret: float, hurst: float, hurst_thresh: float = 0.55) -> "TrendRegime":
        """UP/DOWN/FLAT based on Hurst exponent and cumulative return direction."""
        if hurst < hurst_thresh:
            return self._trend_regime.FLAT
        if cum_ret > 0:
            return self._trend_regime.UP
        if cum_ret < 0:
            return self._trend_regime.DOWN
        return self._trend_regime.FLAT

    def _classify_vol(
        self,
        current_var: float,
        baseline_var: float,
        low_z: float = 0.7,
        high_z: float = 1.5,
        shock_z: float = 3.0,
    ) -> "VolRegime":
        """LOW/NORMAL/HIGH/SHOCK based on EWMA variance ratio."""
        if baseline_var < 1e-12:
            return self._vol_regime.NORMAL

        z = np.sqrt(current_var / baseline_var)

        if z >= shock_z:
            return self._vol_regime.SHOCK
        if z >= high_z:
            return self._vol_regime.HIGH
        if z <= low_z:
            return self._vol_regime.LOW
        return self._vol_regime.NORMAL

    def _classify_liquidity(self, bar: Bar) -> "LiquidityRegime":
        """GOOD/THIN/STRESSED based on rolling percentiles of dollar volume / range."""
        score = self._compute_liquidity_score(bar)
        self.liquidity_history.append(score)

        if len(self.liquidity_history) < self.liquidity_window:
            return self._liquidity_regime.GOOD  # Default until we have a baseline

        p10 = np.percentile(self.liquidity_history, 10)
        p25 = np.percentile(self.liquidity_history, 25)

        if score <= p10:
            return self._liquidity_regime.STRESSED
        if score <= p25:
            return self._liquidity_regime.THIN
        return self._liquidity_regime.GOOD

    def _compute_liquidity_score(self, bar: Bar) -> float:
        """Dollar volume / (range_pct + eps)."""
        dollar_vol = bar.close * bar.volume
        range_pct = (bar.high - bar.low) / (bar.close + 1e-9)
        return dollar_vol / (range_pct + 1e-6)

    def _classify_risk(self) -> "RiskRegime":
        """
        RISK_ON/NEUTRAL/RISK_OFF based on Variance Risk Premium (VRP).
        VRP = IV² - RV². High VRP z-score = market compensating for fear = RISK_ON.
        Falls back to VXX momentum if VRP unavailable, then SPY cumulative return.
        """
        # Primary: VRP-based classification
        if self._vrp_enabled and abs(self._last_vrp_score) > 0:
            if self._last_vrp_score > 1.0:
                return self._risk_regime.RISK_ON
            if self._last_vrp_score < -1.0:
                return self._risk_regime.RISK_OFF
            return self._risk_regime.NEUTRAL

        # Fallback: VXX momentum
        if self.last_vol_ret is not None:
            if self.last_vol_ret > 0.005:
                return self._risk_regime.RISK_OFF
            if self.last_vol_ret < -0.005:
                return self._risk_regime.RISK_ON
            return self._risk_regime.NEUTRAL

        # Last resort: SPY trend direction
        if self.last_cum_ret is not None:
            if self.last_cum_ret > 0.005:
                return self._risk_regime.RISK_ON
            if self.last_cum_ret < -0.005:
                return self._risk_regime.RISK_OFF
        return self._risk_regime.NEUTRAL

    def _classify_session(self, local_time) -> "SessionRegime":
        """Classify session based on wall clock time."""
        hour, minute = local_time.hour, local_time.minute
        time_mins = hour * 60 + minute

        for session, (sh, sm, eh, em) in self.SESSION_BOUNDARIES.items():
            start = sh * 60 + sm
            end = eh * 60 + em
            if start <= time_mins < end:
                return self._session_regime(session)

        # Outside defined sessions
        if time_mins < 4 * 60:
            return self._session_regime.CLOSE
        return self._session_regime.CLOSE

    def _compute_vol_of_vol(self) -> float:
        """Rolling std of realized squared returns."""
        if len(self.sq_returns) < 5:
            return 0.0
        return float(np.std(np.sqrt(self.sq_returns)))

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
