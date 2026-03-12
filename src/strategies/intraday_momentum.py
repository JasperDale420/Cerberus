"""Intraday Momentum Strategy — Gao et al. (2018).

First-half-hour return predicts last-half-hour return.  The strategy accumulates
morning (9:30-10:00 ET) and first-half (9:30-12:30 ET) returns, then enters in
the power-hour window (14:30-15:50 ET) when the morning and first-half returns
agree in direction and magnitude exceeds a configurable threshold.

Six-factor confluence scoring gates entries; a hard exit at 15:50 ET ensures
all positions are closed before the bell.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any, Dict, Optional

from src.core import time_utils
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    SessionRegime,
    Signal,
    SymbolState,
    TrendRegime,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy
from src.strategies.confluence import ConfluenceScorer, score_deviation, score_threshold, score_volume

# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------


@dataclass
class SessionData:
    """Per-symbol intraday session tracking."""

    date: date
    morning_open: Optional[float] = None
    morning_close: Optional[float] = None
    first_half_open: Optional[float] = None
    first_half_close: Optional[float] = None
    morning_volume: float = 0.0
    first_half_volume: float = 0.0
    bar_count: int = 0


# ---------------------------------------------------------------------------
# Time boundaries (Eastern)
# ---------------------------------------------------------------------------

_MORNING_START = time(9, 30)
_MORNING_END = time(10, 0)
_FIRST_HALF_END = time(12, 30)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class IntradayMomentumStrategy(BaseStrategy):
    """Intraday momentum: first-half return predicts last-half return.

    Based on Gao et al. (2018), validated across 45 years and 62 markets.
    Enters during the power-hour window when morning and first-half returns
    agree in sign and magnitude exceeds a minimum threshold.
    """

    name: str = "intraday_momentum"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self._symbol_sessions: Dict[str, SessionData] = {}

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _set_params(self, config: Dict[str, Any]) -> None:
        super()._set_params(config)
        self.min_move_threshold: float = float(config.get("min_move_threshold", 0.003))
        self.confluence_threshold: float = float(config.get("confluence_threshold", 60.0))
        self.time_window_start: str = str(config.get("time_window_start", "14:30"))
        self.time_window_end: str = str(config.get("time_window_end", "15:50"))
        self.max_hold_minutes: int = int(config.get("max_hold_minutes", 80))
        self.stop_atr_mult: float = float(config.get("stop_atr_mult", 1.5))

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def _get_or_reset_session(self, symbol: str, bar_date: date) -> SessionData:
        """Return the session for *symbol*, resetting if the date changed."""
        session = self._symbol_sessions.get(symbol)
        if session is None or session.date != bar_date:
            session = SessionData(date=bar_date)
            self._symbol_sessions[symbol] = session
        return session

    def _update_session(self, session: SessionData, bar: Bar, et_time: time) -> None:
        """Accumulate bar data into the session."""
        session.bar_count += 1

        # Morning window: 9:30-10:00 ET
        if _MORNING_START <= et_time < _MORNING_END:
            if session.morning_open is None:
                session.morning_open = bar.open
            session.morning_close = bar.close
            session.morning_volume += bar.volume

        # First-half window: 9:30-12:30 ET (includes morning)
        if _MORNING_START <= et_time < _FIRST_HALF_END:
            if session.first_half_open is None:
                session.first_half_open = bar.open
            session.first_half_close = bar.close
            session.first_half_volume += bar.volume

    # ------------------------------------------------------------------
    # Return helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_return(open_price: Optional[float], close_price: Optional[float]) -> Optional[float]:
        if open_price is None or close_price is None or open_price == 0.0:
            return None
        return (close_price - open_price) / open_price

    # ------------------------------------------------------------------
    # Regime gate
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_ok(market_state: MarketState) -> bool:
        snap = market_state.regime_snapshot
        if snap is None:
            return False

        # Never trade in SHOCK volatility
        if snap.vol == VolRegime.SHOCK:
            return False

        # Only trade during power hour / close sessions (14:30-16:00)
        if snap.session not in (SessionRegime.POWER_HOUR, SessionRegime.CLOSE):
            return False

        return True

    # ------------------------------------------------------------------
    # Confluence scoring
    # ------------------------------------------------------------------

    def _score_entry(
        self,
        r_morning: float,
        r_first_half: float,
        session: SessionData,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
        et_time: time,
    ) -> ConfluenceScorer:
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)

        # 1. Morning-midday agreement (0.30)
        morning_agree = (r_morning > 0 and r_first_half > 0) or (r_morning < 0 and r_first_half < 0)
        agree_score = 100.0 if morning_agree else 0.0
        scorer.add_factor(
            "morning_midday_agreement",
            raw_value=1.0 if morning_agree else 0.0,
            score=agree_score,
            weight=0.30,
            passed=morning_agree,
        )

        # 2. First-half magnitude (0.25) — scale 0.003 to 0.015
        magnitude_score = score_deviation(r_first_half, 0.003, 0.015)
        scorer.add_factor(
            "first_half_magnitude",
            raw_value=r_first_half,
            score=magnitude_score,
            weight=0.25,
            passed=abs(r_first_half) >= self.min_move_threshold,
        )

        # 3. Volume confirmation (0.15) — above-average volume in first half
        avg_vol = float(symbol_state.meta.get("avg_volume_20", 0) or 0)
        vol_score = score_volume(session.first_half_volume, avg_vol, 1.0, 2.5) if avg_vol > 0 else 50.0
        scorer.add_factor(
            "volume_confirmation",
            raw_value=session.first_half_volume,
            score=vol_score,
            weight=0.15,
            passed=vol_score > 0.0,
        )

        # 4. Regime alignment (0.15) — trend matches signal direction
        snap = market_state.regime_snapshot
        if snap is not None:
            if r_first_half > 0 and snap.trend == TrendRegime.UP:
                regime_score = 100.0
            elif r_first_half < 0 and snap.trend == TrendRegime.DOWN:
                regime_score = 100.0
            elif snap.trend == TrendRegime.FLAT:
                regime_score = 50.0
            else:
                regime_score = 20.0
        else:
            regime_score = 50.0
        scorer.add_factor(
            "regime_alignment",
            raw_value=regime_score,
            score=regime_score,
            weight=0.15,
            passed=regime_score >= 50.0,
        )

        # 5. ADX strength (0.10) — trending market (ADX > 20)
        adx = symbol_state.indicators.get("adx")
        if adx is not None:
            adx_val = float(adx)
            adx_score = score_threshold(adx_val, 20.0, 50.0)
        else:
            adx_score = 50.0  # Neutral when missing
        scorer.add_factor(
            "adx_strength",
            raw_value=float(adx) if adx is not None else 0.0,
            score=adx_score,
            weight=0.10,
            passed=adx_score > 0.0,
        )

        # 6. Session quality (0.05) — power hour (14:30-15:00) scores higher
        if et_time < time(15, 0):
            session_score = 100.0  # Prime power hour
        else:
            session_score = 60.0  # Late power hour
        scorer.add_factor(
            "session_quality",
            raw_value=et_time.hour + et_time.minute / 60.0,
            score=session_score,
            weight=0.05,
            passed=True,
        )

        return scorer

    # ------------------------------------------------------------------
    # Stop / target
    # ------------------------------------------------------------------

    def _compute_stop(
        self,
        side: OrderSide,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> float:
        atr = symbol_state.indicators.get("atr_14")
        if atr is not None and float(atr) > 0:
            atr_val = float(atr)
        else:
            atr_val = bar.close * 0.005  # Fallback 0.5%

        base_distance = self.stop_atr_mult * atr_val
        adjusted = self._apply_regime_volatility_multiplier(base_distance, market_state)

        if side == OrderSide.BUY:
            return bar.close - adjusted
        return bar.close + adjusted

    def _compute_target(
        self,
        side: OrderSide,
        bar: Bar,
        stop_price: float,
    ) -> float:
        """Target at minimum 1.5R from entry."""
        stop_distance = abs(bar.close - stop_price)
        target_distance = stop_distance * 1.5

        if side == OrderSide.BUY:
            return bar.close + target_distance
        return bar.close - target_distance

    # ------------------------------------------------------------------
    # on_bar
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        # Convert to Eastern time
        et_dt = time_utils.to_eastern_time(bar.time)
        et_time = et_dt.time()
        bar_date = et_dt.date()

        # Get or reset session
        session = self._get_or_reset_session(symbol, bar_date)

        # Always accumulate session data
        self._update_session(session, bar, et_time)

        # --- Pre-flight checks ---
        if not self._check_cooldown(symbol, bar.time):
            return None

        # Time window gate: only generate signals 14:30-15:50 ET
        if not time_utils.in_time_window_str(bar.time, self.time_window_start, self.time_window_end):
            return None

        # Regime gate
        if not self._regime_ok(market_state):
            return None

        # Compute returns
        r_morning = self._compute_return(session.morning_open, session.morning_close)
        r_first_half = self._compute_return(session.first_half_open, session.first_half_close)

        if r_morning is None or r_first_half is None:
            self.logger.debug(
                "intraday_momentum: insufficient session data",
                symbol=symbol,
                has_morning=r_morning is not None,
                has_first_half=r_first_half is not None,
            )
            return None

        # Direction agreement gate
        if (r_morning > 0) != (r_first_half > 0):
            return None

        # Minimum move threshold
        if abs(r_first_half) < self.min_move_threshold:
            return None

        # Determine direction
        side = OrderSide.BUY if r_first_half > 0 else OrderSide.SELL

        # --- Confluence scoring ---
        scorer = self._score_entry(
            r_morning=r_morning,
            r_first_half=r_first_half,
            session=session,
            bar=bar,
            symbol_state=symbol_state,
            market_state=market_state,
            et_time=et_time,
        )

        if not scorer.passes_threshold():
            self.logger.debug(
                "intraday_momentum: below confluence threshold",
                symbol=symbol,
                score=round(scorer.score(), 2),
                threshold=self.confluence_threshold,
            )
            return None

        # --- Stop / target ---
        stop_price = self._compute_stop(side, bar, symbol_state, market_state)
        target_price = self._compute_target(side, bar, stop_price)

        # Sanity: stop must be on correct side
        if side == OrderSide.BUY and stop_price >= bar.close:
            return None
        if side == OrderSide.SELL and stop_price <= bar.close:
            return None

        # --- Build meta ---
        meta: Dict[str, Any] = scorer.to_meta()
        meta["exit_config"] = {
            "max_hold_minutes": self.max_hold_minutes,
        }
        meta["r_morning"] = round(r_morning, 6)
        meta["r_first_half"] = round(r_first_half, 6)
        meta["session_bar_count"] = session.bar_count

        # Record cooldown
        self.last_signal_time[symbol] = bar.time

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            size_hint=scorer.conviction_multiplier(),
            meta=meta,
        )
