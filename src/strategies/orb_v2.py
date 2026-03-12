from __future__ import annotations

from datetime import date
from datetime import time as time_type
from typing import Any

from src.core import time_utils
from src.core.domain import (
    Bar,
    MarketState,
    OrderSide,
    SessionRegime,
    Signal,
    SymbolState,
    VolRegime,
)
from src.core.logger import StructuredLogger
from src.data.multi_timeframe import MultiTimeframeAnalyzer
from src.strategies.base import BaseStrategy
from src.strategies.confluence import (
    ConfluenceScorer,
    score_deviation,
    score_threshold,
    score_volume,
)


class ORBV2Strategy(BaseStrategy):
    """Opening Range Breakout v2 -- adaptive range, volume confirmation,
    gap alignment, and confluence scoring."""

    name: str = "orb_v2"

    def __init__(self, config: dict[str, Any], logger: StructuredLogger) -> None:
        super().__init__(config, logger)
        self._orb_state: dict[str, dict[str, Any]] = {}

    def _set_params(self, config: dict[str, Any]) -> None:
        super()._set_params(config)
        overrides = config.get("_optuna_overrides", {})

        self.confluence_threshold: float = float(
            overrides.get("confluence_threshold", config.get("confluence_threshold", 65.0))
        )
        self.min_bars: int = int(config.get("min_bars", 5))
        self.tf_alignment_mode = "trend"

        # Breakout-specific tunable params
        self.vol_gate_mult: float = float(overrides.get("vol_gate_mult", config.get("vol_gate_mult", 1.2)))
        self.target_range_mult: float = float(overrides.get("target_range_mult", config.get("target_range_mult", 2.0)))
        self.trail_min_profit_r: float = float(
            overrides.get("trail_min_profit_r", config.get("trail_min_profit_r", 1.0))
        )
        self.max_hold_minutes: int = int(overrides.get("max_hold_minutes", config.get("max_hold_minutes", 60)))

    # -- regime gate -------------------------------------------------------

    @staticmethod
    def _regime_ok(market_state: MarketState) -> bool:
        snap = market_state.regime_snapshot
        if snap is None:
            return True  # early session, no snapshot yet
        if snap.vol == VolRegime.SHOCK:
            return False
        return snap.session not in (SessionRegime.PREMARKET, SessionRegime.CLOSE)

    # -- per-symbol state --------------------------------------------------

    def _get_state(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._orb_state:
            self._orb_state[symbol] = self._empty_state()
        return self._orb_state[symbol]

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "date": None,
            "range_minutes": 10,
            "range_high": float("-inf"),
            "range_low": float("inf"),
            "range_complete": False,
            "range_bars": 0,
            "breakout_fired": False,
            "gap_pct": 0.0,
        }

    # -- gap / adaptive range ----------------------------------------------

    def _compute_gap_pct(self, bar: Bar, symbol_state: SymbolState) -> float:
        """Overnight gap: (today_open - prev_close) / prev_close."""
        prev_close: float | None = None
        if symbol_state.bars_1d and len(symbol_state.bars_1d) >= 1:
            prev_close = float(symbol_state.bars_1d[-1].close)
        else:
            pc = symbol_state.meta.get("prev_close")
            if pc is not None:
                prev_close = float(pc)
        if prev_close is None or prev_close <= 0:
            return 0.0
        return (bar.open - prev_close) / prev_close

    @staticmethod
    def _adaptive_range_minutes(gap_pct: float) -> int:
        abs_gap = abs(gap_pct)
        if abs_gap < 0.005:
            return 5
        if abs_gap < 0.015:
            return 10
        return 15

    # -- session reset / range building ------------------------------------

    def _maybe_reset(self, state: dict[str, Any], bar: Bar, ss: SymbolState) -> None:
        bar_date: date = time_utils.to_eastern_time(bar.time).date()
        if state["date"] == bar_date:
            return
        gap_pct = self._compute_gap_pct(bar, ss)
        state.update(self._empty_state())
        state["date"] = bar_date
        state["gap_pct"] = gap_pct
        state["range_minutes"] = self._adaptive_range_minutes(gap_pct)

    def _is_in_range_period(self, bar: Bar, state: dict[str, Any]) -> bool:
        t: time_type = time_utils.get_eastern_time_of_day(bar.time)
        if t < time_type(9, 30):
            return False
        end_min = 30 + state["range_minutes"]
        range_end = time_type(9 + end_min // 60, end_min % 60)
        return time_type(9, 30) <= t < range_end

    @staticmethod
    def _update_range(state: dict[str, Any], bar: Bar) -> None:
        state["range_high"] = max(state["range_high"], bar.high)
        state["range_low"] = min(state["range_low"], bar.low)
        state["range_bars"] += 1

    # -- avg volume --------------------------------------------------------

    @staticmethod
    def _avg_volume(ss: SymbolState, lookback: int = 20) -> float:
        cached = ss.indicators.get("sma_vol_1m:20")
        if cached is not None:
            try:
                return float(cached)
            except (TypeError, ValueError):
                pass
        bars = ss.bars_1m
        if not bars:
            return 0.0
        recent = list(bars)[-lookback:]
        return sum(float(b.volume) for b in recent) / len(recent) if recent else 0.0

    # -- confluence scoring ------------------------------------------------

    def _score_entry(
        self,
        side: OrderSide,
        bar: Bar,
        state: dict[str, Any],
        avg_vol: float,
        mtf: MultiTimeframeAnalyzer,
    ) -> ConfluenceScorer:
        scorer = ConfluenceScorer(threshold=self.confluence_threshold)
        rh: float = state["range_high"]
        rl: float = state["range_low"]
        rw = rh - rl
        gap_pct: float = state["gap_pct"]

        # 1. Breakout strength (0.25)
        if rw > 0:
            dev = ((bar.close - rh) if side == OrderSide.BUY else (rl - bar.close)) / rw
            s1 = score_deviation(dev, 0.0, 0.5)
        else:
            dev, s1 = 0.0, 0.0
        scorer.add_factor("breakout_strength", dev, s1, 0.25, passed=s1 > 0)

        # 2. Volume (0.25)
        s2 = score_volume(bar.volume, avg_vol, 1.2, 3.0) if avg_vol > 0 else 0.0
        scorer.add_factor("volume", bar.volume, s2, 0.25, passed=s2 > 0)

        # 3. Gap alignment (0.20)
        if side == OrderSide.BUY:
            s3 = 80.0 if gap_pct > 0.002 else (40.0 if abs(gap_pct) <= 0.002 else 20.0)
        else:
            s3 = 80.0 if gap_pct < -0.002 else (40.0 if abs(gap_pct) <= 0.002 else 20.0)
        scorer.add_factor("gap_alignment", gap_pct, s3, 0.20, passed=s3 >= 60)

        # 4. MTF trend alignment (0.15)
        mtf_raw = mtf.get_trend_alignment(side)
        scorer.add_factor("mtf_alignment", mtf_raw, mtf_raw * 100, 0.15, passed=mtf_raw > 0.3)

        # 5. Range quality -- tighter = better (0.15)
        if bar.close > 0 and rw > 0:
            rwp = rw / bar.close
            s5 = score_threshold(rwp, 0.003, 0.001, invert=True)
        else:
            rwp, s5 = 0.0, 0.0
        scorer.add_factor("range_quality", rwp, s5, 0.15, passed=s5 > 30)
        return scorer

    # -- stop / target -----------------------------------------------------

    def _compute_stop(
        self,
        side: OrderSide,
        state: dict[str, Any],
        ms: MarketState,
    ) -> float:
        base = state["range_high"] - state["range_low"]
        adj = self._apply_regime_volatility_multiplier(base, ms)
        if side == OrderSide.BUY:
            return state["range_high"] - adj
        return state["range_low"] + adj

    def _compute_target(
        self,
        side: OrderSide,
        bar: Bar,
        state: dict[str, Any],
        ss: SymbolState,
    ) -> float:
        rw = state["range_high"] - state["range_low"]
        if side == OrderSide.BUY:
            if ss.bars_1d and len(ss.bars_1d) >= 1:
                ph = float(ss.bars_1d[-1].high)
                if ph > bar.close:
                    return ph
            pdh = ss.meta.get("prior_day_high")
            if pdh is not None and float(pdh) > bar.close:
                return float(pdh)
            return bar.close + self.target_range_mult * rw
        # SELL
        if ss.bars_1d and len(ss.bars_1d) >= 1:
            pl = float(ss.bars_1d[-1].low)
            if pl < bar.close:
                return pl
        pdl = ss.meta.get("prior_day_low")
        if pdl is not None and float(pdl) < bar.close:
            return float(pdl)
        return bar.close - self.target_range_mult * rw

    def _exit_config(self) -> dict[str, Any]:
        return {
            "trailing_enabled": True,
            "trail_timeframe": "5m",
            "trail_lookback": 3,
            "trail_min_profit_r": self.trail_min_profit_r,
            "partial_exits": [(2.0, 0.5)],
            "max_hold_minutes": self.max_hold_minutes,
            "vol_adaptive": True,
        }

    # -- on_bar ------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Signal | None:
        if not bar:
            return None

        state = self._get_state(symbol)
        self._maybe_reset(state, bar, symbol_state)

        # Phase 1: build opening range
        if self._is_in_range_period(bar, state):
            self._update_range(state, bar)
            return None

        # Mark range complete once we leave the range window
        if not state["range_complete"] and state["range_bars"] > 0:
            state["range_complete"] = True

        # Pre-flight checks
        if not state["range_complete"] or state["breakout_fired"]:
            return None
        if state["range_high"] <= state["range_low"]:
            return None
        if not self._check_cooldown(symbol, bar.time):
            return None
        if not self._require_min_bars(symbol_state, self.min_bars, log=False):
            return None
        if not self._regime_ok(market_state):
            return None
        if symbol_state.position and symbol_state.position.strategy == self.name:
            return None

        # Breakout detection with volume gate
        avg_vol = self._avg_volume(symbol_state)
        vol_ok = avg_vol > 0 and bar.volume > avg_vol * self.vol_gate_mult
        side: OrderSide | None = None
        if bar.close > state["range_high"] and vol_ok:
            side = OrderSide.BUY
        elif bar.close < state["range_low"] and vol_ok:
            side = OrderSide.SELL
        if side is None:
            return None

        # Confluence scoring
        mtf = MultiTimeframeAnalyzer(symbol_state)
        scorer = self._score_entry(side, bar, state, avg_vol, mtf)
        if not scorer.passes_threshold():
            self.logger.debug(
                "orb_v2: below threshold",
                symbol=symbol,
                score=round(scorer.score(), 2),
                threshold=self.confluence_threshold,
            )
            return None

        # Stop / target
        stop_price = self._compute_stop(side, state, market_state)
        target_price = self._compute_target(side, bar, state, symbol_state)
        if side == OrderSide.BUY and stop_price >= bar.close:
            return None
        if side == OrderSide.SELL and stop_price <= bar.close:
            return None

        # Build meta
        meta: dict[str, Any] = scorer.to_meta()
        meta["exit_config"] = self._exit_config()
        meta["or_high"] = float(state["range_high"])
        meta["or_low"] = float(state["range_low"])
        meta["range_minutes"] = state["range_minutes"]
        meta["range_bars"] = state["range_bars"]
        meta["gap_pct"] = state["gap_pct"]
        meta["breakout_type"] = "high" if side == OrderSide.BUY else "low"
        meta["vol_ratio"] = round(bar.volume / avg_vol, 2) if avg_vol > 0 else 0.0

        # Record signal
        state["breakout_fired"] = True
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
