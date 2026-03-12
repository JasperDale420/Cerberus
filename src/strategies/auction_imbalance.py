"""
Auction Imbalance Strategy.

Trades NYSE closing auction imbalances based on Morand (2024). At 3:50 PM ET,
consumes NYSE closing auction imbalance data. When imbalance exceeds 3 sigma
of its rolling distribution, enters in the imbalance direction. Distinguishes
mechanical (index rebalance -> expect reversal) vs informational (-> expect
continuation) imbalances.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import date
from datetime import time as time_type
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.domain import Bar, MarketState, OrderSide, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy

# ---------------------------------------------------------------------------
# Config Model (local to strategy, does NOT modify src/config/models.py)
# ---------------------------------------------------------------------------


class AuctionImbalanceConfig(BaseModel):
    """Configuration for the Auction Imbalance strategy."""

    enabled: bool = True
    entry_sigma: float = Field(default=3.0, description="Sigma threshold for entry signal")
    lookback_days: int = Field(default=20, description="Rolling window size in days")
    signal_time_minutes_before_close: int = Field(
        default=10, description="Minutes before 4:00 PM ET to start signaling"
    )
    mechanical_rebal_dates: List[str] = Field(
        default_factory=list,
        description="ISO-format dates of known mechanical rebalances (e.g. '2026-03-20')",
    )
    skip_mechanical: bool = Field(default=True, description="Skip signals on mechanical rebalance dates")
    stop_atr_multiplier: float = Field(default=1.5, description="ATR multiplier for stop distance")
    risk_reward: float = Field(default=2.0, description="Risk-reward ratio for target distance")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

# Market close is 16:00 ET
_MARKET_CLOSE_HOUR = 16
_MARKET_CLOSE_MINUTE = 0


class AuctionImbalanceStrategy(BaseStrategy):
    """
    Auction Imbalance Strategy.

    Monitors NYSE closing auction imbalance data from Alpaca's ``/stocks/auctions``
    endpoint. When the imbalance size exceeds ``entry_sigma`` standard deviations of
    the rolling distribution, a signal is generated in the direction of the imbalance.

    Mechanical rebalance dates (index reconstitution, quad-witching, etc.) are expected
    to produce mean-reverting flow.  By default these dates are skipped; when
    ``skip_mechanical=False`` the signal direction is reversed (fade the imbalance).

    Only active in the final minutes before market close, controlled by
    ``signal_time_minutes_before_close``.
    """

    name = "auction_imbalance"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)

        # Parse typed config
        self._cfg = AuctionImbalanceConfig(
            **{k: v for k, v in config.items() if k in AuctionImbalanceConfig.model_fields}
        )

        # Rolling window of historical imbalance sizes per symbol
        self._imbalance_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self._cfg.lookback_days))

        # Compute the earliest allowed signal time
        self._signal_time = self._compute_signal_time(self._cfg.signal_time_minutes_before_close)

        # Track signals already emitted per symbol per day to avoid duplicates
        self._signaled_today: Dict[str, str] = {}  # symbol -> date iso

        # Pre-parse mechanical rebalance dates
        self._mechanical_dates: set[date] = set()
        for d in self._cfg.mechanical_rebal_dates:
            try:
                self._mechanical_dates.add(date.fromisoformat(d))
            except ValueError:
                self.logger.warning("auction_imbalance: invalid mechanical date", date_str=d)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signal_time(minutes_before_close: int) -> time_type:
        """Return the earliest ``time`` object at which signals are allowed."""
        total_minutes = _MARKET_CLOSE_HOUR * 60 + _MARKET_CLOSE_MINUTE - minutes_before_close
        hour = total_minutes // 60
        minute = total_minutes % 60
        return time_type(hour, minute)

    def _is_past_signal_time(self, bar_time_local: time_type) -> bool:
        """Check whether the bar time is at or after the signal window."""
        return bar_time_local >= self._signal_time

    def _is_mechanical_date(self, d: date) -> bool:
        """Return True if ``d`` is a known mechanical rebalance date."""
        return d in self._mechanical_dates

    def _already_signaled(self, symbol: str, d: date) -> bool:
        """Return True if we already emitted a signal for *symbol* today."""
        return self._signaled_today.get(symbol) == d.isoformat()

    def _record_signal(self, symbol: str, d: date) -> None:
        self._signaled_today[symbol] = d.isoformat()

    def record_imbalance(self, symbol: str, imbalance_size: float) -> None:
        """
        Record a historical auction imbalance observation.

        Called externally (e.g. by a data feed handler) to populate the rolling
        window *before* ``on_bar`` is invoked on the signal bar.
        """
        self._imbalance_history[symbol].append(imbalance_size)

    def get_rolling_stats(self, symbol: str) -> tuple[float, float]:
        """
        Return (mean, std) of absolute imbalance values for *symbol*.

        Returns (0.0, 0.0) when insufficient data.
        """
        history = self._imbalance_history.get(symbol)
        if not history or len(history) < 2:
            return 0.0, 0.0

        abs_vals = [abs(v) for v in history]
        n = len(abs_vals)
        mean = sum(abs_vals) / n
        variance = sum((v - mean) ** 2 for v in abs_vals) / (n - 1)  # sample std
        std = math.sqrt(variance) if variance > 0 else 0.0
        return mean, std

    def classify_imbalance(self, d: date) -> str:
        """
        Classify an imbalance as 'mechanical' or 'informational'.

        Mechanical imbalances occur on known rebalance dates and are expected
        to revert. Informational imbalances are expected to continue.
        """
        if self._is_mechanical_date(d):
            return "mechanical"
        return "informational"

    # ------------------------------------------------------------------
    # Core: on_bar
    # ------------------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        """
        Process a bar and optionally emit an auction-imbalance signal.

        The strategy expects the current auction imbalance size to be stored in
        ``symbol_state.meta["auction_imbalance"]`` (positive = buy imbalance,
        negative = sell imbalance).  This value is populated by the execution
        engine or data feed handler from Alpaca ``/stocks/auctions`` data.

        Returns a Signal when conditions are met, otherwise None.
        """
        if not self._cfg.enabled:
            return None

        # 1. Time gate — only signal in the closing window
        bar_time = bar.time.time() if hasattr(bar.time, "time") else bar.time
        if isinstance(bar_time, time_type) and not self._is_past_signal_time(bar_time):
            return None

        # 2. Cooldown
        if not self._check_cooldown(symbol, bar.time):
            return None

        # 3. One signal per symbol per day
        bar_date = bar.time.date() if hasattr(bar.time, "date") else None
        if bar_date and self._already_signaled(symbol, bar_date):
            return None

        # 4. Read current imbalance from meta
        current_imbalance = symbol_state.meta.get("auction_imbalance")
        if current_imbalance is None:
            return None
        current_imbalance = float(current_imbalance)
        if current_imbalance == 0.0:
            return None

        # 5. Rolling stats check
        mean, std = self.get_rolling_stats(symbol)
        if std == 0.0:
            self.logger.debug(
                "auction_imbalance: zero std, skipping",
                symbol=symbol,
                history_len=len(self._imbalance_history.get(symbol, [])),
            )
            return None

        z_score = (abs(current_imbalance) - mean) / std
        if z_score < self._cfg.entry_sigma:
            return None

        # 6. Classify mechanical vs informational
        classification = self.classify_imbalance(bar_date) if bar_date else "informational"

        if classification == "mechanical" and self._cfg.skip_mechanical:
            self.logger.info(
                "auction_imbalance: skipping mechanical rebalance date",
                symbol=symbol,
                date=bar_date.isoformat() if bar_date else "unknown",
            )
            return None

        # 7. Determine direction
        if classification == "mechanical":
            # Fade the imbalance — expect reversal
            side = OrderSide.SELL if current_imbalance > 0 else OrderSide.BUY
        else:
            # Informational — ride the imbalance direction
            side = OrderSide.BUY if current_imbalance > 0 else OrderSide.SELL

        # 8. Compute stop and target
        atr = float(symbol_state.meta.get("atr", bar.close * 0.01))
        base_stop_distance = atr * self._cfg.stop_atr_multiplier
        stop_distance = self._apply_regime_volatility_multiplier(base_stop_distance, market_state)
        target_distance = stop_distance * self._cfg.risk_reward

        if side == OrderSide.BUY:
            stop_price = bar.close - stop_distance
            target_price = bar.close + target_distance
        else:
            stop_price = bar.close + stop_distance
            target_price = bar.close - target_distance

        # 9. Record and emit
        if bar_date:
            self._record_signal(symbol, bar_date)
        self.last_signal_time[symbol] = bar.time

        meta = {
            "imbalance_size": current_imbalance,
            "z_score": round(z_score, 4),
            "rolling_mean": round(mean, 2),
            "rolling_std": round(std, 2),
            "classification": classification,
            "lookback_days": self._cfg.lookback_days,
        }

        self.logger.info(
            "auction_imbalance: signal generated",
            symbol=symbol,
            side=side.value,
            z_score=round(z_score, 4),
            classification=classification,
            imbalance=current_imbalance,
        )

        return self._create_signal(
            symbol=symbol,
            side=side,
            bar=bar,
            market_state=market_state,
            stop_price=stop_price,
            target_price=target_price,
            meta=meta,
        )
