"""CPPI (Constant Proportion Portfolio Insurance) drawdown-controlled sizing.

Dynamically scales portfolio exposure based on a protective floor that ratchets
up with equity growth. As equity approaches the floor, exposure shrinks toward
min_exposure; as cushion grows, exposure scales back toward max_exposure.

Algorithm:
    floor = max(floor, equity * (1 - max_drawdown))
    cushion = max(0, equity - floor)
    raw_exposure = multiplier * cushion / equity
    exposure = clamp(raw_exposure, min_exposure, max_exposure)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.config.models import CPPIConfig
from src.core.logger import StructuredLogger


@dataclass
class CPPIState:
    """Snapshot of current CPPI state."""

    equity: float
    floor: float
    cushion: float
    exposure_fraction: float
    high_water_mark: float
    drawdown_from_hwm: float


class CPPISizer:
    """CPPI-based position sizing controller.

    Tracks portfolio equity against a protective floor and computes an exposure
    fraction that risk.py uses to scale position sizes. The floor ratchets up
    with new equity highs, compressing exposure as drawdowns accumulate.
    """

    def __init__(
        self,
        config: CPPIConfig,
        initial_equity: float = 100_000.0,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._equity = initial_equity
        self._high_water_mark = initial_equity
        self._floor = initial_equity * (1.0 - config.max_drawdown)
        self._exposure_fraction = self._compute_exposure(initial_equity, self._floor)
        self._last_decay_date: date | None = None

    def update_equity(self, equity: float) -> CPPIState:
        """Update with current equity, recompute floor/cushion/exposure.

        Call after every PnL update to keep the CPPI controller in sync
        with actual portfolio value.
        """
        self._equity = equity

        # Edge case: zero or negative equity
        if equity <= 0:
            self._exposure_fraction = self._config.min_exposure
            return self.get_state()

        # Update high water mark
        self._high_water_mark = max(self._high_water_mark, equity)

        # Apply floor decay once per day BEFORE ratchet, so that
        # ratchet-then-decay doesn't create call-order-dependent floors
        if self._config.floor_decay_rate > 0:
            today = date.today()
            if self._last_decay_date != today:
                self._floor = self._floor * (1.0 - self._config.floor_decay_rate)
                self._last_decay_date = today

        # Ratchet floor up with equity growth
        self._floor = max(self._floor, equity * (1.0 - self._config.max_drawdown))

        # Compute exposure
        self._exposure_fraction = self._compute_exposure(equity, self._floor)

        # Log warnings at low exposure levels
        if self._logger is not None:
            if self._exposure_fraction <= self._config.min_exposure:
                self._logger.critical(
                    "cppi_at_minimum_exposure",
                    equity=equity,
                    floor=self._floor,
                    exposure=self._exposure_fraction,
                    drawdown=self._drawdown_from_hwm(),
                )
            elif self._exposure_fraction < 0.5:
                self._logger.warning(
                    "cppi_low_exposure",
                    equity=equity,
                    floor=self._floor,
                    exposure=self._exposure_fraction,
                    drawdown=self._drawdown_from_hwm(),
                )

        return self.get_state()

    def get_cppi_multiplier(self) -> float:
        """Return current exposure fraction (0.0-1.0) for risk.py position scaling."""
        return self._exposure_fraction

    def reset(self, equity: float) -> None:
        """Reset CPPI to a new starting equity."""
        self._equity = equity
        self._high_water_mark = equity
        self._floor = equity * (1.0 - self._config.max_drawdown)
        self._exposure_fraction = self._compute_exposure(equity, self._floor)

    def get_state(self) -> CPPIState:
        """Return current state snapshot."""
        equity = self._equity
        cushion = max(0.0, equity - self._floor) if equity > 0 else 0.0
        return CPPIState(
            equity=equity,
            floor=self._floor,
            cushion=cushion,
            exposure_fraction=self._exposure_fraction,
            high_water_mark=self._high_water_mark,
            drawdown_from_hwm=self._drawdown_from_hwm(),
        )

    def _compute_exposure(self, equity: float, floor: float) -> float:
        """Compute clamped CPPI exposure fraction."""
        if equity <= 0:
            return self._config.min_exposure

        cushion = max(0.0, equity - floor)
        if cushion <= 0:
            return self._config.min_exposure

        raw_exposure = self._config.multiplier * cushion / equity
        return max(self._config.min_exposure, min(self._config.max_exposure, raw_exposure))

    def _drawdown_from_hwm(self) -> float:
        """Compute current drawdown percentage from high water mark."""
        if self._high_water_mark <= 0:
            return 0.0
        return (self._high_water_mark - self._equity) / self._high_water_mark
