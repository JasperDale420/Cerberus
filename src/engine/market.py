from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.analysis.db import DatabaseDatabase
from src.analysis.regime import Regime, RegimeDetector
from src.analysis.schema import RegimeHistory
from src.core.domain import MarketState, RiskMode
from src.core.logger import StructuredLogger


class MarketStateManager:
    """
    Manages the global market state and regime detection.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: StructuredLogger,
        db: Optional[DatabaseDatabase] = None,
        clock: Optional[Callable[[], datetime]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.logger = logger
        self.db = db
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_error = on_error or (lambda x: None)

        self.state = MarketState(time=self.clock(), regime=Regime.CHOP)
        self.regime_detector = RegimeDetector(
            logger=logger, on_error=lambda: self.on_error("regime")
        )

    def update(self, bar: Any, index_symbol: Optional[str] = None) -> None:
        """
        Updates the market state based on the provided bar.
        Expected to be called with the index symbol's bar.
        """
        idx_sym = index_symbol or self.config.get("index_symbol", "SPY")
        bar_time = getattr(bar, "time", getattr(bar, "timestamp", None))

        # Update Regime
        self.state.regime = self.regime_detector.update(bar)

        # Determinism: Use bar timestamp if available
        self.state.time = bar_time or self.clock()
        self.state.index_symbol = str(idx_sym)
        self.state.index_price = float(getattr(bar, "close", 0.0) or 0.0)
        self.state.index_return = float(
            getattr(self.regime_detector, "last_cum_ret", 0.0) or 0.0
        )
        self.state.realized_vol = float(
            getattr(self.regime_detector, "last_vol", 0.0) or 0.0
        )

        # Expose regime metrics for strategy gating
        try:
            meta = self.state.meta if isinstance(self.state.meta, dict) else {}
            meta = dict(meta)
            meta["trend_score"] = float(
                getattr(self.regime_detector, "last_trend_score", 0.0) or 0.0
            )
            self.state.meta = meta
        except Exception:
            pass

        # Persist regime updates
        if self.db:
            ok = self.db.write(
                "regime_history",
                lambda session: session.add(
                    RegimeHistory(
                        timestamp=self.state.time,
                        regime=self.state.regime.value,
                        index_symbol=self.state.index_symbol,
                        index_price=float(getattr(bar, "close", 0.0) or 0.0),
                        cum_ret=getattr(self.regime_detector, "last_cum_ret", None),
                        trend_score=getattr(
                            self.regime_detector, "last_trend_score", None
                        ),
                        vol=getattr(self.regime_detector, "last_vol", None),
                    )
                ),
            )
            if not ok:
                self.on_error("db")

    def set_risk_mode(self, mode: Any) -> None:
        """
        Updates the risk mode in the market state.
        Accepts string (from config) or RiskMode enum.
        """
        m_str = str(mode or "normal").lower()
        if hasattr(mode, "value") and isinstance(mode.value, str):
            m_str = mode.value.lower()

        try:
            self.state.risk_mode = (
                RiskMode.OFF
                if m_str == "off"
                else RiskMode.REDUCED if m_str == "reduced" else RiskMode.NORMAL
            )
        except Exception:
            pass
