from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from src.analysis.db import DatabaseDatabase
from src.analysis.regime import MarketContextService, Regime
from src.analysis.schema import RegimeHistory
from src.core.domain import MarketState, RiskMode
from src.core.logger import StructuredLogger

# HMM minimum bars required before prediction (rolling windows need warmup)
_HMM_MIN_BARS = 25
# Maximum bars to keep in the rolling buffer (memory-bounded)
_HMM_MAX_BARS = 200


class MarketStateManager:
    """
    Manages the global market state and regime detection.

    Optionally runs a trained HMM regime model in shadow mode alongside the
    rule-based detector and logs both predictions for comparison.
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
        # When False, skip DB writes for regime history (backtest mode).
        self.persist_to_db: bool = True

        self.state = MarketState(time=self.clock(), regime=Regime.CHOP)
        regime_cfg = (config.get("regime") if isinstance(config.get("regime"), dict) else {}) or {}
        tz = str(config.get("timezone", "America/New_York") or "America/New_York")
        idx_sym = str(config.get("index_symbol", "SPY") or "SPY")
        vol_sym = regime_cfg.get("vol_symbol")
        vol_sym = str(vol_sym) if isinstance(vol_sym, str) and vol_sym else None

        self.market_context = MarketContextService(
            window=int(regime_cfg.get("window", 60) or 60),
            min_bars=int(regime_cfg.get("min_bars", 20) or 20),
            vol_baseline_window=int(regime_cfg.get("vol_baseline_window", 120) or 120),
            smooth_k=int(regime_cfg.get("smooth_k", 5) or 5),
            logger=logger,
            tz=tz,
            index_symbol=idx_sym,
            vol_symbol=vol_sym,
        )

        # HMM shadow mode: load trained model if enabled
        self._hmm_service: Any = None
        self._hmm_bar_buffer: deque[dict[str, Any]] = deque(maxlen=_HMM_MAX_BARS)
        self._init_hmm(config)

    def _init_hmm(self, config: Dict[str, Any]) -> None:
        """Load the trained HMM model if enabled in config."""
        hmm_cfg_raw = config.get("hmm_regime", {})
        if not isinstance(hmm_cfg_raw, dict):
            return

        enabled = hmm_cfg_raw.get("enabled", False)
        runtime_cfg = hmm_cfg_raw.get("runtime", {})
        runtime_enabled = runtime_cfg.get("enabled", False) if isinstance(runtime_cfg, dict) else False

        if not (enabled and runtime_enabled):
            return

        try:
            from src.regime_models.hmm.config import HmmConfig
            from src.regime_models.hmm.service import HmmRegimeService

            hmm_config = HmmConfig.model_validate(hmm_cfg_raw)
            artifact_path = Path(hmm_config.artifact_dir)
            if not artifact_path.exists():
                self.logger.warning(
                    "hmm_artifacts_not_found",
                    artifact_dir=str(artifact_path),
                    detail="HMM enabled but no trained artifacts found. Run train_hmm_from_heber.py first.",
                )
                return

            self._hmm_service = HmmRegimeService.load(
                path=artifact_path,
                config=hmm_config,
                logger=self.logger,
            )
            self.logger.info(
                "hmm_shadow_loaded",
                artifact_dir=str(artifact_path),
                mode=hmm_config.runtime.mode,
                n_states=hmm_config.training.n_states,
            )
        except Exception:
            self.logger.warning("hmm_init_failed", exc_info=True)

    def _hmm_shadow_update(self, bar: Any) -> None:
        """Run HMM prediction on accumulated bar history and store in meta."""
        if self._hmm_service is None:
            return

        # Accumulate bar data for the rolling window
        self._hmm_bar_buffer.append(
            {
                "time": getattr(bar, "time", getattr(bar, "timestamp", None)) or self.clock(),
                "open": float(getattr(bar, "open", 0.0) or 0.0),
                "high": float(getattr(bar, "high", 0.0) or 0.0),
                "low": float(getattr(bar, "low", 0.0) or 0.0),
                "close": float(getattr(bar, "close", 0.0) or 0.0),
                "volume": float(getattr(bar, "volume", 0.0) or 0.0),
            }
        )

        if len(self._hmm_bar_buffer) < _HMM_MIN_BARS:
            return

        try:
            frame = pd.DataFrame(list(self._hmm_bar_buffer))
            predictions = self._hmm_service.predict_ohlcv(frame)

            # Take the most recent prediction
            latest = predictions.iloc[-1]
            hmm_result = {
                "state": int(latest["hidden_state"]),
                "label": str(latest["hidden_state_name"]),
                "confidence": float(latest["hidden_state_confidence"]),
            }

            # Store in market state meta for observability
            meta = self.state.meta if isinstance(self.state.meta, dict) else {}
            meta = dict(meta)
            meta["hmm_regime"] = hmm_result
            self.state.meta = meta

            # Log comparison with rule-based regime
            self.logger.info(
                "hmm_shadow_prediction",
                hmm_label=hmm_result["label"],
                hmm_confidence=round(hmm_result["confidence"], 3),
                rule_based_regime=getattr(self.state.regime, "value", str(self.state.regime)),
                bar_count=len(self._hmm_bar_buffer),
            )
        except Exception:
            self.logger.warning("hmm_predict_failed", exc_info=True)

    def update(self, bar: Any, index_symbol: Optional[str] = None) -> None:
        """
        Updates the market state based on the provided bar.
        Expected to be called with the index symbol's bar.
        """
        idx_sym = index_symbol or self.config.get("index_symbol", "SPY")
        bar_time = getattr(bar, "time", getattr(bar, "timestamp", None))

        # Update multi-axis regime snapshot + legacy regime
        snapshot = self.market_context.update(bar)
        self.state.regime_snapshot = snapshot
        self.state.regime = snapshot.legacy_regime

        # Determinism: Use bar timestamp if available
        self.state.time = bar_time or self.clock()
        self.state.index_symbol = str(idx_sym)
        self.state.index_price = float(getattr(bar, "close", 0.0) or 0.0)
        self.state.index_return = float(getattr(self.market_context, "last_cum_ret", 0.0) or 0.0)
        self.state.realized_vol = float(getattr(self.market_context, "last_vol", 0.0) or 0.0)

        # Expose regime metrics for strategy gating
        try:
            meta = self.state.meta if isinstance(self.state.meta, dict) else {}
            meta = dict(meta)
            meta["trend_score"] = float(getattr(self.market_context, "last_trend_score", 0.0) or 0.0)
            meta["regime_tags"] = snapshot.regime_tags if snapshot is not None else meta.get("regime_tags")
            self.state.meta = meta
        except Exception:
            self.logger.warning("market_state_meta_update_failed", exc_info=True)

        # HMM shadow mode: predict alongside rule-based regime
        self._hmm_shadow_update(bar)

        # Persist regime updates (skip in backtest mode)
        if self.db and self.persist_to_db:
            ok = self.db.write(
                "regime_history",
                lambda session: session.add(
                    RegimeHistory(
                        timestamp=self.state.time,
                        regime=self.state.regime.value,
                        index_symbol=self.state.index_symbol,
                        index_price=float(getattr(bar, "close", 0.0) or 0.0),
                        cum_ret=getattr(self.market_context, "last_cum_ret", None),
                        trend_score=getattr(self.market_context, "last_trend_score", None),
                        vol=getattr(self.market_context, "last_vol", None),
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
                RiskMode.OFF if m_str == "off" else RiskMode.REDUCED if m_str == "reduced" else RiskMode.NORMAL
            )
        except Exception:
            self.logger.error("set_risk_mode_failed", mode=str(mode), exc_info=True)

    def update_vol(self, bar: Any) -> None:
        """
        Updates VXX price history for risk axis calculation.
        Call this with VXX bars when available.
        """
        self.market_context.update_vol(bar)

    def update_iv_surface(
        self,
        chain_data: list,
        current_price: float = 0.0,
        risk_free_rate: float = 0.05,
        time_to_expiry: float = 30 / 365,
        term_data: list | None = None,
    ) -> None:
        """Update IV surface dynamics from options chain data."""
        self.market_context.update_iv_surface(
            chain_data,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
            time_to_expiry=time_to_expiry,
            term_data=term_data,
        )

    def update_etf_flow(
        self,
        etf_symbol: str,
        etf_data: dict,
        stock_symbol: str,
        stock_data: dict,
    ) -> None:
        """Update ETF flow propagation from UW-format flow dicts."""
        self.market_context.update_etf_flow(etf_symbol, etf_data, stock_symbol, stock_data)
