from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

from src.core.domain import Bar, MarketState, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from src.core.domain import (
        LiquidityRegime,
        MarketRegimeSnapshot,
        RiskRegime,
        SessionRegime,
        TrendRegime,
        VolRegime,
    )


@dataclass(frozen=True)
class StrategyActivationPolicy:
    """
    Determines when a strategy is allowed to trade based on multi-axis regimes.

    Each field is a tuple of allowed values. Empty tuple means "no constraint".
    Strategy is active only if current regime matches ALL non-empty constraints.
    """

    # Allowed regime axis values (empty = any)
    session: Tuple[SessionRegime, ...] = ()
    trend: Tuple[TrendRegime, ...] = ()
    vol: Tuple[VolRegime, ...] = ()
    liquidity: Tuple[LiquidityRegime, ...] = ()
    risk: Tuple[RiskRegime, ...] = ()

    # Minimum confidence threshold (0.0 = no requirement)
    min_confidence: float = 0.0

    def is_active(self, regime: MarketRegimeSnapshot) -> bool:
        """
        Returns True if all axis constraints pass for the given regime snapshot.
        """
        # Check each axis constraint
        if self.session and regime.session not in self.session:
            return False
        if self.trend and regime.trend not in self.trend:
            return False
        if self.vol and regime.vol not in self.vol:
            return False
        if self.liquidity and regime.liquidity not in self.liquidity:
            return False
        if self.risk and regime.risk not in self.risk:
            return False

        # Check minimum confidence across all axes
        if self.min_confidence > 0.0:
            for axis_conf in regime.confidence.values():
                if axis_conf < self.min_confidence:
                    return False

        return True


@dataclass(frozen=True)
class StrategyRouting:
    """
    Routes strategies based on regime conditions.

    Supports both legacy strategies_by_regime and new activation_policies.
    New activation_policies take precedence when defined.
    """

    # Legacy: simple regime-to-strategy mapping
    strategies_by_regime: Mapping[Regime, List[str]] = field(default_factory=dict)

    # New: per-strategy activation policies
    activation_policies: Mapping[str, StrategyActivationPolicy] = field(
        default_factory=dict
    )


class StrategyEngine:
    def __init__(
        self,
        strategies_by_name: Dict[str, BaseStrategy],
        routing: StrategyRouting,
        logger: StructuredLogger,
        on_error: Optional[Callable[[], None]] = None,
    ) -> None:
        self.strategies_by_name = strategies_by_name
        self.routing = routing
        self.logger = logger
        self.on_error = on_error

    def on_bar(
        self,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> List[Signal]:
        active = self._get_active_strategies(symbol_state, market_state)
        out: List[Signal] = []

        for name in active:
            sig = self._safe_run_strategy(name, symbol, bar, symbol_state, market_state)
            if sig:
                out.append(sig)
        return out

    def _get_active_strategies(
        self, symbol_state: SymbolState, market_state: MarketState
    ) -> List[str]:
        """
        Determines which strategies are active for the current symbol and market state.

        Uses activation policies when available, falls back to legacy regime routing.
        When scanner_bypass is set, skips legacy regime routing but STILL respects
        activation policies (e.g., session filters).
        """
        allowed = set(symbol_state.allowed_strategies)
        meta = symbol_state.meta if isinstance(symbol_state.meta, dict) else {}
        scanner_bypass = bool(meta.get("scanner_bypass", False))

        active: List[str] = []

        for name in allowed:
            # Check if strategy has an activation policy
            policy = self.routing.activation_policies.get(name)

            if policy is not None:
                # New: Use activation policy with regime snapshot
                if market_state.regime_snapshot is not None:
                    if policy.is_active(market_state.regime_snapshot):
                        active.append(name)
                elif scanner_bypass:
                    # scanner_bypass + no snapshot: allow (can't filter without snapshot)
                    active.append(name)
                else:
                    # No snapshot available, fall back to legacy check
                    if self._legacy_regime_allows(name, market_state.regime):
                        active.append(name)
            else:
                # No activation policy for this strategy
                if scanner_bypass:
                    # Skip legacy routing when scanner_bypass is set
                    active.append(name)
                else:
                    # Legacy: Use strategies_by_regime mapping
                    if self._legacy_regime_allows(name, market_state.regime):
                        active.append(name)

            # --- Hard Stop Enforcement ---
            if name in active:
                strat = self.strategies_by_name.get(name)
                hard_stop_fn = (
                    getattr(strat, "is_past_hard_stop", None) if strat else None
                )
                if callable(hard_stop_fn):
                    # Use last known price bar time if available, otherwise current wall clock is risky in backtest
                    # Better to pass current_time explicitly to _get_active_strategies
                    if hard_stop_fn(market_state.time) is True:
                        active.remove(name)

        return sorted(active)

    def _legacy_regime_allows(self, strategy_name: str, regime: Regime) -> bool:
        """Check if strategy is allowed under legacy regime routing."""
        regime_allowed = set(self.routing.strategies_by_regime.get(regime, []))
        return strategy_name in regime_allowed

    def _safe_run_strategy(
        self,
        name: str,
        symbol: str,
        bar: Bar,
        symbol_state: SymbolState,
        market_state: MarketState,
    ) -> Optional[Signal]:
        strat = self.strategies_by_name.get(name)
        if strat is None:
            self.logger.warning("Strategy missing from registry", strategy=name)
            return None

        try:
            sig = strat.on_bar(symbol, bar, symbol_state, market_state)
            if (
                sig is not None
                and hasattr(sig, "feature_snapshot")
                and not sig.feature_snapshot
            ):
                # Inject feature snapshot from metadata if available
                # Note: symbol_state.meta["features"] is usually a SymbolFeatures object
                # We can store it as is or a dict snapshot.
                features = symbol_state.meta.get("features")
                if features:
                    sig.feature_snapshot = features
            return sig
        except Exception as e:
            if self.on_error:
                try:
                    self.on_error()
                except Exception:
                    pass

            from src.core.errors import ErrorCode

            self.logger.error(
                "Strategy error",
                error_code=ErrorCode.STRATEGY_ON_BAR_FAILED.value,
                strategy=name,
                symbol=symbol,
                regime=getattr(market_state.regime, "value", str(market_state.regime)),
                error=str(e),
            )
            return None
