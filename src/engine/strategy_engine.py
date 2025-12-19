from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

from src.core.domain import Bar, MarketState, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


@dataclass(frozen=True)
class StrategyRouting:
    strategies_by_regime: Mapping[Regime, List[str]]


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
        allowed = set(symbol_state.allowed_strategies)
        regime_allowed = set(
            self.routing.strategies_by_regime.get(market_state.regime, [])
        )
        active = sorted(allowed.intersection(regime_allowed))

        out: List[Signal] = []
        for name in active:
            strat = self.strategies_by_name.get(name)
            if strat is None:
                self.logger.warning("Strategy missing from registry", strategy=name)
                continue
            try:
                sig = strat.on_bar(symbol, bar, symbol_state, market_state)
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
                    regime=getattr(
                        market_state.regime, "value", str(market_state.regime)
                    ),
                    error=str(e),
                )
                continue
            if sig is not None:
                out.append(sig)
        return out
