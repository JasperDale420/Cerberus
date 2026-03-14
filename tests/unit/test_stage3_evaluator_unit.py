from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

from src.agent import stage3 as stage3_mod
from src.core.domain import Regime
from src.core.logger import StructuredLogger


def _bar_row(t: datetime, *, o: float, h: float, low: float, c: float, v: float = 0.0) -> Dict[str, Any]:
    return {"t": t.isoformat(), "o": o, "h": h, "l": low, "c": c, "v": v}


class _FakeAlpacaClient:
    def __init__(self) -> None:
        self._bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    def set_bars(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        self._bars_by_symbol[str(symbol).upper()] = list(rows)

    def get_historical_bars(self, symbol: str, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"bars": list(self._bars_by_symbol.get(str(symbol).upper(), []))}


class _DummyConfigLoader:
    def load_config(self, _path: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
        return {}


@pytest.mark.unit
def test_stage3_extract_strategy_class_errors_when_none() -> None:
    logger = StructuredLogger("test_stage3_extract_none", level="INFO")
    ev = stage3_mod.DeterministicStage3Evaluator({}, _DummyConfigLoader(), logger)
    mod = ModuleType("empty_mod")
    with pytest.raises(ValueError, match="No BaseStrategy subclass"):
        ev._extract_strategy_class(mod)


@pytest.mark.unit
def test_stage3_evaluate_code_stop_priority() -> None:
    logger = StructuredLogger("test_stage3_eval", level="INFO")
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cfg = {"agent": {"stage3": {"backtest": {"symbols": ["AAPL"], "window_days": 1}}}}
    ev = stage3_mod.DeterministicStage3Evaluator(cfg, _DummyConfigLoader(), logger, clock=lambda: now)

    fake_alpaca = _FakeAlpacaClient()
    ev.alpaca = fake_alpaca
    fake_alpaca.set_bars(
        "AAPL",
        [
            _bar_row(now, o=100.0, h=100.0, low=100.0, c=100.0),
            # stop (99) and target (102) both crossed; stop wins -> -1R
            _bar_row(now + timedelta(minutes=1), o=100.0, h=103.0, low=98.0, c=101.0),
            _bar_row(now + timedelta(minutes=2), o=101.0, h=101.0, low=101.0, c=101.0),
        ],
    )

    code = """
from datetime import timezone
from src.core.domain import OrderSide, Regime, Signal
from src.strategies.base import BaseStrategy

class CandidateStrategy(BaseStrategy):
    name = "candidate"
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self._fired = set()
    def on_bar(self, symbol, bar, symbol_state, market_state):
        if symbol in self._fired:
            return None
        self._fired.add(symbol)
        return Signal(
            symbol=symbol,
            side=OrderSide.BUY,
            size_hint=1.0,
            entry_price=float(bar.close),
            stop_price=float(bar.close - 1.0),
            target_price=float(bar.close + 2.0),
            strategy="candidate",
            regime=Regime.CHOP,
            generated_at=bar.time,
            meta={},
            correlation_id="",
        )
"""

    m = ev.evaluate_code(code, strategy_params={}, regime=Regime.CHOP, as_of=now)
    assert m.n_trades == 1
    assert m.expectancy == pytest.approx(-1.0)
    assert m.max_drawdown_r == pytest.approx(1.0)
