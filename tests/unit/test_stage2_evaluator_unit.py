from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agent import stage2 as stage2_mod
from src.core.domain import Bar, OrderSide, Regime, Signal
from src.core.logger import StructuredLogger
from src.strategies.base import BaseStrategy


def _bar(
    symbol: str,
    t: datetime,
    *,
    o: float,
    h: float,
    low: float,
    c: float,
    v: float = 0.0,
) -> Dict[str, Any]:
    return {"t": t.isoformat(), "o": o, "h": h, "l": low, "c": c, "v": v}


@dataclass
class _DummyConfigLoader:
    def load_config(self, _path: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
        return {}


class _FakeAlpacaClient:
    def __init__(self, _config_loader: Any, _logger: StructuredLogger):
        self._bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    def set_bars(self, symbol: str, rows: List[Dict[str, Any]]) -> None:
        self._bars_by_symbol[str(symbol).upper()] = list(rows)

    def get_historical_bars(self, symbol: str, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"bars": list(self._bars_by_symbol.get(str(symbol).upper(), []))}


class _OneShotStrategy(BaseStrategy):
    name = "unit_test_strategy"

    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        super().__init__(config, logger)
        self._fired: set[str] = set()

    def on_bar(self, symbol: str, bar: Bar, *_args: Any, **_kwargs: Any) -> Optional[Signal]:
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
            strategy="unit_test_strategy",
            regime=Regime.CHOP,
            generated_at=bar.time,
            meta={},
            correlation_id="",
        )


@pytest.mark.unit
def test_stage2_symbols_prefers_agent_stage2_symbols(monkeypatch) -> None:
    monkeypatch.setattr(stage2_mod, "AlpacaClient", _FakeAlpacaClient)
    logger = StructuredLogger("test_stage2_symbols", level="INFO")
    cfg = {
        "agent": {"stage2": {"symbols": ["aapl", "MSFT", "msft"]}},
        "universe": {"symbols": ["tsla"]},
    }
    ev = stage2_mod.DeterministicStage2Evaluator(cfg, _DummyConfigLoader(), logger)
    assert ev._symbols() == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_stage2_symbols_falls_back_to_universe_symbols(monkeypatch) -> None:
    monkeypatch.setattr(stage2_mod, "AlpacaClient", _FakeAlpacaClient)
    logger = StructuredLogger("test_stage2_symbols_fallback", level="INFO")
    cfg = {
        "agent": {"stage2": {"symbols": []}},
        "universe": {"symbols": ["tsla", "aapl"]},
    }
    ev = stage2_mod.DeterministicStage2Evaluator(cfg, _DummyConfigLoader(), logger)
    assert ev._symbols() == ["AAPL", "TSLA"]


@pytest.mark.unit
def test_stage2_strategy_instance_validates_name(monkeypatch) -> None:
    monkeypatch.setattr(stage2_mod, "AlpacaClient", _FakeAlpacaClient)
    logger = StructuredLogger("test_stage2_strategy_mapping", level="INFO")
    ev = stage2_mod.DeterministicStage2Evaluator({}, _DummyConfigLoader(), logger)
    with pytest.raises(ValueError, match="Unknown strategy"):
        ev._strategy_instance("nope", {})


@pytest.mark.unit
def test_stage2_evaluate_stop_priority_when_both_hit(monkeypatch) -> None:
    monkeypatch.setattr(stage2_mod, "AlpacaClient", _FakeAlpacaClient)
    logger = StructuredLogger("test_stage2_eval_stop_priority", level="INFO")
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cfg = {"agent": {"stage2": {"symbols": ["AAPL"], "window_days": 1}}}
    ev = stage2_mod.DeterministicStage2Evaluator(cfg, _DummyConfigLoader(), logger, clock=lambda: now)

    assert isinstance(ev.alpaca, _FakeAlpacaClient)
    ev.alpaca.set_bars(
        "AAPL",
        [
            _bar("AAPL", now, o=100.0, h=100.0, low=100.0, c=100.0),
            # Next bar: stop (99) and target (102) both crossed; stop must win.
            _bar("AAPL", now + timedelta(minutes=1), o=100.0, h=103.0, low=98.0, c=101.0),
            _bar("AAPL", now + timedelta(minutes=2), o=101.0, h=101.0, low=101.0, c=101.0),
        ],
    )
    monkeypatch.setattr(ev, "_strategy_instance", lambda *_args, **_kwargs: _OneShotStrategy({}, logger))

    m = ev.evaluate("unit_test_strategy", regime=Regime.CHOP, params={}, as_of=now)
    assert m.n_trades == 1
    assert m.expectancy == pytest.approx(-1.0)  # stopped at 99 after entering at 100 with 1R risk
    assert m.max_drawdown_r == pytest.approx(1.0)


@pytest.mark.unit
def test_stage2_evaluate_max_hold_exits_at_close(monkeypatch) -> None:
    monkeypatch.setattr(stage2_mod, "AlpacaClient", _FakeAlpacaClient)
    logger = StructuredLogger("test_stage2_eval_max_hold", level="INFO")
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cfg = {"agent": {"stage2": {"symbols": ["AAPL"], "window_days": 1}}}
    ev = stage2_mod.DeterministicStage2Evaluator(cfg, _DummyConfigLoader(), logger, clock=lambda: now)

    assert isinstance(ev.alpaca, _FakeAlpacaClient)
    ev.alpaca.set_bars(
        "AAPL",
        [
            _bar("AAPL", now, o=100.0, h=100.0, low=100.0, c=100.0),
            _bar("AAPL", now + timedelta(seconds=1), o=100.0, h=100.5, low=99.5, c=100.0),
            # After >= 6 seconds, max-hold should exit at close (101.0).
            _bar("AAPL", now + timedelta(seconds=10), o=100.0, h=101.0, low=99.5, c=101.0),
        ],
    )
    monkeypatch.setattr(ev, "_strategy_instance", lambda *_args, **_kwargs: _OneShotStrategy({}, logger))

    m = ev.evaluate(
        "unit_test_strategy",
        regime=Regime.CHOP,
        params={"max_hold_minutes": 0.1},  # 6 seconds
        as_of=now,
    )
    assert m.n_trades == 1
    assert m.expectancy == pytest.approx(1.0)
    assert m.max_drawdown_r == pytest.approx(0.0)
