from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pytest

from src.core.domain import Bar, MarketState, OrderSide, Regime, Signal, SymbolState
from src.core.logger import StructuredLogger
from src.engine.risk import RiskManager


def _logger(name: str) -> StructuredLogger:
    return StructuredLogger(name, level="INFO")


def _mk_signal(
    *,
    strategy: str = "s",
    regime: Regime = Regime.CHOP,
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
    size_hint: float = 10.0,
) -> Signal:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return Signal(
        symbol="AAPL",
        side=OrderSide.BUY,
        size_hint=size_hint,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        strategy=strategy,
        regime=regime,
        generated_at=now,
        meta={},
        correlation_id="corr-1",
    )


def _mk_state() -> SymbolState:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return SymbolState(
        symbol="AAPL",
        bars=deque(
            [Bar(symbol="AAPL", time=now, open=1, high=1, low=1, close=1, volume=0)],
            maxlen=10,
        ),
        indicators={},
        position=None,
        open_orders={},
        allowed_strategies=[],
        meta={},
    )


@pytest.mark.unit
def test_risk_manager_rejects_strategy_disabled_and_regime_override() -> None:
    cfg = {
        "risk": {"risk_mode": "normal"},
        "strategies": {"s": {"enabled": True, "regimes": {"chop": {"enabled": False}}}},
    }
    rm = RiskManager(cfg, _logger("test_risk_disabled"))
    out = rm.apply(
        _mk_signal(strategy="s", regime=Regime.CHOP),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "REGIME_DISABLED"


@pytest.mark.unit
def test_risk_manager_rejects_risk_mode_off() -> None:
    rm = RiskManager({"risk": {"risk_mode": "off"}}, _logger("test_risk_off"))
    out = rm.apply(
        _mk_signal(),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "RISK_MODE_OFF"


@pytest.mark.unit
def test_risk_manager_rejects_trade_and_order_caps() -> None:
    rm = RiskManager(
        {"risk": {"max_trades_per_day": 1}, "max_orders_per_day": 1},
        _logger("test_caps"),
    )
    rm.daily_entry_count = 1
    out = rm.apply(
        _mk_signal(),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_TRADES_PER_DAY"

    rm.daily_entry_count = 0
    rm.daily_order_count = rm.max_orders_per_day
    out = rm.apply(
        _mk_signal(),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_DAILY_ORDERS"


@pytest.mark.unit
def test_risk_manager_rejects_max_trades_per_strategy() -> None:
    rm = RiskManager({"risk": {"max_trades_per_strategy": 2}}, _logger("test_strat_trade_cap"))
    rm.per_strategy_entry_count["s"] = 2
    out = rm.apply(
        _mk_signal(strategy="s"),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_STRAT_TRADES"


@pytest.mark.unit
def test_risk_manager_rejects_daily_loss_and_invalid_stop() -> None:
    rm = RiskManager({"risk": {"max_daily_loss": 10}}, _logger("test_daily_loss"))
    rm.current_daily_pnl = -10.0
    out = rm.apply(
        _mk_signal(),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_DAILY_LOSS"

    rm.current_daily_pnl = 0.0
    out = rm.apply(
        _mk_signal(entry=100.0, stop=100.0),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "INVALID_STOP"


@pytest.mark.unit
def test_risk_manager_rolls_daily_counters_by_market_time() -> None:
    rm = RiskManager({"risk": {"max_trades_per_day": 1}}, _logger("test_rollover"))

    day1 = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 ET
    day2 = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)  # next session day

    # First entry on day1 allowed.
    out = rm.apply(
        _mk_signal(strategy="s"),
        _mk_state(),
        MarketState(time=day1, regime=Regime.CHOP),
    )
    assert out != []
    assert rm.daily_entry_count == 1
    assert rm.per_strategy_entry_count.get("s") == 1

    # Second entry on day1 rejected by cap.
    out2 = rm.apply(
        _mk_signal(strategy="s"),
        _mk_state(),
        MarketState(time=day1, regime=Regime.CHOP),
    )
    assert out2 == []
    assert rm.last_rejection_reason == "MAX_TRADES_PER_DAY"

    # Next day: counters roll and entry allowed again.
    out3 = rm.apply(
        _mk_signal(strategy="s"),
        _mk_state(),
        MarketState(time=day2, regime=Regime.CHOP),
    )
    assert out3 != []
    assert rm.daily_entry_count == 1
    assert rm.per_strategy_entry_count.get("s") == 1

    # PnL also rolls deterministically when timestamps advance.
    rm.update_pnl(-10.0, as_of=day1)
    assert rm.current_daily_pnl == -10.0
    rm.update_pnl(-5.0, as_of=day2)
    assert rm.current_daily_pnl == -5.0


@pytest.mark.unit
def test_risk_manager_rejects_notional_symbol_and_open_risk_caps() -> None:
    cfg = {
        "risk": {
            "max_notional_per_order": 100.0,
            "max_notional_per_symbol": 150.0,
            "max_open_risk": 5.0,
        }
    }
    rm = RiskManager(cfg, _logger("test_exposure_caps"))

    # Notional per order: 10 * 100 = 1000 > 100 — qty is capped to 1 (not rejected).
    # Risk manager now caps qty to fit notional limit rather than rejecting the signal.
    out = rm.apply(
        _mk_signal(entry=100.0, stop=99.0, size_hint=10),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert len(out) == 1
    assert out[0].qty == 1  # capped: int(100.0 / 100.0) = 1

    # MAX_NOTIONAL rejection when entry_price > cap (capped_qty rounds down to 0).
    rm2 = RiskManager(cfg, _logger("test_exposure_caps2"))
    out2 = rm2.apply(
        _mk_signal(entry=200.0, stop=199.0, size_hint=10),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out2 == []
    assert rm2.last_rejection_reason == "MAX_NOTIONAL"

    # Symbol notional: existing 100 + proposed 100 > 150
    # Cap risk per trade to 1.0 so qty_limit=1 and size_hint=1.0 (full conviction) yields qty=1.
    rm.max_risk_per_trade = 1.0
    st = _mk_state()
    st.position = type("P", (), {"qty": 1.0, "avg_price": 100.0})()
    out = rm.apply(
        _mk_signal(entry=100.0, stop=99.0, size_hint=1),
        st,
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_SYMBOL_NOTIONAL"

    # Open risk cap: proposed risk 1*10=10 > cap after existing open_risk=0
    # Restore max_risk_per_trade so size_hint=10 (absolute) passes qty_limit check.
    rm.max_risk_per_trade = 50.0
    rm.max_notional_per_order = 1e9
    rm.max_notional_per_symbol = 0.0
    out = rm.apply(
        _mk_signal(entry=100.0, stop=99.0, size_hint=10),
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
        current_positions=[type("P", (), {"open_risk": 0.0})()],
    )
    assert out == []
    assert rm.last_rejection_reason == "MAX_OPEN_RISK"


@pytest.mark.unit
def test_risk_manager_approves_and_emits_order_intent() -> None:
    cfg = {
        "risk": {
            "risk_mode": "reduced",
            "max_risk_per_trade": 100.0,
            "time_in_force": "day",
        }
    }
    rm = RiskManager(cfg, _logger("test_risk_approve"))
    sig = _mk_signal(entry=100.0, stop=90.0, target=120.0, size_hint=1000.0)
    intents = rm.apply(
        sig,
        _mk_state(),
        MarketState(time=datetime.now(timezone.utc), regime=Regime.CHOP),
    )
    assert len(intents) == 1
    intent = intents[0]
    assert intent.symbol == "AAPL"
    assert intent.time_in_force == "day"
    # Reduced mode halves max risk; risk/share=10 => qty=5 (floor)
    assert intent.qty == 5
    assert intent.stop_loss == 90.0
    assert intent.take_profit == 120.0
