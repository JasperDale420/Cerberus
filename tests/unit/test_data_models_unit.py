from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.models import Quote, Trade


@pytest.mark.unit
def test_trade_conditions_is_not_shared_between_instances() -> None:
    t1 = Trade(symbol="AAPL", timestamp=datetime.now(timezone.utc), price=1.0, size=1.0)
    t2 = Trade(symbol="AAPL", timestamp=datetime.now(timezone.utc), price=1.0, size=1.0)

    t1.conditions.append("c1")
    assert t1.conditions == ["c1"]
    assert t2.conditions == []


@pytest.mark.unit
def test_quote_conditions_is_not_shared_between_instances() -> None:
    q1 = Quote(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        bid_price=1.0,
        bid_size=1.0,
        ask_price=1.1,
        ask_size=2.0,
    )
    q2 = Quote(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        bid_price=1.0,
        bid_size=1.0,
        ask_price=1.1,
        ask_size=2.0,
    )

    q1.conditions.append("c1")
    assert q1.conditions == ["c1"]
    assert q2.conditions == []
