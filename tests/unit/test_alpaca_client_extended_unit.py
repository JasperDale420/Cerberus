from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.core.logger import StructuredLogger
from src.data import alpaca as alpaca_mod
from src.data.alpaca import AlpacaClient


class _DummyConfigLoader:
    def get_env(self, key: str, default: Optional[str] = None) -> str:
        if key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
            return "test"
        if key == "ALPACA_PAPER":
            return default or "True"
        return default or ""


@dataclass
class _FakeBarsRequest:
    symbol_or_symbols: str
    start: datetime
    end: datetime
    timeframe: Any


class _FakeHistoricalClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.requests: List[_FakeBarsRequest] = []
        self._bars_by_symbol: Dict[str, List[Any]] = {}

    def set_bars(self, symbol: str, bars: List[Any]) -> None:
        self._bars_by_symbol[symbol] = list(bars)

    def get_stock_bars(self, req: _FakeBarsRequest) -> Any:
        self.requests.append(req)
        bars = self._bars_by_symbol.get(req.symbol_or_symbols, [])
        return SimpleNamespace(data={req.symbol_or_symbols: bars})


class _FakeTradingClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _FakeStream:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.subscribed: List[str] = []
        self.unsubscribed: List[str] = []

    def subscribe_bars(self, _handler: Any, symbol: str) -> None:
        self.subscribed.append(symbol)

    def unsubscribe_bars(self, symbol: str) -> None:
        self.unsubscribed.append(symbol)


@pytest.mark.unit
def test_alpaca_client_get_historical_bars_timeframe_mapping_and_shape(
    monkeypatch,
) -> None:
    fake_hist = _FakeHistoricalClient()

    monkeypatch.setattr(alpaca_mod, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(
        alpaca_mod, "StockHistoricalDataClient", lambda *_a, **_k: fake_hist
    )
    monkeypatch.setattr(alpaca_mod, "StockDataStream", _FakeStream)
    monkeypatch.setattr(alpaca_mod, "TradingStream", object)
    monkeypatch.setattr(alpaca_mod, "StockBarsRequest", _FakeBarsRequest)
    monkeypatch.setattr(
        alpaca_mod, "TimeFrame", SimpleNamespace(Minute="Minute", Day="Day")
    )

    logger = StructuredLogger("test_alpaca_historical", level="INFO")
    client = AlpacaClient(_DummyConfigLoader(), logger)  # type: ignore

    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fake_hist.set_bars(
        "AAPL",
        [SimpleNamespace(timestamp=t, open=1, high=2, low=0.5, close=1.5, volume=10)],
    )
    out = client.get_historical_bars("AAPL", t, t, timeframe="1Min")
    assert out == {
        "bars": [{"t": t, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10.0}]
    }

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        client.get_historical_bars("AAPL", t, t, timeframe="5Min")


@pytest.mark.unit
def test_alpaca_client_subscribe_and_unsubscribe_queues_then_subscribes(
    monkeypatch,
) -> None:
    fake_hist = _FakeHistoricalClient()

    monkeypatch.setattr(alpaca_mod, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(
        alpaca_mod, "StockHistoricalDataClient", lambda *_a, **_k: fake_hist
    )
    monkeypatch.setattr(alpaca_mod, "StockBarsRequest", _FakeBarsRequest)
    monkeypatch.setattr(
        alpaca_mod, "TimeFrame", SimpleNamespace(Minute="Minute", Day="Day")
    )

    stream = _FakeStream()
    monkeypatch.setattr(alpaca_mod, "StockDataStream", lambda *_a, **_k: stream)

    logger = StructuredLogger("test_alpaca_subscribe", level="INFO")
    client = AlpacaClient(_DummyConfigLoader(), logger)  # type: ignore

    client.subscribe("AAPL")
    assert "AAPL" in client._pending_symbols
    assert stream.subscribed == []

    client._bar_handler = lambda *_a, **_k: None
    client.subscribe("AAPL")
    assert "AAPL" in stream.subscribed
    assert "AAPL" in client._subscribed_symbols

    client.unsubscribe("AAPL")
    assert stream.unsubscribed == ["AAPL"]
    assert "AAPL" not in client._subscribed_symbols
