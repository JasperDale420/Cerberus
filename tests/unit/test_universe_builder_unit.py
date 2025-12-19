from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.scanner.universe import UniverseBuilder


class _DummyConfigLoader(ConfigLoader):
    def __init__(self, config: Dict[str, Any]):
        self._config = config

    def load_config(self, _path: Optional[str] = None) -> Dict[str, Any]:
        return dict(self._config)


class _FakeAlpaca:
    def __init__(self, volumes_by_symbol: Dict[str, float]):
        self._vols = {k.upper(): float(v) for k, v in volumes_by_symbol.items()}

    def get_historical_bars(
        self, symbol: str, *_args: Any, **_kwargs: Any
    ) -> Dict[str, Any]:
        v = self._vols.get(symbol.upper())
        if v is None:
            return {"bars": []}
        return {"bars": [{"v": float(v)}]}


def _logger() -> StructuredLogger:
    return StructuredLogger("test_universe_builder", level="INFO")


@pytest.mark.unit
def test_universe_builder_combines_sources_and_dedupes(tmp_path) -> None:
    static = tmp_path / "symbols.txt"
    static.write_text(
        "\n".join(
            [
                "# comment",
                "aapl",
                "TSLA,extra_column_is_ignored",
                "  msft ",
                "",
            ]
        )
    )

    cfg = {
        "universe": {
            "symbols": ["aapl", "MSFT"],
            "static_files": [str(static)],
            "dynamic": {
                "previous_day_top_volume": {
                    "enabled": True,
                    "top_n": 1,
                    "candidates": ["msft", "tsla"],
                    "lookback_days": 1,
                }
            },
        }
    }

    builder = UniverseBuilder(
        _DummyConfigLoader(cfg),
        _logger(),
        config=cfg,
        alpaca_client=_FakeAlpaca({"MSFT": 100.0, "TSLA": 200.0}),  # type: ignore
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert builder.build_universe() == ["AAPL", "MSFT", "TSLA"]


@pytest.mark.unit
def test_universe_builder_raises_when_empty() -> None:
    cfg: Dict[str, Any] = {"universe": {"symbols": []}}
    builder = UniverseBuilder(_DummyConfigLoader(cfg), _logger(), config=cfg)
    with pytest.raises(ValueError, match="Universe is empty"):
        builder.build_universe()
