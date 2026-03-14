from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.core.logger import StructuredLogger
from src.scanner.universe import UniverseBuilder


class _FakeUnifiedClient:
    """Fake UnifiedDataClient that returns bars with volume data."""

    def __init__(self, volumes_by_symbol: Dict[str, float]):
        self._vols = {k.upper(): float(v) for k, v in volumes_by_symbol.items()}

    def get_historical_bars(self, symbol: str, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        v = self._vols.get(symbol.upper())
        if v is None:
            return {"bars": []}
        return {"bars": [{"v": float(v)}]}

    def get_most_actives(self, top: int = 20) -> list[str]:
        return []

    def get_movers(self, top: int = 10) -> Dict[str, list[str]]:
        return {"gainers": [], "losers": []}


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
        _FakeUnifiedClient({"MSFT": 100.0, "TSLA": 200.0}),  # type: ignore
        _logger(),
        config=cfg,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert builder.build_universe() == ["AAPL", "MSFT", "TSLA"]


@pytest.mark.unit
def test_universe_builder_raises_when_empty() -> None:
    cfg: Dict[str, Any] = {"universe": {"symbols": []}}
    builder = UniverseBuilder(MagicMock(), _logger(), config=cfg)
    with pytest.raises(ValueError, match="Universe is empty"):
        builder.build_universe()


@pytest.mark.unit
def test_universe_builder_uses_unified_client_for_dynamic_volume_selection() -> None:
    cfg = {
        "universe": {
            "symbols": ["AAPL"],
            "dynamic": {
                "previous_day_top_volume": {
                    "enabled": True,
                    "top_n": 1,
                    "candidates": ["MSFT", "TSLA"],
                    "lookback_days": 1,
                }
            },
        }
    }

    builder = UniverseBuilder(
        _FakeUnifiedClient({"MSFT": 100.0, "TSLA": 300.0}),  # type: ignore
        _logger(),
        config=cfg,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    assert builder.build_universe() == ["AAPL", "TSLA"]
