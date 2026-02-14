from __future__ import annotations

from typing import Any

from structlog.testing import capture_logs

from src.agent.bars_provider import JsonlBarsProvider
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.scanner.universe import UniverseBuilder


def _has_message(logs: list[dict[str, Any]], message: str) -> bool:
    for entry in logs:
        if entry.get("event") == message or entry.get("message") == message:
            return True
    return False


def test_build_universe_falls_back_to_offline_symbols_file(tmp_path) -> None:
    symbols_file = tmp_path / "offline_symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n")

    config = {
        "universe": {
            "symbols": [],
            "static_files": ["data/offline_bars_jan2024/offline_symbols.txt"],
            "dynamic": {},
        }
    }

    builder = UniverseBuilder(
        config_loader=ConfigLoader(),
        logger=StructuredLogger("UniverseBuilderTest"),
        config=config,
        offline_bars_provider=JsonlBarsProvider(tmp_path),
    )

    with capture_logs() as logs:
        universe = builder.build_universe()

    assert "AAPL" in universe
    assert "MSFT" in universe
    assert _has_message(logs, "Universe static file fallback")
