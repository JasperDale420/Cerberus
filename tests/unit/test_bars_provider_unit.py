from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agent.bars_provider import JsonlBarsProvider


@pytest.mark.unit
def test_jsonl_bars_provider_filters_window_and_sorts(tmp_path: Path) -> None:
    p = tmp_path / "AAPL_1Min.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"t":"2025-01-01T00:00:02+00:00","o":1,"h":1,"l":1,"c":1,"v":1}',
                '{"t":"2025-01-01T00:00:01+00:00","o":2,"h":2,"l":2,"c":2,"v":2}',
                '{"t":"2025-01-01T00:00:03+00:00","o":3,"h":3,"l":3,"c":3,"v":3}',
            ]
        )
    )

    provider = JsonlBarsProvider(tmp_path)
    start = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    bars = provider.get_bars("aapl", start, end, "1Min")

    assert [b.time.isoformat() for b in bars] == [
        "2025-01-01T00:00:01+00:00",
        "2025-01-01T00:00:02+00:00",
    ]
    assert bars[0].close == pytest.approx(2.0)
    assert bars[1].close == pytest.approx(1.0)
