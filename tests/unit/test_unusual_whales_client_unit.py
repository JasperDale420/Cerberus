from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.unusual_whales import UnusualWhalesClient


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unusual_whales_client_returns_neutral_flow_when_unconfigured() -> None:
    cfg = MagicMock()
    cfg.get_env.return_value = ""
    logger = MagicMock()

    c = UnusualWhalesClient(cfg, logger)

    out = await c.get_option_flow("AAPL", "2025-01-01")
    assert out == []
    logger.warning.assert_called()
