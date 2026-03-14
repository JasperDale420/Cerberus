from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agent import llm as llm_mod
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class _DummyConfigLoader(ConfigLoader):
    def load_config(self, _path: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
        return {}

    def get_env(self, key: str, default: str = "") -> str:
        return default


@pytest.mark.unit
def test_llm_client_complete_builds_openai_like_messages(monkeypatch) -> None:
    """LLMClient.complete sends correct messages to the LLM HTTP endpoint."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch.object(llm_mod, "create_http_client", return_value=mock_client):
        logger = StructuredLogger("test_llm_client", level="INFO")
        client = llm_mod.LLMClient(_DummyConfigLoader(), logger, model_name="m")

    out = client.complete("hi", system_prompt="sys")
    assert out == "ok"

    mock_client.post.assert_called_once_with(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
