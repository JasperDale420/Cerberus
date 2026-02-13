from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.agent import llm as llm_mod
from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger


class _DummyConfigLoader(ConfigLoader):
    def load_config(self, _path: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
        return {}


class _FakeCentralApiClient:
    def __init__(self, _config_loader: ConfigLoader, _logger: StructuredLogger):
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(self, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.calls.append({"model": model, "messages": messages})
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.unit
def test_llm_client_complete_builds_openai_like_messages(monkeypatch) -> None:
    monkeypatch.setattr(llm_mod, "CentralApiClient", _FakeCentralApiClient)
    logger = StructuredLogger("test_llm_client", level="INFO")
    client = llm_mod.LLMClient(_DummyConfigLoader(), logger, model_name="m")

    out = client.complete("hi", system_prompt="sys")
    assert out == "ok"
    assert client.central_client.calls == [  # type: ignore
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        }
    ]
