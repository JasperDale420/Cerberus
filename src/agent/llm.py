from typing import Optional, cast

from src.core.config import ConfigLoader
from src.core.http_client import create_http_client
from src.core.logger import StructuredLogger


class LLMClient:
    """
    Wrapper for Any-LLM to handle model calls.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        logger: StructuredLogger,
        model_name: str = "deepseek-reasoner",
    ):
        self.logger = logger
        self.model_name = model_name
        llm_base_url = config_loader.get_env(
            "CENTRAL_LLM_API_URL",
            config_loader.get_env("CERBERUS_GATEWAY_URL", "http://localhost:8080"),
        )
        self._llm_client = create_http_client(base_url=llm_base_url, timeout=120.0)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a completion for the given prompt via LLM API.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._llm_client.post(
                "/v1/chat/completions",
                json={"model": self.model_name, "messages": messages},
            )
            response.raise_for_status()
            data = cast(dict, response.json())
            return cast(str, data["choices"][0]["message"]["content"])
        except Exception as e:
            self.logger.error("LLM completion failed", error=str(e))
            raise
