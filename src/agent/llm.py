from typing import Optional, cast

from src.core.config import ConfigLoader
from src.core.logger import StructuredLogger
from src.data.api_client import CentralApiClient


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
        self.central_client = CentralApiClient(config_loader, logger)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a completion for the given prompt via Central API.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.central_client.chat_completion(self.model_name, messages)
            # Response structure matches OpenAI format (from router)
            # choices[0].message.content
            return cast(str, response["choices"][0]["message"]["content"])
        except Exception as e:
            self.logger.error("LLM completion failed", error=str(e))
            raise
