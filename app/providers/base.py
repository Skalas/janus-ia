from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class LLMProvider(ABC):
    """Standard interface for all LLM providers."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """
        Standard interface for all providers.
        Input: messages [{'role': 'user', 'content': '...'}, ...]
        Output: Normalized OpenAI-compatible dict or async generator (when stream=True).
        """
        ...
