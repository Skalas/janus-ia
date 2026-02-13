"""OpenAI provider adapter - uses OpenAI API directly."""

from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.providers.base import LLMProvider
from app.providers.normalize import build_completion_response


class OpenAIAdapter(LLMProvider):
    """OpenAI Chat Completions adapter."""

    def __init__(self) -> None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not configured")
        self.client = AsyncOpenAI(api_key=api_key)

    def _to_openai_messages(
        self, messages: list[dict[str, str | list[Any]]]
    ) -> list[dict[str, Any]]:
        """Convert internal format to OpenAI messages."""
        out: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if isinstance(content, list):
                out.append({"role": role, "content": content})
            else:
                out.append({"role": role, "content": str(content)})
        return out

    async def chat_completion(
        self,
        messages: list[dict[str, str | list[Any]]],
        stream: bool = False,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Call OpenAI chat completions, return normalized response."""
        target_model = model or settings.OPENAI_DEFAULT_MODEL
        oai_messages = self._to_openai_messages(messages)

        if stream:
            return self._stream(oai_messages, target_model, temperature, max_tokens)
        return await self._complete(oai_messages, target_model, temperature, max_tokens)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        resp = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._normalize_response(resp, model)

    async def _stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield {
                        "id": chunk.id or "janus",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta.content},
                                "finish_reason": chunk.choices[0].finish_reason,
                            }
                        ],
                    }

    def _normalize_response(self, resp: Any, model: str) -> dict[str, Any]:
        c = resp.choices[0] if resp.choices else None
        content = c.message.content if c and c.message else ""
        finish_reason = c.finish_reason if c else "stop"
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return build_completion_response(
            id=resp.id or "janus",
            model=model,
            message_content=content,
            finish_reason=finish_reason,
            usage=usage,
        )
