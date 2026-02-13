"""Vertex AI Claude (Anthropic on Vertex) provider adapter."""

import asyncio
from typing import Any, AsyncGenerator

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.providers.base import LLMProvider


class VertexClaudeProvider(LLMProvider):
    """Claude on Vertex AI via Anthropic SDK."""

    def __init__(self) -> None:
        project_id = settings.GCP_PROJECT_ID
        if not project_id:
            raise ProviderError("GCP_PROJECT_ID not configured for Vertex Claude")
        try:
            from anthropic import AnthropicVertex
        except ImportError as e:
            raise ProviderError(
                "anthropic[vertex] not installed. Run: uv add 'anthropic[vertex]'"
            ) from e
        # Claude en Vertex no está en us-central1; usar GCP_LOCATION_CLAUDE (p.ej. "global")
        region = settings.GCP_LOCATION_CLAUDE
        self.client = AnthropicVertex(project_id=project_id, region=region)

    def _to_anthropic_messages(
        self, messages: list[dict[str, str | list[Any]]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract system prompt and convert to Anthropic format."""
        system = ""
        anthropic_msgs: list[dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if isinstance(content, list):
                # Flatten to text for MVP
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                content = " ".join(parts) if parts else ""
            content = str(content)
            if role == "system":
                system = content
            else:
                anthropic_msgs.append(
                    {
                        "role": "user" if role == "user" else "assistant",
                        "content": content,
                    }
                )
        return system, anthropic_msgs

    async def chat_completion(
        self,
        messages: list[dict[str, str | list[Any]]],
        stream: bool = False,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Call Claude on Vertex, return normalized OpenAI-style response."""
        target_model = model or settings.VERTEX_CLAUDE_MODEL
        system, anthropic_msgs = self._to_anthropic_messages(messages)

        if stream:
            return self._stream(
                anthropic_msgs, system, target_model, temperature, max_tokens
            )
        return await self._complete(
            anthropic_msgs, system, target_model, temperature, max_tokens
        )

    async def _complete(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        def _sync() -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            if temperature is not None:
                kwargs["temperature"] = temperature
            return self.client.messages.create(**kwargs)

        resp = await asyncio.to_thread(_sync)
        return self._normalize_response(resp, model)

    async def _stream(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Vertex Claude sync streaming: collect in thread, yield async
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce() -> None:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            if temperature is not None:
                kwargs["temperature"] = temperature
            with self.client.messages.stream(**kwargs) as s:
                for text in s.text_stream:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(_produce))
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield {
                "id": "janus",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                ],
            }

    def _normalize_response(self, resp: Any, model: str) -> dict[str, Any]:
        """Solo incluir bloques de tipo 'text'; ignorar thinking y otros."""
        text_parts = []
        if resp.content:
            for b in resp.content:
                block_type = getattr(b, "type", None) or ""
                if block_type == "text" and hasattr(b, "text") and b.text:
                    text_parts.append(str(b.text))
        text = "".join(text_parts)
        return {
            "id": resp.id or "janus",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": getattr(resp, "usage", None)
                and getattr(resp.usage, "input_tokens", 0)
                or 0,
                "completion_tokens": getattr(resp, "usage", None)
                and getattr(resp.usage, "output_tokens", 0)
                or 0,
                "total_tokens": 0,
            },
        }
