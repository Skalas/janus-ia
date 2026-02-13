"""Shared helpers for OpenAI-compatible response normalization."""

from typing import Any


def build_completion_response(
    *,
    id: str = "janus",
    model: str,
    message_content: str,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a normalized OpenAI-style chat.completion response dict."""
    return {
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message_content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def build_stream_chunk(content_delta: str, *, id: str = "janus") -> dict[str, Any]:
    """Build a normalized OpenAI-style chat.completion.chunk dict."""
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": content_delta}, "finish_reason": None}
        ],
    }
