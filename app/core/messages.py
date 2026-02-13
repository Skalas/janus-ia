"""Shared helpers for message content handling (OpenAI-style messages)."""

from typing import Any


def message_content_to_text(content: str | list[Any]) -> str:
    """
    Convert message content to a single string.
    Handles OpenAI-style content: either a plain string or a list of parts
    with type "text" and a "text" field.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts = [
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    return " ".join(parts) if parts else ""
