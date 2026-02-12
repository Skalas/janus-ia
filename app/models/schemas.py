"""OpenAI-compatible request/response schemas for Janus IA."""

from pydantic import BaseModel, Field
from typing import Any


# --- Request (OpenAI Chat Completions compatible) ---


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str | list[Any] = ""  # str or list of content parts (e.g. vision)


class ChatCompletionRequest(BaseModel):
    model: str = "claude-sonnet"  # Alias or hint; router may override
    messages: list[Message]
    temperature: float | None = 0.7
    max_tokens: int | None = 1024
    stream: bool = False
    # Extras for future: tools, tool_choice, stop, etc.


class MultiChatCompletionRequest(BaseModel):
    """Same prompt to multiple models in parallel."""
    models: list[str]  # Aliases, e.g. ["gemini-pro", "claude-sonnet"]
    messages: list[Message]
    temperature: float | None = 0.7
    max_tokens: int | None = 1024


# --- Response (OpenAI ChatCompletion format) ---


class Choice(BaseModel):
    index: int = 0
    message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = "janus"
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage | None = None
