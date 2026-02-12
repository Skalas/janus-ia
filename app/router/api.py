"""Chat completions API - routes to providers and returns normalized responses."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.core.security import validate_api_key
from app.models.schemas import ChatCompletionRequest, MultiChatCompletionRequest
from app.providers.openai_adapter import OpenAIAdapter
from app.providers.vertex_claude import VertexClaudeProvider
from app.providers.vertex_gemini import VertexGeminiProvider
import asyncio
from app.router.intelligence import ProviderKind, resolve_provider_and_model

router = APIRouter()

# Lazy provider instances (created on first use)
_providers: dict[ProviderKind, Any] = {}


def _get_provider(kind: ProviderKind):
    """Get or create provider instance."""
    if kind not in _providers:
        if kind == ProviderKind.OPENAI:
            _providers[kind] = OpenAIAdapter()
        elif kind == ProviderKind.VERTEX_CLAUDE:
            _providers[kind] = VertexClaudeProvider()
        elif kind == ProviderKind.VERTEX_GEMINI:
            _providers[kind] = VertexGeminiProvider()
    return _providers[kind]


def _messages_to_dict(messages: list) -> list[dict]:
    """Convert Pydantic Message to dict."""
    return [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in messages]


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Optional API key validation
    if settings.JANUS_API_KEY:
        validate_api_key(request)

    messages = _messages_to_dict(body.messages)
    provider_kind, model_id = resolve_provider_and_model(body.model, messages)

    provider = None
    try:
        provider = _get_provider(provider_kind)
    except ProviderError as e:
        if provider_kind != ProviderKind.OPENAI:
            try:
                provider = _get_provider(ProviderKind.OPENAI)
                model_id = settings.OPENAI_DEFAULT_MODEL
            except ProviderError:
                pass
        if provider is None:
            raise HTTPException(status_code=503, detail=str(e)) from e

    stream = body.stream
    temperature = body.temperature or 0.7
    max_tokens = body.max_tokens or 1024

    result = await provider.chat_completion(
        messages,
        stream=stream,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if stream:
        return StreamingResponse(
            _stream_json(result),
            media_type="text/event-stream",
        )

    return result


@router.post("/chat/completions/multi")
async def chat_completions_multi(request: Request, body: MultiChatCompletionRequest):
    """Send the same messages to multiple models in parallel; returns one response per model."""
    if settings.JANUS_API_KEY:
        validate_api_key(request)

    messages = _messages_to_dict(body.messages)
    temperature = body.temperature or 0.7
    max_tokens = body.max_tokens or 1024

    async def _one(model_alias: str):
        provider_kind, model_id = resolve_provider_and_model(model_alias, messages)
        try:
            provider = _get_provider(provider_kind)
        except ProviderError as e:
            return {"error": str(e)}
        try:
            result = await provider.chat_completion(
                messages,
                stream=False,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            return {"error": str(e)}

    results = await asyncio.gather(
        *(_one(alias) for alias in body.models),
        return_exceptions=True,
    )
    model_responses = {}
    for alias, resp in zip(body.models, results):
        if isinstance(resp, Exception):
            model_responses[alias] = {"error": str(resp)}
        else:
            model_responses[alias] = resp

    return {"model_responses": model_responses}


async def _stream_json(agen):
    """Yield SSE-formatted chunks from async generator."""
    async for chunk in agen:
        yield f"data: {json.dumps(chunk)}\n\n"
