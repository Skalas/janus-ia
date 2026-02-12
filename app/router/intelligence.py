"""Router / Triage logic - decides which provider handles each request."""

from enum import Enum
import re

from app.core.config import settings


class ProviderKind(str, Enum):
    OPENAI = "openai"
    VERTEX_CLAUDE = "vertex_claude"
    VERTEX_GEMINI = "vertex_gemini"


# Keywords that suggest code/complex logic -> Claude
CODE_KEYWORDS = re.compile(
    r"\b(def\s|function\s|class\s|import\s|from\s|sql\s|SELECT\s|INSERT\s|"
    r"bug\s|error\s|fix\s|debug|refactor|algorithm|regex|API\s)\b",
    re.IGNORECASE,
)


def resolve_provider_and_model(
    model: str | None,
    messages: list[dict],
    *,
    force_provider: ProviderKind | None = None,
) -> tuple[ProviderKind, str]:
    """
    Tier 1: If model is an explicit alias -> resolve to provider + model.
    Tier 2: Heuristic on last user message -> pick provider + default model.

    Returns (provider_kind, actual_model_id).
    """
    if force_provider:
        provider, resolved = _default_model_for(force_provider)
        return force_provider, resolved

    normalized = (model or "").strip().lower()
    if normalized:
        # Tier 1: explicit model alias
        resolved = settings.MODEL_ALIASES.get(normalized)
        if resolved:
            provider = _provider_for_model_id(resolved)
            return provider, resolved

    # Tier 2: heuristic
    last_user_content = _last_user_content(messages)
    if CODE_KEYWORDS.search(last_user_content):
        return ProviderKind.VERTEX_CLAUDE, settings.VERTEX_CLAUDE_MODEL
    # Default: Gemini (cost-effective, large context)
    return ProviderKind.VERTEX_GEMINI, settings.VERTEX_GEMINI_MODEL


def _provider_for_model_id(model_id: str) -> ProviderKind:
    """Map model ID to provider (by prefix or known IDs)."""
    mid = model_id.lower()
    if mid.startswith("gpt-") or "openai" in mid:
        return ProviderKind.OPENAI
    if "claude" in mid:
        return ProviderKind.VERTEX_CLAUDE
    if "gemini" in mid:
        return ProviderKind.VERTEX_GEMINI
    return ProviderKind.VERTEX_GEMINI  # fallback


def _default_model_for(provider: ProviderKind) -> tuple[ProviderKind, str]:
    """Return (provider, default_model_id)."""
    if provider == ProviderKind.OPENAI:
        return provider, settings.OPENAI_DEFAULT_MODEL
    if provider == ProviderKind.VERTEX_CLAUDE:
        return provider, settings.VERTEX_CLAUDE_MODEL
    return provider, settings.VERTEX_GEMINI_MODEL


def _last_user_content(messages: list[dict]) -> str:
    """Get concatenated text from last user message(s)."""
    if not messages:
        return ""
    # Walk backwards to find last user message
    for m in reversed(messages):
        if str(m.get("role", "")).lower() == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(parts)
            return ""
    return ""
