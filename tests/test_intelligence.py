"""Tests for router intelligence (triage logic)."""

import pytest

from app.router.intelligence import (
    ProviderKind,
    resolve_provider_and_model,
)


def test_tier1_explicit_gpt4o():
    """Explicit model alias routes to correct provider."""
    kind, model = resolve_provider_and_model("gpt-4o", [])
    assert kind == ProviderKind.OPENAI
    assert model == "gpt-4o"


def test_tier1_explicit_claude():
    """Claude alias routes to Vertex Claude."""
    kind, model = resolve_provider_and_model("claude-sonnet", [])
    assert kind == ProviderKind.VERTEX_CLAUDE
    assert "claude" in model.lower()


def test_tier1_explicit_gemini():
    """Gemini alias routes to Vertex Gemini."""
    kind, model = resolve_provider_and_model("gemini-pro", [])
    assert kind == ProviderKind.VERTEX_GEMINI
    assert "gemini" in model.lower()


def test_tier2_heuristic_code():
    """Code keywords route to Claude."""
    messages = [{"role": "user", "content": "Fix this bug in my function def foo():"}]
    kind, _ = resolve_provider_and_model(None, messages)
    assert kind == ProviderKind.VERTEX_CLAUDE


def test_tier2_heuristic_creative():
    """General prompt routes to Gemini (default)."""
    messages = [{"role": "user", "content": "Write a poem about the ocean"}]
    kind, _ = resolve_provider_and_model(None, messages)
    assert kind == ProviderKind.VERTEX_GEMINI
