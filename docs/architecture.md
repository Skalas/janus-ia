# Janus IA: Meta Chat Orchestrator – Architecture

**Author:** Atlas (Senior Backend Architect)  
**Date:** 2026-02-12  
**Status:** Implemented (MVP)  
**Project:** Janus IA — unified LLM orchestration layer.

---

## 1. Executive Summary

**Janus IA** is a unified LLM orchestration layer that decouples the client application from specific model providers. By sitting between the frontend and the LLMs, Janus IA enables dynamic routing based on cost, performance, and capability without requiring client-side changes.

This architecture uses **FastAPI** for high-performance async I/O (crucial for streaming) and a **Provider Adapter Pattern** to normalize the APIs of Vertex AI (Claude/Gemini) and OpenAI into a single OpenAI-compatible interface.

---

## 2. High-Level Architecture

The system follows a **Gateway / Router** pattern.

```mermaid
graph LR
    Client[Frontend / Client App] -->|Unified JSON Request| API[Janus IA API Gateway]
    
    subgraph "Janus IA (e.g. Cloud Run)"
        API --> Auth[Auth & Rate Limit]
        Auth --> Router[Triage / Router Logic]
        
        Router -->|Complex Task| AdapterA[Claude Adapter (Vertex AI)]
        Router -->|Creative/Fast| AdapterB[Gemini Adapter (Vertex AI)]
        Router -->|Reasoning/Standard| AdapterC[OpenAI Adapter]
    end
    
    AdapterA -->|GCP Auth| VertexClaude[Anthropic on Vertex]
    AdapterB -->|GCP Auth| VertexGemini[Gemini on Vertex]
    AdapterC -->|API Key| OAI[OpenAI API]
```

### Data Flow
1. **Ingest:** Client sends a standardized request (OpenAI-compatible schema).
2. **Triage:** The router analyzes the request (explicit model alias or heuristic on last user message).
3. **Adapt:** The request is transformed into the provider’s format (e.g. Anthropic/Gemini/OpenAI).
4. **Execute:** The provider is called (async; Vertex providers use `asyncio.to_thread` for sync SDKs).
5. **Normalize:** The response (stream or bulk) is normalized via shared helpers and returned to the client.

---

## 3. Core Components

### 3.1. Unified Interface (OpenAI-compatible)
Janus IA exposes an **OpenAI-compatible API** so existing tools (LangChain, chat UIs) can switch by changing `base_url`.

* **Request:** Standard `v1/chat/completions` body (`messages`, `model` (alias or hint), `temperature`, `max_tokens`, `stream`).
* **Response:** Standard OpenAI `ChatCompletion` or `ChatCompletionChunk`, built via `app/providers/normalize.py`.

### 3.2. Router / Triage Logic
**Rule-based tiered router** in `app/router/intelligence.py`:

* **Tier 1 (Explicit):** If the client sends a known model alias (e.g. `gpt-4o`, `claude-sonnet`, `gemini-pro`), route to the corresponding provider and model.
* **Tier 2 (Heuristic):** If no explicit model, use the last user message: code-related keywords → **Vertex Claude**; otherwise default → **Vertex Gemini**.
* **Fallback:** If the chosen provider fails to initialize (e.g. missing credentials), the single-completion endpoint can fall back to OpenAI when configured.

### 3.3. Auth & Configuration
* **GCP Vertex (Claude & Gemini):** ADC (Application Default Credentials); env: `GCP_PROJECT_ID`, `GCP_LOCATION_GEMINI`, `GCP_LOCATION_CLAUDE`.
* **OpenAI:** Env: `OPENAI_API_KEY`.
* **Janus API:** Optional `JANUS_API_KEY` for validating incoming requests (Bearer or `X-API-Key`).

---

## 4. Code Structure (Current)

```
janus-ia/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings (SettingsConfigDict)
│   │   ├── exceptions.py      # JanusException, ProviderError
│   │   ├── messages.py        # message_content_to_text() — shared message content handling
│   │   └── security.py        # API key validation for incoming requests
│   ├── models/
│   │   ├── domain.py          # Reserved for internal domain types
│   │   └── schemas.py         # Pydantic Req/Res (OpenAI compat)
│   ├── providers/
│   │   ├── base.py            # LLMProvider (ABC)
│   │   ├── normalize.py       # build_completion_response(), build_stream_chunk()
│   │   ├── openai_adapter.py  # OpenAI implementation
│   │   ├── vertex_claude.py   # Anthropic on Vertex
│   │   └── vertex_gemini.py   # Gemini on Vertex
│   ├── router/
│   │   ├── api.py             # POST /v1/chat/completions, /v1/chat/completions/multi
│   │   └── intelligence.py   # resolve_provider_and_model(), heuristics
│   └── main.py                # FastAPI app entrypoint
├── tests/
├── docs/
│   ├── architecture.md       # This file
│   └── architecture-review.md # Code review and optimization log
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 5. Adapter Pattern

All providers implement `LLMProvider` in `app/providers/base.py`:

* **Input:** `messages` (list of dicts with `role` and `content`), `stream`, `model`, `temperature`, `max_tokens`.
* **Output:** Either a single normalized dict (OpenAI-style completion) or an async generator of normalized chunks (streaming).

Message content (string or list of parts with `type: "text"`) is normalized via `app/core/messages.message_content_to_text()`. Responses and stream chunks are built with `app/providers/normalize.build_completion_response()` and `build_stream_chunk()`.

Provider instances are created lazily and registered in `api.py` via `_PROVIDER_FACTORY`; new providers are added by extending the registry and `ProviderKind` enum.

---

## 6. Roadmap (Reference)

* **Phase 1:** FastAPI skeleton, Docker, OpenAI adapter, `/v1/chat/completions` — done.
* **Phase 2:** Vertex Gemini and Claude adapters, triage — done.
* **Phase 3:** Router logic and heuristics — done. Optional: Streamlit/Chainlit UI, embedding-based routing.

---

## 7. Dependencies

See `pyproject.toml`. Key dependencies: `fastapi`, `uvicorn`, `pydantic-settings`, `openai`, `anthropic[vertex]`, `google-cloud-aiplatform`, `httpx`, `tiktoken` (optional for future token-based routing).
