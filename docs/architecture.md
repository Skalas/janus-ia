# Giorgio: Meta Chat Orchestrator - Architecture Proposal

**Author:** Atlas (Senior Backend Architect)  
**Date:** 2026-02-12  
**Status:** Draft / MVP Proposal  

---

## 1. Executive Summary

"Giorgio" is a unified LLM orchestration layer designed to decouple the client application from specific model providers. By sitting between the frontend and the LLMs, Giorgio enables dynamic routing based on cost, performance, and capability without requiring client-side changes.

This architecture leverages **FastAPI** for high-performance async I/O (crucial for streaming tokens) and adopts a **Provider Adapter Pattern** to normalize the disparate APIs of Vertex AI (Claude/Gemini) and OpenAI.

---

## 2. High-Level Architecture

The system follows a standard **Gateway / Router** pattern.

```mermaid
graph LR
    Client[Frontend / Client App] -->|Unified JSON Request| API[Giorgio API Gateway]
    
    subgraph "Giorgio Orchestrator (Cloud Run)"
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
1.  **Ingest:** Client sends a standardized request (OpenAI-compatible schema recommended).
2.  **Triage:** The Router analyzes the request (complexity, tokens, explicit model selection).
3.  **Adapt:** The request is transformed into the specific provider's format (e.g., protobuf for Google, JSON for OpenAI).
4.  **Execute:** The provider is called (async).
5.  **Normalize:** The response (stream or bulk) is normalized back to the standard schema and returned to the client.

---

## 3. Core Components

### 3.1. Unified Interface (The "Lingua Franca")
To minimize friction, Giorgio will expose an **OpenAI-compatible API**. This allows existing tools (like LangChain or standard UI chat kits) to connect to Giorgio by simply changing the `base_url`.

*   **Request Schema:** Standard `v1/chat/completions` body (`messages`, `model` (optional hint), `temperature`, `stream`).
*   **Response Schema:** Standard OpenAI `ChatCompletion` or `ChatCompletionChunk`.

### 3.2. Router / Triage Logic
For the MVP, we will avoid heavy ML classifiers to keep latency low. We will implement a **Rule-Based Tiered Router**.

*   **Tier 1 (Explicit):** If the client requests a specific model alias (e.g., `model="gpt-4o"`), route directly.
*   **Tier 2 (Heuristic - MVP):**
    *   *Code/Complex Logic:* Keywords (`def `, `function`, `bug`, `sql`) -> **Claude 3.5 Sonnet (Vertex)**.
    *   *Creative/General:* Default -> **Gemini 1.5 Pro (Vertex)** (Cost-effective, large context).
    *   *Fallback:* **OpenAI GPT-4o** if others fail or for specific reasoning tasks.
*   **Future:** Embedding-based semantic routing (using a lightweight vector search to classify intent).

### 3.3. Auth & Configuration
*   **GCP Vertex (Claude & Gemini):**
    *   Use **Workload Identity Federation** or standard `ADC` (Application Default Credentials) when running on Cloud Run. No service account keys in code!
    *   Environment Variables: `GCP_PROJECT_ID`, `GCP_LOCATION` (e.g., `us-central1`).
*   **OpenAI:**
    *   Secret Manager (Google Secret Manager) mapped to env var `OPENAI_API_KEY`.

---

## 4. Code Structure Proposal

We will use a modular "Vertical Slice" architecture where providers are isolated.

```text
giorgio/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings (Env vars, Secrets)
│   │   ├── security.py        # API Key validation for incoming client reqs
│   │   └── exceptions.py      # Unified error handling
│   ├── models/
│   │   ├── domain.py          # Internal normalized message formats
│   │   └── schemas.py         # Pydantic models for Req/Res (OpenAI compat)
│   ├── providers/
│   │   ├── base.py            # Abstract Base Class (LLMProvider)
│   │   ├── openai_adapter.py  # OpenAI Implementation
│   │   ├── vertex_claude.py   # Anthropic on Vertex Implementation
│   │   └── vertex_gemini.py   # Gemini on Vertex Implementation
│   ├── router/
│   │   ├── intelligence.py    # The Triage Logic / Router Brain
│   │   └── api.py             # FastAPI routes (POST /v1/chat/completions)
│   └── main.py                # App entrypoint
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 5. Implementation Details (The Adapter Pattern)

We define a strict protocol so the router doesn't care *who* answers.

### 5.1. Base Provider (Abstract)

```python
# app/providers/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Dict[str, Any] | AsyncGenerator[Any, None]:
        """
        Standard interface for all providers.
        Input: Standard list of messages [{'role': 'user', 'content': '...'}, ...]
        Output: Normalized OpenAI-compatible dict or async generator.
        """
        pass
```

### 5.2. Vertex Claude Adapter (Snippet)

```python
# app/providers/vertex_claude.py
from anthropic import AsyncAnthropicVertex
from app.providers.base import LLMProvider
from app.core.config import settings

class VertexClaudeProvider(LLMProvider):
    def __init__(self):
        # Vertex AI credentials are auto-loaded from environment (ADC)
        self.client = AsyncAnthropicVertex(
            project_id=settings.GCP_PROJECT_ID,
            region=settings.GCP_LOCATION,
        )

    async def chat_completion(self, messages: list, stream: bool = False):
        # 1. Transform Messages (OpenAI format -> Anthropic format)
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        anthropic_msgs = [m for m in messages if m['role'] != 'system']

        # 2. Call Vertex
        if stream:
            async with self.client.messages.stream(
                model="claude-3-5-sonnet-v2@20241022",
                max_tokens=1024,
                messages=anthropic_msgs,
                system=system_prompt,
            ) as stream_response:
                async for text in stream_response.text_stream:
                    # 3. Normalize Chunk to OpenAI Format
                    yield self._normalize_chunk(text)
        else:
            # Handle non-streaming...
            pass
    
    def _normalize_chunk(self, text: str):
        # Return OpenAI-compatible delta chunk
        return {
            "choices": [{"delta": {"content": text}}],
            "object": "chat.completion.chunk"
        }
```

---

## 6. Prototype Roadmap

### Phase 1: The Skeleton (Week 1)
*   **Setup:** Initialize FastAPI project with `uv` or `poetry`.
*   **Docker:** Create `Dockerfile` for Cloud Run (Python 3.11-slim).
*   **OpenAI Only:** Implement the `OpenAIAdapter` first as a baseline (easiest integration).
*   **Endpoint:** Expose `/v1/chat/completions` that proxies to OpenAI.

### Phase 2: The Vertex Integration (Week 2)
*   **Auth:** Configure `gcloud` local auth and Cloud Run Service Account.
*   **Gemini:** Add `VertexGeminiProvider` using `google-cloud-aiplatform`.
*   **Claude:** Add `VertexClaudeProvider` using `anthropic[vertex]`.
*   **Testing:** Verify all 3 providers work with manual switching.

### Phase 3: The Brain (Week 3)
*   **Router Logic:** Implement `app/router/intelligence.py`.
*   **Heuristics:** Add simple regex-based routing (e.g., prompt length > 8k tokens -> Gemini).
*   **Frontend:** Connect a simple Streamlit or Chainlit UI to test the "Unified" feel.

---

## 7. Dependencies (`pyproject.toml`)

```toml
[project]
name = "giorgio"
version = "0.1.0"
description = "Meta Chat Orchestrator"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic-settings>=2.1.0",
    "openai>=1.12.0",              # For OpenAI Adapter
    "anthropic[vertex]>=0.18.0",   # For Claude on Vertex
    "google-cloud-aiplatform>=1.40.0", # For Gemini on Vertex
    "httpx>=0.26.0",               # For async HTTP calls
    "tiktoken>=0.6.0",             # For token counting (routing logic)
]
```
