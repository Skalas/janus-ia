"""Vertex AI Gemini provider adapter."""

import asyncio
import logging
from typing import Any, AsyncGenerator

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.core.messages import message_content_to_text
from app.providers.base import LLMProvider
from app.providers.normalize import build_completion_response, build_stream_chunk


class VertexGeminiProvider(LLMProvider):
    """Gemini on Vertex AI via google-cloud-aiplatform."""

    def __init__(self) -> None:
        project_id = settings.GCP_PROJECT_ID
        if not project_id:
            raise ProviderError("GCP_PROJECT_ID not configured for Vertex Gemini")
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:
            raise ProviderError(
                "google-cloud-aiplatform not installed. Run: uv add google-cloud-aiplatform"
            ) from e
        vertexai.init(project=project_id, location=settings.GCP_LOCATION_GEMINI)
        self.GenerativeModel = GenerativeModel

    def _to_gemini_contents(
        self, messages: list[dict[str, str | list[Any]]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert to Gemini contents format (role + parts)."""
        system = ""
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            text = message_content_to_text(m.get("content", ""))
            if role == "system":
                system = text
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})
        return system, contents

    async def chat_completion(
        self,
        messages: list[dict[str, str | list[Any]]],
        stream: bool = False,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Call Gemini on Vertex, return normalized OpenAI-style response."""
        target_model = model or settings.VERTEX_GEMINI_MODEL
        system, contents = self._to_gemini_contents(messages)

        if stream:
            return self._stream(contents, system, target_model, temperature, max_tokens)
        return await self._complete(
            contents, system, target_model, temperature, max_tokens
        )

    async def _complete(
        self,
        contents: list[dict[str, Any]],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        def _sync() -> Any:
            kwargs: dict[str, Any] = {}
            if system:
                kwargs["system_instruction"] = system
            gen_model = self.GenerativeModel(model, **kwargs)
            gen_config: dict[str, Any] = {"max_output_tokens": max_tokens}
            if temperature is not None:
                gen_config["temperature"] = temperature
            return gen_model.generate_content(
                contents,
                generation_config=gen_config,
            )

        resp = await asyncio.to_thread(_sync)
        return self._normalize_response(resp, model)

    async def _stream(
        self,
        contents: list[dict[str, Any]],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce() -> None:
            kwargs: dict[str, Any] = {}
            if system:
                kwargs["system_instruction"] = system
            gen_model = self.GenerativeModel(model, **kwargs)
            gen_config: dict[str, Any] = {"max_output_tokens": max_tokens}
            if temperature is not None:
                gen_config["temperature"] = temperature
            for chunk in gen_model.generate_content(
                contents,
                generation_config=gen_config,
                stream=True,
            ):
                txt = self._extract_text(chunk)
                if txt:
                    loop.call_soon_threadsafe(queue.put_nowait, txt)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(_produce))
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield build_stream_chunk(chunk)

    def _part_text(self, p: Any) -> list[str]:
        """Extrae todo texto de un part (.text; thought es bool en v1, no objeto con .text)."""
        out = []
        t = getattr(p, "text", None)
        if t is not None and str(t).strip():
            out.append(str(t).strip())
        if hasattr(p, "ListFields"):
            for _fd, value in p.ListFields():
                if _fd.name == "text" and value and str(value).strip():
                    out.append(str(value).strip())
        return out

    def _extract_text_from_parts(self, parts: list) -> str:
        """De una lista de parts (GAPIC o wrapper), toma el último bloque con texto."""
        all_texts = []
        for p in parts:
            all_texts.extend(self._part_text(p))
        return all_texts[-1] if all_texts else ""

    def _extract_text(self, obj: Any) -> str:
        """Extrae texto de candidate/response (wrapper o GAPIC); Gemini 3 puede usar thinking+text."""
        try:
            if hasattr(obj, "text") and obj.text:
                return str(obj.text).strip()
        except (ValueError, AttributeError):
            pass
        # Ruta normal: obj.candidates[0].content.parts
        if hasattr(obj, "candidates") and obj.candidates:
            content = getattr(obj.candidates[0], "content", None)
            if content is not None:
                parts = getattr(content, "parts", []) or []
                if parts:
                    text = self._extract_text_from_parts(parts)
                    if text:
                        return text
        # SDK vertexai devuelve GenerationResponse con _raw_candidates (GAPIC); usar eso si lo normal falla
        raw = getattr(obj, "_raw_candidates", None) or getattr(obj, "_raw_response", None)
        if raw is not None:
            candidates = raw if isinstance(raw, list) else getattr(raw, "candidates", [])
            if candidates:
                c0 = candidates[0]
                content = getattr(c0, "content", None)
                if content is not None:
                    parts = getattr(content, "parts", []) or []
                    if parts:
                        text = self._extract_text_from_parts(parts)
                        if text:
                            return text
        # Último recurso: si el objeto tiene .candidates y _raw_*, puede que content sea el GAPIC
        if hasattr(obj, "candidates") and obj.candidates:
            c0 = obj.candidates[0]
            raw_content = getattr(c0, "_raw_content", None) or getattr(c0, "content", None)
            if raw_content is not None:
                parts = getattr(raw_content, "parts", []) or []
                if parts:
                    text = self._extract_text_from_parts(parts)
                    if text:
                        return text
        log = logging.getLogger(__name__)
        attrs = [a for a in dir(obj) if not a.startswith("__")]
        log.warning(
            "Gemini: sin texto; candidates=%s, attrs=%s",
            getattr(obj, "candidates", None) and len(obj.candidates),
            attrs[:30],
        )
        return ""

    def _normalize_response(self, resp: Any, model: str) -> dict[str, Any]:
        text = self._extract_text(resp)
        usage = None
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }
        return build_completion_response(
            model=model,
            message_content=text,
            usage=usage,
        )
