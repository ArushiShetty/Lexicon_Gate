"""OpenAI-compatible chat completion provider.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint over plain HTTP,
so no vendor SDK is required. Structured tasks ask for JSON; if the model
returns something unparseable we fall back to the deterministic engine rather
than propagating a malformed result into the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ...config import Settings, get_settings
from . import extractive_engine as engine
from .base import LLMProvider, LLMRequest, LLMResponse, LLMTask, estimate_tokens

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

STRUCTURED_TASKS = {
    LLMTask.extract_claims,
    LLMTask.extract_paper_structure,
    LLMTask.rewrite_query,
    LLMTask.reflect,
    LLMTask.retrieval_decision,
    LLMTask.manuscript_assist,
}


class OpenAIProvider(LLMProvider):
    name = "openai"
    deterministic = False

    def __init__(self, settings: Settings | None = None, timeout: float = 60.0) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model
        self._base_url = (
            self.settings.openai_base_url or "https://api.openai.com/v1"
        ).rstrip("/")
        self._timeout = timeout
        self._fallback = None

    @classmethod
    def is_available(cls) -> bool:
        return bool(get_settings().openai_api_key)

    # ------------------------------------------------------------------ core
    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = self._chat(request)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("OpenAI call failed (%s); using deterministic fallback", exc)
            return self._fallback_response(request, note=f"fallback: {exc.__class__.__name__}")

        text, usage = raw
        structured = self._parse_structured(request, text)
        if structured is None and request.task in STRUCTURED_TASKS:
            logger.warning("Unparseable structured response for %s; using fallback", request.task)
            return self._fallback_response(request, note="fallback: unparseable JSON")

        if request.task in (LLMTask.answer, LLMTask.revise):
            structured = {
                "answer": text.strip(),
                "citations": engine.cited_ids(text),
                "selected": [],
                "lead": "",
            }

        return LLMResponse(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", estimate_tokens(request.prompt)),
            completion_tokens=usage.get("completion_tokens", estimate_tokens(text)),
            provider=self.name,
            model=self.model,
            structured=structured,
        )

    def _chat(self, request: LLMRequest) -> tuple[str, dict[str, int]]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.settings.llm_temperature
            ),
            "max_tokens": request.max_tokens or self.settings.llm_max_tokens,
        }
        if request.task in STRUCTURED_TASKS:
            body["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        return text, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse_structured(request: LLMRequest, text: str) -> dict[str, Any] | None:
        if request.task not in STRUCTURED_TASKS:
            return None
        match = _JSON_BLOCK_RE.search(text or "")
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            parsed = {"items": parsed}
        if request.task is LLMTask.extract_claims:
            claims = parsed.get("claims") or parsed.get("items") or []
            normalised = []
            for claim in claims:
                if isinstance(claim, str):
                    normalised.append({"text": claim, "inline_citations": []})
                elif isinstance(claim, dict) and claim.get("text"):
                    normalised.append(
                        {
                            "text": str(claim["text"]),
                            "inline_citations": list(claim.get("inline_citations", [])),
                        }
                    )
            if not normalised:
                return None
            return {"claims": normalised}
        if request.task is LLMTask.extract_paper_structure:
            from ...models.documents import PAPER_STRUCTURE_FIELDS

            return {
                key: str(parsed.get(key) or "").strip()
                for key in PAPER_STRUCTURE_FIELDS
            }
        if request.task is LLMTask.rewrite_query:
            rewrites = parsed.get("rewrites") or parsed.get("items") or []
            rewrites = [str(r) for r in rewrites if str(r).strip()]
            return {"rewrites": rewrites} if rewrites else None
        return parsed

    def _fallback_response(self, request: LLMRequest, note: str) -> LLMResponse:
        if self._fallback is None:
            from .extractive_provider import ExtractiveProvider

            self._fallback = ExtractiveProvider()
        response = self._fallback.generate(request)
        response.provider = f"{self.name}+{response.provider}"
        if response.structured is not None:
            response.structured = {**response.structured, "provider_note": note}
        return response
