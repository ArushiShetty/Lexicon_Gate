"""Local model provider backed by an Ollama server."""

from __future__ import annotations

import logging

import httpx

from ...config import Settings, get_settings
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class LocalModelProvider(OpenAIProvider):
    """Ollama exposes an OpenAI-compatible endpoint, so we reuse that logic."""

    name = "ollama"
    deterministic = False

    def __init__(self, settings: Settings | None = None, timeout: float = 180.0) -> None:
        resolved = settings or get_settings()
        super().__init__(settings=resolved, timeout=timeout)
        self.model = resolved.ollama_model
        self._base_url = f"{resolved.ollama_base_url.rstrip('/')}/v1"

    @classmethod
    def is_available(cls) -> bool:
        settings = get_settings()
        try:
            with httpx.Client(timeout=1.5) as client:
                response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def _chat(self, request):  # type: ignore[override]
        # Ollama ignores the Authorization header but requires it to be absent
        # or arbitrary; reuse the parent implementation with a dummy key.
        if not self.settings.openai_api_key:
            self.settings.openai_api_key = "ollama"
        return super()._chat(request)
