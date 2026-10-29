"""LLM provider resolution.

``SELFRAG_LLM_PROVIDER=auto`` picks the best provider that is actually usable on
this machine, preferring hosted quality but always degrading gracefully to the
deterministic offline provider so the system never hard-fails.
"""

from __future__ import annotations

import logging
from typing import Callable

from ...config import get_settings
from .base import LLMProvider
from .extractive_provider import ExtractiveProvider
from .ollama_provider import LocalModelProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    "openai": OpenAIProvider,
    "ollama": LocalModelProvider,
    "local": LocalModelProvider,
    "extractive": ExtractiveProvider,
}

_AUTO_ORDER = ("openai", "ollama", "extractive")

_cache: dict[str, LLMProvider] = {}


def available_llm_providers() -> dict[str, bool]:
    return {name: factory.is_available() for name, factory in _PROVIDERS.items()}  # type: ignore[attr-defined]


def get_llm_provider(name: str | None = None) -> LLMProvider:
    requested = (name or get_settings().llm_provider or "auto").lower()

    if requested == "auto":
        for candidate in _AUTO_ORDER:
            factory = _PROVIDERS[candidate]
            if factory.is_available():  # type: ignore[attr-defined]
                requested = candidate
                break
        else:  # pragma: no cover - extractive is always available
            requested = "extractive"

    if requested not in _PROVIDERS:
        logger.warning("Unknown LLM provider %r; falling back to extractive", requested)
        requested = "extractive"

    if requested not in _cache:
        factory = _PROVIDERS[requested]
        if not factory.is_available():  # type: ignore[attr-defined]
            logger.warning(
                "LLM provider %r is not available; falling back to extractive", requested
            )
            requested = "extractive"
            factory = _PROVIDERS[requested]
        _cache[requested] = factory()
        logger.info("Using LLM provider: %s (%s)", requested, _cache[requested].model)

    return _cache[requested]


def reset_llm_cache() -> None:
    _cache.clear()
