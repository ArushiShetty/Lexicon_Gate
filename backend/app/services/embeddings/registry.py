"""Embedding provider resolution."""

from __future__ import annotations

import logging

from ...config import get_settings
from .base import EmbeddingProvider
from .lsa_provider import LsaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .sentence_transformer_provider import SentenceTransformerProvider

logger = logging.getLogger(__name__)

_PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,
    "sentence_transformer": SentenceTransformerProvider,
    "lsa": LsaEmbeddingProvider,
}

_AUTO_ORDER = ("sentence_transformer", "openai", "lsa")

_cache: dict[str, EmbeddingProvider] = {}


def available_embedding_providers() -> dict[str, bool]:
    return {name: cls.is_available() for name, cls in _PROVIDERS.items()}


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    requested = (name or get_settings().embedding_provider or "auto").lower()

    if requested == "auto":
        for candidate in _AUTO_ORDER:
            if _PROVIDERS[candidate].is_available():
                requested = candidate
                break
        else:  # pragma: no cover - lsa is always available
            requested = "lsa"

    if requested not in _PROVIDERS:
        logger.warning("Unknown embedding provider %r; falling back to lsa", requested)
        requested = "lsa"

    if requested not in _cache:
        cls = _PROVIDERS[requested]
        if not cls.is_available():
            logger.warning("Embedding provider %r unavailable; using lsa", requested)
            requested = "lsa"
            cls = _PROVIDERS[requested]
        try:
            _cache[requested] = cls()
        except Exception as exc:  # pragma: no cover - model download failures
            logger.warning("Failed to init embedding provider %r (%s); using lsa", requested, exc)
            requested = "lsa"
            _cache[requested] = LsaEmbeddingProvider()
        logger.info("Using embedding provider: %s", _cache[requested].describe())

    return _cache[requested]


def reset_embedding_cache() -> None:
    _cache.clear()
