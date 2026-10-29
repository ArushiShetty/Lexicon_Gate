"""Vector store resolution."""

from __future__ import annotations

import logging

from ...config import get_settings
from .base import VectorStore
from .chroma_store import ChromaVectorStore
from .numpy_store import NumpyVectorStore
from .pinecone_store import PineconeVectorStore

logger = logging.getLogger(__name__)

_STORES = {
    "numpy": NumpyVectorStore,
    "chroma": ChromaVectorStore,
    "pinecone": PineconeVectorStore,
}

_cache: dict[str, VectorStore] = {}


def available_vector_stores() -> dict[str, bool]:
    return {
        name: getattr(cls, "is_available", lambda: True)()  # type: ignore[misc]
        for name, cls in _STORES.items()
    }


def get_vector_store(name: str | None = None, dimension: int | None = None) -> VectorStore:
    requested = (name or get_settings().vector_store or "numpy").lower()
    if requested not in _STORES:
        logger.warning("Unknown vector store %r; using numpy", requested)
        requested = "numpy"

    if requested not in _cache:
        cls = _STORES[requested]
        available = getattr(cls, "is_available", lambda: True)()
        if not available:
            logger.warning("Vector store %r unavailable; using numpy", requested)
            requested = "numpy"
            cls = _STORES[requested]
        try:
            _cache[requested] = (
                cls(dimension=dimension) if requested == "pinecone" else cls()
            )
        except Exception as exc:  # pragma: no cover - external service failure
            logger.warning("Failed to init vector store %r (%s); using numpy", requested, exc)
            requested = "numpy"
            _cache[requested] = NumpyVectorStore()
        logger.info("Using vector store: %s", requested)

    return _cache[requested]


def reset_vector_store_cache() -> None:
    _cache.clear()
