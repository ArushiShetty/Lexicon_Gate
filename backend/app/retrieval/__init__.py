"""Retrieval package.

Keep this module lightweight. Eager imports of HybridRetriever pull in
KnowledgeBase and create a circular import with ``services.store``.
Import from submodules directly (e.g. ``retrieval.hybrid``, ``retrieval.bm25``).
"""

from __future__ import annotations

__all__ = ["FusionStrategy", "HybridRetriever", "RetrievalMode"]


def __getattr__(name: str):
    if name in ("FusionStrategy", "HybridRetriever", "RetrievalMode"):
        from .hybrid import FusionStrategy, HybridRetriever, RetrievalMode

        return {
            "FusionStrategy": FusionStrategy,
            "HybridRetriever": HybridRetriever,
            "RetrievalMode": RetrievalMode,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
