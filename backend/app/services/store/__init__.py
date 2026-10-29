"""Document / knowledge-base store package.

Avoid eager KnowledgeBase import here — importing ``document_store`` (e.g. for
``slugify``) must not load retrieval, which would circularly import this package.
"""

from __future__ import annotations

__all__ = ["KnowledgeBase", "get_knowledge_base", "reset_knowledge_base"]


def __getattr__(name: str):
    if name in __all__:
        from .knowledge_base import KnowledgeBase, get_knowledge_base, reset_knowledge_base

        return {
            "KnowledgeBase": KnowledgeBase,
            "get_knowledge_base": get_knowledge_base,
            "reset_knowledge_base": reset_knowledge_base,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
