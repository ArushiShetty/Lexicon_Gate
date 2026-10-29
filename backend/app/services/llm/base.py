"""LLM provider abstraction.

Every generation step in the system goes through :class:`LLMProvider`. Requests
carry both a natural-language ``prompt`` (used by hosted models) and a
structured ``payload`` (used by the deterministic offline provider), so the same
call site works regardless of which provider is configured.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMTask(str, Enum):
    retrieval_decision = "retrieval_decision"
    answer = "answer"
    revise = "revise"
    rewrite_query = "rewrite_query"
    extract_claims = "extract_claims"
    extract_paper_structure = "extract_paper_structure"
    reflect = "reflect"
    manuscript_assist = "manuscript_assist"


@dataclass(slots=True)
class LLMRequest:
    task: LLMTask
    prompt: str = ""
    system: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = "unknown"
    model: str = "unknown"
    structured: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(abc.ABC):
    """Base class for all LLM providers."""

    name: str = "base"
    model: str = "unknown"
    #: ``True`` when the provider requires no network access and is reproducible.
    deterministic: bool = False

    @abc.abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run a single generation request."""

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover - overridden per provider
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "deterministic": self.deterministic,
        }


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token) for usage accounting."""

    if not text:
        return 0
    return max(1, len(text) // 4)
