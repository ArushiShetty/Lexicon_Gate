"""Embedding provider abstraction."""

from __future__ import annotations

import abc
from typing import Any, Sequence

import numpy as np


class EmbeddingProvider(abc.ABC):
    name: str = "base"
    model: str = "unknown"
    dimension: int = 384
    #: ``True`` when the provider must observe the corpus before encoding.
    requires_fit: bool = False

    @abc.abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into an L2-normalised ``(n, dimension)`` matrix."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def fit(self, texts: Sequence[str]) -> None:  # pragma: no cover - optional
        """Observe the corpus. No-op for providers that do not need it."""

    @property
    def is_fitted(self) -> bool:
        return True

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover - overridden
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "dimension": self.dimension,
            "requires_fit": self.requires_fit,
            "is_fitted": self.is_fitted,
        }


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
