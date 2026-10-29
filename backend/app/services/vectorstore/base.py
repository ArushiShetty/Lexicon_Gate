"""Vector store abstraction."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class VectorMatch:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...

    @abc.abstractmethod
    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]: ...

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "count": self.count()}
