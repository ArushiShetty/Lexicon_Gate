"""Chroma-backed vector store (optional dependency)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Sequence

import numpy as np

from ...config import Settings, get_settings
from .base import VectorMatch, VectorStore

logger = logging.getLogger(__name__)

COLLECTION = "enhanced_self_rag"


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb  # noqa: PLC0415

        self.settings = settings or get_settings()
        path = str(self.settings.chroma_path)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Chroma store ready at %s", path)

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("chromadb") is not None

    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not len(chunk_ids):
            return
        self._collection.upsert(
            ids=list(chunk_ids),
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            metadatas=[_flatten(m) for m in metadatas],
        )

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0 or self.count() == 0:
            return []
        where = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).reshape(-1).tolist()],
            n_results=min(top_k, max(1, self.count())),
            where=where,
            include=["metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        return [
            VectorMatch(chunk_id=cid, score=1.0 - float(dist), metadata=dict(meta or {}))
            for cid, dist, meta in zip(ids, distances, metadatas)
        ]

    def delete_document(self, document_id: str) -> int:
        before = self.count()
        self._collection.delete(where={"document_id": document_id})
        return max(0, before - self.count())

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return int(self._collection.count())


def _flatten(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values."""

    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out
