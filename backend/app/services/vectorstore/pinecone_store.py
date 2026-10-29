"""Pinecone-backed vector store (optional dependency).

Reuses ``SELFRAG_PINECONE_API_KEY`` / ``SELFRAG_PINECONE_INDEX`` if they are
already configured in the environment.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Sequence

import numpy as np

from ...config import Settings, get_settings
from .base import VectorMatch, VectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStore):
    name = "pinecone"

    def __init__(self, settings: Settings | None = None, dimension: int | None = None) -> None:
        from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415

        self.settings = settings or get_settings()
        self._client = Pinecone(api_key=self.settings.pinecone_api_key)
        index_name = self.settings.pinecone_index
        existing = {idx["name"] for idx in self._client.list_indexes()}
        if index_name not in existing:
            self._client.create_index(
                name=index_name,
                dimension=int(dimension or self.settings.embedding_dim),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Created Pinecone index %s", index_name)
        self._index = self._client.Index(index_name)

    @classmethod
    def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("pinecone") is not None
            and bool(get_settings().pinecone_api_key)
        )

    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not len(chunk_ids):
            return
        payload = [
            {"id": cid, "values": np.asarray(vec, dtype=np.float32).tolist(), "metadata": meta}
            for cid, vec, meta in zip(chunk_ids, vectors, metadatas)
        ]
        for start in range(0, len(payload), 100):
            self._index.upsert(vectors=payload[start : start + 100])

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        query_filter = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        result = self._index.query(
            vector=np.asarray(vector, dtype=np.float32).reshape(-1).tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=query_filter,
        )
        raw_matches = getattr(result, "matches", None)
        if raw_matches is None and isinstance(result, dict):
            raw_matches = result.get("matches", [])
        matches = []
        for match in raw_matches or []:
            if hasattr(match, "id"):
                chunk_id = str(match.id)
                score = float(getattr(match, "score", 0.0) or 0.0)
                metadata = dict(getattr(match, "metadata", None) or {})
            else:
                chunk_id = str(match["id"])
                score = float(match.get("score", 0.0) or 0.0)
                metadata = dict(match.get("metadata") or {})
            matches.append(
                VectorMatch(chunk_id=chunk_id, score=score, metadata=metadata)
            )
        return matches

    def delete_document(self, document_id: str) -> int:
        self._index.delete(filter={"document_id": document_id})
        return -1  # Pinecone does not report deleted counts synchronously.

    def clear(self) -> None:
        self._index.delete(delete_all=True)

    def count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            return int(stats.get("total_vector_count", 0))
        except Exception:  # pragma: no cover - network dependent
            return 0
