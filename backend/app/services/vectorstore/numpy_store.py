"""Persisted in-process vector store.

Exact cosine search over an L2-normalised matrix. For research-scale corpora
this is faster and far more reproducible than an approximate index, and it has
zero external dependencies.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...config import Settings, get_settings
from .base import VectorMatch, VectorStore

logger = logging.getLogger(__name__)


class NumpyVectorStore(VectorStore):
    name = "numpy"

    def __init__(self, settings: Settings | None = None, namespace: str = "default") -> None:
        self.settings = settings or get_settings()
        self._dir = Path(self.settings.index_dir) / "vectors"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / f"{namespace}.npy"
        self._meta_path = self._dir / f"{namespace}.json"
        self._lock = threading.RLock()

        self._matrix: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        if not (self._vectors_path.exists() and self._meta_path.exists()):
            return
        try:
            matrix = np.load(self._vectors_path)
            raw = self._meta_path.read_text(encoding="utf-8").strip()
            if not raw:
                return
            payload = json.loads(raw)
            chunk_ids = list(payload.get("chunk_ids", []))
            metadatas = list(payload.get("metadatas", []))
            if matrix.shape[0] != len(chunk_ids):
                raise ValueError("vector/metadata length mismatch")
            self._matrix = matrix.astype(np.float32)
            self._chunk_ids = chunk_ids
            self._metadatas = metadatas
            self._reindex()
            logger.info("Loaded %s vectors from disk", len(chunk_ids))
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("Could not load vector index (%s); starting empty", exc)
            self._matrix = None
            self._chunk_ids = []
            self._metadatas = []
            self._index = {}

    def _persist(self) -> None:
        with self._lock:
            if self._matrix is None or not len(self._chunk_ids):
                self._vectors_path.unlink(missing_ok=True)
                self._meta_path.unlink(missing_ok=True)
                return
            np.save(self._vectors_path, self._matrix)
            tmp_meta = self._meta_path.with_suffix(".tmp")
            tmp_meta.write_text(
                json.dumps(
                    {"chunk_ids": self._chunk_ids, "metadatas": self._metadatas},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp_meta.replace(self._meta_path)

    def _reindex(self) -> None:
        self._index = {cid: i for i, cid in enumerate(self._chunk_ids)}

    # ---------------------------------------------------------------- writes
    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if not (len(chunk_ids) == vectors.shape[0] == len(metadatas)):
            raise ValueError("chunk_ids, vectors and metadatas must be the same length")

        with self._lock:
            if self._matrix is not None and self._matrix.shape[1] != vectors.shape[1]:
                logger.warning(
                    "Embedding dimension changed (%s -> %s); rebuilding index",
                    self._matrix.shape[1],
                    vectors.shape[1],
                )
                self._matrix = None
                self._chunk_ids = []
                self._metadatas = []
                self._index = {}

            new_rows: list[np.ndarray] = []
            for chunk_id, vector, metadata in zip(chunk_ids, vectors, metadatas):
                existing = self._index.get(chunk_id)
                if existing is not None and self._matrix is not None:
                    self._matrix[existing] = vector
                    self._metadatas[existing] = dict(metadata)
                    continue
                self._chunk_ids.append(chunk_id)
                self._metadatas.append(dict(metadata))
                new_rows.append(vector)

            if new_rows:
                stacked = np.vstack(new_rows).astype(np.float32)
                self._matrix = (
                    stacked if self._matrix is None else np.vstack([self._matrix, stacked])
                )
            self._reindex()
            self._persist()

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            if self._matrix is None:
                return 0
            keep = [
                i
                for i, meta in enumerate(self._metadatas)
                if meta.get("document_id") != document_id
            ]
            removed = len(self._chunk_ids) - len(keep)
            if removed == 0:
                return 0
            self._matrix = self._matrix[keep] if keep else None
            self._chunk_ids = [self._chunk_ids[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._reindex()
            self._persist()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._matrix = None
            self._chunk_ids = []
            self._metadatas = []
            self._index = {}
            self._persist()

    # --------------------------------------------------------------- queries
    def count(self) -> int:
        return len(self._chunk_ids)

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        with self._lock:
            if self._matrix is None or not len(self._chunk_ids) or top_k <= 0:
                return []

            query = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(query))
            if norm == 0:
                return []
            query = query / norm

            if self._matrix.shape[1] != query.shape[0]:
                logger.warning(
                    "Query dimension %s does not match index dimension %s",
                    query.shape[0],
                    self._matrix.shape[1],
                )
                return []

            scores = self._matrix @ query

            allowed: np.ndarray | None = None
            if document_ids:
                wanted = set(document_ids)
                allowed = np.array(
                    [meta.get("document_id") in wanted for meta in self._metadatas]
                )
                if not allowed.any():
                    return []
                scores = np.where(allowed, scores, -np.inf)

            limit = min(top_k, int((allowed.sum() if allowed is not None else len(scores))))
            if limit <= 0:
                return []
            order = np.argpartition(-scores, limit - 1)[:limit]
            order = order[np.argsort(-scores[order])]

            return [
                VectorMatch(
                    chunk_id=self._chunk_ids[i],
                    score=float(scores[i]),
                    metadata=self._metadatas[i],
                )
                for i in order
                if np.isfinite(scores[i])
            ]

    def all_metadatas(self) -> list[dict[str, Any]]:
        return list(self._metadatas)
