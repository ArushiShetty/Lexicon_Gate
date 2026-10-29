"""The knowledge base: documents, embeddings, vectors and the BM25 index.

Single seam that the retrieval layer talks to. It owns index consistency: when
the corpus changes, embeddings and the lexical index are rebuilt together so a
query can never see a half-updated view.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, Sequence

from ...config import Settings, get_settings
from ...text_utils import content_tokens, stem, term_bag
from ...models.documents import Chunk, Document
from ...retrieval.bm25 import BM25Hit, BM25Index
from ..embeddings.base import EmbeddingProvider
from ..embeddings.registry import get_embedding_provider
from ..ingestion.loaders import LoadedPage, extract_bibliography, load_bytes, load_document
from ..vectorstore.base import VectorMatch, VectorStore
from ..vectorstore.registry import get_vector_store
from .document_store import DocumentStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        settings: Settings | None = None,
        store: DocumentStore | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or DocumentStore(self.settings)
        self.embedder = embedder or get_embedding_provider()
        self.vectors = vector_store or get_vector_store(dimension=self.embedder.dimension)
        self._bm25_path = Path(self.settings.index_dir) / "bm25.pkl"
        self._bm25 = BM25Index.load(self._bm25_path) or BM25Index()
        self._doc_terms: dict[str, set[str]] = {}
        self._lock = threading.RLock()

        if self._bm25.size == 0 and self.store.chunk_count() > 0:
            logger.info("Lexical index missing; rebuilding from stored chunks")
            self.rebuild_indexes()
        else:
            self._doc_terms = self._build_doc_terms(self.store.all_chunks())

    # ------------------------------------------------------------- ingestion
    def ingest_path(
        self,
        path: Path | str,
        name: str | None = None,
        source: str = "upload",
        source_quality: float = 0.6,
        tags: Iterable[str] | None = None,
        rebuild: bool = True,
        title: str | None = None,
        authors: Iterable[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
    ) -> tuple[Document, list[Chunk]]:
        path = Path(path)
        pages = load_document(path)
        biblio = extract_bibliography(name or path.name, path=path)
        return self.ingest_pages(
            name=name or path.name,
            pages=pages,
            source=source,
            media_type=_media_type(path.suffix),
            source_quality=source_quality,
            tags=tags,
            rebuild=rebuild,
            title=title or biblio.title,
            authors=list(authors) if authors is not None else biblio.authors,
            year=year if year is not None else biblio.year,
            venue=venue if venue is not None else biblio.venue,
            doi=doi if doi is not None else biblio.doi,
        )

    def ingest_bytes(
        self,
        name: str,
        data: bytes,
        source: str = "upload",
        source_quality: float = 0.6,
        tags: Iterable[str] | None = None,
        rebuild: bool = True,
        title: str | None = None,
        authors: Iterable[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
    ) -> tuple[Document, list[Chunk]]:
        pages = load_bytes(name, data)
        biblio = extract_bibliography(name, data=data)
        return self.ingest_pages(
            name=name,
            pages=pages,
            source=source,
            media_type=_media_type(Path(name).suffix),
            source_quality=source_quality,
            tags=tags,
            rebuild=rebuild,
            title=title or biblio.title,
            authors=list(authors) if authors is not None else biblio.authors,
            year=year if year is not None else biblio.year,
            venue=venue if venue is not None else biblio.venue,
            doi=doi if doi is not None else biblio.doi,
        )

    def ingest_pages(
        self,
        name: str,
        pages: Sequence[LoadedPage],
        source: str = "upload",
        media_type: str = "text/plain",
        source_quality: float = 0.6,
        tags: Iterable[str] | None = None,
        rebuild: bool = True,
        title: str | None = None,
        authors: Iterable[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
    ) -> tuple[Document, list[Chunk]]:
        with self._lock:
            document, chunks = self.store.add_document(
                name=name,
                pages=pages,
                source=source,
                media_type=media_type,
                source_quality=source_quality,
                tags=tags,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
            )
            from ..extraction.paper_structure import extract_and_store

            extract_and_store(self.store, document, chunks)
            if rebuild:
                self.rebuild_indexes()
            return document, chunks

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            removed = self.store.delete_document(document_id)
            if removed:
                self.vectors.delete_document(document_id)
                self.rebuild_indexes()
            return removed

    def update_bibliography(self, document_id: str, **fields: object) -> Document | None:
        with self._lock:
            return self.store.update_bibliography(document_id, **fields)

    def reset(self) -> None:
        with self._lock:
            self.store.clear()
            self.vectors.clear()
            self._bm25 = BM25Index()
            self._bm25_path.unlink(missing_ok=True)
            self._doc_terms = {}

    # ----------------------------------------------------------- index build
    def rebuild_indexes(self) -> None:
        """Refit embeddings and rebuild both indexes from the stored chunks."""

        with self._lock:
            chunks = self.store.all_chunks()
            if not chunks:
                self.vectors.clear()
                self._bm25 = BM25Index()
                self._bm25_path.unlink(missing_ok=True)
                self._doc_terms = {}
                return

            texts = [_embedding_text(chunk) for chunk in chunks]

            if self.embedder.requires_fit:
                # Corpus-dependent embedders must see the whole corpus, so every
                # chunk is re-encoded after a refit.
                self.embedder.fit(texts)

            vectors = self.embedder.encode(texts)
            self.vectors.clear()
            self.vectors.upsert(
                chunk_ids=[c.chunk_id for c in chunks],
                vectors=vectors,
                metadatas=[_vector_metadata(c) for c in chunks],
            )

            self._bm25 = BM25Index()
            self._bm25.build(
                [(c.chunk_id, c.metadata.document_id, _embedding_text(c)) for c in chunks]
            )
            self._bm25.save(self._bm25_path)
            self._doc_terms = self._build_doc_terms(chunks)
            logger.info(
                "Indexes rebuilt: %s chunks, %s vectors, bm25 size %s",
                len(chunks),
                self.vectors.count(),
                self._bm25.size,
            )

    # ---------------------------------------------------------------- search
    def dense_search(
        self,
        query: str,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        if not query.strip() or self.vectors.count() == 0:
            return []
        vector = self.embedder.encode_one(query)
        return self.vectors.search(vector, top_k=top_k, document_ids=document_ids)

    def lexical_search(
        self,
        query: str,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[BM25Hit]:
        return self._bm25.search(query, top_k=top_k, document_ids=document_ids)

    # ------------------------------------------------------------- accessors
    @property
    def bm25(self) -> BM25Index:
        return self._bm25

    def idf_map(self) -> dict[str, float]:
        return self._bm25.term_idf_map()

    def document_term_sets(self) -> dict[str, set[str]]:
        return self._doc_terms

    @staticmethod
    def _build_doc_terms(chunks: Sequence[Chunk]) -> dict[str, set[str]]:
        terms: dict[str, set[str]] = {}
        for chunk in chunks:
            terms.setdefault(chunk.metadata.document_id, set()).update(
                term_bag(f"{chunk.metadata.document_name} {chunk.metadata.section or ''} {chunk.text}")
            )
        return terms

    def corpus_terms(self, limit: int = 60) -> list[str]:
        return self._bm25.vocabulary(limit=limit)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.store.get_chunk(chunk_id)

    def stats(self) -> dict[str, object]:
        return {
            "documents": self.store.document_count(),
            "chunks": self.store.chunk_count(),
            "vectors": self.vectors.count(),
            "bm25_size": self._bm25.size,
            "embedding_provider": self.embedder.describe(),
            "vector_store": self.vectors.name,
        }

    def is_empty(self) -> bool:
        return self.store.chunk_count() == 0


def _embedding_text(chunk: Chunk) -> str:
    """Prefix the section heading so headings inform both indexes."""

    section = chunk.metadata.section
    return f"{section}. {chunk.text}" if section else chunk.text


def _vector_metadata(chunk: Chunk) -> dict[str, object]:
    meta = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": meta.document_id,
        "document_name": meta.document_name,
        "page": meta.page,
        "section": meta.section,
        "chunk_index": meta.chunk_index,
        "source": meta.source,
        "source_quality": meta.source_quality,
    }


def _media_type(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".csv": "text/csv",
    }.get(suffix.lower(), "text/plain")


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    """Process-wide knowledge base singleton."""

    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase()
    return _kb


def reset_knowledge_base() -> None:
    global _kb
    with _kb_lock:
        _kb = None
