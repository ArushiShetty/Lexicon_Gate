"""Persistent document and chunk metadata store.

JSON-backed on purpose: the corpora used in this project are research-scale, and
a plain-text store keeps the indexed data auditable, which matters when you need
to explain exactly which passage a citation points at.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Iterable, Sequence

from ...config import Settings, get_settings
from ...models.documents import Chunk, ChunkMetadata, Document, PaperStructure
from ..citations import title_from_filename
from ..ingestion.chunker import chunk_page
from ..ingestion.cleaner import clean_page, strip_repeated_lines
from ..ingestion.loaders import LoadedPage

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", (value or "").lower()).strip("-")
    return (slug[:max_len].strip("-")) or "doc"


class DocumentStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dir = Path(self.settings.index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._documents_path = self._dir / "documents.json"
        self._chunks_path = self._dir / "chunks.json"
        self._extractions_path = self._dir / "paper_extractions.json"
        self._lock = threading.RLock()
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._chunks_by_document: dict[str, list[str]] = {}
        self._extractions: dict[str, PaperStructure] = {}
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            if self._documents_path.exists():
                raw = self._documents_path.read_text(encoding="utf-8").strip()
                if raw:
                    payload = json.loads(raw)
                    self._documents = {}
                    for item in payload:
                        doc = Document.model_validate(item)
                        # If document has no user_id (legacy upload), assign to Arushi Shetty
                        if not doc.user_id:
                            doc.user_id = "92a44eb862cc44ca99402d440f736800"
                        self._documents[doc.document_id] = doc
            if self._chunks_path.exists():
                raw = self._chunks_path.read_text(encoding="utf-8").strip()
                if raw:
                    payload = json.loads(raw)
                    self._chunks = {
                        item["chunk_id"]: Chunk.model_validate(item) for item in payload
                    }
            if self._extractions_path.exists():
                raw = self._extractions_path.read_text(encoding="utf-8").strip()
                if raw:
                    payload = json.loads(raw)
                    self._extractions = {
                        item["document_id"]: PaperStructure.model_validate(item)
                        for item in payload
                    }
            self._rebuild_document_index()
            if self._documents:
                logger.info(
                    "Loaded %s documents / %s chunks from disk",
                    len(self._documents),
                    len(self._chunks),
                )
        except Exception as exc:  # pragma: no cover - corrupt state
            logger.warning("Could not load document store (%s); starting empty", exc)
            self._documents = {}
            self._chunks = {}
            self._chunks_by_document = {}
            self._extractions = {}

    def _persist(self) -> None:
        def _safe_write(path: Path, data: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(data, encoding="utf-8")
            tmp_path.replace(path)

        _safe_write(
            self._documents_path,
            json.dumps([d.model_dump(mode="json") for d in self._documents.values()],
                       ensure_ascii=False, indent=2),
        )
        _safe_write(
            self._chunks_path,
            json.dumps([c.model_dump(mode="json") for c in self._chunks.values()],
                       ensure_ascii=False, indent=2),
        )
        _safe_write(
            self._extractions_path,
            json.dumps(
                [e.model_dump(mode="json") for e in self._extractions.values()],
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _rebuild_document_index(self) -> None:
        self._chunks_by_document = {}
        for chunk in self._chunks.values():
            self._chunks_by_document.setdefault(chunk.metadata.document_id, []).append(
                chunk.chunk_id
            )

    # -------------------------------------------------------------- mutation
    def add_document(
        self,
        name: str,
        pages: Sequence[LoadedPage],
        source: str = "upload",
        media_type: str = "text/plain",
        source_quality: float = 0.6,
        tags: Iterable[str] | None = None,
        document_id: str | None = None,
        title: str | None = None,
        authors: Iterable[str] | None = None,
        year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
        user_id: str | None = None,
    ) -> tuple[Document, list[Chunk]]:
        """Clean, chunk and register a document. Returns the doc and its chunks."""

        with self._lock:
            cleaned_pages = strip_repeated_lines([clean_page(p.text) for p in pages])
            page_numbers = [p.page for p in pages]

            doc_id = document_id or f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
            if doc_id in self._documents:
                self.delete_document(doc_id)

            chunks: list[Chunk] = []
            running_index = 0
            total_chars = 0

            for page_number, page_text in zip(page_numbers, cleaned_pages):
                if not page_text.strip():
                    continue
                total_chars += len(page_text)
                page_chunks = chunk_page(
                    text=page_text,
                    page=page_number,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    min_chunk_chars=self.settings.min_chunk_chars,
                    start_index=running_index,
                )
                running_index += len(page_chunks)
                for piece in page_chunks:
                    chunk_id = f"{doc_id}::p{piece.page}::c{piece.chunk_index}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece.text,
                            metadata=ChunkMetadata(
                                document_id=doc_id,
                                document_name=name,
                                page=piece.page,
                                section=piece.section,
                                chunk_index=piece.chunk_index,
                                source=source,
                                source_quality=source_quality,
                                extra=piece.extra,
                            ),
                        )
                    )

            document = Document(
                document_id=doc_id,
                name=name,
                source=source,
                media_type=media_type,
                n_pages=len({c.metadata.page for c in chunks}) or len(pages),
                n_chunks=len(chunks),
                n_characters=total_chars,
                source_quality=source_quality,
                tags=list(tags or []),
                title=(title or "").strip() or title_from_filename(name),
                authors=[a.strip() for a in (authors or []) if str(a).strip()],
                year=year,
                venue=(venue.strip() if isinstance(venue, str) and venue.strip() else None),
                doi=(doi.strip() if isinstance(doi, str) and doi.strip() else None),
                user_id=(user_id or "").strip(),
            )

            self._documents[doc_id] = document
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
            self._chunks_by_document[doc_id] = [c.chunk_id for c in chunks]
            self._persist()
            logger.info("Indexed document %s (%s chunks)", name, len(chunks))
            return document, chunks

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            for chunk_id in self._chunks_by_document.pop(document_id, []):
                self._chunks.pop(chunk_id, None)
            self._documents.pop(document_id, None)
            self._extractions.pop(document_id, None)
            self._persist()
            return True

    def update_bibliography(self, document_id: str, **fields: object) -> Document | None:
        allowed = {"title", "authors", "year", "venue", "doi"}
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                return None
            payload = {key: value for key, value in fields.items() if key in allowed}
            if "title" in payload:
                title = str(payload["title"] or "").strip()
                payload["title"] = title or document.title or title_from_filename(document.name)
            if "authors" in payload:
                raw = payload["authors"] or []
                payload["authors"] = [str(a).strip() for a in raw if str(a).strip()]
            if "venue" in payload:
                venue = str(payload["venue"] or "").strip()
                payload["venue"] = venue or None
            if "doi" in payload:
                doi = str(payload["doi"] or "").strip()
                payload["doi"] = doi or None
            if "year" in payload and payload["year"] not in (None, ""):
                payload["year"] = int(payload["year"])
            elif "year" in payload:
                payload["year"] = None
            updated = document.model_copy(update=payload)
            self._documents[document_id] = updated
            self._persist()
            return updated

    def clear(self) -> None:
        with self._lock:
            self._documents = {}
            self._chunks = {}
            self._chunks_by_document = {}
            self._extractions = {}
            self._persist()

    # --------------------------------------------------------------- reading
    def list_documents(self, user_id: str | None = None) -> list[Document]:
        docs = sorted(self._documents.values(), key=lambda d: d.created_at)
        if user_id:
            uid = str(user_id).strip()
            # If user_id is provided, only return documents matching this user_id
            return [d for d in docs if d.user_id == uid]
        return docs

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def get_document_by_name(self, name: str) -> Document | None:
        for document in self._documents.values():
            if document.name == name:
                return document
        return None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def all_chunks(self, document_ids: Sequence[str] | None = None) -> list[Chunk]:
        if not document_ids:
            return list(self._chunks.values())
        wanted = set(document_ids)
        return [c for c in self._chunks.values() if c.metadata.document_id in wanted]

    def chunk_count(self) -> int:
        return len(self._chunks)

    def document_count(self) -> int:
        return len(self._documents)

    def upsert_extraction(self, record: PaperStructure) -> PaperStructure:
        with self._lock:
            self._extractions[record.document_id] = record
            self._persist()
            return record

    def get_extraction(self, document_id: str) -> PaperStructure | None:
        return self._extractions.get(document_id)

    def list_extractions(self) -> list[PaperStructure]:
        docs = {d.document_id: d for d in self._documents.values()}
        rows = []
        for record in self._extractions.values():
            if record.document_id not in docs:
                continue
            rows.append(record)
        rows.sort(key=lambda r: docs[r.document_id].created_at)
        return rows
