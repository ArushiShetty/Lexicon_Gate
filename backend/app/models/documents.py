"""Document, chunk and ingestion schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    source: str = "upload"
    source_quality: float = 0.6
    created_at: datetime = Field(default_factory=_utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    @property
    def citation_label(self) -> str:
        page = self.metadata.page
        if page is None:
            return self.metadata.document_name
        return f"{self.metadata.document_name} — page {page}"


class Document(BaseModel):
    document_id: str
    name: str
    source: str = "upload"
    media_type: str = "text/plain"
    n_pages: int = 1
    n_chunks: int = 0
    n_characters: int = 0
    source_quality: float = 0.6
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None


class DocumentBiblioUpdate(BaseModel):
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None


class UploadResponse(BaseModel):
    documents: list[Document]
    total_chunks: int
    warnings: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[Document]
    total_documents: int
    total_chunks: int
    embedding_provider: str
    vector_store: str


class ThemeCluster(BaseModel):
    cluster_id: str
    label: str
    keywords: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    size: int = 0


class ClusterResponse(BaseModel):
    clusters: list[ThemeCluster] = Field(default_factory=list)
    total_documents: int = 0


PAPER_STRUCTURE_FIELDS: tuple[str, ...] = (
    "objective",
    "method",
    "dataset",
    "metric",
    "result",
    "limitation",
)


class StructuredField(BaseModel):
    """One extracted paper attribute, kept even when validation is weak."""

    value: str = ""
    low_confidence: bool = False
    nli_label: str | None = None
    nli_confidence: float | None = None
    source_chunk_id: str | None = None


class PaperStructure(BaseModel):
    """Per-paper structured record, independent of any user question."""

    document_id: str
    document_name: str
    title: str = ""
    fields: dict[str, StructuredField] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=_utcnow)
    extractor: str = ""


class PaperStructureListResponse(BaseModel):
    papers: list[PaperStructure] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=lambda: list(PAPER_STRUCTURE_FIELDS))
    total: int = 0
