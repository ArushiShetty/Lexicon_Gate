"""Public academic + web paper search (not Google Scholar scraping)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PaperSource = Literal["semantic_scholar", "openalex", "crossref", "arxiv", "tavily"]
PaperSearchFilter = Literal["all", "academic", "research_web", "open_access"]
ResultKind = Literal["paper", "web"]


class PaperHit(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citation_count: int = 0
    abstract: str | None = None
    summary: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    source: PaperSource
    open_access: bool = False
    url: str | None = None
    result_kind: ResultKind = "paper"
    full_text_available: bool = False

    @model_validator(mode="after")
    def _set_full_text_flag(self) -> "PaperHit":
        if not self.full_text_available:
            self.full_text_available = bool(self.pdf_url) or bool(self.open_access)
        if self.result_kind == "web" and self.summary and not self.abstract:
            self.abstract = self.summary
        return self


class PaperSearchResponse(BaseModel):
    query: str
    provider: str
    filter: PaperSearchFilter = "all"
    providers_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    papers: list[PaperHit]


class PaperImportRequest(BaseModel):
    paper_id: str
    source: PaperSource
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    summary: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    url: str | None = None
    result_kind: ResultKind | None = None


class PaperImportResponse(BaseModel):
    document: dict
    ingested: str
    warnings: list[str] = Field(default_factory=list)
    paper_url: str | None = None
    pdf_url: str | None = None
