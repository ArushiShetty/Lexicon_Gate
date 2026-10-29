"""Retrieval strategies: dense-only, lexical-only and hybrid with RRF.

All three share one interface so the evaluation can swap the strategy while
holding everything else fixed — that is what makes the "pure vector vs hybrid"
row of the ablation table an actual controlled comparison.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Sequence

from ..models.query import EvidenceItem
from ..services.store.knowledge_base import KnowledgeBase
from .fusion import RankedList, reciprocal_rank_fusion, weighted_score_fusion

logger = logging.getLogger(__name__)


def is_bibliography_chunk(text: str, section: str | None = None) -> bool:
    if section:
        s_lower = section.lower()
        if any(w in s_lower for w in ("reference", "bibliography", "works cited", "literature cited")):
            return True
    lines = [line.strip() for line in (text or "").split("\n") if line.strip()]
    if not lines:
        return False
    cite_patterns = sum(
        1 for line in lines
        if re.match(r"^\[\d+\]\s+", line)
        or ("pp." in line and re.search(r"\b(19\d\d|20\d\d)\b", line))
        or "doi:10." in line.lower()
    )
    return (cite_patterns / len(lines)) >= 0.4


class RetrievalMode(str, Enum):
    dense = "dense"
    bm25 = "bm25"
    hybrid = "hybrid"


class FusionStrategy(str, Enum):
    rrf = "rrf"
    weighted = "weighted"


class HybridRetriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        mode: RetrievalMode = RetrievalMode.hybrid,
        fusion: FusionStrategy = FusionStrategy.rrf,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
        rrf_k: int | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        self.kb = kb
        self.mode = mode
        self.fusion = fusion
        settings = kb.settings
        self.dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)

    # ------------------------------------------------------------------ main
    def retrieve(
        self,
        query: str,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[EvidenceItem]:
        if top_k <= 0 or self.kb.is_empty():
            return []

        pool = top_k * self.candidate_multiplier
        dense_hits = (
            self.kb.dense_search(query, top_k=pool, document_ids=document_ids)
            if self.mode in (RetrievalMode.dense, RetrievalMode.hybrid)
            else []
        )
        lexical_hits = (
            self.kb.lexical_search(query, top_k=pool, document_ids=document_ids)
            if self.mode in (RetrievalMode.bm25, RetrievalMode.hybrid)
            else []
        )

        dense_scores = {hit.chunk_id: float(hit.score) for hit in dense_hits}
        bm25_scores = {hit.chunk_id: float(hit.score) for hit in lexical_hits}
        matched_terms = {hit.chunk_id: hit.matched_terms for hit in lexical_hits}

        if self.mode is RetrievalMode.dense:
            ordered = [(hit.chunk_id, float(hit.score)) for hit in dense_hits][:top_k]
            fused = {cid: score for cid, score in ordered}
            ranking = [cid for cid, _ in ordered]
        elif self.mode is RetrievalMode.bm25:
            ordered = [(hit.chunk_id, float(hit.score)) for hit in lexical_hits][:top_k]
            fused = {cid: score for cid, score in ordered}
            ranking = [cid for cid, _ in ordered]
        else:
            lists = [
                RankedList(
                    name="dense",
                    chunk_ids=[h.chunk_id for h in dense_hits],
                    scores=dense_scores,
                    weight=self.dense_weight,
                ),
                RankedList(
                    name="bm25",
                    chunk_ids=[h.chunk_id for h in lexical_hits],
                    scores=bm25_scores,
                    weight=self.bm25_weight,
                ),
            ]
            fuse = (
                reciprocal_rank_fusion
                if self.fusion is FusionStrategy.rrf
                else weighted_score_fusion
            )
            kwargs = {"k": self.rrf_k} if self.fusion is FusionStrategy.rrf else {}
            results = fuse(lists, top_k=top_k, **kwargs)  # type: ignore[arg-type]
            fused = {r.chunk_id: r.fused_score for r in results}
            ranking = [r.chunk_id for r in results]

        items: list[EvidenceItem] = []
        bib_query = any(w in query.lower() for w in ("reference", "bibliography", "citation", "cited by"))
        for rank, chunk_id in enumerate(ranking, start=1):
            chunk = self.kb.get_chunk(chunk_id)
            if chunk is None:
                continue
            # Exclude bibliography entries from acting as evidence for research claims
            if not bib_query and is_bibliography_chunk(chunk.text, chunk.metadata.section):
                continue
            document = self.kb.store.get_document(chunk.metadata.document_id)
            items.append(
                EvidenceItem(
                    citation_id=0,
                    chunk_id=chunk_id,
                    document_id=chunk.metadata.document_id,
                    document_name=chunk.metadata.document_name,
                    page=chunk.metadata.page,
                    section=chunk.metadata.section,
                    text=chunk.text,
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    bm25_score=bm25_scores.get(chunk_id, 0.0),
                    fused_score=fused.get(chunk_id, 0.0),
                    rank=rank,
                    source_quality=float(chunk.metadata.source_quality),
                    supporting_spans=list(matched_terms.get(chunk_id, [])[:8]),
                    title=document.title if document else chunk.metadata.document_name,
                    authors=list(document.authors) if document else [],
                    year=document.year if document else None,
                    venue=document.venue if document else None,
                    doi=document.doi if document else None,
                )
            )
        return items

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "fusion": self.fusion.value,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "rrf_k": self.rrf_k,
        }
