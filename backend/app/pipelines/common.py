"""Shared pipeline machinery: tracing, numbering citations, building results."""

from __future__ import annotations

import time
import uuid
from typing import Any, Sequence

from ..generation.prompts import INSUFFICIENT_ANSWER
from ..models.query import (
    AnswerStatus,
    Claim,
    ClaimStatus,
    ConfidenceReport,
    ContradictionPair,
    EvidenceGateDecision,
    EvidenceItem,
    HallucinationReport,
    PipelineName,
    PipelineResult,
    PlagiarismReport,
    SystemMetrics,
    TraceEvent,
)
from ..text_utils import split_sentences, stem_set
from ..services.citations import attach_cite_strings


PIPELINE_LABELS = {
    PipelineName.traditional: "Traditional RAG",
    PipelineName.self_rag: "Standard Self-RAG",
    PipelineName.enhanced: "Enhanced Self-RAG",
    PipelineName.no_rag: "Base LLM (no retrieval)",
    PipelineName.rag_verify: "RAG + verification (no retry)",
}


class TraceRecorder:
    """Collects structured, user-visible system events. No hidden reasoning."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._t0 = time.perf_counter()

    def add(
        self,
        stage: str,
        label: str,
        status: str = "ok",
        detail: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        self.events.append(
            TraceEvent(
                step=len(self.events) + 1,
                stage=stage,
                label=label,
                status=status,
                detail=detail,
                metrics=metrics or {},
                duration_ms=round(elapsed, 1),
            )
        )


class UsageCounter:
    def __init__(self) -> None:
        self.retrieval_calls = 0
        self.retrieval_attempts = 0
        self.correction_loops = 0
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.chunks_examined = 0

    def add_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def metrics(self, latency_ms: float) -> SystemMetrics:
        return SystemMetrics(
            latency_ms=round(latency_ms, 1),
            retrieval_calls=self.retrieval_calls,
            retrieval_attempts=self.retrieval_attempts,
            correction_loops=self.correction_loops,
            llm_calls=self.llm_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            chunks_examined=self.chunks_examined,
        )


def number_citations(evidence: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    numbered: list[EvidenceItem] = []
    for index, item in enumerate(evidence, start=1):
        clone = item.model_copy(deep=True)
        clone.citation_id = index
        attach_cite_strings(clone)
        numbered.append(clone)
    return numbered


def evidence_as_dicts(evidence: Sequence[EvidenceItem]) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": item.citation_id,
            "chunk_id": item.chunk_id,
            "document_name": item.document_name,
            "page": item.page,
            "section": item.section,
            "text": item.text,
            "relevance_score": item.relevance_score,
            "fused_score": item.fused_score,
            "evidence_score": item.evidence_score,
        }
        for item in evidence
    ]


def new_query_id() -> str:
    return uuid.uuid4().hex[:12]


def build_result(
    *,
    query_id: str,
    pipeline: PipelineName,
    query: str,
    answer: str,
    status: AnswerStatus,
    evidence: list[EvidenceItem],
    gate_decisions: list[EvidenceGateDecision],
    claims: list[Claim],
    contradictions: list[ContradictionPair],
    confidence: ConfidenceReport,
    hallucination: HallucinationReport,
    trace: list[TraceEvent],
    metrics: SystemMetrics,
    rewritten_queries: list[str] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    unrelated_to_sources: bool = False,
    mismatch_detail: str | None = None,
    plagiarism: PlagiarismReport | None = None,
    uncovered_aspects: list[str] | None = None,
    per_paper_summary: list[dict[str, Any]] | None = None,
    outcome: str | None = None,
) -> PipelineResult:
    if outcome is None:
        if status is AnswerStatus.premise_contradicted:
            computed_outcome = "premise_contradicted"
        elif status is AnswerStatus.conflicting_evidence or contradictions:
            computed_outcome = "conflicting"
        elif status in (AnswerStatus.insufficient_evidence, AnswerStatus.clarification_needed):
            computed_outcome = "not_enough_evidence"
        else:
            computed_outcome = "answered"
    else:
        computed_outcome = outcome

    key_points = [
        {
            "text": claim.text,
            "status": claim.status.value.lower(),
            "citations": [f"[{cid}]" for cid in claim.supporting_citations] or ([f"[{claim.source_chunk_id}]"] if claim.source_chunk_id else []),
        }
        for claim in claims
    ]

    conflicts = [
        {
            "claim_a": pair.claim_a,
            "source_a": f"[{pair.citation_a}]",
            "claim_b": pair.claim_b,
            "source_b": f"[{pair.citation_b}]",
            "context": pair.explanation,
        }
        for pair in contradictions
    ]

    sources = [
        {
            "id": f"[{item.citation_id}]",
            "title": item.title or item.document_name,
            "authors": item.authors or ["Author unavailable"],
            "year": item.year,
            "page": item.page,
            "passage": item.text,
            "section": item.section,
            "highlight": item.supporting_spans[0] if item.supporting_spans else "",
        }
        for item in evidence
    ]

    details = {
        "evidence_support_score": round(confidence.evidence_component * 100, 1) if confidence else None,
        "verification_method": claims[0].verifier if claims and claims[0].verifier else "evidence-grounded claim verification",
        "retrieval_attempts": metrics.retrieval_attempts if metrics else 1,
        "correction_attempts": metrics.correction_loops if metrics else 0,
        "unresolved_claims": [c.text for c in claims if c.status in (ClaimStatus.unsupported, ClaimStatus.contradicted)],
    }

    aspects_list = list(uncovered_aspects or [])

    return PipelineResult(
        query_id=query_id,
        pipeline=pipeline,
        pipeline_label=PIPELINE_LABELS[pipeline],
        query=query,
        answer=answer,
        status=status,
        abstained=status
        in {
            AnswerStatus.insufficient_evidence,
            AnswerStatus.clarification_needed,
        },
        rewritten_queries=rewritten_queries or [],
        evidence=evidence,
        gate_decisions=gate_decisions,
        claims=claims,
        contradictions=contradictions,
        confidence=confidence,
        hallucination=hallucination,
        plagiarism=plagiarism or PlagiarismReport(),
        trace=trace,
        metrics=metrics,
        config_snapshot=config_snapshot or {},
        unrelated_to_sources=unrelated_to_sources,
        mismatch_detail=mismatch_detail,
        uncovered_aspects=aspects_list,
        outcome=computed_outcome,
        key_points=key_points,
        conflicts=conflicts,
        not_found=aspects_list,
        per_paper_summary=per_paper_summary or [],
        sources=sources,
        details=details,
    )


def drop_unsupported_sentences(
    answer: str,
    good_claims: Sequence[Claim],
    bad_claims: Sequence[Claim],
) -> tuple[str, list[Claim]]:
    """Remove sentences that match unverified claims; keep supported ones.

    Used after the self-correct loop so a fluent answer cannot keep red claims.
    """

    if not bad_claims:
        return answer.strip(), list(good_claims)
    sentences = split_sentences(answer, min_chars=8)
    if not sentences:
        sentences = [part.strip() for part in answer.split("\n") if part.strip()]
    kept: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped == INSUFFICIENT_ANSWER:
            continue
        stems = stem_set(stripped)
        bad_score = max((_stem_overlap(stem_set(claim.text), stems) for claim in bad_claims), default=0.0)
        good_score = max((_stem_overlap(stem_set(claim.text), stems) for claim in good_claims), default=0.0)
        if bad_score >= 0.5 and bad_score > good_score:
            continue
        kept.append(stripped)
    text = " ".join(kept).strip()
    return text, list(good_claims)


def _stem_overlap(claim_stems: set[str], sentence_stems: set[str]) -> float:
    if not claim_stems:
        return 0.0
    return len(claim_stems & sentence_stems) / len(claim_stems)
