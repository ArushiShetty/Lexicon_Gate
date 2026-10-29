"""Query, evidence, verification and answer schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineName(str, Enum):
    traditional = "traditional_rag"
    self_rag = "self_rag"
    enhanced = "enhanced_self_rag"
    no_rag = "no_rag"
    rag_verify = "rag_verify"


class ClaimStatus(str, Enum):
    supported = "SUPPORTED"
    partially_supported = "PARTIALLY_SUPPORTED"
    unsupported = "UNSUPPORTED"
    contradicted = "CONTRADICTED"
    conflicting = "CONFLICTING"
    unverifiable = "UNVERIFIABLE"


class AnswerStatus(str, Enum):
    answered = "ANSWERED"
    insufficient_evidence = "INSUFFICIENT_EVIDENCE"
    conflicting_evidence = "CONFLICTING_EVIDENCE"
    clarification_needed = "CLARIFICATION_NEEDED"
    no_retrieval_needed = "NO_RETRIEVAL_NEEDED"
    premise_contradicted = "PREMISE_CONTRADICTED"


class QueryAnalysis(BaseModel):
    """Output of the pre-retrieval query analyser."""

    original_query: str
    normalised_query: str
    needs_retrieval: bool = True
    retrieval_reason: str = ""
    question_type: str = "factual"
    is_multi_hop: bool = False
    is_ambiguous: bool = False
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    suggested_top_k: int = 5


class EvidenceScoreBreakdown(BaseModel):
    """Per-chunk evidence diagnostics produced by the evidence gate."""

    semantic_relevance: float = 0.0
    keyword_overlap: float = 0.0
    entity_overlap: float = 0.0
    source_quality: float = 0.0
    retrieval_score: float = 0.0
    contradiction_indicator: float = 0.0
    chunk_evidence_score: float = 0.0


class EvidenceItem(BaseModel):
    """A retrieved chunk enriched with scoring information."""

    citation_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    section: str | None = None
    text: str
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rank: int = 0
    source_quality: float = 0.6
    relevance_score: float = 0.0
    evidence_score: float = 0.0
    breakdown: EvidenceScoreBreakdown = Field(default_factory=EvidenceScoreBreakdown)
    supporting_spans: list[str] = Field(default_factory=list)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    apa: str = ""
    bibtex: str = ""

    @property
    def citation_label(self) -> str:
        heading = self.title or self.document_name
        if self.page is None:
            return heading
        return f"{heading} — page {self.page}"


class EvidenceGateDecision(BaseModel):
    """Aggregate gate verdict for one retrieval attempt."""

    attempt: int = 1
    query_used: str = ""
    sufficient: bool = False
    evidence_score: float = 0.0
    threshold: float = 0.0
    mean_relevance: float = 0.0
    max_relevance: float = 0.0
    coverage: float = 0.0
    consistency: float = 0.0
    source_quality: float = 0.0
    n_relevant_chunks: int = 0
    n_candidates: int = 0
    uncovered_terms: list[str] = Field(default_factory=list)
    action: str = "proceed"
    rationale: str = ""


class Claim(BaseModel):
    claim_id: str
    text: str
    status: ClaimStatus = ClaimStatus.unsupported
    support_score: float = 0.0
    contradiction_score: float = 0.0
    supporting_citations: list[int] = Field(default_factory=list)
    contradicting_citations: list[int] = Field(default_factory=list)
    best_evidence_span: str | None = None
    rationale: str = ""
    source_chunk_id: str | None = None
    nli_label: str | None = None
    nli_confidence: float | None = None
    verifier: str = ""


class ContradictionPair(BaseModel):
    claim_a: str
    claim_b: str
    citation_a: int
    citation_b: int
    document_a: str
    document_b: str
    similarity: float
    polarity_conflict: float
    numeric_conflict: float
    score: float
    explanation: str
    preferred_citation: int | None = None
    preferred_reason: str | None = None


class ConfidenceReport(BaseModel):
    confidence: float = 0.0
    evidence_component: float = 0.0
    claim_support_component: float = 0.0
    source_agreement_component: float = 0.0
    coverage_component: float = 0.0
    evidence_coverage: float = 0.0
    claims_verified: int = 0
    claims_total: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    label: str = "low"
    caveat: str = (
        "This is a system confidence metric derived from retrieved evidence, "
        "not a guarantee of factual truth."
    )


class HallucinationReport(BaseModel):
    hallucination_detected: bool = False
    unsupported_claim_rate: float = 0.0
    contradicted_claim_rate: float = 0.0
    ungrounded_numeric_tokens: list[str] = Field(default_factory=list)
    ungrounded_entities: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    severity: str = "none"


class PlagiarismMatch(BaseModel):
    text: str
    document_id: str = ""
    document_name: str = ""
    page: int | None = None
    cited: bool = False
    n_words: int = 0


class FlaggedSentence(BaseModel):
    text: str
    matches: list[PlagiarismMatch] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    kind: str = "verbatim"
    section: str = ""


class SectionOriginality(BaseModel):
    title: str
    originality: float = 1.0
    n_sentences: int = 0
    n_flagged: int = 0


class PlagiarismReport(BaseModel):
    originality: float = 1.0
    overlap_ratio: float = 0.0
    risk: str = "low"
    matches: list[PlagiarismMatch] = Field(default_factory=list)
    flagged_sentences: list[FlaggedSentence] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class PlagiarismSourceShare(BaseModel):
    document_id: str
    document_name: str
    n_words: int = 0
    share: float = 0.0
    origin: str = "library"
    url: str | None = None
    doi: str | None = None


class PaperPlagiarismCheck(BaseModel):
    filename: str
    n_words: int = 0
    n_pages: int = 0
    similarity: float = 0.0
    originality: float = 1.0
    risk: str = "low"
    library_empty: bool = False
    matches: list[PlagiarismMatch] = Field(default_factory=list)
    flagged_sentences: list[FlaggedSentence] = Field(default_factory=list)
    sources: list[PlagiarismSourceShare] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    spans: list[FlaggedSentence] = Field(default_factory=list)
    sections: list[SectionOriginality] = Field(default_factory=list)
    oa_full_texts: int = 0
    abstracts_compared: int = 0
    self_matches_skipped: int = 0


class TraceEvent(BaseModel):
    """A single structured, user-visible system event.

    These are deliberately *system decisions and scores* only. No hidden model
    reasoning is ever recorded here.
    """

    step: int
    stage: str
    label: str
    status: str = "ok"
    detail: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None


class SystemMetrics(BaseModel):
    latency_ms: float = 0.0
    retrieval_calls: int = 0
    retrieval_attempts: int = 0
    correction_loops: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    chunks_examined: int = 0


class PipelineResult(BaseModel):
    """Uniform output contract shared by all three pipelines."""

    query_id: str
    pipeline: PipelineName
    pipeline_label: str
    query: str
    answer: str
    status: AnswerStatus = AnswerStatus.answered
    abstained: bool = False
    rewritten_queries: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    gate_decisions: list[EvidenceGateDecision] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)
    hallucination: HallucinationReport = Field(default_factory=HallucinationReport)
    plagiarism: PlagiarismReport = Field(default_factory=PlagiarismReport)
    trace: list[TraceEvent] = Field(default_factory=list)
    metrics: SystemMetrics = Field(default_factory=SystemMetrics)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    unrelated_to_sources: bool = False
    mismatch_detail: str | None = None
    uncovered_aspects: list[str] = Field(default_factory=list)
    outcome: str = "answered"
    key_points: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)
    per_paper_summary: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    # Accepted for backward compatibility; product /query always runs Enhanced Self-RAG.
    pipeline: PipelineName | None = None
    top_k: int | None = None
    document_ids: list[str] | None = None
    evidence_threshold: float | None = None
    include_trace: bool = True
    # Optional ChatGPT-style thread id so Ask history stays grouped in Recents.
    chat_id: str | None = None


class CompareRequest(BaseModel):
    query: str = Field(min_length=1)
    pipelines: list[PipelineName] = Field(
        default_factory=lambda: [
            PipelineName.traditional,
            PipelineName.self_rag,
            PipelineName.enhanced,
        ]
    )
    top_k: int | None = None
    document_ids: list[str] | None = None


class ComparisonRow(BaseModel):
    pipeline: PipelineName
    pipeline_label: str
    answer: str
    status: AnswerStatus
    abstained: bool
    confidence: float
    evidence_score: float
    evidence_coverage: float
    claims_verified: int
    claims_total: int
    claim_support_rate: float
    unsupported_claim_rate: float
    hallucination_detected: bool
    hallucination_flags: list[str]
    contradictions_found: int
    n_sources: int
    citations: list[str]
    retrieval_attempts: int
    correction_loops: int
    latency_ms: float


class CompareResponse(BaseModel):
    query_id: str
    query: str
    rows: list[ComparisonRow]
    results: dict[str, PipelineResult]
    winner: str
    winner_reason: str
