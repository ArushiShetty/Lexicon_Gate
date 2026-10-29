"""Benchmark and evaluation schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .query import PipelineName


class QuestionCategory(str, Enum):
    easy = "easy"
    multi_hop = "multi_hop"
    ambiguous = "ambiguous"
    multi_document = "multi_document"
    unanswerable = "unanswerable"
    conflicting = "conflicting"


class BenchmarkQuestion(BaseModel):
    question_id: str
    question: str
    category: QuestionCategory
    reference_answer: str = ""
    # Facts that a correct answer must contain (used for answer correctness).
    answer_keypoints: list[str] = Field(default_factory=list)
    # Ground-truth relevant chunk ids / document ids for retrieval metrics.
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_documents: list[str] = Field(default_factory=list)
    # True when the correct behaviour is to abstain.
    should_abstain: bool = False
    # True when the corpus intentionally contains conflicting evidence.
    has_conflict: bool = False
    notes: str = ""


class BenchmarkDataset(BaseModel):
    name: str
    description: str = ""
    questions: list[BenchmarkQuestion] = Field(default_factory=list)

    def by_category(self) -> dict[str, list[BenchmarkQuestion]]:
        buckets: dict[str, list[BenchmarkQuestion]] = {}
        for question in self.questions:
            buckets.setdefault(question.category.value, []).append(question)
        return buckets


class RetrievalMetrics(BaseModel):
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 5
    context_relevance: float = 0.0


class GenerationMetrics(BaseModel):
    answer_correctness: float = 0.0
    exact_match: float = 0.0
    keypoint_recall: float = 0.0
    faithfulness: float = 0.0
    citation_accuracy: float = 0.0
    citation_presence: float = 0.0
    claim_support_rate: float = 0.0


class HallucinationMetrics(BaseModel):
    hallucination_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    contradiction_rate: float = 0.0
    abstention_accuracy: float = 0.0
    correct_abstentions: int = 0
    over_abstentions: int = 0
    missed_abstentions: int = 0
    conflict_detection_rate: float = 0.0


class SystemAggregateMetrics(BaseModel):
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_retrieval_calls: float = 0.0
    avg_retrieval_attempts: float = 0.0
    avg_correction_loops: float = 0.0
    avg_total_tokens: float = 0.0
    avg_confidence: float = 0.0
    avg_evidence_coverage: float = 0.0


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    n: int
    mean_confidence: float
    accuracy: float


class PerQuestionOutcome(BaseModel):
    question_id: str
    category: QuestionCategory
    question: str
    answer: str
    status: str
    abstained: bool
    correctness: float
    faithfulness: float
    citation_accuracy: float
    claim_support_rate: float
    unsupported_claim_rate: float
    hallucinated: bool
    confidence: float
    evidence_coverage: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float
    retrieval_attempts: int
    correction_loops: int
    total_tokens: int
    conflict_detected: bool


class SystemEvaluation(BaseModel):
    pipeline: str
    pipeline_label: str
    n_questions: int
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = Field(default_factory=GenerationMetrics)
    hallucination: HallucinationMetrics = Field(default_factory=HallucinationMetrics)
    system: SystemAggregateMetrics = Field(default_factory=SystemAggregateMetrics)
    calibration: list[CalibrationBin] = Field(default_factory=list)
    expected_calibration_error: float = 0.0
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_question: list[PerQuestionOutcome] = Field(default_factory=list)


class SignificanceTest(BaseModel):
    metric: str
    baseline: str
    system: str
    baseline_mean: float
    system_mean: float
    delta: float
    relative_improvement_pct: float
    n_pairs: int
    t_statistic: float
    p_value: float
    significant: bool
    effect_size_cohens_d: float


class EvaluationRun(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str
    n_questions: int
    k: int
    systems: list[SystemEvaluation] = Field(default_factory=list)
    significance: list[SignificanceTest] = Field(default_factory=list)
    ablation: list[SystemEvaluation] = Field(default_factory=list)
    ablation_deltas: list[dict[str, Any]] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    notes: str = ""


class EvaluateRequest(BaseModel):
    pipelines: list[PipelineName] = Field(
        default_factory=lambda: [
            PipelineName.traditional,
            PipelineName.self_rag,
            PipelineName.enhanced,
        ]
    )
    include_ablation: bool = True
    limit: int | None = None
    categories: list[QuestionCategory] | None = None
    k: int = 5
    persist: bool = True
    include_headline: bool = False


class EvaluationSummary(BaseModel):
    run_id: str
    created_at: datetime
    dataset_name: str
    n_questions: int
    headline: dict[str, dict[str, float]] = Field(default_factory=dict)
