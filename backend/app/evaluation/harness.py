"""Evaluation harness.

Runs every selected pipeline (and optionally the ablation suite) against the
same gold-labelled questions, computes retrieval / generation / hallucination /
system metrics, paired significance tests, and confidence calibration.

Nothing in this module is a hard-coded leaderboard. If Enhanced Self-RAG does
not win a metric, the dashboard will show that.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..config import Settings, get_settings
from ..models.evaluation import (
    CalibrationBin,
    EvaluationRun,
    EvaluationSummary,
    GenerationMetrics,
    HallucinationMetrics,
    PerQuestionOutcome,
    RetrievalMetrics,
    SignificanceTest,
    SystemAggregateMetrics,
    SystemEvaluation,
)
from ..models.query import PipelineName, PipelineResult
from ..pipelines.runner import (
    ABLATION_VARIANTS,
    PipelineConfig,
    PipelineRunner,
    enhanced_config,
    no_rag_config,
    self_rag_config,
    traditional_config,
    verify_only_config,
)
from ..services.llm.base import LLMProvider
from ..services.store.knowledge_base import KnowledgeBase
from ..verification.claim_extraction import extract_claims
from ..verification.claim_verifier import ClaimVerifier
from .dataset import BenchmarkDataset, BenchmarkQuestion, benchmark_dataset, load_demo_corpus
from . import metrics as M

logger = logging.getLogger(__name__)

PIPELINE_BUILDERS = {
    PipelineName.traditional: traditional_config,
    PipelineName.self_rag: self_rag_config,
    PipelineName.enhanced: enhanced_config,
    PipelineName.no_rag: no_rag_config,
    PipelineName.rag_verify: verify_only_config,
}

HEADLINE_BUILDERS = (
    no_rag_config,
    traditional_config,
    verify_only_config,
    enhanced_config,
)


class EvaluationHarness:
    def __init__(
        self,
        kb: KnowledgeBase,
        llm: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.kb = kb
        self.llm = llm
        self.settings = settings or get_settings()
        self._verifier = ClaimVerifier(self.settings, idf=kb.idf_map())

    # ------------------------------------------------------------------ public
    def run(
        self,
        pipelines: Sequence[PipelineName] | None = None,
        include_ablation: bool = True,
        limit: int | None = None,
        categories: Sequence[str] | None = None,
        k: int = 5,
        persist: bool = True,
        dataset: BenchmarkDataset | None = None,
    ) -> EvaluationRun:
        t0 = time.perf_counter()
        if self.kb.is_empty():
            logger.info("Knowledge base empty; loading demo corpus for evaluation")
            load_demo_corpus(self.kb)

        dataset = dataset or benchmark_dataset()
        questions = list(dataset.questions)
        if categories:
            wanted = {c if isinstance(c, str) else c.value for c in categories}
            questions = [q for q in questions if q.category.value in wanted]
        if limit is not None:
            questions = questions[: max(0, int(limit))]

        names = list(pipelines) if pipelines else [
            PipelineName.traditional,
            PipelineName.self_rag,
            PipelineName.enhanced,
        ]

        systems: list[SystemEvaluation] = []
        per_system_correctness: dict[str, list[float]] = {}
        per_system_hallucination: dict[str, list[float]] = {}

        for name in names:
            config = PIPELINE_BUILDERS[name]()
            evaluation = self._evaluate_config(config, questions, k)
            systems.append(evaluation)
            per_system_correctness[evaluation.pipeline] = [
                o.correctness for o in evaluation.per_question
            ]
            per_system_hallucination[evaluation.pipeline] = [
                1.0 if o.hallucinated else 0.0 for o in evaluation.per_question
            ]

        significance = self._significance(systems, per_system_correctness, per_system_hallucination)

        ablation: list[SystemEvaluation] = []
        ablation_deltas: list[dict] = []
        if include_ablation:
            ablation, ablation_deltas = self._run_ablation(questions, k, systems)

        run = EvaluationRun(
            run_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc),
            dataset_name=dataset.name,
            n_questions=len(questions),
            k=k,
            systems=systems,
            significance=significance,
            ablation=ablation,
            ablation_deltas=ablation_deltas,
            config_snapshot={
                "llm": self.llm.describe(),
                "embedding": self.kb.embedder.describe(),
                "vector_store": self.kb.vectors.name,
                "evidence_threshold": self.settings.evidence_threshold,
                "initial_top_k": self.settings.initial_top_k,
                "max_retrieval_attempts": self.settings.max_retrieval_attempts,
            },
            duration_seconds=round(time.perf_counter() - t0, 3),
        )
        if persist:
            self._persist(run)
        return run

    def run_headline(
        self,
        limit: int | None = None,
        categories: Sequence[str] | None = None,
        k: int = 5,
        persist: bool = True,
        dataset: BenchmarkDataset | None = None,
    ) -> EvaluationRun:
        """Four-system table: no-RAG, basic RAG, verify-no-retry, full system."""

        t0 = time.perf_counter()
        if self.kb.is_empty():
            logger.info("Knowledge base empty; loading demo corpus for evaluation")
            load_demo_corpus(self.kb)

        dataset = dataset or benchmark_dataset()
        questions = list(dataset.questions)
        if categories:
            wanted = {c if isinstance(c, str) else c.value for c in categories}
            questions = [q for q in questions if q.category.value in wanted]
        if limit is not None:
            questions = questions[: max(0, int(limit))]

        systems: list[SystemEvaluation] = []
        per_system_correctness: dict[str, list[float]] = {}
        per_system_hallucination: dict[str, list[float]] = {}
        for builder in HEADLINE_BUILDERS:
            config = builder()
            evaluation = self._evaluate_config(config, questions, k)
            systems.append(evaluation)
            per_system_correctness[evaluation.pipeline] = [
                o.correctness for o in evaluation.per_question
            ]
            per_system_hallucination[evaluation.pipeline] = [
                1.0 if o.hallucinated else 0.0 for o in evaluation.per_question
            ]

        significance = self._significance(systems, per_system_correctness, per_system_hallucination)
        run = EvaluationRun(
            run_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc),
            dataset_name=dataset.name,
            n_questions=len(questions),
            k=k,
            systems=systems,
            significance=significance,
            config_snapshot={
                "llm": self.llm.describe(),
                "embedding": self.kb.embedder.describe(),
                "vector_store": self.kb.vectors.name,
                "mode": "headline_four_system",
            },
            duration_seconds=round(time.perf_counter() - t0, 3),
            notes="Headline comparison: no-RAG, Traditional RAG, RAG+verify (no retry), Enhanced Self-RAG.",
        )
        if persist:
            self._persist(run)
        return run

    # -------------------------------------------------------------- per system
    def _evaluate_config(
        self, config: PipelineConfig, questions: Sequence[BenchmarkQuestion], k: int
    ) -> SystemEvaluation:
        runner = PipelineRunner(self.kb, self.llm, config, self.settings)
        outcomes: list[PerQuestionOutcome] = []
        logger.info("Evaluating %s on %s questions", config.label or config.name.value, len(questions))

        for question in questions:
            result = runner.run(question.question, top_k=k)
            result = self._ensure_claims(result)
            if result.claims:
                from ..verification.hallucination import detect_hallucinations

                result.hallucination = detect_hallucinations(
                    result.answer, result.claims, result.evidence
                )
            outcomes.append(self._outcome(question, result, k))

        return self._aggregate(config, outcomes, k)

    def _ensure_claims(self, result: PipelineResult) -> PipelineResult:
        """Post-hoc claim extraction so Traditional RAG is scored fairly."""

        if result.claims or result.abstained or not result.answer:
            return result
        claims = extract_claims(result.answer, self.llm)
        verified = self._verifier.verify(claims, result.evidence, result.query)
        result.claims = verified.claims
        return result

    def _outcome(
        self, question: BenchmarkQuestion, result: PipelineResult, k: int
    ) -> PerQuestionOutcome:
        retrieved = [item.chunk_id for item in result.evidence]
        relevant = question.relevant_chunk_ids
        correctness = M.answer_correctness(
            result.answer,
            question.reference_answer,
            question.answer_keypoints,
            question.should_abstain,
            result.abstained,
        )
        faith = M.faithfulness(result)
        cit_acc = M.citation_accuracy(result)
        n_claims = len(result.claims) or 1
        n_unsupported = sum(
            1 for c in result.claims if c.status.value in {"UNSUPPORTED", "CONTRADICTED"}
        )
        unsupported_rate = 0.0 if result.abstained else n_unsupported / n_claims
        support_rate = 1.0 if result.abstained else (
            1.0 - unsupported_rate if result.claims else faith
        )

        hallucinated = _is_hallucination(question, result, unsupported_rate)
        conflict_detected = bool(result.contradictions) or (
            result.status.value == "CONFLICTING_EVIDENCE"
        )

        return PerQuestionOutcome(
            question_id=question.question_id,
            category=question.category,
            question=question.question,
            answer=result.answer,
            status=result.status.value,
            abstained=result.abstained,
            correctness=round(correctness, 4),
            faithfulness=round(faith, 4),
            citation_accuracy=round(cit_acc, 4),
            claim_support_rate=round(support_rate, 4),
            unsupported_claim_rate=round(unsupported_rate, 4),
            hallucinated=hallucinated,
            confidence=result.confidence.confidence,
            evidence_coverage=result.confidence.evidence_coverage,
            precision_at_k=round(M.precision_at_k(retrieved, relevant, k), 4),
            recall_at_k=round(M.recall_at_k(retrieved, relevant, k), 4),
            mrr=round(M.mean_reciprocal_rank(retrieved, relevant), 4),
            ndcg_at_k=round(M.ndcg_at_k(retrieved, relevant, k), 4),
            latency_ms=result.metrics.latency_ms,
            retrieval_attempts=result.metrics.retrieval_attempts,
            correction_loops=result.metrics.correction_loops,
            total_tokens=result.metrics.total_tokens,
            conflict_detected=conflict_detected,
        )

    def _aggregate(
        self, config: PipelineConfig, outcomes: list[PerQuestionOutcome], k: int
    ) -> SystemEvaluation:
        n = len(outcomes) or 1
        retrievable = [o for o in outcomes if o.category.value != "unanswerable"] or list(outcomes)
        retrieval = RetrievalMetrics(
            precision_at_k=M.mean(o.precision_at_k for o in retrievable),
            recall_at_k=M.mean(o.recall_at_k for o in retrievable),
            mrr=M.mean(o.mrr for o in retrievable),
            ndcg_at_k=M.mean(o.ndcg_at_k for o in retrievable),
            k=k,
            context_relevance=M.mean(o.evidence_coverage for o in retrievable),
        )
        generation = GenerationMetrics(
            answer_correctness=M.mean(o.correctness for o in outcomes),
            exact_match=0.0,
            keypoint_recall=M.mean(o.correctness for o in outcomes),
            faithfulness=M.mean(o.faithfulness for o in outcomes),
            citation_accuracy=M.mean(o.citation_accuracy for o in outcomes),
            citation_presence=M.mean(1.0 if "[" in o.answer else 0.0 for o in outcomes),
            claim_support_rate=M.mean(o.claim_support_rate for o in outcomes),
        )

        correct_abstentions = sum(1 for o in outcomes if o.category.value == "unanswerable" and o.abstained)
        missed_abstentions = sum(1 for o in outcomes if o.category.value == "unanswerable" and not o.abstained)
        over_abstentions = sum(1 for o in outcomes if o.category.value != "unanswerable" and o.abstained)
        n_unanswerable = sum(1 for o in outcomes if o.category.value == "unanswerable")
        n_answerable = n - n_unanswerable
        abstention_accuracy = 0.0
        if n_unanswerable or n_answerable:
            recall = correct_abstentions / n_unanswerable if n_unanswerable else 1.0
            precision = (
                correct_abstentions / (correct_abstentions + over_abstentions)
                if (correct_abstentions + over_abstentions)
                else 1.0
            )
            abstention_accuracy = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        n_conflict = sum(1 for o in outcomes if o.category.value == "conflicting")
        n_conflict_detected = sum(
            1 for o in outcomes if o.category.value == "conflicting" and o.conflict_detected
        )

        hallucination = HallucinationMetrics(
            hallucination_rate=M.mean(1.0 if o.hallucinated else 0.0 for o in outcomes),
            unsupported_claim_rate=M.mean(o.unsupported_claim_rate for o in outcomes),
            contradiction_rate=(
                1.0 - (n_conflict_detected / n_conflict) if n_conflict else 0.0
            ),
            abstention_accuracy=abstention_accuracy,
            correct_abstentions=correct_abstentions,
            over_abstentions=over_abstentions,
            missed_abstentions=missed_abstentions,
            conflict_detection_rate=(n_conflict_detected / n_conflict) if n_conflict else 0.0,
        )

        latencies = [o.latency_ms for o in outcomes]
        system = SystemAggregateMetrics(
            avg_latency_ms=M.mean(latencies),
            p95_latency_ms=M.percentile(latencies, 95),
            avg_retrieval_calls=M.mean(float(o.retrieval_attempts) for o in outcomes),
            avg_retrieval_attempts=M.mean(float(o.retrieval_attempts) for o in outcomes),
            avg_correction_loops=M.mean(float(o.correction_loops) for o in outcomes),
            avg_total_tokens=M.mean(float(o.total_tokens) for o in outcomes),
            avg_confidence=M.mean(o.confidence for o in outcomes),
            avg_evidence_coverage=M.mean(o.evidence_coverage for o in outcomes),
        )

        calibration, ece = _calibration(outcomes)

        per_category: dict[str, dict[str, float]] = defaultdict(dict)
        buckets: dict[str, list[PerQuestionOutcome]] = defaultdict(list)
        for outcome in outcomes:
            buckets[outcome.category.value].append(outcome)
        for category, rows in buckets.items():
            per_category[category] = {
                "n": float(len(rows)),
                "accuracy": M.mean(r.correctness for r in rows),
                "faithfulness": M.mean(r.faithfulness for r in rows),
                "hallucination_rate": M.mean(1.0 if r.hallucinated else 0.0 for r in rows),
                "abstention_rate": M.mean(1.0 if r.abstained else 0.0 for r in rows),
            }

        return SystemEvaluation(
            pipeline=config.name.value,
            pipeline_label=config.label or config.name.value,
            n_questions=len(outcomes),
            retrieval=retrieval,
            generation=generation,
            hallucination=hallucination,
            system=system,
            calibration=calibration,
            expected_calibration_error=ece,
            per_category=dict(per_category),
            per_question=outcomes,
        )

    # ---------------------------------------------------------- significance
    def _significance(
        self,
        systems: list[SystemEvaluation],
        correctness: dict[str, list[float]],
        hallucination: dict[str, list[float]],
    ) -> list[SignificanceTest]:
        tests: list[SignificanceTest] = []
        by_name = {s.pipeline: s for s in systems}
        pairs = [
            (PipelineName.traditional.value, PipelineName.enhanced.value),
            (PipelineName.self_rag.value, PipelineName.enhanced.value),
            (PipelineName.traditional.value, PipelineName.self_rag.value),
        ]
        for baseline, system in pairs:
            if baseline not in by_name or system not in by_name:
                continue
            for metric_name, store in (
                ("answer_correctness", correctness),
                ("hallucination_rate", hallucination),
            ):
                a = store[system]
                b = store[baseline]
                t, p = M.paired_ttest(a, b)
                mu_s = M.mean(a)
                mu_b = M.mean(b)
                delta = mu_s - mu_b
                rel = (delta / mu_b * 100.0) if abs(mu_b) > 1e-9 else 0.0
                tests.append(
                    SignificanceTest(
                        metric=metric_name,
                        baseline=baseline,
                        system=system,
                        baseline_mean=round(mu_b, 4),
                        system_mean=round(mu_s, 4),
                        delta=round(delta, 4),
                        relative_improvement_pct=round(rel, 2),
                        n_pairs=len(a),
                        t_statistic=round(t if t != float("inf") and t != float("-inf") else 0.0, 4),
                        p_value=round(p, 4),
                        significant=bool(p < 0.05),
                        effect_size_cohens_d=round(M.cohens_d(a, b), 4),
                    )
                )
        return tests

    # -------------------------------------------------------------- ablation
    def _run_ablation(
        self,
        questions: Sequence[BenchmarkQuestion],
        k: int,
        already: list[SystemEvaluation],
    ) -> tuple[list[SystemEvaluation], list[dict]]:
        by_label = {s.pipeline_label: s for s in already}
        results: list[SystemEvaluation] = []
        for key, config in ABLATION_VARIANTS.items():
            existing = by_label.get(config.label)
            if existing is not None:
                results.append(existing)
                continue
            logger.info("Ablation variant: %s", config.label)
            results.append(self._evaluate_config(config, questions, k))

        order = [c.label for c in ABLATION_VARIANTS.values()]
        results.sort(key=lambda s: order.index(s.pipeline_label) if s.pipeline_label in order else 99)

        deltas: list[dict] = []
        if results:
            baseline = results[0]
            for row in results[1:]:
                deltas.append(
                    {
                        "variant": row.pipeline_label,
                        "accuracy_delta": round(
                            row.generation.answer_correctness - baseline.generation.answer_correctness,
                            4,
                        ),
                        "hallucination_delta": round(
                            row.hallucination.hallucination_rate - baseline.hallucination.hallucination_rate,
                            4,
                        ),
                        "faithfulness_delta": round(
                            row.generation.faithfulness - baseline.generation.faithfulness,
                            4,
                        ),
                    }
                )
        return results, deltas

    # ------------------------------------------------------------- persist
    def _persist(self, run: EvaluationRun) -> Path:
        directory = Path(self.settings.results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        latest = directory / "latest.json"
        latest.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Evaluation run %s saved to %s", run.run_id, path)
        return path


def load_latest_run(settings: Settings | None = None) -> EvaluationRun | None:
    settings = settings or get_settings()
    latest = Path(settings.results_dir) / "latest.json"
    if not latest.exists():
        return None
    return EvaluationRun.model_validate_json(latest.read_text(encoding="utf-8"))


def list_runs(settings: Settings | None = None) -> list[EvaluationSummary]:
    settings = settings or get_settings()
    directory = Path(settings.results_dir)
    if not directory.exists():
        return []
    summaries: list[EvaluationSummary] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        if path.name == "latest.json":
            continue
        try:
            run = EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(summarise(run))
    return summaries


def summarise(run: EvaluationRun) -> EvaluationSummary:
    headline: dict[str, dict[str, float]] = {}
    for system in run.systems:
        headline[system.pipeline] = {
            "answer_accuracy": round(system.generation.answer_correctness, 4),
            "faithfulness": round(system.generation.faithfulness, 4),
            "citation_accuracy": round(system.generation.citation_accuracy, 4),
            "hallucination_rate": round(system.hallucination.hallucination_rate, 4),
            "evidence_coverage": round(system.system.avg_evidence_coverage, 4),
            "avg_retrieval_attempts": round(system.system.avg_retrieval_attempts, 4),
            "avg_response_time_ms": round(system.system.avg_latency_ms, 1),
            "precision_at_k": round(system.retrieval.precision_at_k, 4),
            "avg_confidence": round(system.system.avg_confidence, 4),
            "abstention_accuracy": round(system.hallucination.abstention_accuracy, 4),
        }
    return EvaluationSummary(
        run_id=run.run_id,
        created_at=run.created_at,
        dataset_name=run.dataset_name,
        n_questions=run.n_questions,
        headline=headline,
    )


def comparison_table(run: EvaluationRun) -> list[dict]:
    """Shape used by the RAG Comparison dashboard."""

    rows: list[dict] = []
    metric_names = [
        ("Answer Accuracy", "answer_accuracy"),
        ("Faithfulness", "faithfulness"),
        ("Citation Accuracy", "citation_accuracy"),
        ("Hallucination Rate", "hallucination_rate"),
        ("Evidence Coverage", "evidence_coverage"),
        ("Precision@K", "precision_at_k"),
        ("Abstention Accuracy", "abstention_accuracy"),
        ("Avg Retrieval Attempts", "avg_retrieval_attempts"),
        ("Avg Response Time (ms)", "avg_response_time_ms"),
        ("Avg Confidence", "avg_confidence"),
    ]
    summary = summarise(run).headline
    for label, key in metric_names:
        row = {"metric": label}
        for pipeline, values in summary.items():
            row[pipeline] = values.get(key, 0.0)
        rows.append(row)
    return rows


def headline_table(run: EvaluationRun) -> list[dict]:
    """Compact four-system (or current-run) table for viva / papers."""

    rows: list[dict] = []
    metric_names = [
        ("Hallucination Rate", "hallucination_rate"),
        ("Citation Accuracy", "citation_accuracy"),
        ("Answer Accuracy", "answer_accuracy"),
        ("Faithfulness", "faithfulness"),
    ]
    summary = summarise(run).headline
    for label, key in metric_names:
        row = {"metric": label}
        for pipeline, values in summary.items():
            row[pipeline] = values.get(key, 0.0)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ helpers
_ABSTAIN_IDS: set[str] = set()  # populated lazily; category is the real signal


def _is_hallucination(
    question: BenchmarkQuestion, result: PipelineResult, unsupported_rate: float
) -> bool:
    if question.should_abstain:
        return not result.abstained
    if question.has_conflict:
        detected = bool(result.contradictions) or result.status.value == "CONFLICTING_EVIDENCE"
        return not detected
    if result.abstained:
        return False
    return unsupported_rate >= 0.34


def _calibration(outcomes: Sequence[PerQuestionOutcome], n_bins: int = 5) -> tuple[list[CalibrationBin], float]:
    if not outcomes:
        return [], 0.0
    bins: list[CalibrationBin] = []
    ece = 0.0
    width = 1.0 / n_bins
    for i in range(n_bins):
        lo = i * width
        hi = 1.0 if i == n_bins - 1 else (i + 1) * width
        members = [
            o for o in outcomes if (lo <= o.confidence < hi) or (i == n_bins - 1 and o.confidence == 1.0)
        ]
        if not members:
            bins.append(CalibrationBin(bin_lower=lo, bin_upper=hi, n=0, mean_confidence=0.0, accuracy=0.0))
            continue
        mean_conf = M.mean(o.confidence for o in members)
        acc = M.mean(o.correctness for o in members)
        bins.append(
            CalibrationBin(
                bin_lower=round(lo, 2),
                bin_upper=round(hi, 2),
                n=len(members),
                mean_confidence=round(mean_conf, 4),
                accuracy=round(acc, 4),
            )
        )
        ece += (len(members) / len(outcomes)) * abs(mean_conf - acc)
    return bins, round(ece, 4)
