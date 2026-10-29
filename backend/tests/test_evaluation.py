"""Evaluation metrics and harness produce real, non-hard-coded numbers."""

from __future__ import annotations

from app.evaluation.harness import (
    EvaluationHarness,
    comparison_table,
    headline_table,
    summarise,
)
from app.evaluation.metrics import (
    answer_correctness,
    exact_match,
    keypoint_recall,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.models.query import PipelineName


def test_retrieval_metrics_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "e"]
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 4) == 0.5
    assert ndcg_at_k(["c", "a"], ["c"], 2) == 1.0


def test_answer_correctness_abstention_and_keypoints():
    assert answer_correctness("anything", "ref", ["k"], should_abstain=True, abstained=True) == 1.0
    assert answer_correctness("anything", "ref", ["k"], should_abstain=True, abstained=False) == 0.0
    score = keypoint_recall(
        "Dropout reduces overfitting by preventing co-adaptation.",
        ["reduces overfitting", "co-adaptation"],
    )
    assert score == 1.0
    assert exact_match("The rate is 0.5", "the rate is 0.5") == 1.0


def test_harness_runs_three_systems_and_does_not_hardcode(demo_kb, llm):
    harness = EvaluationHarness(demo_kb, llm, demo_kb.settings)
    run = harness.run(
        pipelines=[PipelineName.traditional, PipelineName.self_rag, PipelineName.enhanced],
        include_ablation=False,
        persist=False,
        k=5,
    )
    assert run.n_questions >= 10
    assert len(run.systems) == 3
    by_name = {s.pipeline: s for s in run.systems}
    trad = by_name[PipelineName.traditional.value]
    enh = by_name[PipelineName.enhanced.value]

    # Real computed values in [0, 1], not the example 82/88/94 from the spec.
    for system in run.systems:
        acc = system.generation.answer_correctness
        hall = system.hallucination.hallucination_rate
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= hall <= 1.0
        assert acc not in {0.82, 0.88, 0.94}

    # Enhanced must beat Traditional RAG on the two headline research metrics.
    assert enh.generation.answer_correctness > trad.generation.answer_correctness
    assert enh.hallucination.hallucination_rate < trad.hallucination.hallucination_rate
    assert enh.hallucination.abstention_accuracy > trad.hallucination.abstention_accuracy

    table = comparison_table(run)
    assert any(row["metric"] == "Answer Accuracy" for row in table)
    summary = summarise(run)
    assert PipelineName.enhanced.value in summary.headline


def test_harness_headline_four_systems(demo_kb, llm):
    harness = EvaluationHarness(demo_kb, llm, demo_kb.settings)
    run = harness.run_headline(persist=False, k=5)
    assert run.n_questions >= 10
    assert len(run.systems) == 4
    assert run.config_snapshot.get("mode") == "headline_four_system"
    names = [s.pipeline for s in run.systems]
    assert names == [
        PipelineName.no_rag.value,
        PipelineName.traditional.value,
        PipelineName.rag_verify.value,
        PipelineName.enhanced.value,
    ]
    by_name = {s.pipeline: s for s in run.systems}
    trad = by_name[PipelineName.traditional.value]
    full = by_name[PipelineName.enhanced.value]
    no_rag = by_name[PipelineName.no_rag.value]

    for system in run.systems:
        hall = system.hallucination.hallucination_rate
        cit = system.generation.citation_accuracy
        acc = system.generation.answer_correctness
        faith = system.generation.faithfulness
        assert 0.0 <= hall <= 1.0
        assert 0.0 <= cit <= 1.0
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= faith <= 1.0
        assert acc not in {0.82, 0.88, 0.94}

    assert full.hallucination.hallucination_rate <= trad.hallucination.hallucination_rate
    assert no_rag.system.avg_retrieval_attempts == 0
    for outcome in no_rag.per_question:
        if not outcome.abstained:
            assert outcome.citation_accuracy == 0.0

    table = headline_table(run)
    metrics = {row["metric"] for row in table}
    assert "Hallucination Rate" in metrics
    assert "Citation Accuracy" in metrics
    assert "Answer Accuracy" in metrics
    for row in table:
        assert PipelineName.no_rag.value in row
        assert PipelineName.enhanced.value in row
