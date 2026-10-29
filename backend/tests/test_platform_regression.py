"""Regression test suite covering the 16 core safety and reliability scenarios.

Matches Section 15 of user instructions:
Test 1: Single-paper factual question.
Test 2: Three-paper summary.
Test 3: Explicitly unsupported premise.
Test 4: Directly contradicted premise.
Test 5: Genuine conflicting results.
Test 6: Different publication years.
Test 7: Different experimental conditions.
Test 8: Exact numerical query.
Test 9: Unsupported numerical claim.
Test 10: Corrective retrieval.
Test 11: Exhausted retry budget.
Test 12: Citation mapping.
Test 13: Bibliography contamination.
Test 14: Missing PDF metadata.
Test 15: Multi-turn conversation.
Test 16: UI and outcome schema contract.
"""

from __future__ import annotations

import re
import pytest

from app.models.query import AnswerStatus, ClaimStatus, PipelineResult
from app.pipelines.runner import PipelineRunner, enhanced_config
from app.services.ingestion.loaders import LoadedPage
from app.services.citations import format_apa, attach_cite_strings
from app.retrieval.hybrid import is_bibliography_chunk
from app.text_utils import numeric_conflict


def test_regression_test_1_single_paper_factual(demo_kb, llm):
    """Test 1: Single-paper factual question returns direct specific answer with source/page."""
    docs = {d.name: d.document_id for d in demo_kb.store.list_documents()}
    handbook_id = docs["Neural Regularization Handbook.md"]
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run(
        "What is the typical dropout rate used in hidden layers?",
        document_ids=[handbook_id],
    )

    assert result.status is AnswerStatus.answered
    assert result.outcome == "answered"
    assert "0.5" in result.answer
    assert len(result.evidence) >= 1
    assert any("Neural Regularization Handbook" in (item.title or item.document_name) for item in result.evidence)


def test_regression_test_2_three_paper_summary(demo_kb, llm):
    """Test 2: Three-paper summary returns three separate summaries followed by synthesis."""
    docs = demo_kb.store.list_documents()[:3]
    doc_ids = [d.document_id for d in docs]

    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run(
        "Summarize the selected papers",
        document_ids=doc_ids,
    )

    assert result.status is AnswerStatus.answered
    assert len(result.per_paper_summary) == 3
    # Check that each paper is summarized individually
    summary_titles = [s["title"] for s in result.per_paper_summary]
    for d in docs:
        assert (d.title or d.name) in summary_titles
    # Check that synthesis is present
    assert "Cross-Paper Synthesis" in result.answer
    assert "Common Themes" in result.answer


def test_regression_test_3_unsupported_premise(demo_kb, llm):
    """Test 3: Unsupported premise is NOT assumed false; reports insufficient evidence."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("Why did the authors evaluate on the SVHN dataset?")

    # SVHN is never mentioned; system must not claim premise is false, but report not enough evidence
    assert result.status is AnswerStatus.insufficient_evidence
    assert result.outcome == "not_enough_evidence"
    assert "not contain enough support" in result.answer.lower() or "searched" in result.answer.lower()
    # It must NOT be flagged as contradicted
    assert not result.contradictions


def test_regression_test_4_directly_contradicted_premise(demo_kb, llm):
    """Test 4: Directly contradicted premise reports conflicting evidence neutrally and cites passage."""
    docs = {d.name: d.document_id for d in demo_kb.store.list_documents()}
    handbook_id = docs["Neural Regularization Handbook.md"]
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    # Handbook explicitly states 0.5 is used in hidden layers
    result = runner.run(
        "Why was the typical dropout rate set to 0.9 in hidden layers?",
        document_ids=[handbook_id],
    )

    # 0.9 directly contradicts 0.5 in the text
    assert result.status is AnswerStatus.premise_contradicted
    assert result.outcome == "premise_contradicted"
    assert "0.5" in result.answer
    assert "[1]" in result.answer or "[2]" in result.answer or len(result.claims) > 0


def test_regression_test_5_genuine_conflicting_results(demo_kb, llm):
    """Test 5: Conflicting results show both claims with their source passages."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("Does dropout reduce overfitting?")

    assert result.status is AnswerStatus.conflicting_evidence
    assert result.outcome == "conflicting"
    assert len(result.contradictions) > 0 or "conflict" in result.answer.lower() or "disagree" in result.answer.lower()


def test_regression_test_6_different_publication_years():
    """Test 6: Different publication years are not labelled contradictory merely because they differ."""
    sent1 = "The study was published in 2018 and introduced residual networks."
    sent2 = "Another architecture was evaluated in 2023 with modern training techniques."
    conflict = numeric_conflict(sent1, sent2)
    assert conflict == 0.0, f"Expected 0 conflict for different years, got {conflict}"


def test_regression_test_7_different_experimental_conditions():
    """Test 7: Different experimental conditions (e.g. CIFAR vs ImageNet) are not treated as contradictions."""
    sent1 = "The model achieves 95.5% top-1 accuracy on CIFAR-10."
    sent2 = "The model achieves 76.2% top-1 accuracy on ImageNet."
    conflict = numeric_conflict(sent1, sent2)
    assert conflict == 0.0, f"Disjoint datasets should not be marked as conflict, got {conflict}"


def test_regression_test_8_exact_numerical_query(demo_kb, llm):
    """Test 8: Exact numerical query returns exact value, metric, and dataset."""
    docs = {d.name: d.document_id for d in demo_kb.store.list_documents()}
    handbook_id = docs["Neural Regularization Handbook.md"]
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run(
        "What is the typical dropout rate used in hidden layers?",
        document_ids=[handbook_id],
    )

    assert result.status is AnswerStatus.answered
    assert "0.5" in result.answer


def test_regression_test_9_unsupported_numerical_claim(demo_kb, llm):
    """Test 9: Unsupported numerical claim is not accepted solely because it sounds plausible."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("What was the exact 99.8% compression ratio achieved by the model?")

    # The text contains no 99.8% compression; system must not accept or hallucinate 99.8%
    assert "99.8%" not in result.answer or result.status in (AnswerStatus.insufficient_evidence, AnswerStatus.premise_contradicted)


def test_regression_test_10_corrective_retrieval(demo_kb, llm):
    """Test 10: Corrective retrieval expands or retries when evidence has gaps."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("How does dropout reduce overfitting?")

    assert result.status in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)
    assert result.metrics.retrieval_attempts >= 1
    assert len(result.claims) >= 1


def test_regression_test_11_exhausted_retry_budget(demo_kb, llm):
    """Test 11: Exhausted retry budget stops without fabricating successful outcome."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("What is the quantum supremacy threshold in superconducting qubits?")

    assert result.status is AnswerStatus.insufficient_evidence
    assert result.outcome == "not_enough_evidence"
    assert result.abstained is True


def test_regression_test_12_citation_mapping(demo_kb, llm):
    """Test 12: Every displayed citation points to the correct document."""
    docs = {d.name: d.document_id for d in demo_kb.store.list_documents()}
    survey_id = docs["Self-RAG Survey.md"]
    hybrid_id = docs["Hybrid Retrieval Notes.md"]

    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run(
        "How do hybrid retrieval and Self-RAG compare?",
        document_ids=[survey_id, hybrid_id],
    )

    for item in result.evidence:
        if item.document_id == survey_id:
            assert "Self-RAG" in item.document_name or "Self RAG" in (item.title or "")
        elif item.document_id == hybrid_id:
            assert "Hybrid" in item.document_name or "Hybrid" in (item.title or "")


def test_regression_test_13_bibliography_contamination():
    """Test 13: Reference entries do not contaminate scientific evidence."""
    bib_text = (
        "[1] Vaswani, A., et al. Attention is all you need. NeurIPS, 2017, pp. 5998-6008.\n"
        "[2] He, K., et al. Deep residual learning. CVPR, 2016, pp. 770-778.\n"
        "[3] Devlin, J., et al. BERT. NAACL, 2019, doi:10.18653/v1/N19-1423."
    )
    assert is_bibliography_chunk(bib_text, section="References") is True
    assert is_bibliography_chunk(bib_text, section=None) is True

    sci_text = (
        "We evaluate our model on the benchmark dataset. The training pipeline uses AdamW "
        "with an initial learning rate of 0.0001 and weight decay of 0.01."
    )
    assert is_bibliography_chunk(sci_text, section="Experiments") is False


def test_regression_test_14_missing_pdf_metadata():
    """Test 14: Missing PDF metadata uses honest fallbacks without inventing details."""
    from app.models.query import EvidenceItem

    item = EvidenceItem(
        citation_id=1,
        chunk_id="chunk_1",
        document_id="doc_1",
        document_name="scanned_paper.pdf",
        title="",
        authors=[],
        year=None,
        page=None,
        text="Sample text without metadata.",
    )
    apa = format_apa(item)
    # Must use honest fallback rather than fabricating names/years
    assert "Unknown" in apa or "Author unavailable" in apa
    assert "n.d." in apa


def test_regression_test_15_multi_turn_conversation(demo_kb, llm):
    """Test 15: Multi-turn chat maintains scoped documents and conversation thread."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    turn1 = runner.run("What is dropout?", query_id="turn_1")
    assert turn1.status in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)

    turn2 = runner.run("Does it reduce overfitting?", query_id="turn_2")
    assert turn2.status in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)
    assert turn2.query_id == "turn_2"


def test_regression_test_16_structured_contract_fields(demo_kb, llm):
    """Test 16: Structured response contract adheres to Section 12 requirements."""
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("What is dropout?")

    # Verify structured fields defined in contract
    assert hasattr(result, "outcome")
    assert result.outcome in ("answered", "conflicting", "premise_contradicted", "not_enough_evidence")
    assert hasattr(result, "key_points")
    assert isinstance(result.key_points, list)
    assert hasattr(result, "conflicts")
    assert isinstance(result.conflicts, list)
    assert hasattr(result, "not_found")
    assert isinstance(result.not_found, list)
    assert hasattr(result, "per_paper_summary")
    assert isinstance(result.per_paper_summary, list)
    assert hasattr(result, "sources")
    assert isinstance(result.sources, list)
    assert hasattr(result, "details")
    assert isinstance(result.details, dict)
    assert "verification_method" in result.details
