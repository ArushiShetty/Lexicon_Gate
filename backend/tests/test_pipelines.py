"""End-to-end pipeline behaviour on the demo corpus."""

from __future__ import annotations

import re

from app.generation.prompts import INSUFFICIENT_ANSWER
from app.models.query import AnswerStatus, Claim, ClaimStatus, PipelineName
from app.pipelines.common import TraceRecorder, drop_unsupported_sentences
from app.pipelines.runner import (
    PipelineRunner,
    enhanced_config,
    no_rag_config,
    self_rag_config,
    traditional_config,
    verify_only_config,
)


def test_traditional_rag_answers_unanswerable_question(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, traditional_config())
    result = runner.run("How does dropout affect Titan's nitrogen atmosphere?")
    assert result.pipeline is PipelineName.traditional
    assert result.abstained is False
    assert result.answer
    assert result.answer != INSUFFICIENT_ANSWER
    assert result.metrics.retrieval_attempts == 1


def test_enhanced_abstains_when_evidence_is_missing(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("How does dropout affect Titan's nitrogen atmosphere?")
    assert result.abstained is True
    assert result.status is AnswerStatus.insufficient_evidence
    assert result.unrelated_to_sources is True
    lowered = result.answer.lower()
    assert "searched" in lowered and "indexed sources" in lowered
    assert "won't invent" in lowered
    assert result.mismatch_detail
    assert "Indexed sources cover:" in result.mismatch_detail
    titles = " ".join(
        (doc.title or doc.name) for doc in demo_kb.store.list_documents()
    )
    assert any(
        (doc.title or doc.name).split(".")[0] in result.mismatch_detail
        for doc in demo_kb.store.list_documents()
    ), titles


def test_enhanced_refusal_names_sources_for_missing_definition(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("what is random recurrent neural network")
    assert result.abstained is True
    assert result.status is AnswerStatus.insufficient_evidence
    lowered = result.answer.lower()
    assert "searched" in lowered and "indexed sources" in lowered
    assert "won't invent a definition" in lowered
    assert "random recurrent neural network" in lowered
    assert result.mismatch_detail
    assert "Indexed sources cover:" in result.mismatch_detail
    assert any(
        (doc.title or doc.name).split(".")[0] in result.mismatch_detail
        for doc in demo_kb.store.list_documents()
    )


def test_enhanced_answers_supported_question_with_citations(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("What is the main advantage of dropout?")
    assert result.abstained is False
    assert result.status in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)
    assert "overfitting" in result.answer.lower()
    assert result.evidence
    assert re.search(r"\[\d+\]", result.answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", result.answer.strip()) if s.strip()]
    assert len(sentences) >= 2
    assert result.confidence.confidence > 0.4
    assert result.trace
    assert result.unrelated_to_sources is False
    assert result.claims
    assert any(claim.status is ClaimStatus.supported for claim in result.claims)


def test_enhanced_multi_document_answer_cites_multiple_sources(demo_kb, llm):
    docs = {doc.name: doc.document_id for doc in demo_kb.store.list_documents()}
    survey = docs["Self-RAG Survey.md"]
    hybrid = docs["Hybrid Retrieval Notes.md"]
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run(
        "How do hybrid retrieval and the Enhanced Self-RAG architecture improve on traditional dense-only RAG?",
        document_ids=[survey, hybrid],
    )
    assert result.abstained is False
    assert result.status in (AnswerStatus.answered, AnswerStatus.conflicting_evidence)
    assert re.search(r"\[\d+\]", result.answer)
    cite_ids = {int(m) for m in re.findall(r"\[(\d+)\]", result.answer)}
    evidence_docs = {item.document_id for item in result.evidence}
    assert survey in evidence_docs
    assert hybrid in evidence_docs
    assert len(evidence_docs) >= 2
    assert len(cite_ids) >= 1
    # Balanced multi-doc retrieve should surface both scoped papers.
    assert {survey, hybrid}.issubset(evidence_docs)
    # Discourse boilerplate must not drive unsupported/hallucination alone.
    flags = " ".join(result.hallucination.flags or [])
    assert "According" not in flags
    if len(cite_ids) >= 2:
        cited_docs = {
            item.document_id
            for item in result.evidence
            if item.citation_id in cite_ids
        }
        assert len(cited_docs) >= 1


def test_balanced_retrieve_covers_each_scoped_document(demo_kb, llm):
    docs = {doc.name: doc.document_id for doc in demo_kb.store.list_documents()}
    survey = docs["Self-RAG Survey.md"]
    hybrid = docs["Hybrid Retrieval Notes.md"]
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    raw = runner._retrieve_for_scope(
        "hybrid retrieval Enhanced Self-RAG architecture",
        top_k=6,
        document_ids=[survey, hybrid],
    )
    found = {item.document_id for item in raw}
    assert survey in found
    assert hybrid in found


def test_enforce_rebuilds_thin_answer_from_supported_claims():
    runner = PipelineRunner.__new__(PipelineRunner)
    answer, status, claims = runner._enforce_supported_answer(
        "Dropout reduces overfitting [1]. Titan has a nitrogen atmosphere [2].",
        AnswerStatus.answered,
        [
            Claim(
                claim_id="c1",
                text="Dropout reduces overfitting.",
                status=ClaimStatus.supported,
                supporting_citations=[1],
            ),
            Claim(
                claim_id="c2",
                text="Dropout prevents co-adaptation of hidden units.",
                status=ClaimStatus.supported,
                supporting_citations=[1],
            ),
            Claim(
                claim_id="c3",
                text="Titan has a nitrogen atmosphere.",
                status=ClaimStatus.unsupported,
                supporting_citations=[2],
            ),
        ],
        TraceRecorder(),
    )
    assert status is AnswerStatus.answered
    assert "overfitting" in answer.lower()
    assert "co-adaptation" in answer.lower() or "hidden units" in answer.lower()
    assert "titan" not in answer.lower()
    assert all(c.status is ClaimStatus.supported for c in claims)
    assert re.search(r"\[\d+\]", answer)


def test_enhanced_detects_dropout_conflict(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("Does dropout reduce overfitting?")
    assert result.contradictions or result.status is AnswerStatus.conflicting_evidence


def test_no_rag_never_retrieves(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, no_rag_config())
    result = runner.run("What is the main advantage of dropout?")
    assert result.pipeline is PipelineName.no_rag
    assert result.evidence == []
    assert result.metrics.retrieval_calls == 0
    assert result.metrics.retrieval_attempts == 0


def test_verify_only_labels_claims_without_correction(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, verify_only_config())
    result = runner.run("What is the main advantage of dropout?")
    assert result.pipeline is PipelineName.rag_verify
    assert result.evidence
    assert result.metrics.correction_loops == 0
    stages = {event.stage for event in result.trace}
    assert "claims" in stages or "verify" in stages or result.claims


def test_self_rag_can_skip_retrieval_for_greeting(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, self_rag_config())
    result = runner.run("hello")
    assert result.status is AnswerStatus.no_retrieval_needed
    assert result.metrics.retrieval_calls == 0


def test_compare_three_systems_same_question(demo_kb, llm):
    query = "What is the main advantage of the proposed architecture?"
    results = {
        name: PipelineRunner(demo_kb, llm, cfg()).run(query)
        for name, cfg in (
            ("traditional", traditional_config),
            ("self_rag", self_rag_config),
            ("enhanced", enhanced_config),
        )
    }
    assert all(r.answer for r in results.values())
    enhanced = results["enhanced"]
    assert enhanced.abstained is False
    lowered = enhanced.answer.lower()
    assert "sufficient" in lowered or "verif" in lowered or "claim" in lowered or "evidence" in lowered


def test_drop_unsupported_sentences_keeps_supported_only():
    good = [
        Claim(claim_id="1", text="Dropout reduces overfitting.", status=ClaimStatus.supported),
    ]
    bad = [
        Claim(claim_id="2", text="Titan has a nitrogen atmosphere.", status=ClaimStatus.unsupported),
    ]
    answer = "Dropout reduces overfitting [1]. Titan has a nitrogen atmosphere [2]."
    stripped, kept = drop_unsupported_sentences(answer, good, bad)
    assert "overfitting" in stripped.lower()
    assert "titan" not in stripped.lower()
    assert kept == good


def test_enhanced_answered_has_no_red_claims(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    result = runner.run("What is the main advantage of dropout?")
    if result.status is AnswerStatus.answered:
        assert all(claim.status is ClaimStatus.supported for claim in result.claims)


def test_partial_claims_do_not_produce_answered(demo_kb, llm):
    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    answer, status, claims = runner._enforce_supported_answer(
        "Dropout might reduce overfitting in some settings.",
        AnswerStatus.answered,
        [
            Claim(
                claim_id="c1",
                text="Dropout might reduce overfitting in some settings.",
                status=ClaimStatus.partially_supported,
            )
        ],
        TraceRecorder(),
    )
    assert status is AnswerStatus.insufficient_evidence
    assert INSUFFICIENT_ANSWER in answer
    assert claims


def test_claim_expand_retrieves_more_chunks_before_unsupported(demo_kb, llm):
    """Thin first-pass evidence can miss a claim; expand should pull supporting chunks."""

    from app.models.query import EvidenceItem
    from app.pipelines.common import UsageCounter

    runner = PipelineRunner(demo_kb, llm, enhanced_config())
    thin = [
        EvidenceItem(
            citation_id=1,
            chunk_id="thin-only",
            document_id="x",
            document_name="thin.md",
            page=1,
            text="Batch normalization stabilizes activations and allows higher learning rates.",
            fused_score=0.4,
            rank=1,
            source_quality=0.8,
        )
    ]
    answer = (
        "The main advantage of dropout is that it reduces overfitting by preventing "
        "co-adaptation of hidden units."
    )
    claims = [
        Claim(
            claim_id="c1",
            text="The main advantage of dropout is that it reduces overfitting by preventing co-adaptation of hidden units.",
            status=ClaimStatus.unsupported,
        )
    ]
    usage = UsageCounter()
    trace = TraceRecorder()
    out_answer, status, out_claims = runner._verify_and_correct(
        "What is the main advantage of dropout?",
        thin,
        answer,
        AnswerStatus.answered,
        claims,
        usage,
        trace,
        document_ids=None,
    )
    stages = {event.stage for event in trace.events}
    assert "claim_expand" in stages or any(
        c.status is ClaimStatus.supported for c in out_claims
    )
    # After expand + verify, dropout claim should be supportable from the demo corpus.
    assert any(c.status is ClaimStatus.supported for c in out_claims) or status is AnswerStatus.answered
    assert "overfitting" in out_answer.lower() or any(
        "overfitting" in (c.text or "").lower() for c in out_claims if c.status is ClaimStatus.supported
    )


def test_multi_aspect_answer_flags_uncovered_when_sources_thin(kb, llm):
    from app.services.ingestion.loaders import LoadedPage

    intro = (
        "Deep neural networks have achieved remarkable results across vision and language. "
        "This survey introduces recent architectures at a high level."
    )
    kb.ingest_pages("intro.md", [LoadedPage(page=1, text=intro)], rebuild=True)
    query = (
        "Explain the main mathematical concepts used in deep learning "
        "and how they support neural network training."
    )
    result = PipelineRunner(kb, llm, enhanced_config()).run(query)
    # May abstain or answer thinly; if answered, incompleteness must be explicit.
    if not result.abstained and result.answer:
        assert result.uncovered_aspects or "Not covered by the indexed sources" in result.answer


def test_multi_aspect_compose_preserves_citations_in_pipeline(kb, llm):
    from app.services.ingestion.loaders import LoadedPage

    math_text = (
        "Linear algebra is central to deep learning: vectors and matrices represent "
        "activations and weights, and matrix multiplication implements linear layers. "
        "Calculus supplies gradients via partial derivatives and the chain rule so "
        "backpropagation can update parameters during neural network training."
    )
    kb.ingest_pages("math.md", [LoadedPage(page=1, text=math_text)], rebuild=True)
    query = (
        "Explain the main mathematical concepts used in deep learning "
        "and how they support neural network training."
    )
    result = PipelineRunner(kb, llm, enhanced_config()).run(query)
    if result.abstained:
        return
    assert re.search(r"\[\d+\]", result.answer)
    lowered = result.answer.lower()
    assert "matrix" in lowered or "gradient" in lowered or "linear" in lowered
