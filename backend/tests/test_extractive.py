"""Extractive lead and definition scoring for concept questions."""

from __future__ import annotations

import re

from app.services.ingestion.loaders import LoadedPage
from app.services.llm.extractive_engine import compose_answer, declarative_stem
from app.text_utils import (
    first_question_clause,
    is_concept_definition_query,
    normalise_query_text,
)
from app.models.query import AnswerStatus
from app.pipelines.runner import PipelineRunner, enhanced_config


QUERY = "what is deeplearning how is it related to machine learning"

MRI_TEXT = (
    "A combined deeplearning and deformable-model approach to fully automatic "
    "segmentation of the left ventricle in cardiac MRI. The method uses imaging "
    "from the lung image database consortium (LIDC) and reports ventricle contours."
)

DL_TEXT = (
    "Deep learning is a subset of machine learning that uses multi-layer neural "
    "networks. It is related to machine learning as a specialised form of "
    "representation learning trained on large datasets."
)


def test_normalise_splits_glued_compounds():
    assert "deep learning" in normalise_query_text("what is deeplearning")
    assert "machine learning" in normalise_query_text("machinelearning")
    assert is_concept_definition_query(QUERY)


def test_declarative_stem_uses_first_clause_only():
    normalised = normalise_query_text(QUERY)
    stem, mode = declarative_stem(normalised)
    assert mode == "copula"
    assert "deep learning is" in stem.lower()
    assert "how is it related" not in stem.lower()
    assert "ventricle" not in first_question_clause(normalised).lower()


def test_compose_prefers_definition_over_mri_title():
    evidence = [
        {"citation_id": 1, "text": MRI_TEXT, "fused_score": 0.9, "relevance_score": 0.9},
        {"citation_id": 2, "text": DL_TEXT, "fused_score": 0.5, "relevance_score": 0.5},
    ]
    result = compose_answer(QUERY, evidence, question_type="definition")
    answer = result["answer"].lower()
    assert "subset" in answer or "machine learning" in answer
    assert "ventricle" not in answer
    assert "lidc" not in answer
    assert not answer.startswith("deeplearning how is it related")


def test_enhanced_mixed_corpus_answers_definition(kb, llm):
    kb.ingest_pages("mri.md", [LoadedPage(page=1, text=MRI_TEXT)], rebuild=False)
    kb.ingest_pages("dl.md", [LoadedPage(page=1, text=DL_TEXT)], rebuild=True)
    result = PipelineRunner(kb, llm, enhanced_config()).run(QUERY)
    lowered = result.answer.lower()
    assert result.abstained is False
    assert "ventricle" not in lowered
    assert "lidc" not in lowered
    assert "subset" in lowered or "machine learning" in lowered


def test_enhanced_mri_only_abstains_on_what_is_deeplearning(kb, llm):
    kb.ingest_pages("mri.md", [LoadedPage(page=1, text=MRI_TEXT)], rebuild=True)
    result = PipelineRunner(kb, llm, enhanced_config()).run(QUERY)
    assert result.abstained is True
    assert result.status is AnswerStatus.insufficient_evidence
    assert "ventricle" not in result.answer.lower()
    assert "how is it related to machine learning is a combined" not in result.answer.lower()


RESNET_TEXT = (
    "Deeper neural networks are more difficult to train. We present a residual "
    "learning framework to ease the training of networks that are substantially "
    "deeper than those used previously."
)


def test_compose_answers_method_intro_for_what_is_residual_learning():
    evidence = [
        {"citation_id": 1, "text": RESNET_TEXT, "fused_score": 0.9, "relevance_score": 0.9},
    ]
    result = compose_answer(
        "What is residual learning?",
        evidence,
        question_type="definition",
    )
    lowered = result["answer"].lower()
    assert "residual learning framework" in lowered
    assert '{"answer"' not in result["answer"]


def test_enhanced_answers_residual_learning_from_paper(kb, llm):
    kb.ingest_pages("resnet.md", [LoadedPage(page=1, text=RESNET_TEXT)], rebuild=True)
    result = PipelineRunner(kb, llm, enhanced_config()).run("What is residual learning?")
    lowered = result.answer.lower()
    assert result.status is not AnswerStatus.insufficient_evidence
    assert "residual learning framework" in lowered
    assert '{"answer"' not in result.answer


def test_extractive_provider_empty_compose_is_not_json():
    from app.services.llm.base import LLMRequest, LLMTask
    from app.services.llm.extractive_provider import ExtractiveProvider

    provider = ExtractiveProvider()
    response = provider.generate(
        LLMRequest(
            task=LLMTask.answer,
            payload={
                "query": "What is residual learning?",
                "evidence": [],
                "question_type": "definition",
            },
        )
    )
    assert response.structured["answer"] == ""
    assert response.text == ""
    assert "{" not in response.text


LLM_USAGE_TEXT = (
    "Modern chatbots rely on LLMs to generate fluent replies. Fine-tuning an LLM "
    "on domain data improves factual accuracy for specialised assistants."
)

LLM_DEFINITION_TEXT = (
    "A Large Language Model (LLM) is a neural network trained on vast text corpora "
    "to predict the next token. LLMs underpin conversational AI systems."
)


def test_compose_answers_what_is_llm_from_acronym_expansion():
    evidence = [
        {
            "citation_id": 1,
            "text": LLM_USAGE_TEXT,
            "fused_score": 0.95,
            "relevance_score": 0.95,
        },
        {
            "citation_id": 2,
            "text": LLM_DEFINITION_TEXT,
            "fused_score": 0.4,
            "relevance_score": 0.4,
        },
    ]
    result = compose_answer("What is LLM?", evidence, question_type="definition")
    lowered = result["answer"].lower()
    assert "large language model" in lowered
    assert "neural network" in lowered or "trained" in lowered


def test_enhanced_answers_what_is_llm(kb, llm):
    kb.ingest_pages("llm-usage.md", [LoadedPage(page=1, text=LLM_USAGE_TEXT)], rebuild=False)
    kb.ingest_pages(
        "llm-def.md", [LoadedPage(page=1, text=LLM_DEFINITION_TEXT)], rebuild=True
    )
    result = PipelineRunner(kb, llm, enhanced_config()).run("What is LLM?")
    lowered = result.answer.lower()
    assert result.abstained is False
    assert result.status is not AnswerStatus.insufficient_evidence
    assert "large language model" in lowered


MATH_CONCEPTS_TEXT = (
    "Linear algebra is central to deep learning: vectors and matrices represent "
    "activations and weights, and matrix multiplication implements linear layers. "
    "Calculus supplies gradients via partial derivatives and the chain rule so "
    "backpropagation can update parameters. Optimization methods such as gradient "
    "descent minimise a loss function during neural network training."
)

TRAINING_ROLE_TEXT = (
    "Gradient-based optimization supports neural network training by repeatedly "
    "adjusting weights in the direction that reduces the training loss."
)

INTRO_ONLY_TEXT = (
    "Deep neural networks have achieved remarkable results across vision and language. "
    "This paper surveys recent architectures."
)


def test_compose_multi_aspect_emits_sections_and_citations():
    query = (
        "Explain the main mathematical concepts used in deep learning "
        "and how they support neural network training."
    )
    evidence = [
        {
            "citation_id": 1,
            "text": MATH_CONCEPTS_TEXT,
            "fused_score": 0.9,
            "relevance_score": 0.9,
            "chunk_id": "c1",
            "document_name": "math.md",
        },
        {
            "citation_id": 2,
            "text": TRAINING_ROLE_TEXT,
            "fused_score": 0.85,
            "relevance_score": 0.85,
            "chunk_id": "c2",
            "document_name": "train.md",
        },
    ]
    result = compose_answer(query, evidence, question_type="explanatory")
    answer = result["answer"]
    assert re.search(r"\[\d+\]", answer)
    lowered = answer.lower()
    assert "1." in answer or "linear" in lowered or "gradient" in lowered
    assert "matrix" in lowered or "gradient" in lowered or "loss" in lowered


def test_compose_skips_figure_asides_and_repeated_lead():
    evidence = [
        {
            "citation_id": 1,
            "text": (
                "Image as you can see the architecture has three blocks. "
                "Residual learning is a framework that adds shortcut connections "
                "so deeper networks can be trained."
            ),
            "fused_score": 0.9,
            "relevance_score": 0.9,
        }
    ]
    result = compose_answer(
        "What is residual learning?",
        evidence,
        question_type="definition",
    )
    answer = result["answer"]
    lowered = answer.lower()
    assert "shortcut" in lowered or "framework" in lowered
    assert not lowered.startswith("image")
    assert "as you can see" not in lowered
    # The same claim should not be stated twice.
    assert lowered.count("shortcut") <= 1
    assert lowered.count("residual learning is a framework") <= 1
    from app.text_utils import uncovered_aspects

    query = (
        "Explain the main mathematical concepts used in deep learning "
        "and how they support neural network training."
    )
    thin = "Deep neural networks have achieved remarkable results across vision."
    gaps = uncovered_aspects(query, thin)
    assert gaps
    assert any("mathematical" in g.lower() or "train" in g.lower() for g in gaps)

    off_topic = "Saturn is a gas giant with prominent rings."
    assert uncovered_aspects(query, off_topic)
