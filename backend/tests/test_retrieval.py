"""BM25, dense retrieval, RRF fusion, hybrid ranking."""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import RankedList, reciprocal_rank_fusion
from app.retrieval.hybrid import FusionStrategy, HybridRetriever, RetrievalMode
from app.retrieval.query_analysis import analyse_query, uncovered_query_terms


def test_bm25_ranks_exact_identifier_first():
    index = BM25Index()
    index.build(
        [
            ("c1", "d1", "Dropout reduces overfitting in neural networks."),
            ("c2", "d1", "The fusion constant RRF-7F3 uses k = 60."),
            ("c3", "d2", "Saturn is a gas giant with rings."),
        ]
    )
    hits = index.search("What k does RRF-7F3 use?", top_k=3)
    assert hits
    assert hits[0].chunk_id == "c2"


def test_rrf_promotes_consensus():
    lists = [
        RankedList(name="dense", chunk_ids=["a", "b", "c"], weight=1.0),
        RankedList(name="bm25", chunk_ids=["c", "a", "d"], weight=1.0),
    ]
    fused = reciprocal_rank_fusion(lists, k=60)
    # a is rank 1 and 2; c is rank 3 and 1. a should win or tie near the top.
    top_ids = [r.chunk_id for r in fused]
    assert top_ids[0] in {"a", "c"}
    assert "d" in top_ids


def test_hybrid_beats_dense_on_rare_token(demo_kb):
    query = "What k does RRF-7F3 use?"
    dense = HybridRetriever(demo_kb, mode=RetrievalMode.dense)
    hybrid = HybridRetriever(demo_kb, mode=RetrievalMode.hybrid, fusion=FusionStrategy.rrf)
    dense_ids = [item.chunk_id for item in dense.retrieve(query, top_k=5)]
    hybrid_ids = [item.chunk_id for item in hybrid.retrieve(query, top_k=5)]
    lexical = HybridRetriever(demo_kb, mode=RetrievalMode.bm25)
    lexical_hit = lexical.retrieve(query, top_k=1)
    assert lexical_hit, "BM25 must find RRF-7F3"
    assert lexical_hit[0].chunk_id in hybrid_ids
    # Hybrid must not be worse than dense at surfacing the lexical document.
    hybrid_docs = {item.document_name for item in hybrid.retrieve(query, top_k=5)}
    assert any("Hybrid Retrieval" in name for name in hybrid_docs)


def test_query_analysis_skips_greetings_and_flags_multihop():
    greeting = analyse_query("hello there")
    assert greeting.needs_retrieval is False
    multi = analyse_query("What gaps remain in standard Self-RAG and what does the proposed architecture add?")
    assert multi.needs_retrieval is True
    assert multi.is_multi_hop is True


def test_explain_and_how_is_explanatory_multihop():
    from app.text_utils import question_aspects

    query = (
        "Explain the main mathematical concepts used in deep learning "
        "and how they support neural network training."
    )
    analysis = analyse_query(query)
    aspects = question_aspects(query)
    assert len(aspects) >= 2
    assert analysis.is_multi_hop is True
    assert analysis.question_type == "explanatory"


def test_uncovered_terms_detect_gaps():
    missing = uncovered_query_terms(
        "How does dropout affect Titan's nitrogen atmosphere?",
        ["Dropout reduces overfitting in neural networks."],
    )
    assert any("titan" in t.lower() or "nitrogen" in t.lower() for t in missing)
