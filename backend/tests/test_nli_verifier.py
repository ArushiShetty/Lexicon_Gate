"""NLI claim-verification tests. The live transformer test is skipped without weights."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models.query import Claim, ClaimStatus, EvidenceItem
from app.verification.claim_verifier import ClaimVerifier
from app.verification.nli_verifier import (
    nli_verify,
    reset_nli_runtime,
    set_nli_score_override,
    verify_claim,
)

# Hand-written (claim, evidence, expected_label) triples.
NLI_TRIPLES = [
    (
        "Dropout reduces overfitting.",
        "Dropout is a regularization technique that reduces overfitting by preventing co-adaptation of neurons.",
        "supported",
    ),
    (
        "Dropout was invented on Titan in 1901.",
        "Dropout is applied only during training of neural networks.",
        "unsupported",
    ),
    (
        "Dropout increases overfitting on held-out data.",
        "Dropout reduces overfitting and improves generalization on held-out data.",
        "contradicted",
    ),
]


@pytest.fixture(autouse=True)
def _reset_nli():
    reset_nli_runtime()
    yield
    reset_nli_runtime()


def _item(chunk_id: str, text: str, citation_id: int) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id="doc",
        document_name="handbook.pdf",
        text=text,
        fused_score=0.9,
        rank=citation_id,
    )


def _override_from_triples(claim: str, evidence: str) -> tuple[str, float]:
    mapping = {
        "supported": ("entailment", 0.93),
        "unsupported": ("neutral", 0.88),
        "contradicted": ("contradiction", 0.91),
    }
    for text, chunk, label in NLI_TRIPLES:
        if claim == text and evidence == chunk:
            return mapping[label]
    return "neutral", 0.40


def test_verify_claim_maps_mnli_labels():
    set_nli_score_override(_override_from_triples)
    for claim, evidence, expected in NLI_TRIPLES:
        out = verify_claim(claim, evidence)
        assert out["label"] == expected
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["confidence"] >= 0.8


def test_nli_verify_stores_label_confidence_and_chunk_id():
    set_nli_score_override(_override_from_triples)
    claim, chunk, expected = NLI_TRIPLES[0]
    claims = [Claim(claim_id="c1", text=claim)]
    evidence = [_item("chunk-a", chunk, 1)]
    result = nli_verify(claims, evidence, "What does dropout do?")
    verified = result.claims[0]
    assert verified.nli_label == expected
    assert verified.source_chunk_id == "chunk-a"
    assert verified.nli_confidence is not None and verified.nli_confidence >= 0.8
    assert verified.status is ClaimStatus.supported
    assert verified.verifier == "nli"
    assert 1 in verified.supporting_citations


def test_nli_verify_contradicted_and_unsupported():
    set_nli_score_override(_override_from_triples)
    contra_claim, contra_chunk, _ = NLI_TRIPLES[2]
    miss_claim, miss_chunk, _ = NLI_TRIPLES[1]
    claims = [
        Claim(claim_id="c1", text=contra_claim),
        Claim(claim_id="c2", text=miss_claim),
    ]
    evidence = [
        _item("chunk-contra", contra_chunk, 1),
        _item("chunk-miss", miss_chunk, 2),
    ]
    result = nli_verify(claims, evidence)
    by_id = {c.claim_id: c for c in result.claims}
    assert by_id["c1"].status is ClaimStatus.contradicted
    assert by_id["c1"].nli_label == "contradicted"
    assert by_id["c1"].source_chunk_id == "chunk-contra"
    assert by_id["c2"].status is ClaimStatus.unsupported
    assert by_id["c2"].nli_label == "unsupported"


def test_nli_verbatim_excerpt_is_supported_despite_neutral():
    set_nli_score_override(lambda claim, chunk: ("neutral", 0.99))
    excerpt = (
        "We present a residual learning framework to ease the training of networks "
        "that are substantially deeper than those used previously."
    )
    claims = [Claim(claim_id="c1", text=excerpt)]
    evidence = [_item("chunk-res", excerpt + " This result won ILSVRC 2015.", 1)]
    result = nli_verify(claims, evidence, "What is residual learning?")
    verified = result.claims[0]
    assert verified.status is ClaimStatus.supported
    assert verified.nli_label == "supported"
    assert verified.verifier == "verbatim"
    assert verified.source_chunk_id == "chunk-res"


def test_pipeline_uses_nli_when_llm_judge_disabled(tmp_path):
    set_nli_score_override(_override_from_triples)
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        index_dir=tmp_path / "i",
        results_dir=tmp_path / "r",
        demo_dir=tmp_path / "d",
        use_llm_judge=False,
        nli_allow_download=False,
    )
    claim, chunk, _ = NLI_TRIPLES[0]
    claims = [Claim(claim_id="c1", text=claim)]
    evidence = [_item("chunk-a", chunk, 1)]
    result = ClaimVerifier(settings=settings, idf={}).verify(claims, evidence)
    assert result.claims[0].verifier == "nli"
    assert result.claims[0].status is ClaimStatus.supported


def test_llm_judge_flag_keeps_lexical_baseline(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        index_dir=tmp_path / "i",
        results_dir=tmp_path / "r",
        demo_dir=tmp_path / "d",
        use_llm_judge=True,
    )
    claims = [
        Claim(
            claim_id="c1",
            text="The main advantage of dropout is that it reduces overfitting.",
        )
    ]
    evidence = [
        _item(
            "c1",
            "The main advantage of dropout is that it reduces overfitting by preventing co-adaptation.",
            1,
        )
    ]
    result = ClaimVerifier(settings=settings, idf={}).verify(
        claims, evidence, "What is the advantage of dropout?"
    )
    assert result.claims[0].verifier == "lexical_baseline"
    assert result.claims[0].status in {
        ClaimStatus.supported,
        ClaimStatus.partially_supported,
    }
    assert result.claims[0].source_chunk_id == "c1"


def test_live_nli_checkpoint_on_handwritten_triples():
    from app.verification import nli_verifier as mod

    reset_nli_runtime()
    if not mod.nli_is_available():
        pytest.skip(
            "transformers/torch or cached NLI weights missing "
            "(install backend/requirements-nli.txt and allow a one-time download)"
        )
    failures: list[str] = []
    for claim, evidence, expected in NLI_TRIPLES:
        out = verify_claim(claim, evidence)
        if out["label"] != expected:
            failures.append(f"{expected=} got={out}")
    assert not failures, failures
