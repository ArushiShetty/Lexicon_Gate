"""Evidence gate, query rewriting, claim verification, contradiction, confidence."""

from __future__ import annotations

from app.models.query import EvidenceItem, QueryAnalysis
from app.retrieval.adaptive import AdaptiveController
from app.services.llm.extractive_engine import extract_claims, rewrite_queries
from app.verification.claim_verifier import ClaimVerifier
from app.verification.confidence import score_confidence
from app.verification.contradiction import detect_contradictions
from app.verification.evidence_gate import EvidenceGate
from app.verification.hallucination import detect_hallucinations
from app.models.query import Claim, ClaimStatus


def _item(cid: str, text: str, doc: str, citation: int, quality: float = 0.8) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation,
        chunk_id=cid,
        document_id=doc,
        document_name=doc,
        page=1,
        text=text,
        fused_score=0.5,
        rank=citation,
        source_quality=quality,
    )


def test_evidence_gate_accepts_on_topic_passage(kb):
    gate = EvidenceGate(kb.embedder, settings=kb.settings, threshold=0.4)
    evidence = [
        _item("c1", "The main advantage of dropout is that it reduces overfitting by preventing co-adaptation of hidden units.", "handbook", 1, 0.9),
        _item("c2", "Dropout is applied only during training. At test time every unit is kept.", "handbook", 2, 0.9),
    ]
    outcome = gate.evaluate(
        "What is the main advantage of dropout?",
        evidence,
        attempt=1,
        attempts_remaining=2,
    )
    assert outcome.decision.sufficient is True
    assert outcome.decision.action in {"proceed", "conflict"}


def test_evidence_gate_abstains_on_unrelated_mixture(kb):
    gate = EvidenceGate(kb.embedder, settings=kb.settings, threshold=0.55)
    evidence = [
        _item("c1", "Dropout reduces overfitting by preventing co-adaptation of hidden units.", "handbook", 1),
        _item("c2", "Titan is Saturn's largest moon and has a dense nitrogen atmosphere.", "astro", 2),
    ]
    outcome = gate.evaluate(
        "How does dropout affect Titan's nitrogen atmosphere?",
        evidence,
        attempt=1,
        attempts_remaining=0,
    )
    assert outcome.decision.sufficient is False
    assert outcome.decision.action == "abstain"
    from app.verification.evidence_gate import decision_is_unrelated

    assert decision_is_unrelated(outcome.decision) is True


def test_query_rewriting_targets_gaps():
    rewrites = rewrite_queries(
        "What are the advantages of dropout?",
        uncovered_terms=["regularization", "overfitting"],
        entities=["dropout"],
        limit=3,
    )
    assert rewrites
    joined = " ".join(rewrites).lower()
    assert "dropout" in joined
    assert any(term in joined for term in ("overfitting", "regularization", "benefits", "performance"))


def test_definition_rewrite_uses_subject_noun_phrase():
    rewrites = rewrite_queries(
        "what is random recurrent neural network",
        question_type="definition",
        limit=3,
    )
    assert rewrites
    assert rewrites[0].lower() == "random recurrent neural network"
    assert "what is" not in rewrites[0].lower()


def test_adaptive_controller_caps_attempts(settings):
    controller = AdaptiveController(settings)
    analysis = QueryAnalysis(
        original_query="q",
        normalised_query="q",
        needs_retrieval=True,
        suggested_top_k=5,
    )
    from app.models.query import EvidenceGateDecision

    decision = EvidenceGateDecision(
        attempt=1,
        sufficient=False,
        evidence_score=0.2,
        threshold=0.55,
        action="expand",
        rationale="thin",
    )
    plan = controller.plan(decision, current_top_k=5, analysis=analysis, attempts_used=1, rewrites=[])
    assert plan.stop is False
    assert plan.next_top_k > 5
    exhausted = controller.plan(
        decision, current_top_k=12, analysis=analysis, attempts_used=settings.max_retrieval_attempts, rewrites=[]
    )
    assert exhausted.stop is True


def test_claim_extraction_and_verification():
    answer = (
        "The main advantage of dropout is that it reduces overfitting. [1] "
        "Dropout is applied only during training. [1] "
        "Dropout was invented on Titan in 1901."
    )
    raw = extract_claims(answer)
    assert len(raw) >= 2
    claims = [
        Claim(claim_id=f"c{i}", text=item["text"], supporting_citations=item.get("inline_citations", []))
        for i, item in enumerate(raw, start=1)
    ]
    evidence = [
        _item(
            "c1",
            "The main advantage of dropout is that it reduces overfitting by preventing co-adaptation. Dropout is applied only during training.",
            "handbook",
            1,
        )
    ]
    result = ClaimVerifier(idf={}).llm_judge_verify(
        claims, evidence, "What is the advantage of dropout?"
    )
    statuses = {c.status for c in result.claims}
    assert ClaimStatus.supported in statuses or ClaimStatus.partially_supported in statuses
    assert any(
        c.status in {ClaimStatus.unsupported, ClaimStatus.partially_supported}
        for c in result.claims
        if "1901" in c.text or "Titan" in c.text
    )


def test_contradiction_detection_on_opposing_sources():
    evidence = [
        _item(
            "c1",
            "Dropout reduces overfitting and improves generalization performance on held-out data.",
            "handbook",
            1,
            0.92,
        ),
        _item(
            "c2",
            "A small internal study found that dropout increased overfitting rather than reducing it.",
            "memo",
            2,
            0.4,
        ),
    ]
    pairs = detect_contradictions(evidence)
    assert pairs, "opposing dropout claims must be flagged"
    assert pairs[0].preferred_citation == 1


def test_hallucination_flags_ungrounded_numbers():
    claims = [
        Claim(claim_id="c1", text="Dropout uses a rate of 0.9.", status=ClaimStatus.unsupported)
    ]
    evidence = [_item("c1", "The typical dropout rate used in hidden layers is 0.5.", "handbook", 1)]
    report = detect_hallucinations("Dropout uses a rate of 0.9.", claims, evidence)
    assert report.hallucination_detected is True
    assert "0.9" in report.ungrounded_numeric_tokens or report.unsupported_claim_rate > 0


def test_confidence_is_evidence_derived_not_arbitrary():
    evidence = [
        _item("c1", "Dropout reduces overfitting.", "handbook", 1, 0.9),
    ]
    claims = [
        Claim(claim_id="c1", text="Dropout reduces overfitting.", status=ClaimStatus.supported, support_score=0.9)
    ]
    from app.models.query import EvidenceGateDecision

    gate = EvidenceGateDecision(
        sufficient=True, evidence_score=0.88, coverage=0.9, consistency=1.0, threshold=0.55
    )
    high = score_confidence(evidence, claims, gate, [])
    low = score_confidence([], [], None, [])
    assert high.confidence > 0.7
    assert low.confidence == 0.0
    assert "not a guarantee" in high.caveat.lower()


def test_hallucination_ignores_according_discourse_lead():
    from app.text_utils import extract_entities

    answer = "According to the sources, Dropout reduces overfitting by preventing co-adaptation."
    evidence = [
        _item(
            "c1",
            "The main advantage of dropout is that it reduces overfitting by preventing co-adaptation of hidden units.",
            "handbook",
            1,
        )
    ]
    entities = [e.lower() for e in extract_entities(answer)]
    assert "according" not in entities
    claims = [
        Claim(
            claim_id="c1",
            text="Dropout reduces overfitting by preventing co-adaptation.",
            status=ClaimStatus.supported,
            support_score=0.9,
        )
    ]
    report = detect_hallucinations(answer, claims, evidence)
    assert "According" not in report.ungrounded_entities
    assert not any("According" in flag for flag in report.flags)
    assert report.hallucination_detected is False


def test_hallucination_ignores_in_this_context_boilerplate():
    answer = "In this context, Dropout reduces overfitting."
    evidence = [
        _item("c1", "Dropout reduces overfitting by preventing co-adaptation.", "handbook", 1)
    ]
    claims = [
        Claim(
            claim_id="c1",
            text="Dropout reduces overfitting.",
            status=ClaimStatus.supported,
            support_score=0.9,
        )
    ]
    report = detect_hallucinations(answer, claims, evidence)
    assert not any(e.lower() in {"context", "contrast"} for e in report.ungrounded_entities)
    assert report.hallucination_detected is False


def test_hallucination_ignores_in_contrast_boilerplate():
    answer = "In contrast, Dropout reduces overfitting."
    evidence = [
        _item("c1", "Dropout reduces overfitting by preventing co-adaptation.", "handbook", 1)
    ]
    claims = [
        Claim(
            claim_id="c1",
            text="Dropout reduces overfitting.",
            status=ClaimStatus.supported,
            support_score=0.9,
        )
    ]
    report = detect_hallucinations(answer, claims, evidence)
    assert "Contrast" not in report.ungrounded_entities
    assert report.hallucination_detected is False


def test_paraphrase_claim_not_unsupported_by_lexical_verifier():
    claims = [
        Claim(
            claim_id="c1",
            text="Dropout's primary benefit is reducing overfitting and improving generalization on held-out data.",
        )
    ]
    evidence = [
        _item(
            "c1",
            "The main advantage of dropout is that it reduces overfitting by preventing co-adaptation of hidden units. Empirically, dropout improves generalization performance on held-out data.",
            "handbook",
            1,
        )
    ]
    result = ClaimVerifier(idf={}).llm_judge_verify(
        claims, evidence, "What is the main advantage of dropout?"
    )
    assert result.claims[0].status in (
        ClaimStatus.supported,
        ClaimStatus.partially_supported,
    )
    assert result.claims[0].status is not ClaimStatus.unsupported
