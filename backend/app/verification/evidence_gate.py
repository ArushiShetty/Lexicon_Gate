"""The Evidence Gate.

Sits between retrieval and generation and answers one question: *is this
evidence good enough to answer from?* Traditional RAG never asks it, which is
the single largest source of confident-but-ungrounded answers.

Scoring is two-tiered.

**Per chunk** we combine six signals — semantic relevance, IDF-weighted keyword
overlap, entity overlap, source quality, retrieval score and a contradiction
indicator — into a chunk evidence score.

**Per attempt** we aggregate those into

``evidence_score = w_r·relevance + w_q·source_quality + w_c·coverage + w_s·consistency``

where ``relevance`` deliberately favours the best chunk over the mean (one
excellent passage can answer a narrow question), ``coverage`` is the
IDF-weighted share of query terms the evidence actually mentions, and
``consistency`` is the inverse of internal disagreement.

The gate then emits an **action**: proceed, expand (retrieve deeper), rewrite
(the query is the problem, not the depth), flag a conflict, or abstain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from ..models.query import (
    EvidenceGateDecision,
    EvidenceItem,
    EvidenceScoreBreakdown,
    QueryAnalysis,
)
from ..retrieval.query_analysis import uncovered_query_terms
from ..services.embeddings.base import EmbeddingProvider
from ..text_utils import (
    clamp,
    content_tokens,
    definition_subject,
    extract_entities,
    idf_weighted_containment,
    is_definitional_sentence,
    is_concept_definition_query,
    numeric_conflict,
    polarity_conflict,
    split_sentences,
    stem,
    stem_set,
    term_bag,
)

logger = logging.getLogger(__name__)

GateAction = str  # "proceed" | "expand" | "rewrite" | "conflict" | "abstain"

QUESTION_FUNCTION_WORDS = {
    "affect", "effect", "using", "used", "make", "makes", "take", "takes",
    "give", "gives", "come", "comes", "main", "proposed", "standard",
    "typical", "does", "doing", "list", "relate", "remaining", "matter",
    "cost", "improve", "improves", "explain", "describe", "happen",
    "happens", "mean", "means", "into", "what", "when", "why", "how",
    "which", "where", "who", "many", "much",
}


@dataclass(slots=True)
class GateOutcome:
    decision: EvidenceGateDecision
    scored_evidence: list[EvidenceItem]
    kept_evidence: list[EvidenceItem]


class EvidenceGate:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
        threshold: float | None = None,
        idf: dict[str, float] | None = None,
        corpus_doc_terms: dict[str, set[str]] | None = None,
    ) -> None:
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.threshold = (
            threshold if threshold is not None else self.settings.evidence_threshold
        )
        self.idf = idf or {}
        self.corpus_doc_terms = corpus_doc_terms or {}

    # ------------------------------------------------------------------ main
    def evaluate(
        self,
        query: str,
        evidence: Sequence[EvidenceItem],
        analysis: QueryAnalysis | None = None,
        attempt: int = 1,
        attempts_remaining: int = 0,
        query_used: str | None = None,
    ) -> GateOutcome:
        settings = self.settings
        query_used = query_used or query

        if not evidence:
            decision = EvidenceGateDecision(
                attempt=attempt,
                query_used=query_used,
                sufficient=False,
                evidence_score=0.0,
                threshold=self.threshold,
                n_candidates=0,
                action="rewrite" if attempts_remaining > 0 else "abstain",
                rationale="Retrieval returned no candidate passages.",
            )
            return GateOutcome(decision=decision, scored_evidence=[], kept_evidence=[])

        scored = self._score_chunks(query, list(evidence))
        relevances = [item.relevance_score for item in scored]
        mean_relevance = float(np.mean(relevances)) if relevances else 0.0
        max_relevance = float(np.max(relevances)) if relevances else 0.0

        coverage, uncovered = self._coverage(query, scored)
        split_terms = self._topic_split(query, scored)
        if split_terms:
            coverage *= 0.45
        consistency, pair_conflict = self._consistency(scored)
        source_quality = self._source_quality(scored)

        relevance_component = 0.6 * max_relevance + 0.4 * mean_relevance

        weights = np.array(
            [
                settings.ev_weight_relevance,
                settings.ev_weight_source_quality,
                settings.ev_weight_coverage,
                settings.ev_weight_consistency,
            ],
            dtype=float,
        )
        weights = weights / (weights.sum() or 1.0)
        components = np.array(
            [relevance_component, source_quality, coverage, consistency], dtype=float
        )
        evidence_score = float(np.dot(weights, components))

        kept = [
            item
            for item in scored
            if item.relevance_score >= settings.chunk_relevance_floor
        ]
        n_relevant = len(kept)
        if not kept:
            kept = scored[:1]

        has_definitional = None
        if analysis and (
            analysis.question_type == "definition" or is_concept_definition_query(query)
        ):
            has_definitional = _passages_define_subject(query, kept)

        sufficient, action, rationale = self._decide(
            evidence_score=evidence_score,
            max_relevance=max_relevance,
            coverage=coverage,
            consistency=consistency,
            n_relevant=n_relevant,
            n_candidates=len(scored),
            attempts_remaining=attempts_remaining,
            pair_conflict=pair_conflict,
            analysis=analysis,
            uncovered=uncovered,
            split_terms=split_terms,
            has_definitional=has_definitional,
        )

        decision = EvidenceGateDecision(
            attempt=attempt,
            query_used=query_used,
            sufficient=sufficient,
            evidence_score=round(evidence_score, 4),
            threshold=self.threshold,
            mean_relevance=round(mean_relevance, 4),
            max_relevance=round(max_relevance, 4),
            coverage=round(coverage, 4),
            consistency=round(consistency, 4),
            source_quality=round(source_quality, 4),
            n_relevant_chunks=n_relevant,
            n_candidates=len(scored),
            uncovered_terms=uncovered,
            action=action,
            rationale=rationale,
        )
        return GateOutcome(decision=decision, scored_evidence=scored, kept_evidence=kept)

    # -------------------------------------------------------- chunk scoring
    def _score_chunks(self, query: str, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        texts = [item.text for item in evidence]
        semantic = self._semantic_relevance(query, texts)

        query_tokens = [stem(tok) for tok in content_tokens(query)]
        stemmed_idf = {stem(term): value for term, value in self.idf.items()}
        query_entities = {e.lower() for e in extract_entities(query)}

        fused = [item.fused_score for item in evidence]
        max_fused = max(fused) if fused else 0.0
        min_fused = min(fused) if fused else 0.0
        spread = (max_fused - min_fused) or 1.0

        contradiction_indicators = self._chunk_contradiction_indicators(evidence)

        for index, item in enumerate(evidence):
            chunk_stems = stem_set(item.text)
            keyword_overlap = idf_weighted_containment(query_tokens, chunk_stems, stemmed_idf)
            chunk_entities = {e.lower() for e in extract_entities(item.text)}
            entity_overlap = (
                len(query_entities & chunk_entities) / len(query_entities)
                if query_entities
                else 0.0
            )
            source_quality = _source_quality_for(item)
            retrieval_score = (item.fused_score - min_fused) / spread
            contradiction = contradiction_indicators[index]

            relevance = clamp(
                0.55 * semantic[index] + 0.32 * keyword_overlap + 0.13 * entity_overlap
            )

            chunk_evidence = clamp(
                0.58 * relevance
                + 0.14 * source_quality
                + 0.18 * retrieval_score
                - 0.12 * contradiction
                + 0.10 * min(1.0, len(content_tokens(item.text)) / 60.0)
            )

            item.relevance_score = round(relevance, 4)
            item.evidence_score = round(chunk_evidence, 4)
            item.breakdown = EvidenceScoreBreakdown(
                semantic_relevance=round(float(semantic[index]), 4),
                keyword_overlap=round(keyword_overlap, 4),
                entity_overlap=round(entity_overlap, 4),
                source_quality=round(source_quality, 4),
                retrieval_score=round(retrieval_score, 4),
                contradiction_indicator=round(contradiction, 4),
                chunk_evidence_score=round(chunk_evidence, 4),
            )

        evidence.sort(key=lambda i: (-i.evidence_score, i.rank))
        for rank, item in enumerate(evidence, start=1):
            item.rank = rank
        return evidence

    def _semantic_relevance(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        try:
            vectors = self.embedder.encode([query, *texts])
        except Exception as exc:  # pragma: no cover - provider failure
            logger.warning("Embedding failed during gating (%s); lexical only", exc)
            return [0.0] * len(texts)
        query_vector = vectors[0]
        similarities = vectors[1:] @ query_vector
        # Map cosine into [0, 1]; negative similarity carries no evidence value.
        return [float(clamp((value + 0.05) / 1.05)) for value in similarities]

    # ---------------------------------------------------- aggregate signals
    def _coverage(
        self, query: str, evidence: Sequence[EvidenceItem]
    ) -> tuple[float, list[str]]:
        """How well the evidence jointly covers the query.

        Union coverage across unrelated documents is a known false friend
        ("dropout" in one paper, "Titan" in another looks fully covered).
        We therefore blend three views:

        * union of all passages (upper bound)
        * best single passage
        * best single *document* (legitimate multi-hop inside one source)
        """

        query_tokens = [stem(tok) for tok in content_tokens(query)]
        if not query_tokens:
            return 0.0, []
        stemmed_idf = {stem(term): value for term, value in self.idf.items()}

        union: set[str] = set()
        per_doc: dict[str, set[str]] = {}
        best_single = 0.0
        for item in evidence:
            stems = stem_set(item.text)
            union |= stems
            per_doc.setdefault(item.document_id, set()).update(stems)
            best_single = max(
                best_single, idf_weighted_containment(query_tokens, stems, stemmed_idf)
            )
        union_cov = idf_weighted_containment(query_tokens, union, stemmed_idf)
        best_doc = max(
            (idf_weighted_containment(query_tokens, stems, stemmed_idf) for stems in per_doc.values()),
            default=0.0,
        )
        coverage = 0.25 * union_cov + 0.35 * best_single + 0.40 * best_doc
        uncovered = uncovered_query_terms(
            query, [item.text for item in evidence], self.idf, limit=6
        )
        return clamp(coverage), uncovered

    def _topic_split(self, query: str, evidence: Sequence[EvidenceItem]) -> list[str]:
        """Return distinctive query terms that never co-occur in one *corpus* document.

        A relational question is unanswerable when those terms live in disjoint
        sources. Checking the whole document (not just the retrieved slices)
        avoids false abstentions when the right chunk simply wasn't ranked yet.
        """

        terms = [
            tok for tok in dict.fromkeys(content_tokens(query))
            if tok not in QUESTION_FUNCTION_WORDS and len(tok) > 3
        ]
        if len(terms) < 2:
            return []

        per_doc = self.corpus_doc_terms or {}
        if not per_doc:
            for item in evidence:
                per_doc.setdefault(item.document_id, set()).update(
                    stem(t) for t in content_tokens(item.text)
                )
        if not per_doc:
            return []

        stemmed_terms = [(t, stem(t)) for t in terms]
        present = [
            (t, s) for t, s in stemmed_terms
            if any(s in stems or t in stems or t[:5] in stems for stems in per_doc.values())
        ]
        missing: list[str] = []
        for i, (t1, s1) in enumerate(present):
            for t2, s2 in present[i + 1 :]:
                together = any(
                    (s1 in stems or t1 in stems or t1[:5] in stems)
                    and (s2 in stems or t2 in stems or t2[:5] in stems)
                    for stems in per_doc.values()
                )
                if not together:
                    missing.extend((t1, t2))
        seen: set[str] = set()
        ordered: list[str] = []
        for term in missing:
            if term not in seen:
                seen.add(term)
                ordered.append(term)
        return ordered[:6]

    def _consistency(self, evidence: Sequence[EvidenceItem]) -> tuple[float, float]:
        """1 − internal disagreement among the top passages."""

        top = list(evidence)[:5]
        if len(top) < 2:
            return 1.0, 0.0

        conflicts: list[float] = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                if top[i].document_id == top[j].document_id:
                    continue
                conflicts.append(self._pair_conflict(top[i].text, top[j].text))
        if not conflicts:
            return 1.0, 0.0
        mean_conflict = float(np.mean(conflicts))
        peak_conflict = float(np.max(conflicts))
        # Weight the peak: one flat contradiction matters more than mild
        # average disagreement across many pairs.
        conflict = clamp(0.4 * mean_conflict + 0.6 * peak_conflict)
        return clamp(1.0 - conflict), conflict

    def _pair_conflict(self, text_a: str, text_b: str) -> float:
        sentences_a = split_sentences(text_a)[:6]
        sentences_b = split_sentences(text_b)[:6]
        if not sentences_a or not sentences_b:
            return 0.0

        best = 0.0
        for sentence_a in sentences_a:
            stems_a = stem_set(sentence_a)
            if len(stems_a) < 3:
                continue
            for sentence_b in sentences_b:
                stems_b = stem_set(sentence_b)
                if len(stems_b) < 3:
                    continue
                overlap = len(stems_a & stems_b) / min(len(stems_a), len(stems_b))
                if len(stems_a & stems_b) < 2 or overlap < 0.25:
                    continue  # different topics cannot contradict each other
                polarity = polarity_conflict(sentence_a, sentence_b)
                numeric = numeric_conflict(sentence_a, sentence_b)
                conflict = max(polarity, numeric)
                if conflict <= 0:
                    continue
                best = max(best, clamp(0.4 * min(1.0, overlap / 0.4) + 0.6 * conflict))
        return clamp(best)

    def _chunk_contradiction_indicators(
        self, evidence: Sequence[EvidenceItem]
    ) -> list[float]:
        indicators: list[float] = []
        for index, item in enumerate(evidence):
            others = [
                other
                for j, other in enumerate(evidence)
                if j != index and other.document_id != item.document_id
            ]
            if not others:
                indicators.append(0.0)
                continue
            indicators.append(
                max((self._pair_conflict(item.text, other.text) for other in others[:4]),
                    default=0.0)
            )
        return indicators

    @staticmethod
    def _source_quality(evidence: Sequence[EvidenceItem]) -> float:
        if not evidence:
            return 0.0
        # Weight by rank: the quality of the passages we will actually cite.
        weights = [1.0 / (1.0 + 0.3 * index) for index in range(len(evidence))]
        values = [_source_quality_for(item) for item in evidence]
        total = sum(weights) or 1.0
        return clamp(sum(w * v for w, v in zip(weights, values)) / total)

    # -------------------------------------------------------------- decision
    def _decide(
        self,
        evidence_score: float,
        max_relevance: float,
        coverage: float,
        consistency: float,
        n_relevant: int,
        n_candidates: int,
        attempts_remaining: int,
        pair_conflict: float,
        analysis: QueryAnalysis | None,
        uncovered: list[str] | None = None,
        split_terms: list[str] | None = None,
        has_definitional: bool | None = None,
    ) -> tuple[bool, GateAction, str]:
        settings = self.settings
        min_support = settings.min_supporting_chunks
        uncovered = uncovered or []
        split_terms = split_terms or []
        fallback = "rewrite" if attempts_remaining > 0 else "abstain"

        if split_terms:
            return (
                False,
                fallback,
                "Query terms "
                + ", ".join(split_terms[:4])
                + " never co-occur in a single source, so the retrieved set cannot "
                "support a joint answer.",
            )

        if has_definitional is False:
            return (
                False,
                fallback,
                "No definitional passage for the query subject; "
                "abstaining rather than answering from application papers.",
            )

        distinctive_uncovered = [
            t
            for t in uncovered
            if len(t) > 3 and t not in QUESTION_FUNCTION_WORDS
        ]
        if distinctive_uncovered:
            corpus_vocab: set[str] = set()
            for stems in self.corpus_doc_terms.values():
                corpus_vocab |= stems
            missing_from_corpus = [
                t for t in distinctive_uncovered
                if stem(t) not in corpus_vocab and t not in corpus_vocab and t[:5] not in corpus_vocab
            ]
            specific_missing = [
                t for t in missing_from_corpus
                if any(ch.isdigit() for ch in t) or "-" in t or len(t) >= 8
            ]
            if len(missing_from_corpus) >= 2 or specific_missing:
                return (
                    False,
                    fallback,
                    "Query terms absent from the corpus: " + ", ".join(missing_from_corpus) + ".",
                )
            if attempts_remaining > 0:
                return (
                    False,
                    "expand",
                    "Query terms not covered by the current evidence: "
                    + ", ".join(distinctive_uncovered)
                    + "; retrieving more passages.",
                )

        strong_single = max_relevance >= 0.62 and coverage >= 0.7

        if evidence_score >= self.threshold and (n_relevant >= min_support or strong_single):
            if pair_conflict >= settings.contradiction_threshold:
                return (
                    True,
                    "conflict",
                    f"Evidence is sufficient (score {evidence_score:.2f}) but the sources "
                    f"disagree (consistency {consistency:.2f}); surfacing the conflict.",
                )
            return (
                True,
                "proceed",
                f"Evidence score {evidence_score:.2f} ≥ threshold {self.threshold:.2f} with "
                f"{n_relevant} relevant passage(s); proceeding to generation.",
            )

        if attempts_remaining <= 0:
            return (
                False,
                "abstain",
                f"Evidence score {evidence_score:.2f} stayed below threshold "
                f"{self.threshold:.2f} after exhausting retrieval attempts; "
                f"abstaining instead of answering unsupported.",
            )

        # Below the floor the retrieved passages are simply about something else:
        # retrieving deeper into the same ranking will not help, so reformulate.
        if evidence_score < settings.evidence_low_threshold or max_relevance < 0.3:
            return (
                False,
                "rewrite",
                f"Evidence score {evidence_score:.2f} is far below threshold and best "
                f"passage relevance is only {max_relevance:.2f}; rewriting the query.",
            )

        if coverage < 0.6 and (analysis is None or not analysis.is_ambiguous):
            return (
                False,
                "expand",
                f"Query-term coverage is only {coverage:.2f}; retrieving more passages "
                f"to fill the gaps.",
            )

        if n_relevant < min_support:
            if attempts_remaining > 0 and n_candidates > n_relevant:
                return (
                    False,
                    "expand",
                    f"Only {n_relevant} passage(s) cleared the relevance floor "
                    f"({settings.chunk_relevance_floor:.2f}); increasing retrieval depth.",
                )
            if evidence_score >= self.threshold and n_relevant >= 1:
                return (
                    True,
                    "proceed",
                    f"Evidence score {evidence_score:.2f} ≥ threshold {self.threshold:.2f} "
                    f"with {n_relevant} relevant passage(s) and no further unique hits; "
                    f"proceeding to generation.",
                )
            return (
                False,
                fallback,
                f"Only {n_relevant} passage(s) cleared the relevance floor "
                f"({settings.chunk_relevance_floor:.2f}); increasing retrieval depth."
                if attempts_remaining > 0
                else (
                    f"Only {n_relevant} passage(s) cleared the relevance floor "
                    f"after exhausting retrieval attempts."
                ),
            )

        return (
            False,
            "rewrite",
            f"Evidence score {evidence_score:.2f} is short of threshold "
            f"{self.threshold:.2f}; reformulating the query.",
        )


def decision_is_unrelated(decision: EvidenceGateDecision | None) -> bool:
    """True when the gate abstained because the question is not about the corpus."""

    if decision is None or decision.action != "abstain":
        return False
    rationale = (decision.rationale or "").lower()
    if "absent from the corpus" in rationale or "never co-occur" in rationale:
        return True
    if "definitional passage" in rationale:
        return True
    if decision.max_relevance < 0.3:
        return True
    if decision.uncovered_terms and decision.evidence_score < 0.35:
        return True
    return False


def _passages_define_subject(query: str, items: Sequence[EvidenceItem]) -> bool:
    subject = definition_subject(query)
    subject_stems = stem_set(subject)
    if not subject_stems:
        return True
    for item in items:
        if is_definitional_sentence(item.text, subject_stems, subject=subject):
            return True
        for sentence in split_sentences(item.text, min_chars=12):
            if is_definitional_sentence(sentence, subject_stems, subject=subject):
                return True
    return False


def _source_quality_for(item: EvidenceItem) -> float:
    """Resolve a source-quality prior for a retrieved passage.

    Prefers the value recorded at ingestion time; falls back to a structural
    heuristic (documents with real section headings and prose-like passages are
    more trustworthy evidence than fragmentary extractions).
    """

    recorded = getattr(item, "source_quality", None)
    if recorded is not None and recorded > 0:
        return clamp(float(recorded))

    structural = 0.55
    if item.section:
        structural += 0.1
    tokens = len(content_tokens(item.text))
    if tokens >= 40:
        structural += 0.1
    if tokens < 15:
        structural -= 0.15
    return clamp(structural)
