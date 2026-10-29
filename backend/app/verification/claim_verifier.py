"""Claim-to-evidence verification.

For every claim extracted from a draft answer we search the retrieved evidence
for supporting and contradicting spans, then classify the claim as

* ``SUPPORTED``          – a passage entails the claim
* ``PARTIALLY_SUPPORTED`` – key terms overlap but the assertion is incomplete
* ``UNSUPPORTED``        – no passage covers the claim
* ``CONTRADICTED``       – a passage asserts the opposite

The historical classifier is lexical + polarity + numeric (``llm_judge_verify``).
It remains the baseline/fallback. The default path is a pretrained NLI
cross-encoder (see ``nli_verifier.verify_claim``), gated by ``use_llm_judge``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..models.query import Claim, ClaimStatus, EvidenceItem
from ..text_utils import (
    build_idf,
    clamp,
    content_tokens,
    extract_entities,
    extract_numbers,
    idf_weighted_containment,
    numeric_conflict,
    polarity_conflict,
    split_sentences,
    stem,
    stem_set,
    truncate,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ClaimVerificationResult:
    claims: list[Claim]
    support_rate: float
    n_supported: int
    n_partial: int
    n_unsupported: int
    n_contradicted: int
    important_unsupported: list[Claim]


class ClaimVerifier:
    def __init__(self, settings: Settings | None = None, idf: dict[str, float] | None = None) -> None:
        self.settings = settings or get_settings()
        self.idf = idf or {}

    def verify(
        self, claims: list[Claim], evidence: list[EvidenceItem], query: str = ""
    ) -> ClaimVerificationResult:
        if self.settings.use_llm_judge:
            return self.llm_judge_verify(claims, evidence, query)
        try:
            from .nli_verifier import nli_is_available, nli_verify

            if nli_is_available():
                return nli_verify(claims, evidence, query, settings=self.settings)
        except Exception:
            logger.exception("NLI claim verification failed; using lexical baseline")
        return self.llm_judge_verify(claims, evidence, query)

    def llm_judge_verify(
        self, claims: list[Claim], evidence: list[EvidenceItem], query: str = ""
    ) -> ClaimVerificationResult:
        """Lexical / heuristic baseline (not an LLM call). Kept as fallback."""

        if not claims:
            return ClaimVerificationResult(
                claims=[],
                support_rate=1.0 if evidence else 0.0,
                n_supported=0,
                n_partial=0,
                n_unsupported=0,
                n_contradicted=0,
                important_unsupported=[],
            )

        corpus_idf = self.idf or build_idf([item.text for item in evidence] + [query])
        stemmed_idf = {stem(k): v for k, v in corpus_idf.items()}
        query_stems = stem_set(query)

        for claim in claims:
            self._verify_one(claim, evidence, stemmed_idf, query_stems)

        n_supported = sum(1 for c in claims if c.status is ClaimStatus.supported)
        n_partial = sum(1 for c in claims if c.status is ClaimStatus.partially_supported)
        n_unsupported = sum(1 for c in claims if c.status is ClaimStatus.unsupported)
        n_contradicted = sum(1 for c in claims if c.status is ClaimStatus.contradicted)
        # Partial credit for partial support: half a claim is better than none.
        support_rate = (
            (n_supported + 0.5 * n_partial) / len(claims) if claims else 0.0
        )
        important = [
            c
            for c in claims
            if c.status in (ClaimStatus.unsupported, ClaimStatus.contradicted)
            and _is_important(c, query_stems)
        ]
        return ClaimVerificationResult(
            claims=claims,
            support_rate=round(support_rate, 4),
            n_supported=n_supported,
            n_partial=n_partial,
            n_unsupported=n_unsupported,
            n_contradicted=n_contradicted,
            important_unsupported=important,
        )

    def _verify_one(
        self,
        claim: Claim,
        evidence: list[EvidenceItem],
        stemmed_idf: dict[str, float],
        query_stems: set[str],
    ) -> None:
        claim_tokens = [stem(tok) for tok in content_tokens(claim.text)]
        claim_stems = set(claim_tokens)
        claim_entities = {e.lower() for e in extract_entities(claim.text)}
        claim_numbers = set(extract_numbers(claim.text))

        best_support = 0.0
        best_span: str | None = None
        best_citation: int | None = None
        supporting: list[int] = []
        contradicting: list[int] = []
        best_contradiction = 0.0

        for item in evidence:
            span, support, contradiction = _best_span_for_claim(
                claim_text=claim.text,
                claim_tokens=claim_tokens,
                claim_stems=claim_stems,
                claim_entities=claim_entities,
                claim_numbers=claim_numbers,
                passage=item.text,
                stemmed_idf=stemmed_idf,
            )
            if support > best_support:
                best_support = support
                best_span = span
                best_citation = item.citation_id
            if support >= self.settings.claim_partial_threshold:
                if item.citation_id not in supporting:
                    supporting.append(item.citation_id)
            if contradiction >= self.settings.contradiction_threshold:
                if item.citation_id not in contradicting:
                    contradicting.append(item.citation_id)
                best_contradiction = max(best_contradiction, contradiction)

        claim.support_score = round(best_support, 4)
        claim.contradiction_score = round(best_contradiction, 4)
        claim.best_evidence_span = truncate(best_span, 280) if best_span else None
        claim.supporting_citations = supporting or (
            [best_citation] if best_citation is not None and best_support >= self.settings.claim_partial_threshold else []
        )
        claim.contradicting_citations = contradicting
        claim.verifier = "lexical_baseline"
        if best_citation is not None:
            for item in evidence:
                if item.citation_id == best_citation:
                    claim.source_chunk_id = item.chunk_id
                    break

        if best_contradiction >= self.settings.contradiction_threshold and best_contradiction > best_support:
            claim.status = ClaimStatus.contradicted
            claim.rationale = (
                f"A retrieved passage contradicts this claim "
                f"(contradiction {best_contradiction:.2f} > support {best_support:.2f})."
            )
        elif best_support >= self.settings.claim_support_threshold:
            claim.status = ClaimStatus.supported
            claim.rationale = f"Supported by passage [{best_citation}] (score {best_support:.2f})."
        elif best_support >= self.settings.claim_partial_threshold:
            claim.status = ClaimStatus.partially_supported
            claim.rationale = (
                f"Partially supported by passage [{best_citation}] "
                f"(score {best_support:.2f}); some details are missing."
            )
        else:
            claim.status = ClaimStatus.unsupported
            claim.rationale = (
                f"No retrieved passage covers this claim (best support {best_support:.2f})."
            )


def _best_span_for_claim(
    claim_text: str,
    claim_tokens: list[str],
    claim_stems: set[str],
    claim_entities: set[str],
    claim_numbers: set[str],
    passage: str,
    stemmed_idf: dict[str, float],
) -> tuple[str, float, float]:
    sentences = split_sentences(passage, min_chars=12) or [passage]
    best_support = 0.0
    best_contradiction = 0.0
    best_span = sentences[0] if sentences else passage

    for sentence in sentences:
        sentence_stems = stem_set(sentence)
        if not sentence_stems:
            continue
        lexical = idf_weighted_containment(claim_tokens, sentence_stems, stemmed_idf)
        sentence_entities = {e.lower() for e in extract_entities(sentence)}
        entity = (
            len(claim_entities & sentence_entities) / len(claim_entities)
            if claim_entities
            else 0.0
        )
        sentence_numbers = set(extract_numbers(sentence))
        numeric = (
            len(claim_numbers & sentence_numbers) / len(claim_numbers)
            if claim_numbers
            else 0.0
        )
        # Prefer IDF/stem containment over brittle entity string matches so
        # paraphrases stay supported when meaning overlaps the passage.
        support = clamp(0.78 * lexical + 0.10 * entity + 0.12 * numeric)
        if claim_numbers and not (claim_numbers & sentence_numbers):
            support *= 0.55

        polarity = polarity_conflict(claim_text, sentence)
        numeric_c = numeric_conflict(claim_text, sentence)
        overlap = (
            len(claim_stems & sentence_stems) / len(claim_stems) if claim_stems else 0.0
        )
        contradiction = overlap * max(polarity, numeric_c)

        if support > best_support:
            best_support = support
            best_span = sentence
        best_contradiction = max(best_contradiction, contradiction)

    return best_span, best_support, best_contradiction


def _is_important(claim: Claim, query_stems: set[str]) -> bool:
    """A claim is important if it shares content with the question or carries a number."""

    claim_stems = stem_set(claim.text)
    if query_stems and len(claim_stems & query_stems) >= 2:
        return True
    if extract_numbers(claim.text):
        return True
    return len(content_tokens(claim.text)) >= 8
