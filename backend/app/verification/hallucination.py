"""Hallucination detection over a generated answer.

A hallucination here is an *assertion that is not entailed by the retrieved
evidence*. We never treat fluency as a signal. The detector looks at:

* unsupported / contradicted claims (from the claim verifier)
* numeric tokens in the answer that never appear in the evidence
* named entities in the answer that never appear in the evidence
* a lead sentence whose content is not covered by any supporting sentence

Severity is ``none`` / ``low`` / ``medium`` / ``high`` so the dashboard can
show a flag without collapsing everything into a boolean.
"""

from __future__ import annotations

import re

from ..models.query import Claim, ClaimStatus, EvidenceItem, HallucinationReport
from ..services.llm.extractive_engine import strip_citations
from ..text_utils import extract_entities, extract_numbers, is_content_entity, stem_set

_ATTRIBUTION_PREFIX_RE = re.compile(
    r"^(?:according\s+to\s+(?:the\s+)?(?:retrieved\s+)?sources?,?\s*"
    r"|based\s+on\s+(?:the\s+)?(?:retrieved\s+)?(?:evidence|sources?),?\s*"
    r"|from\s+the\s+(?:retrieved\s+)?(?:evidence|sources?),?\s*"
    r"|in\s+this\s+context,?\s*"
    r"|in\s+contrast,?\s*"
    r"|on\s+the\s+other\s+hand,?\s*"
    r"|by\s+contrast,?\s*)",
    re.I,
)


def detect_hallucinations(
    answer: str,
    claims: list[Claim],
    evidence: list[EvidenceItem],
) -> HallucinationReport:
    flags: list[str] = []
    n_claims = len(claims)
    n_unsupported = sum(1 for c in claims if c.status is ClaimStatus.unsupported)
    n_contradicted = sum(1 for c in claims if c.status is ClaimStatus.contradicted)
    if n_claims:
        unsupported_rate = n_unsupported / n_claims
        contradicted_rate = n_contradicted / n_claims
    else:
        unsupported_rate = 0.0
        contradicted_rate = 0.0

    if n_unsupported:
        flags.append(f"{n_unsupported} claim(s) unsupported by retrieved evidence")
    if n_contradicted:
        flags.append(f"{n_contradicted} claim(s) contradicted by retrieved evidence")

    evidence_text = " ".join(item.text for item in evidence)
    evidence_stems = stem_set(evidence_text)
    evidence_numbers = set(extract_numbers(evidence_text))
    evidence_entities = {e.lower() for e in extract_entities(evidence_text)}

    answer_clean = _strip_attribution_prefixes(strip_citations(answer))
    ungrounded_numbers = [
        n for n in extract_numbers(answer_clean) if n not in evidence_numbers
    ]
    ungrounded_entities = [
        e
        for e in extract_entities(answer_clean)
        if is_content_entity(e)
        and e.lower() not in evidence_entities
        and e.lower() not in evidence_stems
    ]

    if ungrounded_numbers:
        flags.append(
            "Answer contains numeric values absent from the retrieved evidence: "
            + ", ".join(ungrounded_numbers[:5])
        )
    if ungrounded_entities:
        flags.append(
            "Answer mentions entities absent from the retrieved evidence: "
            + ", ".join(ungrounded_entities[:5])
        )

    # A lead sentence that shares almost no content with the rest of the answer
    # (and is unsupported) is a typical abstractive hallucination.
    sentences = [c.text for c in claims]
    if sentences and claims and claims[0].status in (
        ClaimStatus.unsupported,
        ClaimStatus.contradicted,
    ):
        flags.append("Opening assertion is not supported by retrieved evidence")

    severity = _severity(unsupported_rate, contradicted_rate, flags)
    detected = severity in {"medium", "high"} or n_contradicted > 0 or unsupported_rate >= 0.34

    return HallucinationReport(
        hallucination_detected=detected,
        unsupported_claim_rate=round(unsupported_rate, 4),
        contradicted_claim_rate=round(contradicted_rate, 4),
        ungrounded_numeric_tokens=ungrounded_numbers[:8],
        ungrounded_entities=ungrounded_entities[:8],
        flags=flags,
        severity=severity,
    )


def _strip_attribution_prefixes(text: str) -> str:
    """Remove boilerplate attribution leads before entity scanning."""

    remaining = (text or "").strip()
    # Strip repeatedly in case of multiple short clauses.
    for _ in range(3):
        updated = _ATTRIBUTION_PREFIX_RE.sub("", remaining, count=1).lstrip(" ,;—-")
        if updated == remaining:
            break
        remaining = updated
    return remaining


def _severity(
    unsupported_rate: float, contradicted_rate: float, flags: list[str]
) -> str:
    if contradicted_rate >= 0.34 or unsupported_rate >= 0.6:
        return "high"
    if contradicted_rate > 0 or unsupported_rate >= 0.34 or len(flags) >= 2:
        return "medium"
    if unsupported_rate > 0 or flags:
        return "low"
    return "none"
