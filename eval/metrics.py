"""Metrics for the labeled paper eval (eval/test_set.json).

Retrieval scores are only defined when ``relevant_chunk_ids`` is non-empty.
Hallucination vs gold is only defined when ``expected_facts`` is non-empty.
Citation precision and passage relevance are never auto-filled.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

PENDING_MANUAL_REVIEW = "pending_manual_review"
NOT_APPLICABLE = "n/a"


def _stem_set(text: str) -> set[str]:
    from app.text_utils import stem_set

    return stem_set(text)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float | None:
    gold = [cid for cid in relevant if cid]
    if not gold:
        return None
    if k <= 0:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float | None:
    gold = [cid for cid in relevant if cid]
    if not gold:
        return None
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in set(gold)) / len(top)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float | None:
    gold = set(cid for cid in relevant if cid)
    if not gold:
        return None
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def claim_supported_by_expected_facts(claim_text: str, expected_facts: Sequence[str]) -> bool | None:
    """True if the claim overlaps a labeled fact. None if there are no gold facts."""

    facts = [f for f in expected_facts if str(f).strip()]
    if not facts:
        return None
    claim_stems = _stem_set(claim_text)
    if not claim_stems:
        return False
    for fact in facts:
        fact_stems = _stem_set(str(fact))
        if not fact_stems:
            continue
        need_fact = max(1, math.ceil(0.6 * len(fact_stems)))
        need_claim = max(1, math.ceil(0.6 * len(claim_stems)))
        overlap = len(claim_stems & fact_stems)
        if overlap >= min(need_fact, need_claim):
            return True
    return False


def hallucination_rate_for_answer(
    claim_texts: Sequence[str], expected_facts: Sequence[str]
) -> float | None:
    """Unsupported claims / total claims, vs labeled facts (not retrieved chunks)."""

    facts = [f for f in expected_facts if str(f).strip()]
    if not facts:
        return None
    claims = [c for c in claim_texts if str(c).strip()]
    if not claims:
        return None
    unsupported = 0
    for text in claims:
        supported = claim_supported_by_expected_facts(text, facts)
        if supported is not True:
            unsupported += 1
    return unsupported / len(claims)


def mean_skip_none(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def format_metric(value: float | None, *, pending: bool = False) -> str:
    if pending:
        return PENDING_MANUAL_REVIEW
    if value is None:
        return NOT_APPLICABLE
    return f"{value:.4f}"
