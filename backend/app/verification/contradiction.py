"""Cross-source contradiction detection.

Distinct from claim-level contradiction (claim vs. evidence). Here we look for
two retrieved *sources* that assert opposite things about the same topic, so
the system can surface the disagreement instead of silently picking a winner.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models.query import ContradictionPair, EvidenceItem
from ..text_utils import (
    clamp,
    numeric_conflict,
    polarity_conflict,
    split_sentences,
    stem_set,
    truncate,
)


def detect_contradictions(
    evidence: list[EvidenceItem],
    settings: Settings | None = None,
    max_pairs: int = 4,
) -> list[ContradictionPair]:
    settings = settings or get_settings()
    if len(evidence) < 2:
        return []

    pairs: list[ContradictionPair] = []
    seen: set[tuple[int, int]] = set()

    for i, item_a in enumerate(evidence):
        sentences_a = split_sentences(item_a.text)[:6]
        for j, item_b in enumerate(evidence):
            if j <= i:
                continue
            if item_a.document_id == item_b.document_id:
                continue
            key = (min(item_a.citation_id, item_b.citation_id), max(item_a.citation_id, item_b.citation_id))
            if key in seen:
                continue

            best = _best_conflict(sentences_a, split_sentences(item_b.text)[:6])
            if best is None or best[0] < settings.contradiction_threshold:
                continue
            seen.add(key)

            score, sent_a, sent_b, polarity, numeric, overlap = best
            preferred, reason = _prefer(item_a, item_b)
            pairs.append(
                ContradictionPair(
                    claim_a=truncate(sent_a, 240),
                    claim_b=truncate(sent_b, 240),
                    citation_a=item_a.citation_id,
                    citation_b=item_b.citation_id,
                    document_a=item_a.document_name,
                    document_b=item_b.document_name,
                    similarity=round(overlap, 4),
                    polarity_conflict=round(polarity, 4),
                    numeric_conflict=round(numeric, 4),
                    score=round(score, 4),
                    explanation=(
                        f"[{item_a.citation_id}] {item_a.document_name} states that "
                        f"{truncate(sent_a, 140)} "
                        f"[{item_b.citation_id}] {item_b.document_name} states that "
                        f"{truncate(sent_b, 140)}"
                    ),
                    preferred_citation=preferred.citation_id if preferred else None,
                    preferred_reason=reason,
                )
            )

    pairs.sort(key=lambda p: -p.score)
    return pairs[:max_pairs]


def _best_conflict(
    sentences_a: list[str], sentences_b: list[str]
) -> tuple[float, str, str, float, float, float] | None:
    best: tuple[float, str, str, float, float, float] | None = None
    for sent_a in sentences_a:
        stems_a = stem_set(sent_a)
        if len(stems_a) < 3:
            continue
        for sent_b in sentences_b:
            stems_b = stem_set(sent_b)
            if len(stems_b) < 3:
                continue
            overlap = len(stems_a & stems_b) / min(len(stems_a), len(stems_b))
            if len(stems_a & stems_b) < 2 or overlap < 0.25:
                continue
            polarity = polarity_conflict(sent_a, sent_b)
            numeric = numeric_conflict(sent_a, sent_b)
            conflict = max(polarity, numeric)
            if conflict <= 0:
                continue
            score = clamp(0.4 * min(1.0, overlap / 0.4) + 0.6 * conflict)
            if best is None or score > best[0]:
                best = (score, sent_a, sent_b, polarity, numeric, overlap)
    return best


def _prefer(
    item_a: EvidenceItem, item_b: EvidenceItem
) -> tuple[EvidenceItem | None, str | None]:
    """Rank conflicting sources by source quality, then by evidence score."""

    qa = item_a.source_quality
    qb = item_b.source_quality
    if abs(qa - qb) >= 0.15:
        winner = item_a if qa > qb else item_b
        return winner, (
            f"{winner.document_name} is ranked higher by source quality "
            f"({max(qa, qb):.2f} vs {min(qa, qb):.2f})."
        )
    if abs(item_a.evidence_score - item_b.evidence_score) >= 0.1:
        winner = item_a if item_a.evidence_score > item_b.evidence_score else item_b
        return winner, (
            f"{winner.document_name} has a higher evidence score "
            f"({max(item_a.evidence_score, item_b.evidence_score):.2f})."
        )
    return None, "Sources of similar quality; both views are shown."
