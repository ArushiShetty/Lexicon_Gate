"""Rank fusion strategies for combining dense and lexical result lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(slots=True)
class RankedList:
    """One retriever's output: chunk ids in descending relevance order."""

    name: str
    chunk_ids: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(slots=True)
class FusedResult:
    chunk_id: str
    fused_score: float
    rank: int
    contributions: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    lists: Sequence[RankedList],
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Reciprocal Rank Fusion.

    ``score(d) = Σ_l weight_l / (k + rank_l(d))``

    Rank-based rather than score-based, so cosine similarities and BM25 scores
    can be combined without needing a shared scale.
    """

    totals: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}

    for ranked in lists:
        for position, chunk_id in enumerate(ranked.chunk_ids, start=1):
            contribution = ranked.weight / (k + position)
            totals[chunk_id] = totals.get(chunk_id, 0.0) + contribution
            contributions.setdefault(chunk_id, {})[ranked.name] = contribution
            ranks.setdefault(chunk_id, {})[ranked.name] = position

    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        FusedResult(
            chunk_id=chunk_id,
            fused_score=score,
            rank=rank,
            contributions=contributions.get(chunk_id, {}),
            ranks=ranks.get(chunk_id, {}),
        )
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def weighted_score_fusion(
    lists: Sequence[RankedList],
    top_k: int | None = None,
) -> list[FusedResult]:
    """Min-max normalise each list's scores, then take a weighted sum.

    Provided as an alternative to RRF so the fusion strategy itself can be
    varied in the ablation study.
    """

    normalised: list[tuple[RankedList, dict[str, float]]] = []
    for ranked in lists:
        values = [ranked.scores.get(cid, 0.0) for cid in ranked.chunk_ids]
        if not values:
            normalised.append((ranked, {}))
            continue
        low, high = min(values), max(values)
        spread = (high - low) or 1.0
        normalised.append(
            (
                ranked,
                {
                    cid: (ranked.scores.get(cid, 0.0) - low) / spread
                    for cid in ranked.chunk_ids
                },
            )
        )

    totals: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for ranked, scores in normalised:
        for chunk_id, value in scores.items():
            weighted = ranked.weight * value
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weighted
            contributions.setdefault(chunk_id, {})[ranked.name] = weighted

    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        FusedResult(
            chunk_id=chunk_id,
            fused_score=score,
            rank=rank,
            contributions=contributions.get(chunk_id, {}),
        )
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def normalise_scores(values: Iterable[float]) -> list[float]:
    values = list(values)
    if not values:
        return []
    low, high = min(values), max(values)
    spread = (high - low) or 1.0
    return [(v - low) / spread for v in values]
