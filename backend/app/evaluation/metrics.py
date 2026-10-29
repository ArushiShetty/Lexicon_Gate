"""Metric implementations.

Every function here is a pure function of (prediction, gold). None of them
read hard-coded leaderboard numbers. The dashboard displays whatever these
functions return after a real run.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

from ..models.query import ClaimStatus, PipelineResult
from ..text_utils import clamp, content_tokens, stem, stem_set

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the retrieved sources provide conflicting information\.\s*)", re.I
)


def normalise_answer(text: str) -> str:
    text = _ANSWER_PREFIX_RE.sub("", text or "")
    tokens = content_tokens(text)
    return " ".join(stem(t) for t in tokens)


def exact_match(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    return 1.0 if normalise_answer(prediction) == normalise_answer(reference) else 0.0


def keypoint_recall(prediction: str, keypoints: Sequence[str]) -> float:
    """Fraction of gold keypoints whose content tokens appear in the prediction."""

    if not keypoints:
        return 1.0 if prediction.strip() else 0.0
    pred_stems = stem_set(prediction)
    hits = 0
    for point in keypoints:
        point_stems = stem_set(point)
        if not point_stems:
            continue
        if len(point_stems & pred_stems) >= max(1, math.ceil(0.6 * len(point_stems))):
            hits += 1
    return hits / len(keypoints)


def answer_correctness(
    prediction: str,
    reference: str,
    keypoints: Sequence[str],
    should_abstain: bool,
    abstained: bool,
) -> float:
    """Task-level correctness.

    * Unanswerable questions: 1 iff the system abstained, else 0.
    * Answerable questions: keypoint recall (with a small exact-match bonus).
      An incorrect abstention scores 0.
    """

    if should_abstain:
        return 1.0 if abstained else 0.0
    if abstained:
        return 0.0
    recall = keypoint_recall(prediction, keypoints)
    em = exact_match(prediction, reference)
    return clamp(0.85 * recall + 0.15 * em)


def faithfulness(result: PipelineResult) -> float:
    """Fraction of generated claims supported by retrieved evidence.

    Falls back to token containment against the evidence if no claims were
    extracted (Traditional RAG does not run the claim extractor at generation
    time, so evaluation extracts them post-hoc — see the harness).
    """

    if result.abstained:
        return 1.0
    claims = result.claims
    if claims:
        n = len(claims)
        supported = sum(
            1 for c in claims if c.status is ClaimStatus.supported
        ) + 0.5 * sum(1 for c in claims if c.status is ClaimStatus.partially_supported)
        return clamp(supported / n)
    evidence_stems = stem_set(" ".join(item.text for item in result.evidence))
    answer_stems = stem_set(result.answer)
    if not answer_stems:
        return 0.0
    return clamp(len(answer_stems & evidence_stems) / len(answer_stems))


def citation_accuracy(result: PipelineResult) -> float:
    """Fraction of cited passages that actually share content with the answer."""

    if not result.evidence:
        return 1.0 if result.abstained else 0.0
    answer_stems = stem_set(result.answer)
    if not answer_stems:
        return 0.0
    cited = [item for item in result.evidence if f"[{item.citation_id}]" in result.answer]
    pool = cited or result.evidence[:3]
    hits = 0
    for item in pool:
        item_stems = stem_set(item.text)
        if not item_stems:
            continue
        overlap = len(answer_stems & item_stems) / min(len(answer_stems), len(item_stems))
        if overlap >= 0.12:
            hits += 1
    return hits / len(pool) if pool else 0.0


def citation_presence(result: PipelineResult) -> float:
    if result.abstained:
        return 1.0
    if not result.evidence:
        return 0.0
    return 1.0 if re.search(r"\[\d+\]", result.answer or "") else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold = set(relevant)
    if not gold:
        return 1.0 if not retrieved else 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in gold) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 1.0
    top = set(list(retrieved)[:k])
    return len(top & gold) / len(gold)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    gold = set(relevant)
    if not gold:
        return 1.0 if not retrieved else 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    gold = set(relevant)
    if k <= 0:
        return 0.0
    if not gold:
        return 1.0 if not retrieved else 0.0

    def dcg(ranking: Sequence[str]) -> float:
        score = 0.0
        for i, cid in enumerate(ranking[:k], start=1):
            rel = 1.0 if cid in gold else 0.0
            score += rel / math.log2(i + 1)
        return score

    ideal = dcg(list(gold)[:k])
    if ideal <= 0:
        return 0.0
    return dcg(list(retrieved)) / ideal


def context_relevance(retrieved_texts: Sequence[str], question: str) -> float:
    if not retrieved_texts:
        return 0.0
    q = stem_set(question)
    if not q:
        return 0.0
    scores = []
    for text in retrieved_texts:
        stems = stem_set(text)
        scores.append(len(q & stems) / len(q) if stems else 0.0)
    return sum(scores) / len(scores)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * (p / 100.0)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return float(ordered[lo])
    weight = index - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test. Returns ``(t_statistic, p_value)``."""

    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    try:
        from scipy import stats  # noqa: PLC0415

        result = stats.ttest_rel(list(a), list(b), nan_policy="omit")
        t = float(result.statistic) if result.statistic is not None else 0.0
        p = float(result.pvalue) if result.pvalue is not None else 1.0
        if math.isnan(t) or math.isnan(p):
            return 0.0, 1.0
        return t, p
    except Exception:
        diffs = [x - y for x, y in zip(a, b)]
        n = len(diffs)
        mu = sum(diffs) / n
        var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
        if var <= 0:
            return (0.0, 1.0) if abs(mu) < 1e-12 else (math.inf if mu > 0 else -math.inf, 0.0)
        t = mu / math.sqrt(var / n)
        return float(t), 1.0


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mu = sum(diffs) / n
    var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
    if var <= 0:
        return 0.0
    return mu / math.sqrt(var)
