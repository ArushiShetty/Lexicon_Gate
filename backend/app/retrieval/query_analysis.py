"""Pre-retrieval query analysis.

Decides whether retrieval is needed at all (the Self-RAG ``Retrieve`` token),
classifies the question, and estimates how much evidence it will take to answer
— which the adaptive controller uses to pick the initial ``top_k``.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models.query import QueryAnalysis
from ..text_utils import (
    STOPWORDS,
    content_tokens,
    extract_entities,
    is_concept_definition_query,
    normalise_query_text,
    normalise_whitespace,
    question_aspects,
)

# Questions that need no corpus lookup: greetings, meta-questions, arithmetic.
_NO_RETRIEVAL_PATTERNS = (
    re.compile(r"^\s*(hi|hey|hello|thanks|thank you|good (morning|evening|afternoon))\b", re.I),
    re.compile(r"^\s*(who are you|what can you do|help|how do you work)\??\s*$", re.I),
    re.compile(r"^\s*(what is|calculate|compute)\s+[\d\s+\-*/^().]+\??\s*$", re.I),
)

_MULTI_HOP_MARKERS = (
    "and also", "as well as", "compare", "comparison", "versus", " vs ",
    "difference between", "both", "each of", "relationship between",
    "how does .* affect", "combined", "together with", "trade-off", "tradeoff",
    "and what", "and how", "as well as what",
)

_AMBIGUOUS_MARKERS = (
    "it", "this", "that", "they", "them", "these", "those", "the thing",
    "the system", "the model", "the approach", "better", "best", "good",
)

_QUESTION_TYPES = (
    # Prefer concept definitions only via is_concept_definition_query above;
    # keep explicit define/meaning cues here, not bare "what is the …".
    ("definition", (r"\bdefine\b", r"\bmeaning of\b", r"^what (?:does|do)\b.*\bmean\b")),
    ("explanatory", (r"^(?:please\s+)?(?:explain|describe|discuss|outline)\b", r"\bexplain\b")),
    ("causal", (r"^why\b", r"\bcause", r"\bbecause\b", r"\bleads? to\b", r"\breason\b")),
    ("procedural", (r"^how (?:do|does|to|can)\b", r"\bsteps?\b", r"\bprocedure\b", r"\bprocess\b")),
    ("comparative", (r"\bcompare\b", r"\bversus\b", r"\bvs\b", r"\bdifference\b", r"\bbetter than\b")),
    ("quantitative", (r"^how (?:many|much)\b", r"\bpercent", r"\bnumber of\b", r"\brate\b", r"\bscore\b")),
    ("temporal", (r"^when\b", r"\byear\b", r"\bdate\b")),
    ("enumerative", (r"^(?:list|name|which)\b", r"\ball (?:the )?\w+ (?:that|which)\b")),
)


def analyse_query(
    query: str,
    initial_top_k: int = 5,
    max_top_k: int = 12,
    corpus_is_empty: bool = False,
) -> QueryAnalysis:
    original = query.strip()
    normalised = normalise_query_text(normalise_whitespace(original))
    lowered = normalised.lower()

    needs_retrieval = True
    reason = "Document-grounded question: corpus lookup required."

    if corpus_is_empty:
        needs_retrieval = False
        reason = "No documents are indexed, so there is nothing to retrieve."
    else:
        for pattern in _NO_RETRIEVAL_PATTERNS:
            if pattern.search(normalised):
                needs_retrieval = False
                reason = "Conversational or self-contained request: retrieval adds no evidence."
                break
        if needs_retrieval and not content_tokens(normalised) and not extract_entities(normalised):
            needs_retrieval = False
            reason = "Query is too short to form a retrievable information need."

    question_type = "factual"
    if is_concept_definition_query(normalised):
        question_type = "definition"
    else:
        for label, patterns in _QUESTION_TYPES:
            if any(re.search(pattern, lowered) for pattern in patterns):
                question_type = label
                break

    aspects = question_aspects(normalised)
    is_multi_hop = _detect_multi_hop(lowered) or len(aspects) >= 2
    is_ambiguous = _detect_ambiguity(normalised, lowered)

    entities = extract_entities(normalised)
    keywords = [tok for tok in content_tokens(normalised) if tok not in STOPWORDS]

    suggested = initial_top_k
    if is_multi_hop:
        suggested = min(max_top_k, initial_top_k + 3)
    if question_type == "enumerative":
        suggested = min(max_top_k, suggested + 2)
    if question_type == "explanatory" and len(aspects) >= 2:
        suggested = min(max_top_k, suggested + 2)
    if question_type == "definition" and not is_multi_hop:
        suggested = max(3, initial_top_k - 1)

    return QueryAnalysis(
        original_query=original,
        normalised_query=normalised,
        needs_retrieval=needs_retrieval,
        retrieval_reason=reason,
        question_type=question_type,
        is_multi_hop=is_multi_hop,
        is_ambiguous=is_ambiguous,
        entities=entities[:8],
        keywords=keywords[:12],
        suggested_top_k=suggested,
    )


def _detect_multi_hop(lowered: str) -> bool:
    for marker in _MULTI_HOP_MARKERS:
        if marker.startswith("how does"):
            if re.search(marker, lowered):
                return True
        elif marker in lowered:
            return True
    # Two or more distinct question clauses also implies multiple hops.
    if lowered.count("?") > 1:
        return True
    if len(re.findall(r"\band\b", lowered)) >= 2:
        return True
    return False


def _detect_ambiguity(normalised: str, lowered: str) -> bool:
    tokens = lowered.split()
    if len(tokens) <= 4 and not extract_entities(normalised):
        return True
    # A leading pronoun with no named anchor cannot be resolved from the query.
    leading = tokens[1] if len(tokens) > 1 else ""
    if leading in {"it", "this", "that", "they", "them"} and not extract_entities(normalised):
        return True
    vague = sum(1 for marker in _AMBIGUOUS_MARKERS if f" {marker} " in f" {lowered} ")
    return vague >= 2 and not extract_entities(normalised)


def uncovered_query_terms(
    query: str,
    evidence_texts: Sequence[str],
    idf: dict[str, float] | None = None,
    limit: int = 6,
) -> list[str]:
    """Query terms that the retrieved evidence never mentions.

    These are the concrete gaps that query rewriting should target.
    """

    from ..text_utils import stem

    covered: set[str] = set()
    for text in evidence_texts:
        covered.update(stem(tok) for tok in content_tokens(text))

    missing: list[tuple[float, str]] = []
    for token in dict.fromkeys(content_tokens(query)):
        if stem(token) in covered:
            continue
        weight = (idf or {}).get(token, 1.0)
        missing.append((weight, token))

    missing.sort(key=lambda pair: -pair[0])
    return [token for _weight, token in missing[:limit]]
