"""Dependency-free linguistic helpers shared across the pipeline.

The evidence gate, claim verifier and contradiction detector all need the same
primitives: tokenisation, sentence segmentation, negation/polarity detection,
numeric extraction and lexical similarity. Keeping them in one place means the
components stay consistent with one another.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all also am an and any are aren't as at be
because been before being below between both but by can cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he her here hers herself him himself
his how however i if in into is isn't it its itself just let's me more most
must mustn't my myself no nor not of off on once only or other ought our ours
ourselves out over own same shan't she should shouldn't so some such than that
the their theirs them themselves then there these they this those through to
too under until up upon very was wasn't we were weren't what when where which
while who whom why with won't would wouldn't you your yours yourself yourselves
what's whats does doe using used use may might shall will can't
""".split()
)

# Sentence-initial / connective words that look like proper nouns under _PROPER_RE.
DISCOURSE_MARKERS: frozenset[str] = frozenset(
    """
according additionally alternatively consequently conversely finally firstly
furthermore hence however indeed instead lastly likewise meanwhile moreover namely
next nonetheless notably overall particularly previously similarly specifically
still subsequently therefore thus typically ultimately yes context contrast
""".split()
)

NEGATION_TOKENS: frozenset[str] = frozenset(
    """
no not never none neither nor cannot can't cant don't dont doesn't doesnt
didn't didnt won't wont isn't isnt aren't arent wasn't wasnt weren't werent
without lacks lacking fails failed fail unable rarely seldom nothing nowhere
false incorrect untrue unsupported absent excludes excluded prevents prevented
""".split()
)

ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("improve", "degrade"),
    ("improves", "degrades"),
    ("improved", "degraded"),
    ("higher", "lower"),
    ("high", "low"),
    ("more", "less"),
    ("faster", "slower"),
    ("fast", "slow"),
    ("better", "worse"),
    ("best", "worst"),
    ("reduce", "increase"),
    ("reduces", "increases"),
    ("reduced", "increased"),
    ("reducing", "increasing"),
    ("reduce", "raise"),
    ("reduces", "raises"),
    ("effective", "ineffective"),
    ("efficient", "inefficient"),
    ("safe", "unsafe"),
    ("stable", "unstable"),
    ("supported", "unsupported"),
    ("beneficial", "harmful"),
    ("helps", "hurts"),
    ("enable", "prevent"),
    ("enables", "prevents"),
    ("possible", "impossible"),
    ("required", "optional"),
    ("always", "never"),
    ("outperforms", "underperforms"),
    ("superior", "inferior"),
    ("gain", "loss"),
    ("positive", "negative"),
    ("accepted", "rejected"),
    ("recommended", "discouraged"),
    ("significant", "negligible"),
)

_ANTONYM_LOOKUP: dict[str, set[str]] = {}
for _a, _b in ANTONYM_PAIRS:
    _ANTONYM_LOOKUP.setdefault(_a, set()).add(_b)
    _ANTONYM_LOOKUP.setdefault(_b, set()).add(_a)

HEDGE_TOKENS: frozenset[str] = frozenset(
    """
may might possibly perhaps probably likely unlikely apparently seemingly
suggests suggest appears arguably potentially roughly approximately about
""".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']*|\d+(?:\.\d+)?%?")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent|x|ms|s|gb|mb|k|m|b)?", re.I)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z])(?<!\b[A-Z]\.[A-Z])(?<!\be\.g)(?<!\bi\.e)(?<!\betc)"
    r"(?<!\bFig)(?<!\bEq)(?<!\bNo)(?<!\bvs)(?<=[.!?])[\s\n]+"
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,9}\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_COMPOUND_SPLITS: dict[str, tuple[str, ...]] = {
    "deeplearning": ("deep", "learning"),
    "machinelearning": ("machine", "learning"),
    "neuralnetwork": ("neural", "network"),
    "neuralnetworks": ("neural", "networks"),
}
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _COMPOUND_SPLITS) + r")\b",
    re.I,
)
DEFINITION_CUES: tuple[str, ...] = (
    "is a",
    "is an",
    "is the",
    "are a",
    "are an",
    "are the",
    "refers to",
    "refer to",
    "subset of",
    "type of",
    "defined as",
    "means",
    "known as",
    "also known as",
    "also called",
    "stands for",
    "short for",
    "abbreviation for",
    "abbreviated as",
    "denotes",
    "denote",
    "branch of",
    "form of",
    "related to",
    "we present",
    "we introduce",
    "we propose",
    "framework for",
    "framework to",
)
_USAGE_TAIL_PREFIXES: tuple[str, ...] = (
    "used",
    "applied",
    "trained",
    "based",
    "shown",
    "discussed",
    "described",
    "presented",
    "studied",
    "evaluated",
    "compared",
)
_CLAUSE_CUT_RE = re.compile(
    r"\s+(?:how|and how|and what|and why|;|\?)\s+",
    re.I,
)
_EXPLAIN_AND_RE = re.compile(
    r"^(?P<head>(?:please\s+)?(?:explain|describe|discuss|outline)\b.+?)\s+and\s+(?P<tail>.+)$",
    re.I,
)
_GENERIC_EXTRA_STEMS: frozenset[str] = frozenset(
    """
    approach method model using used combined automatic fully section goal
    provide introduction concept paper study result results based
    """.split()
)


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_query_text(text: str) -> str:
    """Space known glued compounds and camelCase so retrieval and stems match."""

    text = normalise_whitespace(text or "")
    text = _CAMEL_RE.sub(r"\1 \2", text)

    def _split(match: re.Match[str]) -> str:
        parts = _COMPOUND_SPLITS[match.group(0).lower()]
        return " ".join(parts)

    return normalise_whitespace(_COMPOUND_RE.sub(_split, text))


def first_question_clause(text: str) -> str:
    stripped = normalise_query_text(text).rstrip("?").strip()
    parts = _CLAUSE_CUT_RE.split(stripped, maxsplit=1)
    return (parts[0] if parts else stripped).strip()


def remainder_question_clause(text: str) -> str:
    stripped = normalise_query_text(text).rstrip("?").strip()
    parts = _CLAUSE_CUT_RE.split(stripped, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def question_aspects(query: str, *, max_aspects: int = 4) -> list[str]:
    """Split a user question into ordered aspects / subquestions.

    Uses clause cuts (``and how`` / ``and what`` / ``;`` / ``?``) and a light
    ``explain … and …`` split. Falls back to the full query when no split applies.
    Does not inject a fixed curriculum of concepts.
    """

    stripped = normalise_query_text(query).rstrip("?").strip()
    if not stripped:
        return []

    aspects: list[str] = []
    remaining = stripped
    while remaining and len(aspects) < max_aspects:
        parts = _CLAUSE_CUT_RE.split(remaining, maxsplit=1)
        head = parts[0].strip()
        if head:
            aspects.append(head)
        if len(parts) < 2:
            break
        remaining = parts[1].strip()
        if len(aspects) >= max_aspects and remaining:
            break

    if len(aspects) <= 1:
        match = _EXPLAIN_AND_RE.match(stripped)
        if match:
            head = match.group("head").strip()
            tail = match.group("tail").strip()
            if head and tail and len(content_tokens(tail)) >= 2:
                aspects = [head, tail]

    # Drop tiny fragments; keep unique order.
    cleaned: list[str] = []
    seen: set[str] = set()
    for aspect in aspects:
        key = aspect.lower()
        if key in seen or len(content_tokens(aspect)) < 2:
            continue
        seen.add(key)
        cleaned.append(aspect)
        if len(cleaned) >= max_aspects:
            break

    return cleaned or [stripped]


def aspect_is_covered(aspect: str, answer_text: str, *, min_overlap: float = 0.4) -> bool:
    """True when answer content stems overlap the aspect enough to count as addressed."""

    aspect_stems = stem_set(aspect)
    if not aspect_stems:
        return True
    # Drop ultra-generic stems so "explain" alone does not count as covered.
    generic = {
        "explain",
        "describe",
        "discuss",
        "outline",
        "main",
        "concept",
        "concepts",
        "used",
        "use",
        "using",
        "support",
        "supports",
        "supporting",
        "please",
        "they",
        "them",
        "their",
    }
    focus = {s for s in aspect_stems if s not in generic and len(s) >= 3}
    if len(focus) < 2:
        focus = {s for s in aspect_stems if len(s) >= 3} or aspect_stems
    answer_stems = stem_set(answer_text)
    if not answer_stems:
        return False
    shared = focus & answer_stems
    if len(focus) >= 3 and len(shared) < 2:
        return False
    # Prefer that longer topical stems are present (e.g. train, mathemat).
    distinctive = {s for s in focus if len(s) >= 5}
    if len(distinctive) >= 2:
        dist_hit = len(distinctive & answer_stems) / len(distinctive)
        if dist_hit < 0.45:
            return False
    return (len(shared) / len(focus)) >= min_overlap


def uncovered_aspects(query: str, answer_text: str) -> list[str]:
    """Return aspect phrases from the query that the answer does not address."""

    aspects = question_aspects(query)
    if len(aspects) <= 1:
        return []
    return [aspect for aspect in aspects if not aspect_is_covered(aspect, answer_text)]


def definition_subject(query: str) -> str:
    """Noun phrase after ``what is/are`` on the first clause only."""

    clause = first_question_clause(query)
    match = re.match(
        r"^(?:please\s+)?(?:what|which)\s+(?:is|are|was|were)\s+(.+)$",
        clause,
        re.I,
    )
    if match:
        return match.group(1).strip()
    return ""


def is_definition_query(query: str) -> bool:
    clause = first_question_clause(query)
    if re.match(r"^what\s+(?:is|are|was|were)\s+the\b", clause, re.I):
        return False
    return bool(re.match(r"^what\s+(?:is|are|was|were)\b", clause, re.I))


def is_concept_definition_query(query: str) -> bool:
    if not is_definition_query(query):
        return False
    tokens = set(content_tokens(definition_subject(query)))
    skip = {
        "advantage",
        "disadvantage",
        "effect",
        "purpose",
        "role",
        "result",
        "difference",
        "limitation",
    }
    return not (tokens & skip)


def extract_acronym_definition(subject: str, text: str) -> str:
    """Return the expanded form of an acronym subject when present in ``text``.

    Handles ``LLM (Large Language Model)``, ``Large Language Models (LLMs)``,
    and ``LLM stands for Large Language Model``.
    """

    subj = (subject or "").strip()
    body = text or ""
    if not subj or not body:
        return ""

    variants: list[str] = []
    for base in (subj, subj.upper(), subj.lower(), subj.capitalize()):
        if base and base not in variants:
            variants.append(base)
        plural = f"{base}s"
        if base and plural not in variants:
            variants.append(plural)

    for variant in variants:
        esc = re.escape(variant)
        # Subject (Full Form …)
        match = re.search(rf"\b{esc}\s*\(([^)]{{3,120}})\)", body, re.I)
        if match:
            expansion = match.group(1).strip(" .,;:")
            if _looks_like_expansion(variant, expansion):
                return expansion
        # Full Form (Subject)
        match = re.search(
            rf"\b([A-Z][A-Za-z0-9][^()]{{2,100}}?)\s*\({esc}\)",
            body,
        )
        if match:
            expansion = match.group(1).strip(" .,;:")
            if _looks_like_expansion(variant, expansion):
                return expansion
        match = re.search(
            rf"\b{esc}\s+(?:stands for|short for|is short for|means|denotes)\s+"
            rf"([^.;,\n]{{3,120}})",
            body,
            re.I,
        )
        if match:
            return match.group(1).strip(" .,;:")
    return ""


def _looks_like_expansion(acronym: str, expansion: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (expansion or "").strip())
    if len(cleaned) < 4:
        return False
    letters = re.sub(r"[^A-Za-z]", "", acronym)
    if letters and cleaned.lower() == letters.lower():
        return False
    words = [w for w in re.split(r"\s+", cleaned) if w]
    if len(words) >= 2:
        return True
    return len(cleaned) >= max(8, len(letters) + 3)


def extract_copula_definition(subject: str, text: str) -> str:
    """Return the predicate after ``subject is/are …`` when it looks definitional."""

    subj = (subject or "").strip()
    body = text or ""
    if not subj or not body:
        return ""
    variants = {subj, f"{subj}s", subj.upper(), f"{subj.upper()}s"}
    for variant in variants:
        match = re.search(
            rf"\b{re.escape(variant)}\b\s+(?:is|are|was|were)\s+(.+?)(?:[.!?;]|$)",
            body,
            re.I,
        )
        if not match:
            continue
        tail = match.group(1).strip(" .,;:")
        lowered = tail.lower()
        if any(lowered.startswith(prefix) for prefix in _USAGE_TAIL_PREFIXES):
            continue
        if len(tail) < 4:
            continue
        return tail
    return ""


def is_definitional_sentence(text: str, subject_stems: set[str], subject: str = "") -> bool:
    if not subject_stems:
        return False
    lowered = (text or "").lower()
    if not (subject_stems & stem_set(text)):
        return False
    if any(cue in lowered for cue in DEFINITION_CUES):
        return True
    subject_hint = (subject or "").strip()
    if not subject_hint:
        # Recover short acronym-like subjects from stems (e.g. {"llm"} → "llm").
        for stem_token in subject_stems:
            if 2 <= len(stem_token) <= 12 and stem_token.isalpha():
                subject_hint = stem_token
                break
    if subject_hint and extract_acronym_definition(subject_hint, text):
        return True
    if subject_hint and extract_copula_definition(subject_hint, text):
        return True
    return False


def off_topic_penalty(sentence: str, query: str) -> float:
    """Down-rank application sentences whose distinctive nouns are not in the query."""

    extra = stem_set(sentence) - stem_set(query) - _GENERIC_EXTRA_STEMS
    extra = {token for token in extra if len(token) >= 3}
    if len(extra) < 2:
        return 0.0
    return min(0.55, 0.11 * len(extra))


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text or ""):
        spaced = _CAMEL_RE.sub(r"\1 \2", tok)
        for part in spaced.split():
            key = part.lower()
            if key in _COMPOUND_SPLITS:
                out.extend(_COMPOUND_SPLITS[key])
            else:
                out.append(key)
    return out


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1]


@lru_cache(maxsize=8192)
def content_token_set(text: str) -> frozenset[str]:
    return frozenset(content_tokens(text))


def stem(token: str) -> str:
    """A small, conservative suffix stripper.

    A full Porter stemmer is unnecessary here and its aggressive rewrites hurt
    entity matching (e.g. ``dropout`` -> ``dropou``).
    """

    for suffix in ("ization", "isation", "ations", "ation", "ements", "ement",
                   "ingly", "edly", "ness", "ities", "ity", "ies", "ing",
                   "ers", "er", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            base = token[: -len(suffix)]
            if suffix in {"ies", "ity", "ities"}:
                base += "y" if suffix != "ity" else ""
            return base
    return token


def stem_set(text: str) -> set[str]:
    return {stem(tok) for tok in content_tokens(text)}


def term_bag(text: str) -> set[str]:
    """Token set used for coverage / co-occurrence, with light stemming.

    Includes the raw token, a conservative stem, and a 5-character prefix so
    ``retrieval``/``retrieved`` and ``improve``/``improves`` still match.
    """

    bag: set[str] = set()
    for token in content_tokens(text):
        bag.add(token)
        stemmed = stem(token)
        bag.add(stemmed)
        if len(token) >= 5:
            bag.add(token[:5])
        if len(stemmed) >= 5:
            bag.add(stemmed[:5])
    return bag


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    text = normalise_whitespace(text)
    if not text:
        return []
    raw_parts: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        raw_parts.extend(part.strip() for part in _SENTENCE_SPLIT_RE.split(block))
    sentences: list[str] = []
    buffer = ""
    for part in raw_parts:
        if not part:
            continue
        candidate = f"{buffer} {part}".strip() if buffer else part
        if len(candidate) < min_chars:
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""
    if buffer:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {buffer}".strip()
        else:
            sentences.append(buffer)
    return sentences


def extract_entities(text: str) -> list[str]:
    """Heuristic entity extraction: acronyms, proper nouns and hyphenated terms.

    Discourse connectives and sentence-initial attribution leads (e.g. ``According``)
    are excluded so they are not treated as named entities.
    """

    found: list[str] = []
    seen: set[str] = set()
    raw = text or ""
    acronyms = {m.lower() for m in _ACRONYM_RE.findall(raw)}

    for match in _ACRONYM_RE.findall(raw):
        key = match.lower()
        if key in seen or key in STOPWORDS or key in DISCOURSE_MARKERS:
            continue
        seen.add(key)
        found.append(match)

    for match in _PROPER_RE.finditer(raw):
        token = match.group(0)
        key = token.lower()
        if key in seen or key in STOPWORDS or key in DISCOURSE_MARKERS:
            continue
        is_multi = " " in token
        # Skip sentence-initial singles that look like connectives/attribution,
        # not real paper entities (keeps mid-sentence "Dropout", "Transformer").
        if (
            not is_multi
            and key not in acronyms
            and _is_sentence_initial(raw, match.start())
            and _looks_like_discourse_lead(raw, match)
        ):
            continue
        seen.add(key)
        found.append(token)

    for match in re.findall(r"\b[a-z]+-[a-z]+(?:-[a-z]+)?\b", raw.lower()):
        if match in seen or match in STOPWORDS or match in DISCOURSE_MARKERS:
            continue
        seen.add(match)
        found.append(match)
    return found


def _is_sentence_initial(text: str, start: int) -> bool:
    """True when ``start`` is the first non-space char of the text or of a sentence."""

    if start <= 0:
        return True
    before = text[:start].rstrip()
    if not before:
        return True
    return before[-1] in ".!?"


def _looks_like_discourse_lead(text: str, match: re.Match[str]) -> bool:
    """True for connective / attribution leads such as ``According to``."""

    key = match.group(0).lower()
    if key in DISCOURSE_MARKERS:
        return True
    after = text[match.end() : match.end() + 16].lower()
    if after.startswith(" to ") or after.startswith(" to,") or after.startswith(" to\n"):
        return True
    return False


def is_content_entity(entity: str) -> bool:
    """True for acronyms, multi-token names, or hyphenated terms — not lone discourse words."""

    text = (entity or "").strip()
    if not text:
        return False
    key = text.lower()
    if key in STOPWORDS or key in DISCOURSE_MARKERS:
        return False
    if "-" in text:
        return True
    if " " in text:
        return True
    if _ACRONYM_RE.fullmatch(text):
        return True
    # Retained single proper nouns still count as content (e.g. Dropout, Titan).
    return bool(_PROPER_RE.fullmatch(text))


def extract_numbers(text: str) -> list[str]:
    out: list[str] = []
    for value, unit in _NUMBER_RE.findall(text or ""):
        unit = (unit or "").lower()
        unit = "%" if unit in {"%", "percent"} else unit
        out.append(f"{value}{unit}")
    return out


def numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for value, _unit in _NUMBER_RE.findall(text or ""):
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def negation_count(text: str) -> int:
    return sum(1 for tok in tokenize(text) if tok in NEGATION_TOKENS)


def has_hedge(text: str) -> bool:
    return any(tok in HEDGE_TOKENS for tok in tokenize(text))


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def containment(needle: set[str] | frozenset[str], haystack: set[str] | frozenset[str]) -> float:
    """Fraction of ``needle`` tokens present in ``haystack``.

    Asymmetric on purpose: a short claim can be fully contained in a long
    passage, and that is exactly what "supported" should mean.
    """

    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def idf_weighted_containment(
    needle_tokens: list[str],
    haystack: set[str] | frozenset[str],
    idf: dict[str, float],
    default_idf: float = 1.0,
) -> float:
    """Containment where rare terms count for more than common ones."""

    if not needle_tokens:
        return 0.0
    total = 0.0
    matched = 0.0
    for token in set(needle_tokens):
        weight = idf.get(token, default_idf)
        total += weight
        if token in haystack:
            matched += weight
    if total <= 0:
        return 0.0
    return matched / total


def cosine(vec_a, vec_b) -> float:
    """Cosine similarity for plain sequences (no numpy requirement)."""

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(vec_a, vec_b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def polarity_conflict(text_a: str, text_b: str) -> float:
    """Estimate whether two statements assert opposite things.

    Combines a negation-parity signal with antonym co-occurrence. Returns a
    value in ``[0, 1]``.
    """

    neg_a = negation_count(text_a)
    neg_b = negation_count(text_b)
    # Contrastive constructions ("rather than", "instead of", "as opposed to")
    # flip polarity even without an explicit negation token.
    contrast = ("rather than", "instead of", "as opposed to", "contrary to", "the opposite")
    lowered_a = (text_a or "").lower()
    lowered_b = (text_b or "").lower()
    if any(marker in lowered_a for marker in contrast):
        neg_a += 1
    if any(marker in lowered_b for marker in contrast):
        neg_b += 1
    parity = 1.0 if (neg_a % 2) != (neg_b % 2) else 0.0

    stems_a = stem_set(text_a)
    stems_b = stem_set(text_b)
    antonym_hits = 0
    for token in stems_a | set(tokenize(text_a)):
        opposites = _ANTONYM_LOOKUP.get(token, set()) | {
            stem(opp) for opp in _ANTONYM_LOOKUP.get(token, set())
        }
        if opposites & (stems_b | set(tokenize(text_b))):
            antonym_hits += 1
    antonym_signal = min(1.0, antonym_hits / 1.0)

    if antonym_signal and parity:
        return min(1.0, 0.65 * antonym_signal + 0.35 * parity)
    if antonym_signal:
        return min(1.0, 0.80 * antonym_signal)
    # Parity alone (differing negation count) without any antonym or direct opposition
    # does NOT establish contradiction; different papers describing different mechanisms
    # are not contradictory.
    return 0.0


_METADATA_NUM_RE = re.compile(
    r"\b(?:19\d\d|20\d\d)\b"
    r"|\b(?:p\.|page|pages|pp\.|fig\.|figure|table|eq\.|equation|section|sec\.|vol\.|volume|no\.)\s*\d+(?:\.\d+)*\b"
    r"|\[\d+\]",
    re.I,
)

_COMMON_METRICS = frozenset({
    "accuracy", "acc", "bleu", "rouge", "f1", "f1-score", "loss",
    "perplexity", "latency", "throughput", "error", "wer", "mae", "rmse",
    "map", "mrr", "auc", "roc", "precision", "recall", "params", "parameters",
    "rate", "ratio", "probability", "prob", "score", "threshold", "percentage",
    "size", "layers", "epochs", "units",
})


def substantive_numeric_values(text: str) -> list[float]:
    """Extract substantive experimental numbers, excluding years, pages, figures and citations."""
    if not text:
        return []
    cleaned = _METADATA_NUM_RE.sub(" ", text)
    values: list[float] = []
    for value, unit in _NUMBER_RE.findall(cleaned):
        try:
            val = float(value)
            if 1900 <= val <= 2099 and "." not in value and "%" not in unit:
                continue
            values.append(val)
        except ValueError:
            continue
    return values


def numeric_conflict(text_a: str, text_b: str) -> float:
    """Detect incompatible numeric assertions between two statements.
    
    Excludes publication years, page numbers, citation numbers, and different
    experimental conditions or metrics.
    """

    nums_a = substantive_numeric_values(text_a)
    nums_b = substantive_numeric_values(text_b)
    if not nums_a or not nums_b:
        return 0.0
    if set(nums_a) & set(nums_b):
        return 0.0

    toks_a = {t.lower() for t in tokenize(text_a)}
    toks_b = {t.lower() for t in tokenize(text_b)}

    # Require at least one shared metric or 2+ shared substantive content tokens
    shared_metrics = (toks_a & _COMMON_METRICS) & (toks_b & _COMMON_METRICS)
    shared_content = (set(content_tokens(text_a)) & set(content_tokens(text_b))) - frozenset({
        "paper", "model", "method", "result", "propos", "studi", "use", "show"
    })
    if not shared_metrics and len(shared_content) < 2:
        # Without a shared metric or shared entities, differing numbers are from different setups
        return 0.0

    entities_a = {e.lower() for e in extract_entities(text_a)}
    entities_b = {e.lower() for e in extract_entities(text_b)}
    # If disjoint named entities (e.g. CIFAR vs ImageNet) are mentioned,
    # these represent different experimental conditions, not contradictions.
    if entities_a and entities_b and not (entities_a & entities_b):
        return 0.0

    conflicts = 0
    comparisons = 0
    for a in nums_a:
        for b in nums_b:
            comparisons += 1
            scale = max(abs(a), abs(b), 1e-6)
            if abs(a - b) / scale > 0.20:
                conflicts += 1
    if comparisons == 0:
        return 0.0
    return conflicts / comparisons


def build_idf(documents: list[str]) -> dict[str, float]:
    """Smoothed inverse document frequency over token *sets*."""

    n_docs = max(1, len(documents))
    df: Counter[str] = Counter()
    for doc in documents:
        for token in set(content_tokens(doc)):
            df[token] += 1
    return {
        token: math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)
        for token, count in df.items()
    }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def truncate(text: str, limit: int = 320) -> str:
    text = normalise_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
