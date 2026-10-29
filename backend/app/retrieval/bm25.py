"""BM25 (Okapi) lexical retrieval index.

Implemented directly rather than pulled from a library so the scoring is
transparent, inspectable in the evaluation, and free of extra dependencies.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..text_utils import content_tokens, stem

K1 = 1.5
B = 0.75


@dataclass(slots=True)
class BM25Hit:
    chunk_id: str
    score: float
    rank: int
    matched_terms: list[str]


class BM25Index:
    """Sparse inverted index with Okapi BM25 scoring."""

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._lengths: list[int] = []
        self._term_freqs: list[dict[str, int]] = []
        self._postings: dict[str, list[int]] = {}
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    # ------------------------------------------------------------------ build
    @staticmethod
    def analyse(text: str) -> list[str]:
        return [stem(token) for token in content_tokens(text)]

    def build(self, chunks: Sequence[tuple[str, str, str]]) -> None:
        """Build from ``(chunk_id, document_id, text)`` triples."""

        self._chunk_ids = []
        self._doc_ids = []
        self._lengths = []
        self._term_freqs = []
        self._postings = {}
        self._df = Counter()

        for position, (chunk_id, document_id, text) in enumerate(chunks):
            tokens = self.analyse(text)
            freqs = Counter(tokens)
            self._chunk_ids.append(chunk_id)
            self._doc_ids.append(document_id)
            self._lengths.append(len(tokens))
            self._term_freqs.append(dict(freqs))
            for term in freqs:
                self._postings.setdefault(term, []).append(position)
                self._df[term] += 1

        total = sum(self._lengths)
        self._avg_len = (total / len(self._lengths)) if self._lengths else 0.0

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def idf(self, term: str) -> float:
        n_docs = self.size
        if n_docs == 0:
            return 0.0
        df = self._df.get(term, 0)
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def term_idf_map(self) -> dict[str, float]:
        return {term: self.idf(term) for term in self._df}

    def vocabulary(self, limit: int | None = None) -> list[str]:
        terms = sorted(self._df, key=lambda t: -self._df[t])
        return terms[:limit] if limit else terms

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int,
        document_ids: Iterable[str] | None = None,
    ) -> list[BM25Hit]:
        if self.size == 0 or top_k <= 0:
            return []

        query_terms = self.analyse(query)
        if not query_terms:
            return []

        allowed: set[str] | None = set(document_ids) if document_ids else None
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}

        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0:
                continue
            for position in postings:
                if allowed is not None and self._doc_ids[position] not in allowed:
                    continue
                freq = self._term_freqs[position].get(term, 0)
                length = self._lengths[position] or 1
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / (self._avg_len or 1.0))
                )
                if denominator <= 0:
                    continue
                scores[position] = scores.get(position, 0.0) + idf * (
                    freq * (self.k1 + 1.0) / denominator
                )
                matched.setdefault(position, set()).add(term)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            BM25Hit(
                chunk_id=self._chunk_ids[position],
                score=float(score),
                rank=rank,
                matched_terms=sorted(matched.get(position, set())),
            )
            for rank, (position, score) in enumerate(ranked, start=1)
        ]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "chunk_ids": self._chunk_ids,
                    "doc_ids": self._doc_ids,
                    "lengths": self._lengths,
                    "term_freqs": self._term_freqs,
                    "postings": self._postings,
                    "df": self._df,
                    "avg_len": self._avg_len,
                    "k1": self.k1,
                    "b": self.b,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                state = pickle.load(handle)
        except Exception:
            return None
        index = cls(k1=state.get("k1", K1), b=state.get("b", B))
        index._chunk_ids = state["chunk_ids"]
        index._doc_ids = state["doc_ids"]
        index._lengths = state["lengths"]
        index._term_freqs = state["term_freqs"]
        index._postings = state["postings"]
        index._df = state["df"]
        index._avg_len = state["avg_len"]
        return index
