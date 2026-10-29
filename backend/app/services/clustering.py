"""Theme clustering over library documents.

Documents are clustered, not chunks: each paper is encoded from title,
bibliography and a capped sample of its passages using the same embedder
already fitted for retrieval. Groups are formed with agglomerative average
linkage on cosine similarity (numpy only). Labels come from distinctive
title/section phrases, not a second model call.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ..config import Settings, get_settings
from ..models.documents import Chunk, ClusterResponse, Document, ThemeCluster
from ..text_utils import STOPWORDS, content_tokens, tokenize
from .embeddings.base import l2_normalise
from .store.knowledge_base import KnowledgeBase

_MAX_PHRASE = 4
_MIN_PHRASE = 2


def cluster_documents(
    kb: KnowledgeBase,
    settings: Settings | None = None,
) -> ClusterResponse:
    settings = settings or kb.settings or get_settings()
    documents = kb.store.list_documents()
    if not documents:
        return ClusterResponse(clusters=[], total_documents=0)

    if len(documents) == 1:
        doc = documents[0]
        chunks = kb.store.all_chunks(document_ids=[doc.document_id])
        label, keywords = _label_cluster([doc], {doc.document_id: chunks}, kb)
        return ClusterResponse(
            clusters=[
                ThemeCluster(
                    cluster_id="theme-1",
                    label=label,
                    keywords=keywords,
                    document_ids=[doc.document_id],
                    size=1,
                )
            ],
            total_documents=1,
        )

    texts = []
    for document in documents:
        chunks = kb.store.all_chunks(document_ids=[document.document_id])
        texts.append(_document_text(document, chunks, settings.cluster_sample_chars))

    vectors = l2_normalise(kb.embedder.encode(texts))
    groups = agglomerative_groups(
        vectors,
        min_similarity=settings.cluster_min_similarity,
        max_themes=max(1, settings.cluster_max_themes),
    )

    chunks_by_doc: dict[str, list[Chunk]] = {
        document.document_id: kb.store.all_chunks(document_ids=[document.document_id])
        for document in documents
    }

    clusters: list[ThemeCluster] = []
    for rank, indices in enumerate(sorted(groups, key=len, reverse=True), start=1):
        members = [documents[i] for i in indices]
        label, keywords = _label_cluster(members, chunks_by_doc, kb)
        clusters.append(
            ThemeCluster(
                cluster_id=f"theme-{rank}",
                label=label,
                keywords=keywords,
                document_ids=[item.document_id for item in members],
                size=len(members),
            )
        )

    return ClusterResponse(clusters=clusters, total_documents=len(documents))


def agglomerative_groups(
    vectors: np.ndarray,
    min_similarity: float,
    max_themes: int,
) -> list[list[int]]:
    """Average-linkage agglomerative clustering on L2-normalised rows.

    Pairs merge while mean pairwise cosine stays at or above ``min_similarity``.
    If more groups remain than ``max_themes``, the closest pair is still merged
    until the cap is met.
    """

    if vectors.size == 0:
        return []
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    n = matrix.shape[0]
    if n == 1:
        return [[0]]

    sim = matrix @ matrix.T
    groups: list[list[int]] = [[i] for i in range(n)]
    max_themes = max(1, int(max_themes))

    while len(groups) > 1:
        best = -np.inf
        pair: tuple[int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                score = _average_link(sim, groups[i], groups[j])
                if score > best:
                    best = score
                    pair = (i, j)
        if pair is None:
            break
        over_cap = len(groups) > max_themes
        if best < min_similarity and not over_cap:
            break
        i, j = pair
        merged = groups[i] + groups[j]
        groups = [group for k, group in enumerate(groups) if k not in {i, j}]
        groups.append(merged)

    return groups


def _average_link(sim: np.ndarray, left: Sequence[int], right: Sequence[int]) -> float:
    block = sim[np.ix_(list(left), list(right))]
    return float(np.mean(block))


def _document_text(document: Document, chunks: Sequence[Chunk], sample_chars: int) -> str:
    parts: list[str] = [document.title or document.name]
    if document.authors:
        parts.append(", ".join(document.authors))
    if document.year:
        parts.append(str(document.year))
    if document.venue:
        parts.append(document.venue)
    used = 0
    body: list[str] = []
    ordered = sorted(chunks, key=lambda chunk: chunk.metadata.chunk_index)
    for chunk in ordered:
        section = chunk.metadata.section
        piece = f"{section}. {chunk.text}" if section else chunk.text
        body.append(piece)
        used += len(piece)
        if used >= sample_chars:
            break
    if body:
        parts.append(" ".join(body)[:sample_chars])
    return "\n".join(part for part in parts if part)


def _label_cluster(
    documents: Sequence[Document],
    chunks_by_doc: dict[str, list[Chunk]],
    kb: KnowledgeBase,
) -> tuple[str, list[str]]:
    title_blobs = []
    for document in documents:
        bits = [document.title or document.name]
        for chunk in chunks_by_doc.get(document.document_id, []):
            if chunk.metadata.section:
                bits.append(chunk.metadata.section)
        title_blobs.append(" ".join(bits))

    phrase_scores = _phrase_scores(title_blobs, kb)
    keywords = [phrase for phrase, _ in phrase_scores[:5]]
    if keywords:
        return _title_case(keywords[0]), keywords

    fallback = documents[0].title or documents[0].name
    tokens = content_tokens(fallback)[:4]
    label = _title_case(" ".join(tokens)) if tokens else fallback
    return label, tokens[:3]


def _phrase_scores(texts: Sequence[str], kb: KnowledgeBase) -> list[tuple[str, float]]:
    idf = kb.idf_map()
    n_docs = max(1, len(texts))
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for text in texts:
        phrases = _phrases(text)
        tf.update(phrases)
        df.update(set(phrases))

    scored: list[tuple[str, float]] = []
    seen_stems: set[str] = set()
    for phrase, count in tf.items():
        words = phrase.split()
        if not words:
            continue
        idf_score = float(np.mean([idf.get(word, 1.0) for word in words]))
        coverage = df[phrase] / n_docs
        length_bonus = 1.0 + 0.25 * (len(words) - 1)
        score = count * coverage * idf_score * length_bonus
        key = " ".join(words)
        if key in seen_stems:
            continue
        seen_stems.add(key)
        scored.append((phrase, score))

    scored.sort(key=lambda item: (-item[1], -len(item[0].split()), item[0]))
    return scored


def _phrases(text: str) -> list[str]:
    tokens = [tok for tok in tokenize(text) if not tok.isdigit()]
    phrases: list[str] = []
    for n in range(_MIN_PHRASE, _MAX_PHRASE + 1):
        for i in range(0, len(tokens) - n + 1):
            window = tokens[i : i + n]
            if window[0] in STOPWORDS or window[-1] in STOPWORDS:
                continue
            phrases.append(" ".join(window))
    phrases.extend(tok for tok in tokens if tok not in STOPWORDS and len(tok) >= 5)
    return phrases


def _title_case(value: str) -> str:
    parts = value.split()
    styled: list[str] = []
    for index, part in enumerate(parts):
        if part.isupper() and len(part) > 1:
            styled.append(part)
        elif index > 0 and part in STOPWORDS:
            styled.append(part)
        else:
            styled.append(part.capitalize())
    return " ".join(styled)
