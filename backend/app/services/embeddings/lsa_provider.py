"""Offline semantic embeddings via latent semantic analysis.

Combines word-level and character-level TF-IDF with a truncated SVD projection.
Unlike a pure bag-of-words baseline this captures term *co-occurrence*, so
queries retrieve passages that share meaning without sharing vocabulary — which
is what makes the dense-vs-BM25 comparison in the evaluation meaningful.

Fitted state is persisted so restarts do not change retrieval behaviour.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from ...config import Settings, get_settings
from .base import EmbeddingProvider, l2_normalise

logger = logging.getLogger(__name__)


class LsaEmbeddingProvider(EmbeddingProvider):
    name = "lsa"
    model = "tfidf-svd-offline-v1"
    requires_fit = True

    def __init__(self, settings: Settings | None = None, dimension: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.dimension = int(dimension or min(self.settings.embedding_dim, 256))
        self._vectoriser: FeatureUnion | None = None
        self._svd: TruncatedSVD | None = None
        self._fallback = HashingVectorizer(
            n_features=2**15,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            stop_words="english",
        )
        self._state_path = Path(self.settings.index_dir) / "lsa_embedder.pkl"
        self._load()

    # ----------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        return self._svd is not None and self._vectoriser is not None

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with self._state_path.open("rb") as handle:
                state = pickle.load(handle)
            self._vectoriser = state["vectoriser"]
            self._svd = state["svd"]
            self.dimension = int(state["dimension"])
            logger.info("Loaded persisted LSA embedder (dim=%s)", self.dimension)
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("Could not load LSA embedder state: %s", exc)
            self._vectoriser = None
            self._svd = None

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._state_path.open("wb") as handle:
                pickle.dump(
                    {
                        "vectoriser": self._vectoriser,
                        "svd": self._svd,
                        "dimension": self.dimension,
                    },
                    handle,
                )
        except Exception as exc:  # pragma: no cover - disk issues
            logger.warning("Could not persist LSA embedder state: %s", exc)

    # ------------------------------------------------------------------- fit
    def fit(self, texts: Sequence[str]) -> None:
        corpus = [t for t in texts if t and t.strip()]
        if len(corpus) < 2:
            logger.info("Corpus too small to fit LSA embedder (%s docs)", len(corpus))
            return

        vectoriser = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        max_df=0.95,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        min_df=2,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
            ]
        )
        matrix = vectoriser.fit_transform(corpus)
        n_components = int(min(self.dimension, max(2, min(matrix.shape) - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm="randomized")
        svd.fit(matrix)

        self._vectoriser = vectoriser
        self._svd = svd
        self.dimension = n_components
        explained = float(svd.explained_variance_ratio_.sum())
        logger.info(
            "Fitted LSA embedder on %s chunks (dim=%s, explained variance=%.3f)",
            len(corpus),
            n_components,
            explained,
        )
        self._persist()

    # ---------------------------------------------------------------- encode
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if not self.is_fitted:
            # Deterministic lexical fallback so the system still functions
            # before any corpus has been indexed.
            sparse = self._fallback.transform(texts)
            dense = np.asarray(sparse.todense(), dtype=np.float32)
            projected = _fold(dense, self.dimension)
            return l2_normalise(projected)

        assert self._vectoriser is not None and self._svd is not None
        matrix = self._vectoriser.transform(texts)
        projected = self._svd.transform(matrix).astype(np.float32)
        return l2_normalise(projected)

    @classmethod
    def is_available(cls) -> bool:
        return True


def _fold(dense: np.ndarray, dimension: int) -> np.ndarray:
    """Deterministically fold a wide sparse-ish matrix into ``dimension`` dims."""

    n_features = dense.shape[1]
    pad = (-n_features) % dimension
    if pad:
        dense = np.hstack([dense, np.zeros((dense.shape[0], pad), dtype=np.float32)])
    return dense.reshape(dense.shape[0], -1, dimension).sum(axis=1)
