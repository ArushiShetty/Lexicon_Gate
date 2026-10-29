"""Sentence-Transformers embedding provider (optional dependency)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Sequence

import numpy as np

from ...config import Settings, get_settings
from .base import EmbeddingProvider, l2_normalise

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    name = "sentence_transformer"
    requires_fit = False

    def __init__(self, settings: Settings | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.settings = settings or get_settings()
        self.model = self.settings.sentence_transformer_model
        self._model = SentenceTransformer(self.model)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info("Loaded SentenceTransformer %s (dim=%s)", self.model, self.dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return l2_normalise(vectors)

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None
