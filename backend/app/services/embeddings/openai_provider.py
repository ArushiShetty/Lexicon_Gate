"""OpenAI embeddings provider (HTTP, no vendor SDK)."""

from __future__ import annotations

import logging
from typing import Sequence

import httpx
import numpy as np

from ...config import Settings, get_settings
from .base import EmbeddingProvider, l2_normalise

logger = logging.getLogger(__name__)

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"
    requires_fit = False

    def __init__(self, settings: Settings | None = None, timeout: float = 60.0) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_embedding_model
        self.dimension = _DIMENSIONS.get(self.model, 1536)
        self._base_url = (
            self.settings.openai_base_url or "https://api.openai.com/v1"
        ).rstrip("/")
        self._timeout = timeout

    @classmethod
    def is_available(cls) -> bool:
        return bool(get_settings().openai_api_key)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t and t.strip() else " " for t in texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors: list[list[float]] = []
        batch_size = 96
        with httpx.Client(timeout=self._timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                for item in sorted(payload["data"], key=lambda d: d["index"]):
                    vectors.append(item["embedding"])

        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1]
        return l2_normalise(matrix)
