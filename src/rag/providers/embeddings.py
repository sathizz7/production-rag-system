from __future__ import annotations

from collections.abc import Sequence

import litellm

Vector = list[float]


class LiteLLMEmbeddingProvider:
    """Embeddings via LiteLLM. Default model is Google text-embedding-004 (768-d)."""

    def __init__(self, model: str, dim: int, batch_size: int = 96) -> None:
        self.model = model
        self.dim = dim
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            resp = litellm.embedding(model=self.model, input=batch)
            out.extend([list(item["embedding"]) for item in resp.data])
        return out
