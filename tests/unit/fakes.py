from __future__ import annotations

from collections.abc import Sequence

from rag.models import Completion

Vector = list[float]


class FakeEmbedder:
    """Deterministic embeddings for tests. Optionally maps specific texts to vectors."""

    def __init__(self, dim: int = 3, mapping: dict[str, Vector] | None = None) -> None:
        self.model = "fake-embed"
        self.dim = dim
        self._mapping = mapping or {}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> Vector:
        if text in self._mapping:
            return list(self._mapping[text])
        # deterministic, length-based fallback vector
        seed = float(len(text) % 7 + 1)
        return [seed] * self.dim


class FakeLLMProvider:
    """Returns a canned completion; records the messages it was given."""

    def __init__(self, reply: str = "Answer grounded in context [1].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **opts: object) -> Completion:
        self.calls.append(messages)
        return Completion(
            text=self.reply,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
