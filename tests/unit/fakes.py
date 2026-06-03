from __future__ import annotations

from collections.abc import Sequence

from rag.models import Chunk, Completion, MetadataFilter, Provenance, ScoredChunk

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


def make_chunk(chunk_id: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=0,
        page=1,
        char_start=0,
        char_end=len(text),
        token_count=1,
    )


class FakeRetriever:
    """Returns a fixed ScoredChunk list; records the args of the last call."""

    def __init__(self, chunks: list[Chunk], provenance: Provenance = Provenance.dense) -> None:
        self._chunks = chunks
        self._provenance = provenance
        self.last_call: tuple[str, int, MetadataFilter | None] | None = None

    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        self.last_call = (query, k, filt)
        return [
            ScoredChunk(chunk=c, score=1.0 / (i + 1), provenance=self._provenance)
            for i, c in enumerate(self._chunks[:k])
        ]
