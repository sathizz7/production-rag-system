from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from rag.models import (
    AssembledContext,
    Chunk,
    Completion,
    Document,
    MetadataFilter,
    RawDocument,
    ScoredChunk,
    UpsertStats,
    Watermark,
)

Vector = list[float]


@runtime_checkable
class SourceAdapter(Protocol):
    source_type: str

    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, raw: RawDocument) -> Document: ...


@runtime_checkable
class Cleaner(Protocol):
    def clean(self, doc: Document) -> Document: ...


@runtime_checkable
class Chunker(Protocol):
    name: str
    version: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


@runtime_checkable
class ChunkRepository(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> UpsertStats: ...
    def soft_delete(self, doc_ids: list[str]) -> int: ...
    def get_watermark(self, source_id: str) -> Watermark | None: ...
    def set_watermark(self, source_id: str, wm: Watermark) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class ContextAssembler(Protocol):
    def assemble(
        self, query: str, chunks: list[Chunk], token_budget: int
    ) -> AssembledContext: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, object]], **opts: object) -> Completion: ...
