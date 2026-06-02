from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

PAGE_SEPARATOR = "\n\n"


class RetrievalScope(StrEnum):
    corpus_only = "corpus_only"
    web_allowed = "web_allowed"
    web_required = "web_required"


class Provenance(StrEnum):
    dense = "dense"
    lexical = "lexical"
    fused = "fused"
    rerank = "rerank"


class SourceKind(StrEnum):
    corpus = "corpus"
    web = "web"


class RawDocument(BaseModel):
    source_id: str
    source_type: str
    uri: str
    text: str | None = None
    raw_bytes: bytes | None = None
    fetched_at: datetime | None = None
    source_etag: str | None = None
    source_last_modified: str | None = None
    license: str | None = None
    source_meta: dict[str, object] = Field(default_factory=dict)


class Page(BaseModel):
    number: int  # 1-based
    text: str


class Document(BaseModel):
    doc_id: str
    source_id: str
    source_type: str
    uri: str
    document_version: int = 1
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    content_hash: str = ""
    license: str | None = None
    deleted_at: datetime | None = None

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(p.text for p in self.pages)

    def page_spans(self) -> list[tuple[int, int, int]]:
        """Returns (page_number, char_start, char_end) into self.text for each page."""
        spans: list[tuple[int, int, int]] = []
        pos = 0
        last = len(self.pages) - 1
        for i, p in enumerate(self.pages):
            start = pos
            end = pos + len(p.text)
            spans.append((p.number, start, end))
            pos = end + (len(PAGE_SEPARATOR) if i < last else 0)
        return spans

    def page_at(self, offset: int) -> int:
        spans = self.page_spans()
        for number, start, end in spans:
            if start <= offset < end:
                return number
        return spans[-1][0] if spans else 0


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_uri: str
    text: str
    ordinal: int
    page: int
    char_start: int
    char_end: int
    token_count: int
    metadata: dict[str, object] = Field(default_factory=dict)
    content_hash: str = ""
    chunker_name: str = ""
    chunker_version: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    provenance: Provenance


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    doc_id: str
    chunk_id: str
    source_uri: str
    source_kind: SourceKind = SourceKind.corpus
    page: int
    char_start: int
    char_end: int


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    usage: dict[str, object] = Field(default_factory=dict)
    trace_id: str = ""
    retrieval_scope: RetrievalScope = RetrievalScope.corpus_only


class AssembledContext(BaseModel):
    text: str
    chunks: list[Chunk]  # order defines citation markers: chunks[i] -> "[i+1]"
    token_count: int


class Completion(BaseModel):
    text: str
    usage: dict[str, object] = Field(default_factory=dict)


class Watermark(BaseModel):
    source_id: str
    etag: str | None = None
    last_modified: str | None = None
    updated_at: datetime | None = None


class MetadataFilter(BaseModel):
    region: str | None = None
    crop: str | None = None
    season: str | None = None

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self.model_dump().items() if v is not None]


class UpsertStats(BaseModel):
    documents_upserted: int = 0
    chunks_upserted: int = 0
    skipped_unchanged: int = 0
