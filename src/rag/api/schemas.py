from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from rag.models import Answer, MetadataFilter


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    filter: MetadataFilter | None = None

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class CitationOut(BaseModel):
    marker: str
    doc_id: str
    chunk_id: str
    source_uri: str
    source_kind: str
    page: int
    char_start: int
    char_end: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    usage: dict[str, object]
    trace_id: str
    retrieval_scope: str

    @classmethod
    def from_answer(cls, answer: Answer) -> QueryResponse:
        return cls(
            answer=answer.text,
            citations=[CitationOut(**c.model_dump(mode="json")) for c in answer.citations],
            usage=answer.usage,
            trace_id=answer.trace_id,
            retrieval_scope=answer.retrieval_scope.value,
        )
