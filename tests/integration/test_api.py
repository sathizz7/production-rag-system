import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk
from tests.unit.fakes import FakeLLMProvider

pytestmark = pytest.mark.integration


class _FakeRetriever:
    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        chunk = Chunk(
            chunk_id="a", doc_id="d1", source_uri="file:///x.pdf",
            text="Nitrogen helps maize.", ordinal=0, page=1, char_start=0, char_end=21,
            token_count=4, chunker_name="fixed", chunker_version="1",
        )
        return [ScoredChunk(chunk=chunk, score=1.0, provenance=Provenance.dense)]


def _client() -> TestClient:
    answerer = StraightLineAnswerer(
        retriever=_FakeRetriever(),
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Nitrogen helps maize [1]."),
        token_budget=1000,
        retrieval_k=5,
    )
    return TestClient(create_app(answerer=answerer))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_query_returns_cited_answer() -> None:
    resp = _client().post("/query", json={"query": "does nitrogen help maize?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Nitrogen helps maize [1]."
    assert body["citations"][0]["chunk_id"] == "a"
    assert body["citations"][0]["page"] == 1
    assert body["trace_id"]


def test_query_validates_empty_query() -> None:
    resp = _client().post("/query", json={"query": "   "})
    assert resp.status_code == 422
