import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def test_ui_index_is_served() -> None:
    answerer = StreamingAnswerer(
        retriever=FakeRetriever([make_chunk("a", text="alpha")]),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(),
        token_budget=1000,
        retrieval_k=3,
    )
    client = TestClient(create_app(answerer=answerer))
    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "query/stream" in resp.text          # the page wires to the SSE endpoint
