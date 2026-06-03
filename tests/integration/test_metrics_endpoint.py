import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def test_metrics_endpoint_reports_request_counts() -> None:
    answerer = StreamingAnswerer(
        retriever=FakeRetriever([make_chunk("a", text="alpha")]),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(tokens=["alpha ", "[1]"]),
        token_budget=1000,
        retrieval_k=3,
    )
    client = TestClient(create_app(answerer=answerer))
    client.post("/query", json={"query": "alpha?"})
    body = client.get("/metrics").text
    assert "rag_requests_total" in body
    assert "rag_stage_latency_seconds" in body
