import json

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def _client() -> TestClient:
    chunks = [make_chunk("a", text="alpha")]
    answerer = StreamingAnswerer(
        retriever=FakeRetriever(chunks),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(tokens=["alpha ", "is true ", "[1]"]),
        token_budget=1000,
        retrieval_k=3,
    )
    return TestClient(create_app(answerer=answerer))


def test_stream_endpoint_emits_token_then_done_events() -> None:
    stream_req = {"query": "is alpha true?"}
    with (
        _client() as client,
        client.stream("POST", "/query/stream", json=stream_req) as resp,
    ):
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    # SSE frames: "event: <type>\ndata: <json>\n\n"
    assert "event: token" in body
    assert "event: done" in body
    done_payload = body.split("event: done")[1].split("data: ")[1].splitlines()[0]
    done = json.loads(done_payload)
    assert done["citations"][0]["chunk_id"] == "a"
    assert "[1]" in done["answer"]


def test_json_query_still_works_with_streaming_answerer() -> None:
    with _client() as client:
        resp = client.post("/query", json={"query": "is alpha true?"})
    assert resp.status_code == 200
    assert resp.json()["citations"][0]["chunk_id"] == "a"
