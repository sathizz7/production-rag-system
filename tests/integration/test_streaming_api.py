import json
import re

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


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse SSE the way a browser client must — strip CR, split frames on \\n\\n.

    sse-starlette frames are CRLF-separated (event: ...\\r\\ndata: ...\\r\\n\\r\\n);
    a client that splits on a bare \\n\\n recovers nothing. This mirrors ui/index.html
    so the framing contract the UI depends on is locked by a test.
    """
    events: list[tuple[str, dict]] = []
    for frame in body.replace("\r", "").split("\n\n"):
        ev = re.search(r"^event: (.*)$", frame, re.M)
        data = re.search(r"^data: (.*)$", frame, re.M)
        if ev and data:
            events.append((ev.group(1), json.loads(data.group(1))))
    return events


def test_stream_endpoint_emits_token_then_done_events() -> None:
    stream_req = {"query": "is alpha true?"}
    with (
        _client() as client,
        client.stream("POST", "/query/stream", json=stream_req) as resp,
    ):
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())

    events = _parse_sse(body)
    types = [t for t, _ in events]
    assert "token" in types and types[-1] == "done"        # tokens then a terminal done
    tokens = "".join(p["text"] for t, p in events if t == "token")
    assert tokens == "alpha is true [1]"                   # client reassembles the stream
    done = next(p for t, p in events if t == "done")
    assert done["citations"][0]["chunk_id"] == "a"
    assert "[1]" in done["answer"]


def test_json_query_still_works_with_streaming_answerer() -> None:
    with _client() as client:
        resp = client.post("/query", json={"query": "is alpha true?"})
    assert resp.status_code == 200
    assert resp.json()["citations"][0]["chunk_id"] == "a"
