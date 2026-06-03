from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from rag.models import DoneEvent, ErrorEvent, TokenEvent
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk


def _answerer(llm, retriever):
    return StreamingAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=llm,
        token_budget=1000,
        retrieval_k=3,
    )


def test_stream_emits_tokens_then_done_with_validated_citations() -> None:
    chunks = [make_chunk("a", text="alpha"), make_chunk("b", text="beta")]
    llm = FakeStreamingLLM(tokens=["alpha ", "is true ", "[1]", " [9]"])  # [9] is invalid
    events = list(_answerer(llm, FakeRetriever(chunks)).answer_stream("q"))

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    done = events[-1]
    assert [t.text for t in tokens] == ["alpha ", "is true ", "[1]", " [9]"]
    assert isinstance(done, DoneEvent)
    assert "[9]" not in done.answer                      # invalid marker stripped
    assert [c.marker for c in done.citations] == ["[1]"]
    assert done.citations[0].chunk_id == "a"
    assert done.usage["completion_tokens_est"] >= 1
    assert done.trace_id


def test_stream_empty_retrieval_yields_no_context_done() -> None:
    events = list(_answerer(FakeStreamingLLM(), FakeRetriever([])).answer_stream("q"))
    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)
    assert events[0].answer == "I don't have relevant context to answer that."
    assert events[0].citations == []


def test_stream_provider_error_flushes_partial_then_error_event() -> None:
    class _BoomLLM:
        def stream(self, messages, **opts):
            yield "partial "
            raise RuntimeError("provider exploded")

        def complete(self, messages, **opts):  # unused
            raise NotImplementedError

    chunks = [make_chunk("a", text="alpha")]
    events = list(_answerer(_BoomLLM(), FakeRetriever(chunks)).answer_stream("q"))
    assert isinstance(events[0], TokenEvent) and events[0].text == "partial "
    assert isinstance(events[-1], ErrorEvent)
    assert "provider exploded" in events[-1].message


def test_answer_collects_stream_into_answer_object() -> None:
    chunks = [make_chunk("a", text="alpha")]
    llm = FakeStreamingLLM(tokens=["alpha ", "[1]"])
    answer = _answerer(llm, FakeRetriever(chunks)).answer("q")
    assert answer.text.strip().endswith("[1]")
    assert answer.citations[0].chunk_id == "a"
