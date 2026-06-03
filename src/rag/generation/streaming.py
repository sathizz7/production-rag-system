from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from importlib import resources

import structlog

from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.citations import validate_and_build_citations
from rag.models import (
    Answer,
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    MetadataFilter,
    RetrievalScope,
    TokenEvent,
)
from rag.observability.metrics import observe_stage
from rag.protocols import LLMProvider, Retriever
from rag.util.tokens import count_tokens

log = structlog.get_logger()

_NO_CONTEXT = "I don't have relevant context to answer that."


def _load_prompt() -> str:
    return (
        resources.files("rag.generation.prompts")
        .joinpath("answer_v1.md")
        .read_text(encoding="utf-8")
    )


class StreamingAnswerer:
    """Streaming read path: retrieve → assemble → stream → validate.

    Emits ``TokenEvent`` per delta for live UX, then a terminal ``DoneEvent`` whose
    ``answer``/``citations`` come from the SAME post-gen validator as the JSON path
    (invalid markers stripped). A provider error flushes the partial stream and ends
    with an ``ErrorEvent`` (spec §16). ``answer()`` collects the stream into an
    ``Answer`` so the JSON ``/query`` route and tests share one implementation.

    Rerank is NOT a parameter here — inject a ``RerankedRetriever`` as ``retriever``
    and this class is unchanged. It only ever sees "a retriever".
    """

    def __init__(
        self,
        retriever: Retriever,
        assembler: TokenBudgetAssembler,
        llm: LLMProvider,
        token_budget: int,
        retrieval_k: int,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._llm = llm
        self._token_budget = token_budget
        self._retrieval_k = retrieval_k
        self._prompt = _load_prompt()

    def answer_stream(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Iterator[AnswerEvent]:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        bound_log = log.bind(trace_id=trace_id)

        scored = self._retriever.retrieve(query, self._retrieval_k, filt)
        if not scored:
            bound_log.info("no_context")
            yield DoneEvent(answer=_NO_CONTEXT, trace_id=trace_id, retrieval_scope=scope)
            return

        chunks = [s.chunk for s in scored]
        with observe_stage("assemble"):
            context = self._assembler.assemble(query, chunks, self._token_budget)
        messages: list[dict[str, object]] = [
            {"role": "user", "content": self._prompt.format(context=context.text, question=query)}
        ]

        parts: list[str] = []
        try:
            with observe_stage("generate"):
                for delta in self._llm.stream(messages):
                    parts.append(delta)
                    yield TokenEvent(text=delta)
        except Exception as exc:  # flush partial, then surface as a terminal event
            bound_log.warning("stream_error", error=str(exc))
            yield ErrorEvent(message=str(exc), trace_id=trace_id)
            return

        full_text = "".join(parts)
        cleaned, citations = validate_and_build_citations(full_text, context)
        usage = {
            "completion_tokens_est": count_tokens(full_text),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        bound_log.info("answered_stream", citations=len(citations), latency_ms=usage["latency_ms"])
        yield DoneEvent(
            answer=cleaned,
            citations=citations,
            usage=usage,
            trace_id=trace_id,
            retrieval_scope=scope,
        )

    def answer(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Answer:
        done: DoneEvent | None = None
        for event in self.answer_stream(query, filt, scope):
            if isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            if isinstance(event, DoneEvent):
                done = event
        assert done is not None  # answer_stream always ends with a DoneEvent on success
        return Answer(
            text=done.answer,
            citations=done.citations,
            usage=done.usage,
            trace_id=done.trace_id,
            retrieval_scope=done.retrieval_scope,
        )
