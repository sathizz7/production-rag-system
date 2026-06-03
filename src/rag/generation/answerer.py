from __future__ import annotations

import time
import uuid
from importlib import resources

import structlog

from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.citations import validate_and_build_citations
from rag.models import Answer, MetadataFilter, RetrievalScope
from rag.protocols import LLMProvider, Retriever

log = structlog.get_logger()

_NO_CONTEXT = "I don't have relevant context to answer that."


def _load_prompt() -> str:
    return (
        resources.files("rag.generation.prompts")
        .joinpath("answer_v1.md")
        .read_text(encoding="utf-8")
    )


class StraightLineAnswerer:
    """P0a read path: retrieve -> assemble -> generate -> validate citations (no loop)."""

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

    def answer(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Answer:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        bound_log = log.bind(trace_id=trace_id)

        scored = self._retriever.retrieve(query, self._retrieval_k, filt)
        if not scored:
            bound_log.info("no_context")
            return Answer(text=_NO_CONTEXT, citations=[], trace_id=trace_id, retrieval_scope=scope)

        context = self._assembler.assemble(
            query, [s.chunk for s in scored], self._token_budget
        )
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": self._prompt.format(context=context.text, question=query),
            }
        ]
        completion = self._llm.complete(messages)
        text, citations = validate_and_build_citations(completion.text, context)

        usage = {
            **completion.usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        bound_log.info("answered", citations=len(citations), latency_ms=usage["latency_ms"])
        return Answer(
            text=text,
            citations=citations,
            usage=usage,
            trace_id=trace_id,
            retrieval_scope=scope,
        )
