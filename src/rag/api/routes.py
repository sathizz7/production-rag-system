from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from rag.api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    answerer = request.app.state.answerer
    answer = await run_in_threadpool(
        answerer.answer, body.query, body.filter
    )
    return QueryResponse.from_answer(answer)


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> EventSourceResponse:
    answerer = request.app.state.answerer

    async def event_publisher() -> AsyncIterator[dict[str, str]]:
        # answer_stream is a sync generator (LiteLLM stream is sync); run it in a
        # threadpool so the event loop is never blocked.
        gen = answerer.answer_stream(body.query, body.filter)
        async for event in iterate_in_threadpool(gen):
            yield {"event": event.type, "data": event.model_dump_json()}

    return EventSourceResponse(event_publisher())
