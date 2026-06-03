from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

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
