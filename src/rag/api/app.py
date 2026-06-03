from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest

from rag.api.routes import router
from rag.config import apply_provider_env, get_settings
from rag.db import get_engine
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from rag.observability.tracing import configure_observability
from rag.protocols import Retriever
from rag.providers.embeddings import LiteLLMEmbeddingProvider
from rag.providers.llm import LiteLLMProvider
from rag.providers.rerank import CohereReranker
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.lexical import LexicalRetriever
from rag.retrieval.reranked import RerankedRetriever


def _build_answerer() -> StreamingAnswerer:
    settings = get_settings()
    apply_provider_env(settings)
    configure_observability(settings)
    engine = get_engine(settings.database_url)
    embedder = LiteLLMEmbeddingProvider(
        model=settings.embedding_model, dim=settings.embedding_dim
    )
    retriever: Retriever = HybridRetriever(
        dense=DenseRetriever(engine=engine, embedder=embedder),
        lexical=LexicalRetriever(engine=engine),   # FTS frozen to english (migration 0002)
        rrf_k=settings.rrf_k,
        candidate_k=settings.candidate_k,
    )
    if settings.rerank_enabled:
        # Decorate the hybrid retriever — the answerer never knows rerank happened.
        retriever = RerankedRetriever(
            base=retriever,
            reranker=CohereReranker(model=settings.rerank_model),
            candidate_k=settings.candidate_k,
        )
    return StreamingAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=LiteLLMProvider(
            model=settings.generation_model,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        ),
        token_budget=settings.context_token_budget,
        retrieval_k=settings.retrieval_k,
    )


def create_app(answerer: StreamingAnswerer | None = None) -> FastAPI:
    app = FastAPI(title="Production RAG — P0b")
    app.state.answerer = answerer if answerer is not None else _build_answerer()
    app.include_router(router)

    @app.middleware("http")
    async def _record_metrics(request: StarletteRequest, call_next):  # type: ignore[no-untyped-def]
        start = perf_counter()
        response = await call_next(request)
        # Raw path as a label is safe for our fixed route set; if path-param routes are
        # added later, switch to the matched route template to avoid Prometheus
        # label-cardinality blow-up from unique paths / scanner traffic.
        endpoint = request.url.path
        from rag.observability import metrics

        metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - start)
        metrics.REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        return response

    ui_dir = Path(__file__).resolve().parents[3] / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app
