from __future__ import annotations

from fastapi import FastAPI

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
    return app
