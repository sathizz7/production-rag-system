from __future__ import annotations

from fastapi import FastAPI

from rag.api.routes import router
from rag.config import apply_provider_env, get_settings
from rag.db import get_engine
from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.providers.embeddings import LiteLLMEmbeddingProvider
from rag.providers.llm import LiteLLMProvider
from rag.retrieval.dense import DenseRetriever


def _build_answerer() -> StraightLineAnswerer:
    settings = get_settings()
    apply_provider_env(settings)
    engine = get_engine(settings.database_url)
    embedder = LiteLLMEmbeddingProvider(
        model=settings.embedding_model, dim=settings.embedding_dim
    )
    retriever = DenseRetriever(engine=engine, embedder=embedder)
    return StraightLineAnswerer(
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


def create_app(answerer: StraightLineAnswerer | None = None) -> FastAPI:
    app = FastAPI(title="Production RAG — P0a")
    app.state.answerer = answerer if answerer is not None else _build_answerer()
    app.include_router(router)
    return app
