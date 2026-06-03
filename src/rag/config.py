import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env at the project root so the app works from any working directory
# (e.g. when run from src/). This file is src/rag/config.py → parents[2] is root.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets / infra
    gemini_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    test_database_url: str = ""   # integration tests only; e.g. .../rag_test

    # Models (config-driven; swap provider by changing the string)
    generation_model: str = "gemini/gemini-2.5-pro"
    grader_model: str = "gemini/gemini-2.5-flash"
    embedding_model: str = "gemini/gemini-embedding-001"
    embedding_dim: int = 768

    # Chunking
    chunk_tokens: int = 512
    chunk_overlap: int = 64

    # Retrieval / assembly
    retrieval_k: int = 10
    context_token_budget: int = 6000

    # Generation
    generation_temperature: float = 0.0
    generation_max_tokens: int = 4096  # headroom for gemini-2.5 "thinking" + the answer

    # Hybrid retrieval
    rrf_k: int = 60                       # RRF damping constant
    candidate_k: int = 30                 # over-fetch per leg before fusion; rerank pool size
    # NOTE: Postgres FTS language is frozen to 'english' in P0b. It MUST match the
    # STORED generated tsvector column (migration 0002: to_tsvector('english', ...)).
    # A runtime language knob would silently desync the query language from the column,
    # so there is intentionally none — multi-language is a future migration, not config.

    # Rerank (Cohere via LiteLLM); disabled until a key is present
    cohere_api_key: str = ""
    rerank_model: str = "cohere/rerank-english-v3.0"
    rerank_enabled: bool = False

    # Eval (dedicated DB so the harness NEVER writes the app/dev database)
    eval_database_url: str = ""           # e.g. .../rag_eval ; empty -> rag-eval errors out

    # Observability (all optional; absent keys → features no-op)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def apply_provider_env(settings: Settings | None = None) -> None:
    """Export provider API keys from Settings (.env) into ``os.environ`` for LiteLLM.

    LiteLLM and the Langfuse callback read keys from the process environment, but
    pydantic-settings only loads ``.env`` into the Settings object. Call this at
    app/CLI startup so plain ``uv run`` works without ``--env-file``. Pre-existing
    environment values win (``setdefault``).
    """
    settings = settings or get_settings()
    pairs = {
        "GEMINI_API_KEY": settings.gemini_api_key,
        "COHERE_API_KEY": settings.cohere_api_key,
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_HOST": settings.langfuse_host,
    }
    for key, value in pairs.items():
        if value:
            os.environ.setdefault(key, value)
