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


@lru_cache
def get_settings() -> Settings:
    return Settings()


def apply_provider_env(settings: Settings | None = None) -> None:
    """Export provider API keys from Settings (.env) into ``os.environ``.

    LiteLLM reads keys such as ``GEMINI_API_KEY`` from the process environment, but
    pydantic-settings only loads ``.env`` into the Settings object. Call this at
    app/CLI startup so a key placed in ``.env`` reaches LiteLLM without requiring
    ``uv run --env-file``. Pre-existing environment values win (``setdefault``).
    """
    settings = settings or get_settings()
    if settings.gemini_api_key:
        os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
