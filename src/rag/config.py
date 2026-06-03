from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets / infra
    gemini_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    test_database_url: str = ""   # integration tests only; e.g. .../rag_test

    # Models (config-driven; swap provider by changing the string)
    generation_model: str = "gemini/gemini-2.5-pro"
    grader_model: str = "gemini/gemini-2.5-flash"
    embedding_model: str = "gemini/text-embedding-004"
    embedding_dim: int = 768

    # Chunking
    chunk_tokens: int = 512
    chunk_overlap: int = 64

    # Retrieval / assembly
    retrieval_k: int = 10
    context_token_budget: int = 6000

    # Generation
    generation_temperature: float = 0.0
    generation_max_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
