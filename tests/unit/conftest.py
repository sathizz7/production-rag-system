"""Unit-test fixtures and environment isolation helpers."""
from __future__ import annotations

import pytest

# Env vars that map to Settings fields. Importing litellm (transitively pulled in by
# many modules) auto-loads the project .env into os.environ via python-dotenv, and
# Settings(_env_file=None) still reads os.environ — so without this cleanup a developer
# whose .env defines e.g. EVAL_DATABASE_URL would see "default/unset" assertions fail.
# Clearing them per-test keeps unit tests hermetic regardless of the local .env.
_RAG_ENV_VARS = (
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "DATABASE_URL",
    "TEST_DATABASE_URL",
    "EVAL_DATABASE_URL",
    "GENERATION_MODEL",
    "GRADER_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RRF_K",
    "CANDIDATE_K",
    "RERANK_MODEL",
    "RERANK_ENABLED",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


@pytest.fixture(autouse=True)
def _isolate_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip RAG env vars before each unit test so .env never leaks into Settings."""
    for key in _RAG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
