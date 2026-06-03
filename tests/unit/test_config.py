from rag.config import Settings


def test_settings_defaults_use_gemini() -> None:
    s = Settings(
        _env_file=None, gemini_api_key="x", database_url="postgresql+psycopg://u:p@h:5432/db"
    )
    assert s.generation_model == "gemini/gemini-2.5-pro"
    assert s.grader_model == "gemini/gemini-2.5-flash"
    assert s.embedding_model == "gemini/gemini-embedding-001"
    assert s.embedding_dim == 768


def test_settings_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "envkey")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GENERATION_MODEL", "gemini/gemini-2.5-flash")
    s = Settings()
    assert s.gemini_api_key == "envkey"
    assert s.generation_model == "gemini/gemini-2.5-flash"


def test_settings_has_p0b_fields() -> None:
    from rag.config import Settings

    s = Settings(_env_file=None)
    assert s.rrf_k == 60
    assert s.rerank_model == "cohere/rerank-english-v3.0"
    assert s.rerank_enabled is False          # off until a Cohere key is present
    assert s.candidate_k == 30                 # over-fetch pool for fusion + rerank
    assert s.eval_database_url == ""           # isolated eval DB; empty until configured
    assert s.langfuse_host == "https://cloud.langfuse.com"


def test_apply_provider_env_exports_optional_keys(monkeypatch) -> None:
    import os

    from rag.config import Settings, apply_provider_env

    for var in ("GEMINI_API_KEY", "COHERE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(
        _env_file=None,
        gemini_api_key="g",
        cohere_api_key="c",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    apply_provider_env(s)
    assert os.environ["GEMINI_API_KEY"] == "g"
    assert os.environ["COHERE_API_KEY"] == "c"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk"
