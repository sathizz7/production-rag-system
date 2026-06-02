from rag.config import Settings


def test_settings_defaults_use_gemini() -> None:
    s = Settings(gemini_api_key="x", database_url="postgresql+psycopg://u:p@h:5432/db")
    assert s.generation_model == "gemini/gemini-2.5-pro"
    assert s.grader_model == "gemini/gemini-2.5-flash"
    assert s.embedding_model == "gemini/text-embedding-004"
    assert s.embedding_dim == 768


def test_settings_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "envkey")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GENERATION_MODEL", "gemini/gemini-2.5-flash")
    s = Settings()
    assert s.gemini_api_key == "envkey"
    assert s.generation_model == "gemini/gemini-2.5-flash"
