from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from sqlalchemy import Engine, text


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """A deterministic 2-page text PDF used across ingestion tests."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Maize responds strongly to nitrogen fertilizer.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Wheat yields decline under prolonged drought stress.")
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    """Local Postgres+pgvector (TEST_DATABASE_URL): reset schema, migrate, yield engine."""
    from rag.config import get_settings
    from rag.db import get_engine, run_migrations

    url = get_settings().test_database_url           # from .env; empty -> skip
    if not url:
        pytest.skip("TEST_DATABASE_URL not set — integration tests need local Postgres+pgvector")

    reset = get_engine(url)
    with reset.begin() as conn:                      # reset the DEDICATED test DB to empty
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    reset.dispose()

    run_migrations(url)
    engine = get_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def clean_db(migrated_engine: Engine) -> Engine:
    with migrated_engine.begin() as conn:
        conn.execute(text("TRUNCATE chunks, documents, source_watermarks"))
    return migrated_engine
