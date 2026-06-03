import pytest
from sqlalchemy import Engine, inspect, text

pytestmark = pytest.mark.integration


def test_migration_creates_tables_and_hnsw_index(migrated_engine: Engine) -> None:
    insp = inspect(migrated_engine)
    assert {"documents", "chunks", "source_watermarks"} <= set(insp.get_table_names())

    with migrated_engine.connect() as conn:
        idx = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
        ).scalars().all()
    assert "ix_chunks_embedding_hnsw" in idx


def test_chunks_has_fts_column_and_index(migrated_engine) -> None:
    from sqlalchemy import text

    with migrated_engine.connect() as conn:
        col = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='chunks' AND column_name='text_search'"
            )
        ).scalar()
        idx = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_chunks_text_search'")
        ).scalar()
    assert col == "tsvector"
    assert idx is not None and "gin" in idx.lower()
