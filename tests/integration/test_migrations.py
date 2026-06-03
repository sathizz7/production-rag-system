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
