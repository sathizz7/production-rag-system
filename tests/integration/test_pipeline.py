from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.repository import PgChunkRepository
from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def _pipeline(engine: Engine) -> IngestionPipeline:
    return IngestionPipeline(
        repository=PgChunkRepository(engine),
        embedder=FakeEmbedder(dim=768),
        chunk_tokens=64,
        overlap=8,
    )


def test_pipeline_ingests_pdf_into_postgres(clean_db: Engine, sample_pdf_path: Path) -> None:
    stats = _pipeline(clean_db).ingest_paths([sample_pdf_path])

    assert stats.documents_upserted == 1
    assert stats.chunks_upserted >= 1
    with clean_db.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        dim = conn.execute(text("SELECT embedding_dim FROM chunks LIMIT 1")).scalar_one()
    assert n == stats.chunks_upserted
    assert dim == 768


def test_pipeline_is_idempotent(clean_db: Engine, sample_pdf_path: Path) -> None:
    pipe = _pipeline(clean_db)
    first = pipe.ingest_paths([sample_pdf_path])
    pipe.ingest_paths([sample_pdf_path])
    with clean_db.connect() as conn:
        docs = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
        n_chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    assert docs == 1
    assert n_chunks == first.chunks_upserted  # stable doc_id => no duplicate chunks
