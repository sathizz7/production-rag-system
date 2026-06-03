import pytest
from sqlalchemy import Engine, text

from rag.ingestion.repository import PgChunkRepository
from rag.models import Chunk, Watermark

pytestmark = pytest.mark.integration


def _vec(*head: float) -> list[float]:
    """768-dim test vector from leading components (rest zero-padded)."""
    v = [0.0] * 768
    for i, x in enumerate(head):
        v[i] = x
    return v


def _chunk(doc_id: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-c{ordinal}",
        doc_id=doc_id,
        source_uri="file:///x.pdf",
        text=f"text {ordinal}",
        ordinal=ordinal,
        page=1,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + 6,
        token_count=2,
        content_hash=f"h{doc_id}",
        chunker_name="fixed",
        chunker_version="1",
        embedding_model="fake",
        embedding_dim=768,
    )


def test_upsert_is_idempotent(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    chunks = [_chunk("d1", 0), _chunk("d1", 1)]
    vectors = [_vec(0.1, 0.2, 0.3), _vec(0.4, 0.5, 0.6)]

    repo.upsert(chunks, vectors)
    repo.upsert(chunks, vectors)  # second run must not duplicate

    with clean_db.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        d = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
    assert n == 2
    assert d == 1


def test_soft_delete_tombstones_document_and_chunks(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    repo.upsert([_chunk("d1", 0)], [_vec(0.1, 0.2, 0.3)])

    deleted = repo.soft_delete(["d1"])
    assert deleted == 1

    with clean_db.connect() as conn:
        live = conn.execute(
            text("SELECT count(*) FROM chunks WHERE deleted_at IS NULL")
        ).scalar_one()
    assert live == 0


def test_watermark_roundtrip(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    assert repo.get_watermark("s1") is None
    repo.set_watermark("s1", Watermark(source_id="s1", etag="abc"))
    wm = repo.get_watermark("s1")
    assert wm is not None and wm.etag == "abc"
