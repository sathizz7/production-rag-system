import pytest
from sqlalchemy import Engine

from rag.ingestion.repository import PgChunkRepository
from rag.models import Chunk, MetadataFilter, Provenance
from rag.retrieval.dense import DenseRetriever
from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def _vec(*head: float) -> list[float]:
    """768-dim test vector from leading components (rest zero-padded)."""
    v = [0.0] * 768
    for i, x in enumerate(head):
        v[i] = x
    return v


def _chunk(cid: str, text: str, meta: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=int(cid[-1]),
        page=1,
        char_start=0,
        char_end=len(text),
        token_count=2,
        metadata=meta or {},
        content_hash=cid,
        chunker_name="fixed",
        chunker_version="1",
        embedding_model="fake",
        embedding_dim=768,
    )


def _seed(engine: Engine) -> FakeEmbedder:
    # explicit vectors: 'near' is closest to the query vector _vec(1, 0, 0)
    mapping = {
        "near": _vec(1, 0, 0),
        "mid": _vec(0.7, 0.7, 0),
        "far": _vec(0, 0, 1),
        "QUERY": _vec(1, 0, 0),
    }
    embedder = FakeEmbedder(dim=768, mapping=mapping)
    repo = PgChunkRepository(engine)
    chunks = [_chunk("c0", "near"), _chunk("c1", "mid"), _chunk("c2", "far")]
    repo.upsert(chunks, [mapping["near"], mapping["mid"], mapping["far"]])
    return embedder


def test_dense_retrieval_orders_by_cosine(clean_db: Engine) -> None:
    embedder = _seed(clean_db)
    retriever = DenseRetriever(engine=clean_db, embedder=embedder)
    results = retriever.retrieve("QUERY", k=3, filt=None)

    assert [r.chunk.text for r in results] == ["near", "mid", "far"]
    assert results[0].provenance == Provenance.dense
    assert results[0].score >= results[1].score


def test_dense_retrieval_applies_metadata_filter(clean_db: Engine) -> None:
    mapping = {
        "south": _vec(1, 0, 0),
        "north": _vec(0.9, 0.1, 0),
        "QUERY": _vec(1, 0, 0),
    }
    embedder = FakeEmbedder(dim=768, mapping=mapping)
    repo = PgChunkRepository(clean_db)
    repo.upsert(
        [_chunk("c0", "south", {"region": "south"}), _chunk("c1", "north", {"region": "north"})],
        [mapping["south"], mapping["north"]],
    )
    retriever = DenseRetriever(engine=clean_db, embedder=embedder)
    results = retriever.retrieve("QUERY", k=5, filt=MetadataFilter(region="south"))
    assert [r.chunk.text for r in results] == ["south"]
