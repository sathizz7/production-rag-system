import pytest
from sqlalchemy import Engine

from rag.ingestion.repository import PgChunkRepository
from rag.models import Provenance
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.lexical import LexicalRetriever
from tests.integration.test_dense_retrieval import _chunk, _vec
from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def test_hybrid_surfaces_lexical_only_and_dense_only_hits(clean_db: Engine) -> None:
    # 'c0' is the dense nearest; 'c1' shares the query's lexical term ("nitrogen").
    mapping = {
        "Maize nitrogen response is strong.": _vec(1, 0, 0),
        "Irrigation scheduling for nitrogen uptake timing.": _vec(0, 0, 1),
        "nitrogen": _vec(1, 0, 0),   # the query vector → nearest to c0
    }
    embedder = FakeEmbedder(dim=768, mapping=mapping)
    repo = PgChunkRepository(clean_db)
    c0 = _chunk("c0", "Maize nitrogen response is strong.")
    c1 = _chunk("c1", "Irrigation scheduling for nitrogen uptake timing.")
    repo.upsert([c0, c1], [mapping[c0.text], mapping[c1.text]])

    hybrid = HybridRetriever(
        dense=DenseRetriever(engine=clean_db, embedder=embedder),
        lexical=LexicalRetriever(engine=clean_db),
        rrf_k=60,
    )
    results = hybrid.retrieve("nitrogen", k=5, filt=None)
    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"c0", "c1"}                       # dense-only + lexical-only both present
    assert all(r.provenance == Provenance.fused for r in results)
