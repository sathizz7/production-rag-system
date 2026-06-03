import pytest
from sqlalchemy import Engine

from rag.models import Provenance
from rag.retrieval.lexical import LexicalRetriever
from tests.integration.test_dense_retrieval import _chunk, _vec

pytestmark = pytest.mark.integration


def _seed(engine: Engine) -> None:
    from rag.ingestion.repository import PgChunkRepository

    repo = PgChunkRepository(engine)
    chunks = [
        _chunk("c0", "Nitrogen fertilizer increases maize yield substantially."),
        _chunk("c1", "Drought stress reduces wheat grain filling."),
        _chunk("c2", "Phosphorus supports early root development in maize."),
    ]
    repo.upsert(chunks, [_vec(1.0), _vec(1.0), _vec(1.0)])


def test_lexical_matches_query_terms(clean_db: Engine) -> None:
    _seed(clean_db)
    retriever = LexicalRetriever(engine=clean_db)
    results = retriever.retrieve("nitrogen maize yield", k=5, filt=None)

    texts = [r.chunk.text for r in results]
    assert any("Nitrogen" in t for t in texts)
    assert results[0].provenance == Provenance.lexical
    assert all("Drought" not in t for t in texts)  # no shared lexical terms


def test_lexical_returns_empty_on_no_match(clean_db: Engine) -> None:
    _seed(clean_db)
    retriever = LexicalRetriever(engine=clean_db)
    assert retriever.retrieve("zzzqxnonsense", k=5, filt=None) == []
