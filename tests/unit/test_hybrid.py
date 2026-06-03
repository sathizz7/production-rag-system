from rag.models import Provenance
from rag.retrieval.hybrid import HybridRetriever
from tests.unit.fakes import FakeRetriever, make_chunk


def test_hybrid_fuses_both_sources_and_marks_provenance() -> None:
    a, b, c, d = (make_chunk(x) for x in "abcd")
    dense = FakeRetriever([a, b, c], provenance=Provenance.dense)
    lexical = FakeRetriever([b, a, d], provenance=Provenance.lexical)
    hybrid = HybridRetriever(dense=dense, lexical=lexical, rrf_k=60, candidate_k=3)

    results = hybrid.retrieve("q", k=3, filt=None)

    assert all(r.provenance == Provenance.fused for r in results)
    assert {r.chunk.chunk_id for r in results[:2]} == {"a", "b"}
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    # both legs over-fetched the candidate pool (max(k, candidate_k) = 3 here)
    assert dense.last_call == ("q", 3, None)
    assert lexical.last_call == ("q", 3, None)


def test_hybrid_over_fetches_candidate_pool_beyond_k() -> None:
    pool = [make_chunk(str(i)) for i in range(10)]
    dense = FakeRetriever(pool)
    lexical = FakeRetriever(list(reversed(pool)))
    hybrid = HybridRetriever(dense=dense, lexical=lexical, rrf_k=60, candidate_k=8)
    results = hybrid.retrieve("q", k=3, filt=None)
    assert dense.last_call == ("q", 8, None)      # asked for the pool, not just k
    assert len(results) == 3                        # but returns the fused top-k


def test_hybrid_returns_empty_when_both_empty() -> None:
    hybrid = HybridRetriever(
        dense=FakeRetriever([]), lexical=FakeRetriever([]), rrf_k=60
    )
    assert hybrid.retrieve("q", k=5, filt=None) == []
