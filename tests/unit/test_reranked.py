from rag.models import Provenance
from rag.retrieval.reranked import RerankedRetriever
from tests.unit.fakes import FakeReranker, FakeRetriever, make_chunk


def test_reranked_over_fetches_pool_then_returns_reranked_top_k() -> None:
    pool = [make_chunk(str(i)) for i in range(10)]
    base = FakeRetriever(pool)
    reranker = FakeReranker()
    rr = RerankedRetriever(base=base, reranker=reranker, candidate_k=8)

    results = rr.retrieve("q", k=3, filt=None)

    assert base.last_call == ("q", 8, None)             # over-fetched candidate_k (pool of 8)
    assert reranker.calls == [("q", 3)]                  # reranked down to k
    # base returns top-8 (ids 0..7); FakeReranker reverses → top-3 = 7,6,5
    assert [r.chunk.chunk_id for r in results] == ["7", "6", "5"]
    assert all(r.provenance == Provenance.rerank for r in results)


def test_reranked_uses_k_when_larger_than_candidate_k() -> None:
    base = FakeRetriever([make_chunk(str(i)) for i in range(5)])
    rr = RerankedRetriever(base=base, reranker=FakeReranker(), candidate_k=2)
    rr.retrieve("q", k=4, filt=None)
    assert base.last_call == ("q", 4, None)              # pool = max(k, candidate_k)


def test_reranked_empty_base_short_circuits() -> None:
    reranker = FakeReranker()
    rr = RerankedRetriever(base=FakeRetriever([]), reranker=reranker, candidate_k=8)
    assert rr.retrieve("q", k=3, filt=None) == []
    assert reranker.calls == []                           # never reranks nothing
