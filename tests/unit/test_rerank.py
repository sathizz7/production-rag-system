import litellm

from rag.models import Provenance
from rag.providers.rerank import CohereReranker
from tests.unit.fakes import make_chunk


class _FakeRerankResponse:
    # LiteLLM returns an object with a .results list of {index, relevance_score}
    def __init__(self, results: list[dict]) -> None:
        self.results = results


def test_rerank_reorders_by_relevance_and_truncates(monkeypatch) -> None:
    chunks = [make_chunk("a", "alpha"), make_chunk("b", "beta"), make_chunk("c", "gamma")]
    captured: dict = {}

    def fake_rerank(*, model, query, documents, top_n):
        captured.update(model=model, query=query, documents=documents, top_n=top_n)
        # provider says doc index 2 is best, then 0; drops index 1
        return _FakeRerankResponse([{"index": 2, "relevance_score": 0.9},
                                    {"index": 0, "relevance_score": 0.4}])

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    reranker = CohereReranker(model="cohere/rerank-english-v3.0", top_n=2)
    out = reranker.rerank("which is best?", chunks, top_n=2)

    assert [s.chunk.chunk_id for s in out] == ["c", "a"]
    assert [s.score for s in out] == [0.9, 0.4]
    assert all(s.provenance == Provenance.rerank for s in out)
    assert captured["documents"] == ["alpha", "beta", "gamma"]
    assert captured["top_n"] == 2


def test_rerank_empty_input_short_circuits(monkeypatch) -> None:
    def boom(**kwargs):  # must NOT be called
        raise AssertionError("litellm.rerank should not run on empty input")

    monkeypatch.setattr(litellm, "rerank", boom)
    assert CohereReranker().rerank("q", [], top_n=5) == []
