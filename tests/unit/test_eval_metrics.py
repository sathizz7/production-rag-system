import math

from rag.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k() -> None:
    assert hit_at_k(["d2", "d1", "d3"], {"d1"}, k=2) == 1.0
    assert hit_at_k(["d2", "d3", "d1"], {"d1"}, k=2) == 0.0   # d1 is at rank 3
    assert hit_at_k([], {"d1"}, k=5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["d2", "d1", "d3"], {"d1"}) == 0.5  # first relevant at rank 2
    assert reciprocal_rank(["d1"], {"d1"}) == 1.0
    assert reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0


def test_ndcg_at_k() -> None:
    # one relevant doc at rank 2 → DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["d2", "d1", "d3"], {"d1"}, k=3), expected, rel_tol=1e-9)
    # perfect ranking → 1.0
    assert math.isclose(ndcg_at_k(["d1", "d2"], {"d1", "d2"}, k=2), 1.0, rel_tol=1e-9)
    assert ndcg_at_k(["d2"], {"d1"}, k=1) == 0.0
