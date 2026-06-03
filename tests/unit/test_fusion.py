from rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_high_in_both_lists() -> None:
    dense = ["a", "b", "c"]
    lexical = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ids = [item_id for item_id, _ in fused]
    # 'a' (ranks 0,1) and 'b' (ranks 1,0) outrank singletons 'c' and 'd'
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids[2:]) == {"c", "d"}


def test_rrf_scores_are_descending_and_use_k() -> None:
    fused = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert fused[0] == ("x", 1.0 / 61)
    assert fused[1] == ("y", 1.0 / 62)
    assert fused[0][1] > fused[1][1]


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
