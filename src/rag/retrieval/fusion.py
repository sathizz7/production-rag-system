from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists by Reciprocal Rank Fusion.

    Each list is in best-first order. An id at 0-based rank ``r`` in a list
    contributes ``1 / (k + r + 1)``; contributions sum across lists. Returns
    ``(id, score)`` pairs sorted by score descending. Rank-based, so it needs no
    score normalisation between dense and lexical sources (spec §4).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
