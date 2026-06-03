from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2


def hit_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    relevant = set(relevant_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant id (1-based), or 0.0 if none retrieved."""
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k. IDCG assumes all relevant docs ranked first."""
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / log2(index + 2)
        for index, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
