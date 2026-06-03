from __future__ import annotations

from rag.models import MetadataFilter, ScoredChunk
from rag.protocols import Reranker, Retriever


class RerankedRetriever:
    """A Retriever decorator: over-fetch from a base retriever, then rerank.

    ``retrieve(query, k, filt)`` fetches ``max(k, candidate_k)`` candidates from the
    base (e.g. a HybridRetriever), reranks them with a cross-encoder, and returns
    the reranked top-``k`` (``provenance=rerank``). Because it IS a Retriever, the
    answerer never knows rerank is happening, and the eval A/B measures rerank lift
    by swapping the base retriever for this one — a clean before/after.
    """

    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        pool = max(k, self._candidate_k)
        scored = self._base.retrieve(query, pool, filt)
        if not scored:
            return []
        return self._reranker.rerank(query, [s.chunk for s in scored], top_n=k)
