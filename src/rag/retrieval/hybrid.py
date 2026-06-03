from __future__ import annotations

from rag.models import MetadataFilter, Provenance, ScoredChunk
from rag.observability.metrics import observe_stage
from rag.protocols import Retriever
from rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Runs dense + lexical retrieval and fuses their rankings with RRF.

    Each leg is over-fetched to ``max(k, candidate_k)`` candidates so RRF fuses a
    rich pool (fusing only top-k from each leg loses recall); the fused top-``k``
    are returned as ``ScoredChunk`` with ``provenance=fused`` and the RRF score.
    The same ``filt`` passes through to both legs — the geospatial metadata edge
    works identically on each.
    """

    def __init__(
        self, dense: Retriever, lexical: Retriever, rrf_k: int = 60, candidate_k: int = 30
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        pool = max(k, self._candidate_k)
        with observe_stage("dense"):
            dense = self._dense.retrieve(query, pool, filt)
        with observe_stage("lexical"):
            lexical = self._lexical.retrieve(query, pool, filt)
        with observe_stage("fusion"):
            by_id = {sc.chunk.chunk_id: sc.chunk for sc in (*dense, *lexical)}
            fused = reciprocal_rank_fusion(
                [[sc.chunk.chunk_id for sc in dense], [sc.chunk.chunk_id for sc in lexical]],
                k=self._rrf_k,
            )
            return [
                ScoredChunk(chunk=by_id[chunk_id], score=score, provenance=Provenance.fused)
                for chunk_id, score in fused[:k]
            ]
