from __future__ import annotations

import litellm

from rag.models import Chunk, Provenance, ScoredChunk


class CohereReranker:
    """Cross-encoder rerank via ``litellm.rerank`` (Cohere by default).

    Sends the chunk texts as documents and reorders by the provider's relevance
    score, keeping the top ``top_n``. A local ``bge-reranker-v2-m3`` is the
    offline swap — change ``model`` only (spec §8). Verify the response shape
    against the current LiteLLM docs; results expose ``index`` + ``relevance_score``
    (attribute or mapping access are both handled below).
    """

    def __init__(self, model: str = "cohere/rerank-english-v3.0", top_n: int = 8) -> None:
        self.model = model
        self._top_n = top_n

    def rerank(
        self, query: str, chunks: list[Chunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        n = min(top_n or self._top_n, len(chunks))
        resp = litellm.rerank(
            model=self.model,
            query=query,
            documents=[c.text for c in chunks],
            top_n=n,
        )
        out: list[ScoredChunk] = []
        for r in resp.results:
            idx = r["index"] if isinstance(r, dict) else r.index
            score = r["relevance_score"] if isinstance(r, dict) else r.relevance_score
            out.append(
                ScoredChunk(chunk=chunks[idx], score=float(score), provenance=Provenance.rerank)
            )
        return out
