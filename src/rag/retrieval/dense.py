from __future__ import annotations

from sqlalchemy import Engine, select

from rag.db import chunks as chunks_t
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk
from rag.protocols import EmbeddingProvider


class DenseRetriever:
    """pgvector cosine KNN over live (non-tombstoned) chunks, with optional filter."""

    def __init__(self, engine: Engine, embedder: EmbeddingProvider) -> None:
        self._engine = engine
        self._embedder = embedder

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        qvec = self._embedder.embed([query])[0]
        distance = chunks_t.c.embedding.cosine_distance(qvec)
        stmt = (
            select(chunks_t, distance.label("distance"))
            .where(chunks_t.c.deleted_at.is_(None))
            .order_by(distance)
            .limit(k)
        )
        if filt is not None:
            for key, value in filt.as_pairs():
                stmt = stmt.where(chunks_t.c.metadata[key].as_string() == value)

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [self._to_scored(row) for row in rows]

    @staticmethod
    def _to_scored(row) -> ScoredChunk:  # type: ignore[no-untyped-def]
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_uri=row["source_uri"],
            text=row["text"],
            ordinal=row["ordinal"],
            page=row["page"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            metadata=row["metadata"],
            content_hash=row["content_hash"],
            chunker_name=row["chunker_name"],
            chunker_version=row["chunker_version"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
        )
        score = 1.0 - float(row["distance"])  # cosine similarity
        return ScoredChunk(chunk=chunk, score=score, provenance=Provenance.dense)
