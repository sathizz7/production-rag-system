from __future__ import annotations

from sqlalchemy import Engine, func, select

from rag.db import chunks as chunks_t
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk


class LexicalRetriever:
    """Postgres full-text search over live chunks via the generated tsvector.

    Uses ``websearch_to_tsquery`` (tolerant of free-form user input) and ranks
    with ``ts_rank_cd``. The language is frozen to ``'english'`` to match the STORED
    generated tsvector column (migration 0002) — querying with a different config
    than the column was built with silently returns wrong matches. NOTE: Postgres
    FTS rank is tf-idf-ish, not BM25 — the README documents this; hybrid fusion
    (RRF) ignores absolute scores anyway.
    """

    _LANGUAGE = "english"  # MUST equal the column's to_tsvector config (migration 0002)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        tsquery = func.websearch_to_tsquery(self._LANGUAGE, query)
        rank = func.ts_rank_cd(chunks_t.c.text_search, tsquery)
        stmt = (
            select(chunks_t, rank.label("rank"))
            .where(chunks_t.c.deleted_at.is_(None))
            .where(chunks_t.c.text_search.op("@@")(tsquery))
            .order_by(rank.desc())
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
        return ScoredChunk(chunk=chunk, score=float(row["rank"]), provenance=Provenance.lexical)
