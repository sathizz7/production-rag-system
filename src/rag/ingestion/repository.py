from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag.db import chunks as chunks_t
from rag.db import documents as documents_t
from rag.db import watermarks as watermarks_t
from rag.models import Chunk, UpsertStats, Watermark

Vector = list[float]


class PgChunkRepository:
    """The only component that touches Postgres. Idempotent via content_hash/ordinal."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> UpsertStats:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must align")
        now = datetime.now(UTC)
        stats = UpsertStats()
        by_doc: dict[str, list[tuple[Chunk, Vector]]] = {}
        for ch, vec in zip(chunks, vectors, strict=True):
            by_doc.setdefault(ch.doc_id, []).append((ch, vec))

        with self._engine.begin() as conn:
            for doc_id, items in by_doc.items():
                first = items[0][0]
                doc_stmt = (
                    pg_insert(documents_t)
                    .values(
                        doc_id=doc_id,
                        source_id=first.metadata.get("source_id", "pdf-corpus"),
                        source_type="pdf",
                        uri=first.source_uri,
                        document_version=1,
                        content_hash=first.metadata.get(
                            "document_content_hash", first.content_hash
                        ),
                        metadata=first.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=["doc_id"],
                        set_={
                            "content_hash": first.metadata.get(
                                "document_content_hash", first.content_hash
                            ),
                            "metadata": first.metadata,
                            "updated_at": now,
                            "deleted_at": None,
                        },
                    )
                )
                conn.execute(doc_stmt)
                stats.documents_upserted += 1

                for ch, vec in items:
                    chunk_stmt = (
                        pg_insert(chunks_t)
                        .values(
                            chunk_id=ch.chunk_id,
                            doc_id=ch.doc_id,
                            source_uri=ch.source_uri,
                            text=ch.text,
                            ordinal=ch.ordinal,
                            page=ch.page,
                            char_start=ch.char_start,
                            char_end=ch.char_end,
                            token_count=ch.token_count,
                            metadata=ch.metadata,
                            content_hash=ch.content_hash,
                            chunker_name=ch.chunker_name,
                            chunker_version=ch.chunker_version,
                            embedding_model=ch.embedding_model,
                            embedding_dim=ch.embedding_dim,
                            embedding=vec,
                            created_at=now,
                        )
                        .on_conflict_do_update(
                            constraint="uq_chunks_doc_ordinal",
                            set_={
                                "text": ch.text,
                                "embedding": vec,
                                "content_hash": ch.content_hash,
                                "char_start": ch.char_start,
                                "char_end": ch.char_end,
                                "page": ch.page,
                                "token_count": ch.token_count,
                                "deleted_at": None,
                            },
                        )
                    )
                    conn.execute(chunk_stmt)
                    stats.chunks_upserted += 1
        return stats

    def soft_delete(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        now = datetime.now(UTC)
        with self._engine.begin() as conn:
            conn.execute(
                chunks_t.update()
                .where(chunks_t.c.doc_id.in_(doc_ids))
                .values(deleted_at=now)
            )
            result = conn.execute(
                documents_t.update()
                .where(documents_t.c.doc_id.in_(doc_ids))
                .values(deleted_at=now)
            )
            return int(result.rowcount or 0)

    def get_watermark(self, source_id: str) -> Watermark | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(watermarks_t).where(watermarks_t.c.source_id == source_id)
            ).mappings().first()
        if row is None:
            return None
        return Watermark(
            source_id=row["source_id"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            updated_at=row["updated_at"],
        )

    def set_watermark(self, source_id: str, wm: Watermark) -> None:
        now = datetime.now(UTC)
        stmt = (
            pg_insert(watermarks_t)
            .values(
                source_id=source_id,
                etag=wm.etag,
                last_modified=wm.last_modified,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["source_id"],
                set_={"etag": wm.etag, "last_modified": wm.last_modified, "updated_at": now},
            )
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
