from __future__ import annotations

from pathlib import Path

import structlog

from rag.ingestion.chunking.fixed import FixedTokenChunker
from rag.ingestion.clean import BasicCleaner
from rag.ingestion.parse import PdfParser
from rag.ingestion.repository import PgChunkRepository
from rag.ingestion.sources.pdf import PdfSourceAdapter
from rag.models import Chunk, RawDocument, UpsertStats
from rag.protocols import EmbeddingProvider

log = structlog.get_logger()


class IngestionPipeline:
    """Wires the write path: PDF -> parse -> clean -> chunk -> embed -> upsert."""

    def __init__(
        self,
        repository: PgChunkRepository,
        embedder: EmbeddingProvider,
        chunk_tokens: int = 512,
        overlap: int = 64,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._parser = PdfParser()
        self._cleaner = BasicCleaner()
        self._chunker = FixedTokenChunker(chunk_tokens=chunk_tokens, overlap=overlap)

    def ingest_paths(self, paths: list[Path]) -> UpsertStats:
        adapter = PdfSourceAdapter(paths=paths)
        total = UpsertStats()
        for raw in adapter.fetch(since=None):
            try:
                stats = self._ingest_one(raw)
            except Exception as exc:  # per-doc isolation (spec §16)
                log.error("ingest_failed", uri=raw.uri, error=str(exc))
                continue
            total.documents_upserted += stats.documents_upserted
            total.chunks_upserted += stats.chunks_upserted
        return total

    def _ingest_one(self, raw: RawDocument) -> UpsertStats:
        doc = self._cleaner.clean(self._parser.parse(raw))
        chunks: list[Chunk] = self._chunker.chunk(doc)
        if not chunks:
            log.warning("no_chunks", uri=raw.uri)
            return UpsertStats()
        for ch in chunks:
            ch.embedding_model = self._embedder.model
            ch.embedding_dim = self._embedder.dim
            ch.metadata = {
                **ch.metadata,
                "source_id": doc.source_id,
                "document_content_hash": doc.content_hash,
            }
        vectors = self._embedder.embed([c.text for c in chunks])
        stats = self._repo.upsert(chunks, vectors)
        log.info("ingested", uri=raw.uri, chunks=stats.chunks_upserted)
        return stats
