from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from rag.config import get_settings
from rag.db import get_engine
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.repository import PgChunkRepository
from rag.providers.embeddings import LiteLLMEmbeddingProvider

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG store.")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or directories")
    args = parser.parse_args()

    settings = get_settings()
    engine = get_engine(settings.database_url)
    pipeline = IngestionPipeline(
        repository=PgChunkRepository(engine),
        embedder=LiteLLMEmbeddingProvider(
            model=settings.embedding_model, dim=settings.embedding_dim
        ),
        chunk_tokens=settings.chunk_tokens,
        overlap=settings.chunk_overlap,
    )
    stats = pipeline.ingest_paths(args.paths)
    log.info("ingest_complete", documents=stats.documents_upserted, chunks=stats.chunks_upserted)


if __name__ == "__main__":
    main()
