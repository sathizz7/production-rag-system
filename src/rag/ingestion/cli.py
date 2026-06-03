from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from rag.config import apply_provider_env, get_settings
from rag.db import get_engine
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.repository import PgChunkRepository
from rag.providers.embeddings import LiteLLMEmbeddingProvider

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG store.")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or directories")
    args = parser.parse_args()

    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        log.warning("paths_not_found", paths=missing, cwd=str(Path.cwd()))

    settings = get_settings()
    apply_provider_env(settings)
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
    if stats.documents_upserted == 0:
        log.warning(
            "no_documents_ingested",
            hint="no readable PDFs found at the given path(s)",
            resolved=[str(p.resolve()) for p in args.paths],
        )
    log.info("ingest_complete", documents=stats.documents_upserted, chunks=stats.chunks_upserted)


if __name__ == "__main__":
    main()
