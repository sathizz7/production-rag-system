from __future__ import annotations

import os

from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)

from alembic import command

EMBEDDING_DIM = 768  # P0a default (text-embedding-004). A model swap = a new migration.

metadata_obj = MetaData()

documents = Table(
    "documents",
    metadata_obj,
    Column("doc_id", String, primary_key=True),
    Column("source_id", String, nullable=False),
    Column("source_type", String, nullable=False),
    Column("uri", Text, nullable=False),
    Column("document_version", Integer, nullable=False, default=1),
    Column("content_hash", String, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("license", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

chunks = Table(
    "chunks",
    metadata_obj,
    Column("chunk_id", String, primary_key=True),
    Column("doc_id", String, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("page", Integer, nullable=False),
    Column("char_start", Integer, nullable=False),
    Column("char_end", Integer, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("content_hash", String, nullable=False),
    Column("chunker_name", String, nullable=False),
    Column("chunker_version", String, nullable=False),
    Column("embedding_model", String, nullable=False),
    Column("embedding_dim", Integer, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),
)

watermarks = Table(
    "source_watermarks",
    metadata_obj,
    Column("source_id", String, primary_key=True),
    Column("etag", String, nullable=True),
    Column("last_modified", String, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=True),
)


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True, pool_size=10, max_overflow=5)


def run_migrations(database_url: str) -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    prev = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url      # env.py reads this
    try:
        command.upgrade(cfg, "head")
    finally:
        if prev is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev
