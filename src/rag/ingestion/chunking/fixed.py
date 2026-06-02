from __future__ import annotations

import uuid

from rag.models import Chunk, Document
from rag.util.hashing import content_hash
from rag.util.tokens import get_encoder


class FixedTokenChunker:
    """Sliding fixed-size token windows with overlap.

    Char offsets are recovered from token offsets via the concatenative property
    of byte-level BPE: ``decode(tokens[:i])`` is exactly ``text[:char_start]``.
    """

    name = "fixed"
    version = "1"

    def __init__(self, chunk_tokens: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_tokens:
            raise ValueError("overlap must be smaller than chunk_tokens")
        self._chunk_tokens = chunk_tokens
        self._overlap = overlap
        self._enc = get_encoder()

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        stride = self._chunk_tokens - self._overlap
        chunks: list[Chunk] = []
        for ordinal, i in enumerate(range(0, len(tokens), stride)):
            window = tokens[i : i + self._chunk_tokens]
            char_start = len(self._enc.decode(tokens[:i]))
            chunk_text = self._enc.decode(window)
            char_end = char_start + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}:{ordinal}")),
                    doc_id=doc.doc_id,
                    source_uri=doc.uri,
                    text=chunk_text,
                    ordinal=ordinal,
                    page=doc.page_at(char_start),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(window),
                    metadata=dict(doc.metadata),
                    content_hash=content_hash(chunk_text),
                    chunker_name=self.name,
                    chunker_version=self.version,
                )
            )
            if i + self._chunk_tokens >= len(tokens):
                break
        return chunks
