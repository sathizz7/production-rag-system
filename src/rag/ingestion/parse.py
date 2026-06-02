from __future__ import annotations

import uuid

import fitz  # PyMuPDF

from rag.models import Document, Page, RawDocument
from rag.util.hashing import content_hash


class PdfParser:
    """Extracts per-page text from a PDF's text layer (no OCR — that is P0c)."""

    def parse(self, raw: RawDocument) -> Document:
        if raw.raw_bytes is None:
            raise ValueError(f"PdfParser requires raw_bytes; got none for {raw.uri}")

        pages: list[Page] = []
        with fitz.open(stream=raw.raw_bytes, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text("text") or ""
                pages.append(Page(number=i + 1, text=text.strip()))

        # uuid5 is deterministic: same URI always yields the same doc_id -> idempotent re-ingest
        doc = Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_URL, raw.uri)),
            source_id=raw.source_id,
            source_type=raw.source_type,
            uri=raw.uri,
            pages=pages,
            license=raw.license,
            metadata=dict(raw.source_meta),
        )
        doc.content_hash = content_hash(doc.text)
        return doc
