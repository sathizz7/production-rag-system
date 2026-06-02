from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from rag.models import RawDocument, Watermark


class PdfSourceAdapter:
    """Yields one RawDocument per .pdf file. Phase-0 full-scan only (since is ignored)."""

    source_type = "pdf"

    def __init__(self, paths: list[Path], source_id: str = "pdf-corpus") -> None:
        self._paths = paths
        self._source_id = source_id

    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]:
        for pdf_path in self._iter_pdf_files():
            yield RawDocument(
                source_id=self._source_id,
                source_type=self.source_type,
                uri=pdf_path.as_uri(),
                raw_bytes=pdf_path.read_bytes(),
                fetched_at=datetime.now(UTC),
                source_meta={"filename": pdf_path.name},
            )

    def _iter_pdf_files(self) -> Iterator[Path]:
        for p in self._paths:
            if p.is_dir():
                yield from sorted(p.rglob("*.pdf"))
            elif p.suffix.lower() == ".pdf":
                yield p
