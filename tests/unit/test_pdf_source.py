from pathlib import Path

from rag.ingestion.sources.pdf import PdfSourceAdapter


def test_pdf_adapter_yields_one_rawdocument_per_file(sample_pdf_path: Path) -> None:
    adapter = PdfSourceAdapter(paths=[sample_pdf_path])
    docs = list(adapter.fetch(since=None))
    assert len(docs) == 1
    raw = docs[0]
    assert raw.source_type == "pdf"
    assert raw.uri == sample_pdf_path.as_uri()
    assert raw.raw_bytes is not None and len(raw.raw_bytes) > 0


def test_pdf_adapter_expands_directory(tmp_path: Path, sample_pdf_path: Path) -> None:
    adapter = PdfSourceAdapter(paths=[sample_pdf_path.parent])
    docs = list(adapter.fetch(since=None))
    assert len(docs) == 1
