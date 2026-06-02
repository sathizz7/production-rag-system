from pathlib import Path

from rag.ingestion.parse import PdfParser
from rag.ingestion.sources.pdf import PdfSourceAdapter


def test_parser_builds_document_with_pages(sample_pdf_path: Path) -> None:
    raw = next(PdfSourceAdapter(paths=[sample_pdf_path]).fetch(since=None))
    doc = PdfParser().parse(raw)

    assert len(doc.pages) == 2
    assert "nitrogen" in doc.pages[0].text
    assert "drought" in doc.pages[1].text
    assert doc.source_type == "pdf"
    assert doc.content_hash != ""


def test_parser_page_offsets_are_consistent(sample_pdf_path: Path) -> None:
    raw = next(PdfSourceAdapter(paths=[sample_pdf_path]).fetch(since=None))
    doc = PdfParser().parse(raw)
    # page_at on the first char of page 2 must return 2
    _, start_p2, _ = doc.page_spans()[1]
    assert doc.page_at(start_p2) == 2
