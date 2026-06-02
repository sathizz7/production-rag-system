from pathlib import Path

import fitz  # PyMuPDF
import pytest


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """A deterministic 2-page text PDF used across ingestion tests."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Maize responds strongly to nitrogen fertilizer.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Wheat yields decline under prolonged drought stress.")
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path
