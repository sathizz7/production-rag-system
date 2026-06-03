from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def render_corpus_to_pdfs(corpus_dir: Path, out_dir: Path) -> list[Path]:
    """Render each eval/corpus/*.md into a one-page PDF named <stem>.pdf.

    Keeps the eval corpus diffable as text while exercising the real PDF pipeline.
    The doc_id assigned at ingest is a hash of the URI; retrieval is scored by the
    file stem parsed from source_uri, which equals these <stem>.pdf names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []
    for md_path in sorted(corpus_dir.glob("*.md")):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), md_path.read_text(encoding="utf-8"))
        pdf_path = out_dir / f"{md_path.stem}.pdf"
        doc.save(str(pdf_path))
        doc.close()
        pdf_paths.append(pdf_path)
    return pdf_paths
