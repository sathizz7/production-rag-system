from rag.models import Document, Page, RetrievalScope, SourceKind


def test_document_text_joins_pages_with_separator() -> None:
    doc = _doc(["alpha", "bravo"])
    assert doc.text == "alpha\n\nbravo"


def test_page_at_maps_offsets_to_page_numbers() -> None:
    doc = _doc(["alpha", "bravo"])  # 'alpha'=0..5, sep=5..7, 'bravo'=7..12
    assert doc.page_at(0) == 1
    assert doc.page_at(4) == 1
    assert doc.page_at(7) == 2
    assert doc.page_at(11) == 2


def test_enums_are_string_valued() -> None:
    assert RetrievalScope.corpus_only.value == "corpus_only"
    assert SourceKind.corpus.value == "corpus"


def _doc(page_texts: list[str]) -> Document:
    return Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=i + 1, text=t) for i, t in enumerate(page_texts)],
    )
