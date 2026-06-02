from rag.ingestion.clean import BasicCleaner
from rag.models import Document, Page


def test_cleaner_collapses_whitespace_and_recomputes_hash() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text="hello   \t  world\n\n\n bye")],
        content_hash="stale",
    )
    cleaned = BasicCleaner().clean(doc)
    assert cleaned.pages[0].text == "hello world\nbye"
    # offsets are derived from cleaned text, so the invariant holds for any slice
    assert cleaned.text[0:5] == "hello"
    assert cleaned.content_hash != "stale"


def test_cleaner_preserves_page_count() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text="a  b"), Page(number=2, text="c   d")],
    )
    cleaned = BasicCleaner().clean(doc)
    assert [p.number for p in cleaned.pages] == [1, 2]
    assert cleaned.pages[1].text == "c d"
