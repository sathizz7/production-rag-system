from rag.ingestion.chunking.fixed import FixedTokenChunker
from rag.models import Document, Page
from rag.util.tokens import count_tokens


def _doc(text: str, pages: int = 1) -> Document:
    return Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text=text)],
    )


def test_chunk_spans_reconstruct_source_text() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    doc = _doc(text)
    chunks = FixedTokenChunker(chunk_tokens=50, overlap=10).chunk(doc)

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.text == doc.text[ch.char_start : ch.char_end]  # the invariant
        assert ch.token_count == count_tokens(ch.text)
        assert ch.chunker_name == "fixed"
        assert ch.chunker_version == "1"


def test_chunks_are_ordinal_and_within_budget() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = FixedTokenChunker(chunk_tokens=50, overlap=10).chunk(_doc(text))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for ch in chunks:
        assert ch.token_count <= 50


def test_overlap_creates_shared_text() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunks = FixedTokenChunker(chunk_tokens=40, overlap=10).chunk(_doc(text))
    # consecutive chunks overlap in character space
    assert chunks[1].char_start < chunks[0].char_end


def test_pages_assigned_from_offsets() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[
            Page(number=1, text=" ".join(f"a{i}" for i in range(100))),
            Page(number=2, text=" ".join(f"b{i}" for i in range(100))),
        ],
    )
    chunks = FixedTokenChunker(chunk_tokens=30, overlap=5).chunk(doc)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
