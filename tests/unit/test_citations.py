from rag.generation.citations import validate_and_build_citations
from rag.models import AssembledContext, Chunk


def _ctx(chunk_ids: list[str]) -> AssembledContext:
    chunks = [
        Chunk(
            chunk_id=cid,
            doc_id="d1",
            source_uri="file:///x.pdf",
            text=f"text {cid}",
            ordinal=i,
            page=i + 1,
            char_start=i * 10,
            char_end=i * 10 + 6,
            token_count=2,
            chunker_name="fixed",
            chunker_version="1",
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return AssembledContext(text="ctx", chunks=chunks, token_count=1)


def test_valid_markers_become_citations() -> None:
    ctx = _ctx(["a", "b"])
    text, citations = validate_and_build_citations("Nitrogen helps [1]. Drought hurts [2].", ctx)

    assert text == "Nitrogen helps [1]. Drought hurts [2]."
    assert [c.marker for c in citations] == ["[1]", "[2]"]
    assert citations[0].chunk_id == "a"
    assert citations[0].page == 1
    assert citations[1].chunk_id == "b"


def test_out_of_range_marker_is_stripped() -> None:
    ctx = _ctx(["a"])  # only [1] is valid
    text, citations = validate_and_build_citations("Real [1]. Fabricated [2][3].", ctx)

    assert "[2]" not in text and "[3]" not in text
    assert text == "Real [1]. Fabricated ."
    assert [c.marker for c in citations] == ["[1]"]


def test_each_chunk_cited_once_even_if_repeated() -> None:
    ctx = _ctx(["a", "b"])
    _, citations = validate_and_build_citations("A [1]. Again [1]. B [2].", ctx)
    assert [c.marker for c in citations] == ["[1]", "[2]"]
