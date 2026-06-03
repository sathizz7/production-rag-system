from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk


def _chunk(cid: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=ordinal,
        page=ordinal + 1,
        char_start=0,
        char_end=len(text),
        token_count=0,
        chunker_name="fixed",
        chunker_version="1",
    )


def test_assembler_numbers_chunks_and_builds_context() -> None:
    chunks = [_chunk("a", "alpha fact", 0), _chunk("b", "bravo fact", 1)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=1000)

    assert ctx.chunks[0].chunk_id == "a"
    assert "[1]" in ctx.text and "[2]" in ctx.text
    assert "alpha fact" in ctx.text


def test_assembler_dedupes_by_chunk_id() -> None:
    chunks = [_chunk("a", "alpha", 0), _chunk("a", "alpha", 0)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=1000)
    assert len(ctx.chunks) == 1


def test_assembler_truncates_to_budget() -> None:
    chunks = [_chunk(str(i), f"sentence number {i} here", i) for i in range(50)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=20)
    assert len(ctx.chunks) < 50
    assert ctx.token_count <= 20 or len(ctx.chunks) == 1  # always keep at least one
