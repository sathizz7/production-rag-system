from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk
from tests.unit.fakes import FakeLLMProvider


class _FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0, provenance=Provenance.dense) for c in self._chunks]


def _chunk(cid: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d1", source_uri="file:///x.pdf", text=text, ordinal=ordinal,
        page=ordinal + 1, char_start=0, char_end=len(text), token_count=2,
        chunker_name="fixed", chunker_version="1",
    )


def test_answerer_returns_cited_answer() -> None:
    retriever = _FakeRetriever([_chunk("a", "Nitrogen helps maize.", 0)])
    answerer = StraightLineAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Nitrogen helps maize [1]."),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("does nitrogen help maize?", filt=None)

    assert answer.text == "Nitrogen helps maize [1]."
    assert [c.chunk_id for c in answer.citations] == ["a"]
    assert answer.trace_id != ""
    assert answer.usage["total_tokens"] == 15


def test_answerer_strips_hallucinated_citation() -> None:
    retriever = _FakeRetriever([_chunk("a", "Nitrogen helps maize.", 0)])
    answerer = StraightLineAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Helps [1]. Made up [2]."),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("q", filt=None)
    assert "[2]" not in answer.text
    assert [c.marker for c in answer.citations] == ["[1]"]


def test_answerer_no_context_path() -> None:
    answerer = StraightLineAnswerer(
        retriever=_FakeRetriever([]),
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="should not be used"),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("q", filt=None)
    assert answer.text == "I don't have relevant context to answer that."
    assert answer.citations == []
