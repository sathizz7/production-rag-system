from rag.eval.golden import GoldenItem
from rag.eval.runner import evaluate
from rag.eval.scorecard import StratumScore


class _RetrieverByQuestion:
    """Returns chunk ids keyed by question string."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def retrieve(self, query, k, filt):
        from rag.models import Provenance, ScoredChunk
        from tests.unit.fakes import make_chunk

        ids = self._mapping.get(query, [])
        return [
            ScoredChunk(chunk=make_chunk(doc_id, text=doc_id), score=1.0 / (i + 1),
                        provenance=Provenance.dense)
            for i, doc_id in enumerate(ids)
        ]


class _FixedJudge:
    def score(self, question, answer, context):
        from rag.eval.judge import JudgeScore

        return JudgeScore(faithfulness=0.9, answer_relevance=0.8)


class _FixedAnswerer:
    def answer(self, query, filt=None, scope=None):
        from rag.models import Answer

        return Answer(text="grounded answer [1].")


def test_evaluate_produces_per_stratum_scores() -> None:
    items = [
        GoldenItem(
            id="q1", question="q1", reference_answer="",
            relevant_doc_ids=["maize_nitrogen"], stratum="easy",
        ),
        GoldenItem(
            id="q2", question="q2", reference_answer="",
            relevant_doc_ids=[], stratum="out_of_corpus",
        ),
    ]
    retriever = _RetrieverByQuestion({"q1": ["maize_nitrogen", "x"], "q2": ["y"]})
    scores = evaluate(
        items=items,
        retriever=retriever,
        answerer=_FixedAnswerer(),
        judge=_FixedJudge(),
        k=5,
        seed=0,
        # test ids are stems on doc_id; live uses source_uri stem
        doc_key=lambda sc: sc.chunk.doc_id,
    )
    by_stratum = {s.stratum: s for s in scores}
    assert isinstance(by_stratum["easy"], StratumScore)
    assert by_stratum["easy"].hit_at_k[0] == 1.0          # q1 retrieved a relevant doc
    assert by_stratum["easy"].faithfulness[0] == 0.9
    assert by_stratum["out_of_corpus"].n == 1
    assert by_stratum["ALL"].n == 2                        # aggregate pools every item
