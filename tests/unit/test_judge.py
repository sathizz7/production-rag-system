from rag.eval.judge import FaithfulnessJudge, JudgeScore


class _CannedLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages, **opts):
        from rag.models import Completion

        self.calls.append(messages)
        return Completion(text=self._reply, usage={})

    def stream(self, messages, **opts):  # unused by the judge
        yield ""


def test_judge_parses_scores_from_json_reply() -> None:
    llm = _CannedLLM('Here is my rating: {"faithfulness": 0.8, "answer_relevance": 0.6}')
    score = FaithfulnessJudge(llm).score("q?", "answer", "context")
    assert isinstance(score, JudgeScore)
    assert score.faithfulness == 0.8
    assert score.answer_relevance == 0.6
    assert "context" in llm.calls[0][0]["content"]


def test_judge_clamps_and_defaults_on_unparseable_reply() -> None:
    score = FaithfulnessJudge(_CannedLLM("the model rambled with no json")).score("q", "a", "c")
    assert score.faithfulness == 0.0
    assert score.answer_relevance == 0.0
