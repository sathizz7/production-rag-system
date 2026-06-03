from __future__ import annotations

import json
import re
from importlib import resources

from pydantic import BaseModel

from rag.protocols import LLMProvider

_JSON = re.compile(r"\{[^{}]*\}", re.DOTALL)


class JudgeScore(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0


def _load_prompt() -> str:
    return (
        resources.files("rag.eval.prompts").joinpath("judge_v1.md").read_text(encoding="utf-8")
    )


class FaithfulnessJudge:
    """Cheap-model judge (temperature 0) scoring faithfulness + answer-relevance.

    Routes through the same ``LLMProvider.complete`` seam as the rest of the system
    so the judge model is config-driven and pinned. Unparseable replies default to
    0.0 (a non-answer is not credited) — see spec §15 on determinism.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._prompt = _load_prompt()

    def score(self, question: str, answer: str, context: str) -> JudgeScore:
        content = self._prompt.format(question=question, context=context, answer=answer)
        reply = self._llm.complete([{"role": "user", "content": content}]).text
        match = _JSON.search(reply)
        if not match:
            return JudgeScore()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeScore()
        return JudgeScore(
            faithfulness=_clamp(data.get("faithfulness", 0.0)),
            answer_relevance=_clamp(data.get("answer_relevance", 0.0)),
        )


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
