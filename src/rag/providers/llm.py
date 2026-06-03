from __future__ import annotations

import litellm

from rag.models import Completion


class LiteLLMProvider:
    """Non-streaming completion via LiteLLM (P0a). Streaming is added in P0b."""

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict[str, object]], **opts: object) -> Completion:
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text: str = resp.choices[0].message.content or ""
        usage: dict[str, object] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "cost_usd": self._safe_cost(resp),
        }
        return Completion(text=text, usage=usage)

    @staticmethod
    def _safe_cost(resp: object) -> float | None:
        try:
            return float(litellm.completion_cost(completion_response=resp))
        except Exception:
            return None
