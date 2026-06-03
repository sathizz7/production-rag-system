from types import SimpleNamespace

from rag.providers.llm import LiteLLMProvider


def test_complete_extracts_text_and_usage(monkeypatch) -> None:
    def fake_completion(model: str, messages: list[dict], **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )

    monkeypatch.setattr("rag.providers.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.providers.llm.litellm.completion_cost", lambda **k: 0.0009)

    provider = LiteLLMProvider(model="gemini/gemini-2.5-pro", temperature=0.0)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result.text == "hello world"
    assert result.usage["total_tokens"] == 15
    assert result.usage["cost_usd"] == 0.0009


def test_complete_survives_cost_failure(monkeypatch) -> None:
    def fake_completion(model: str, messages: list[dict], **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def boom(**kwargs):
        raise RuntimeError("no cost table")

    monkeypatch.setattr("rag.providers.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.providers.llm.litellm.completion_cost", boom)

    result = LiteLLMProvider(model="m").complete([{"role": "user", "content": "hi"}])
    assert result.usage["cost_usd"] is None


def test_litellm_provider_stream_yields_content_deltas(monkeypatch) -> None:
    import litellm

    from rag.providers.llm import LiteLLMProvider

    class _Delta:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        return iter([_Chunk("Nitro"), _Chunk("gen"), _Chunk(None), _Chunk(" helps.")])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = LiteLLMProvider(model="gemini/gemini-2.5-pro")
    assert list(provider.stream([{"role": "user", "content": "q"}])) == ["Nitro", "gen", " helps."]
