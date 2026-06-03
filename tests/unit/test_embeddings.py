from types import SimpleNamespace

from rag.providers.embeddings import LiteLLMEmbeddingProvider


def test_embed_returns_vectors_and_batches(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_embedding(model: str, input: list[str], **kwargs):  # noqa: A002
        calls.append(list(input))
        data = [{"embedding": [float(len(t))] * 3} for t in input]
        return SimpleNamespace(data=data)

    monkeypatch.setattr("rag.providers.embeddings.litellm.embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(
        model="gemini/text-embedding-004", dim=3, batch_size=2
    )
    vectors = provider.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert calls == [["a", "bb"], ["ccc"]]  # batched in groups of 2


def test_embed_empty_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.providers.embeddings.litellm.embedding",
        lambda **k: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    assert LiteLLMEmbeddingProvider(model="m", dim=3).embed([]) == []
