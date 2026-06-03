def test_configure_observability_sets_litellm_callbacks_when_keys_present(monkeypatch) -> None:
    import litellm

    from rag.config import Settings
    from rag.observability.tracing import configure_observability

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [], raising=False)
    s = Settings(_env_file=None, langfuse_public_key="pk", langfuse_secret_key="sk")

    assert configure_observability(s) is True
    assert "langfuse" in litellm.success_callback
    assert "langfuse" in litellm.failure_callback


def test_configure_observability_is_noop_without_keys(monkeypatch) -> None:
    import litellm

    from rag.config import Settings
    from rag.observability.tracing import configure_observability

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    s = Settings(_env_file=None)  # no langfuse keys
    assert configure_observability(s) is False
    assert litellm.success_callback == []
