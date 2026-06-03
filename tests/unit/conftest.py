"""Unit-test fixtures and environment isolation helpers."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove LANGFUSE_* env vars before each unit test so tests are isolated.

    configure_observability calls apply_provider_env which writes LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY into os.environ via setdefault.  Without cleanup the
    'no-op' test (which runs second) sees stale values and fails.
    """
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
