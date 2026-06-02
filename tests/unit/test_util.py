from rag.util.hashing import content_hash
from rag.util.tokens import count_tokens


def test_content_hash_is_stable_and_sensitive() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
    assert len(content_hash("hello")) == 64  # sha256 hex


def test_count_tokens_monotonic() -> None:
    assert count_tokens("") == 0
    assert count_tokens("one two three") > 0
    assert count_tokens("a much longer sentence with more words") > count_tokens("short")
