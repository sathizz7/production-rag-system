import hashlib


def content_hash(text: str) -> str:
    """Stable SHA-256 hex digest of UTF-8 text — drives dedup and idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
