from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Default process registry — one set of metrics for the whole app.
REQUESTS = Counter(
    "rag_requests_total", "HTTP requests handled", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end HTTP latency", ["endpoint"]
)
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Per-stage read-path latency",
    ["stage"],  # dense | lexical | fusion | rerank | assemble | generate
)


@contextmanager
def observe_stage(stage: str) -> Iterator[None]:
    """Time a read-path stage into ``rag_stage_latency_seconds{stage=...}``."""
    start = perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(perf_counter() - start)


def render_latest() -> tuple[str, str]:
    """Return (exposition_text, content_type) for the /metrics endpoint."""
    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
