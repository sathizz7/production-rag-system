def test_observe_stage_records_a_sample() -> None:
    from rag.observability import metrics

    before = metrics.STAGE_LATENCY.labels(stage="dense")._sum.get()
    with metrics.observe_stage("dense"):
        pass
    after = metrics.STAGE_LATENCY.labels(stage="dense")._sum.get()
    assert after >= before
    # a count sample exists for the labelled stage — derived from cumulative bucket totals
    count = sum(b.get() for b in metrics.STAGE_LATENCY.labels(stage="dense")._buckets)
    assert count >= 1.0


def test_render_latest_returns_prometheus_text() -> None:
    from rag.observability import metrics

    metrics.REQUESTS.labels(endpoint="/query", status="200").inc()
    text, content_type = metrics.render_latest()
    assert "rag_requests_total" in text
    assert "text/plain" in content_type
