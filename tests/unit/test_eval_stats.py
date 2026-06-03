from rag.eval.stats import bootstrap_ci


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    mean, low, high = bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=7)
    assert abs(mean - 0.7) < 1e-9
    assert low <= mean <= high
    assert 0.0 <= low <= high <= 1.0
    # same seed → identical interval
    assert bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=7) == (mean, low, high)


def test_bootstrap_ci_empty_is_zeros() -> None:
    assert bootstrap_ci([], n_resamples=100) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_single_value_has_zero_width() -> None:
    mean, low, high = bootstrap_ci([0.5], n_resamples=100, seed=1)
    assert mean == low == high == 0.5
