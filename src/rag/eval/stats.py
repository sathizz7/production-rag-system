from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_mean, ci_low, ci_high) via percentile bootstrap of the mean.

    Deterministic given ``seed`` (so eval runs are reproducible and CI bands are
    stable — spec §15). Empty input → all zeros; a single value → zero-width CI.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    point = float(arr.mean())
    if arr.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low = float(np.percentile(means, tail * 100))
    high = float(np.percentile(means, (1.0 - tail) * 100))
    return (point, low, high)
