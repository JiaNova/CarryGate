"""Paired bootstrap confidence intervals for head comparisons."""

from __future__ import annotations

from typing import Callable

import numpy as np


def paired_bootstrap_ci(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap ``metric(a) - metric(b)`` on paired test predictions."""

    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    if not (len(y_true) == len(p_a) == len(p_b)):
        raise ValueError("paired bootstrap inputs must have equal length")
    if len(y_true) == 0:
        raise ValueError("cannot bootstrap an empty test set")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(n_boot)):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        value = metric_fn(y_true[indices], p_a[indices]) - metric_fn(
            y_true[indices], p_b[indices]
        )
        if np.isfinite(value):
            deltas.append(float(value))
    if not deltas:
        raise ValueError("metric was non-finite for every bootstrap sample")
    values = np.asarray(deltas, dtype=float)
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "ci95": (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ),
        "n_boot": int(n_boot),
        "n_valid": int(len(values)),
    }
