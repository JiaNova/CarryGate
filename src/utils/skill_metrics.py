"""Persistence-relative forecast skill and the mandatory Phase 3 views."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEGENERATE_EPS = 1e-9
MIN_TEST_EVENTS = 50


def persistence_skill(pred, target, variable, x_last):
    """Return persistence skill, per-variable rows, and excluded variables.

    This is the registered Phase 3 primary metric:

    ``S_v = 1 - sum(abs(pred-target)) / sum(abs(x_last-target))``.

    Variables with a degenerate persistence denominator or fewer than 50 test
    events are excluded from the mean and returned explicitly.
    """

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    variable = np.asarray(variable, dtype=np.int64).reshape(-1)
    x_last = np.asarray(x_last, dtype=np.float64).reshape(-1)
    if not (pred.shape == target.shape == variable.shape == x_last.shape):
        raise ValueError("pred, target, variable, and x_last must have equal size")

    rows = []
    excluded = []
    for value in np.unique(variable):
        selected = variable == value
        n_test = int(selected.sum())
        persistence_error = float(np.abs(x_last[selected] - target[selected]).sum())
        if persistence_error <= DEGENERATE_EPS or n_test < MIN_TEST_EVENTS:
            excluded.append(int(value))
            continue
        model_error = float(np.abs(pred[selected] - target[selected]).sum())
        rows.append(
            {
                "variable": int(value),
                "n_test": n_test,
                "mae_model": float(np.abs(pred[selected] - target[selected]).mean()),
                "mae_persist": float(
                    np.abs(x_last[selected] - target[selected]).mean()
                ),
                "skill": float(1.0 - model_error / persistence_error),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["variable", "n_test", "mae_model", "mae_persist", "skill"],
    )
    skill = float(frame["skill"].mean()) if not frame.empty else float("nan")
    return skill, frame, excluded


def _mean_absolute_error_by_variable(target, variable, train_mean) -> dict[int, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    variable = np.asarray(variable, dtype=np.int64).reshape(-1)
    train_mean = np.asarray(train_mean, dtype=np.float64).reshape(-1)
    result = {}
    for value in np.unique(variable):
        selected = variable == value
        if int(value) >= len(train_mean):
            raise ValueError(f"variable {int(value)} is outside train_mean")
        result[int(value)] = float(np.abs(target[selected] - train_mean[int(value)]).mean())
    return result


def stratified_report(
    frame: pd.DataFrame,
    staleness,
    variable,
    pred,
    target,
    x_last,
    train_mean=None,
):
    """Return dynamics, staleness, and per-variable skill tables.

    Dynamics are split by the registered persistence/mean baseline ratio.  If
    ``train_mean`` is omitted, the function keeps the pseudocode-compatible
    fallback ratio ``mae_persist / mae_model``; production report generation
    always supplies the train-only mean so the documented ratio is exact.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be the per-variable DataFrame from persistence_skill")
    result = frame.copy()
    if train_mean is None:
        mean_mae = result["mae_model"].replace(0, np.nan)
    else:
        mean_by_variable = _mean_absolute_error_by_variable(target, variable, train_mean)
        mean_mae = result["variable"].map(mean_by_variable)
    result["mae_mean"] = mean_mae
    result["persist_ratio"] = result["mae_persist"] / result["mae_mean"].replace(0, np.nan)
    if result.empty:
        result["dynamics"] = pd.Series(dtype="object")
    elif result["persist_ratio"].nunique(dropna=True) < 3:
        result["dynamics"] = "mixed"
    else:
        try:
            result["dynamics"] = pd.qcut(
                result["persist_ratio"],
                3,
                labels=["persistence-dominated", "mixed", "mean-reverting"],
            )
        except ValueError:
            result["dynamics"] = "mixed"
    by_dynamics = (
        result.groupby("dynamics", observed=False)["skill"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    staleness = np.asarray(staleness, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    x_last = np.asarray(x_last, dtype=np.float64).reshape(-1)
    if not (len(staleness) == len(pred) == len(target) == len(x_last)):
        raise ValueError("staleness and forecast arrays must have equal size")
    bins = [0.0, 1.0, 6.0, 24.0, np.inf]
    labels = ["<1h", "1-6h", "6-24h", ">24h"]
    bucket = pd.cut(staleness, bins=bins, labels=labels, right=False)
    by_stale = []
    for label in labels:
        selected = np.asarray(bucket == label, dtype=bool)
        if int(selected.sum()) < MIN_TEST_EVENTS:
            continue
        denominator = float(np.abs(x_last[selected] - target[selected]).sum())
        if denominator <= DEGENERATE_EPS:
            continue
        by_stale.append(
            {
                "bucket": label,
                "n": int(selected.sum()),
                "skill": float(
                    1.0
                    - np.abs(pred[selected] - target[selected]).sum() / denominator
                ),
            }
        )
    return by_dynamics, pd.DataFrame(by_stale), result
