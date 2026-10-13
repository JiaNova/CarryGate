"""Per-variable normalized forecasting metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _arrays(prediction, target, variable_id, mask=None):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    variable_id = np.asarray(variable_id, dtype=np.int64).reshape(-1)
    if not (prediction.shape == target.shape == variable_id.shape):
        raise ValueError("prediction, target, and variable_id must have equal size")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(prediction):
            raise ValueError("mask has the wrong size")
        prediction, target, variable_id = prediction[mask], target[mask], variable_id[mask]
    return prediction, target, variable_id


def normalized_forecast_metrics(
    prediction,
    target,
    variable_id,
    train_mean,
    *,
    n_variables: int | None = None,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Return nMAE/nRMSE averaged across variables with real test events.

    The denominator is the error of predicting that variable's training mean,
    which makes the documented ``< 1`` interpretation explicit.  Targets in
    Phase 1 are already robustly normalized, but the function is agnostic to
    the original units as long as ``train_mean`` uses the same scale.
    """

    prediction, target, variable_id = _arrays(prediction, target, variable_id)
    means = np.asarray(train_mean, dtype=np.float64).reshape(-1)
    if n_variables is None:
        n_variables = len(means)
    if len(means) < int(n_variables):
        raise ValueError("train_mean has fewer entries than n_variables")
    per_variable = {}
    mae_values = []
    rmse_values = []
    undefined_variables = []
    for variable in range(int(n_variables)):
        selected = variable_id == variable
        if not np.any(selected):
            continue
        error = prediction[selected] - target[selected]
        baseline = target[selected] - means[variable]
        baseline_mae = float(np.sum(np.abs(baseline)))
        baseline_mse = float(np.sum(baseline**2))
        mae = float(np.sum(np.abs(error)))
        mse = float(np.sum(error**2))
        if baseline_mae <= eps or baseline_mse <= eps:
            # A constant target has no meaningful normalized denominator.
            # Keep its absolute errors visible, but exclude it from the
            # normalized average instead of allowing epsilon to create a
            # numerically arbitrary 1e10 score (e.g. MechVent in P12).
            undefined_variables.append(variable)
            per_variable[variable] = {
                "n_events": int(selected.sum()),
                "nmae": float("nan"),
                "nrmse": float("nan"),
                "mae": float(mae / max(int(selected.sum()), 1)),
                "rmse": float(np.sqrt(mse / max(int(selected.sum()), 1))),
                "baseline_defined": False,
            }
            continue
        normalized_mae = mae / max(baseline_mae, eps)
        normalized_rmse = np.sqrt(mse / max(baseline_mse, eps))
        per_variable[variable] = {
            "n_events": int(selected.sum()),
            "nmae": float(normalized_mae),
            "nrmse": float(normalized_rmse),
            "mae": float(mae / max(int(selected.sum()), 1)),
            "rmse": float(np.sqrt(mse / max(int(selected.sum()), 1))),
            "baseline_defined": True,
        }
        mae_values.append(normalized_mae)
        rmse_values.append(normalized_rmse)
    return {
        "nmae": float(np.mean(mae_values)) if mae_values else float("nan"),
        "nrmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "n_variables_observed": int(len(mae_values)),
        "n_variables_undefined": int(len(undefined_variables)),
        "undefined_variables": undefined_variables,
        "per_variable": per_variable,
        "n_events": int(len(target)),
    }


def nmae(*args, **kwargs) -> float:
    return float(normalized_forecast_metrics(*args, **kwargs)["nmae"])


def nrmse(*args, **kwargs) -> float:
    return float(normalized_forecast_metrics(*args, **kwargs)["nrmse"])
