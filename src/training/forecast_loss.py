"""Masked losses for real prediction-window observations only."""

from __future__ import annotations

import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MAE over true events; padded targets contribute exactly zero."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    mask = mask.to(dtype=prediction.dtype)
    count = mask.sum()
    # Keep a gradient path even for a rare batch with no prediction-window
    # event, so the optimizer loop remains numerically well-defined.
    return (torch.abs(prediction - target) * mask).sum() / count.clamp_min(1.0)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute event-masked MSE over true forecast-window observations."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    mask = mask.to(dtype=prediction.dtype)
    count = mask.sum()
    return (((prediction - target) ** 2) * mask).sum() / count.clamp_min(1.0)


def masked_variable_balanced_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    variable: torch.Tensor,
) -> torch.Tensor:
    """Average per-variable MAE over variables with observed target events."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if variable.shape != prediction.shape:
        raise ValueError("variable must have the same shape as prediction")
    valid = mask.to(dtype=torch.bool).reshape(-1)
    error = torch.abs(prediction - target).reshape(-1)
    variable = variable.to(dtype=torch.long).reshape(-1)
    if not bool(valid.any()):
        return error.sum() * 0.0
    error = error[valid]
    variable = variable[valid]
    n_variables = int(variable.max().item()) + 1
    sums = torch.zeros(n_variables, dtype=error.dtype, device=error.device)
    counts = torch.zeros(n_variables, dtype=error.dtype, device=error.device)
    sums.scatter_add_(0, variable, error)
    counts.scatter_add_(0, variable, torch.ones_like(error))
    present = counts > 0
    return (sums[present] / counts[present]).mean()


forecast_l1_loss = masked_l1_loss
