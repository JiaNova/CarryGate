"""Leakage-safe carry-forward baselines for Phase 3."""

from __future__ import annotations

from typing import Mapping

import torch


def build_persistence_baseline(
    obs_var: torch.Tensor,
    obs_time: torch.Tensor,
    obs_value: torch.Tensor,
    tgt_var: torch.Tensor,
    tgt_time: torch.Tensor,
    t_obs: torch.Tensor | float,
    train_median: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``x_last`` and staleness using observations strictly before T_obs.

    The tensors represent one unpadded patient example.  In particular, the
    cutoff is ``t_obs`` rather than each target timestamp; using the latter
    would leak forecast-window observations into the persistence baseline.
    ``train_median`` is in the same (standardized) value space as
    ``obs_value``.
    """

    if not all(t.ndim == 1 for t in (obs_var, obs_time, obs_value, tgt_var, tgt_time)):
        raise ValueError("persistence baseline expects one-dimensional event tensors")
    if obs_var.shape != obs_time.shape or obs_var.shape != obs_value.shape:
        raise ValueError("observation tensors must have equal shapes")
    if tgt_var.shape != tgt_time.shape:
        raise ValueError("target tensors must have equal shapes")
    train_median = torch.as_tensor(
        train_median, dtype=obs_value.dtype, device=obs_value.device
    ).reshape(-1)
    if train_median.numel() == 0:
        raise ValueError("train_median must contain at least one variable")
    cutoff = torch.as_tensor(t_obs, dtype=obs_time.dtype, device=obs_time.device)
    if cutoff.numel() != 1:
        raise ValueError("t_obs must be a scalar for one patient example")

    x_last = torch.full_like(tgt_time, float("nan"), dtype=obs_value.dtype)
    staleness = torch.full_like(tgt_time, float("nan"), dtype=tgt_time.dtype)
    for variable in torch.unique(tgt_var):
        variable_id = int(variable.item())
        if variable_id < 0 or variable_id >= train_median.numel():
            raise ValueError(f"target variable is outside train_median: {variable_id}")
        prior = (obs_var == variable) & (obs_time < cutoff)
        selected = tgt_var == variable
        if bool(prior.any()):
            prior_indices = torch.where(prior)[0]
            latest = prior_indices[torch.argmax(obs_time[prior])]
            x_last[selected] = obs_value[latest]
            staleness[selected] = tgt_time[selected] - obs_time[latest]

    missing = torch.isnan(x_last)
    x_last[missing] = train_median[tgt_var[missing]]
    # The documented fallback is time since admission (time zero).
    staleness[missing] = tgt_time[missing]
    return x_last, staleness


def build_persistence_batch(
    batch: Mapping[str, torch.Tensor],
    train_median: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply :func:`build_persistence_baseline` to a padded forecast batch."""

    required = {
        "obs_var",
        "obs_time",
        "obs_value",
        "obs_mask",
        "target_var",
        "target_time",
        "target_mask",
        "t_obs",
    }
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"batch is missing persistence fields: {sorted(missing)}")
    target_var = batch["target_var"].to(dtype=torch.long)
    target_time = batch["target_time"].to(dtype=torch.float32)
    obs_var = batch["obs_var"].to(dtype=torch.long)
    obs_time = batch["obs_time"].to(dtype=torch.float32)
    obs_value = batch["obs_value"].to(dtype=torch.float32)
    obs_mask = batch["obs_mask"].to(dtype=torch.bool)
    target_mask = batch["target_mask"].to(dtype=torch.bool)
    median = torch.as_tensor(
        train_median, dtype=obs_value.dtype, device=obs_value.device
    ).reshape(-1)
    prior = median[target_var]
    x_last = prior.clone()
    staleness = torch.zeros_like(target_time)
    for row in range(target_var.shape[0]):
        obs_valid = obs_mask[row]
        target_valid = target_mask[row]
        if not bool(target_valid.any()):
            continue
        row_x_last, row_staleness = build_persistence_baseline(
            obs_var[row][obs_valid],
            obs_time[row][obs_valid],
            obs_value[row][obs_valid],
            target_var[row][target_valid],
            target_time[row][target_valid],
            batch["t_obs"][row],
            median,
        )
        x_last[row, target_valid] = row_x_last
        staleness[row, target_valid] = row_staleness
    return x_last, staleness
