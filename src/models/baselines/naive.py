"""NAIVE-1/2/3 baselines required by the Phase 1 protocol."""

from __future__ import annotations

from typing import Mapping

import torch


def naive_predictions(
    batch: Mapping[str, torch.Tensor],
    strategy: str,
    train_mean,
) -> torch.Tensor:
    """Generate one prediction for every padded target slot in ``batch``."""

    name = str(strategy).upper().replace("_", "-")
    if name in {"NAIVE1", "NAIVE-1", "LAST"}:
        name = "NAIVE-1"
    elif name in {"NAIVE2", "NAIVE-2", "MEAN"}:
        name = "NAIVE-2"
    elif name in {"NAIVE3", "NAIVE-3", "GLOBAL"}:
        name = "NAIVE-3"
    else:
        raise ValueError("strategy must be NAIVE-1, NAIVE-2, or NAIVE-3")

    obs_var = batch["obs_var"].to(dtype=torch.long)
    obs_value = batch["obs_value"].to(dtype=torch.float32)
    obs_mask = batch.get("obs_mask", ~batch["obs_pad_mask"]).to(dtype=torch.bool)
    target_var = batch["target_var"].to(dtype=torch.long)
    result = torch.zeros(
        target_var.shape,
        dtype=obs_value.dtype,
        device=obs_value.device,
    )
    global_mean = torch.as_tensor(train_mean, dtype=obs_value.dtype, device=obs_value.device)
    for row in range(obs_var.shape[0]):
        for variable in torch.unique(target_var[row]).tolist():
            variable = int(variable)
            target_selected = target_var[row] == variable
            observed = obs_mask[row] & (obs_var[row] == variable)
            if name == "NAIVE-3" or not bool(observed.any()):
                value = global_mean[variable]
            elif name == "NAIVE-1":
                value = obs_value[row, torch.nonzero(observed, as_tuple=False)[-1, 0]]
            else:
                value = obs_value[row, observed].mean()
            result[row, target_selected] = value
    return result


def naive_1(batch: Mapping[str, torch.Tensor], train_mean) -> torch.Tensor:
    return naive_predictions(batch, "NAIVE-1", train_mean)


def naive_2(batch: Mapping[str, torch.Tensor], train_mean) -> torch.Tensor:
    return naive_predictions(batch, "NAIVE-2", train_mean)


def naive_3(batch: Mapping[str, torch.Tensor], train_mean) -> torch.Tensor:
    return naive_predictions(batch, "NAIVE-3", train_mean)
