"""Training-only per-variable descriptors for Phase 4 G3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .persistence_baseline import build_persistence_batch


@dataclass
class VariableDescriptors:
    """Standardised descriptors fitted from one training loader only."""

    FIELDS = ("persist_ratio", "log_n_obs", "volatility", "sampling_irregularity")

    values: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    raw: torch.Tensor
    source: str = "train"

    def __post_init__(self) -> None:
        self.values = torch.as_tensor(self.values, dtype=torch.float32)
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(-1)
        self.std = torch.as_tensor(self.std, dtype=torch.float32).reshape(-1)
        self.raw = torch.as_tensor(self.raw, dtype=torch.float32)
        if self.values.ndim != 2 or self.values.shape[1] != len(self.FIELDS):
            raise ValueError("values must have shape (n_vars, 4)")
        if self.raw.shape != self.values.shape:
            raise ValueError("raw and values must have identical shapes")
        if self.mean.numel() != len(self.FIELDS) or self.std.numel() != len(self.FIELDS):
            raise ValueError("descriptor statistics must contain four fields")
        if not torch.isfinite(self.values).all():
            raise ValueError("standardised descriptors must be finite")

    @property
    def n_vars(self) -> int:
        return int(self.values.shape[0])

    @classmethod
    def fit(
        cls,
        train_loader: Iterable[Mapping[str, object]],
        n_vars: int,
        train_prior: torch.Tensor | np.ndarray,
    ) -> "VariableDescriptors":
        """Compute descriptors from target/observation events in ``train_loader``.

        The function deliberately accepts a loader rather than a full bundle:
        callers cannot accidentally pass validation or test examples without
        making that split choice explicitly.  Patient volatility and sampling
        gaps use observation-window events only; persistence/prior errors use
        training targets only.
        """

        n_vars = int(n_vars)
        if n_vars < 1:
            raise ValueError("n_vars must be positive")
        prior = torch.as_tensor(train_prior, dtype=torch.float32).reshape(-1)
        if prior.numel() != n_vars:
            raise ValueError("train_prior must contain one value per variable")
        persist_err = np.zeros(n_vars, dtype=np.float64)
        prior_err = np.zeros(n_vars, dtype=np.float64)
        count = np.zeros(n_vars, dtype=np.float64)
        within_std: list[list[float]] = [[] for _ in range(n_vars)]
        gaps: list[list[float]] = [[] for _ in range(n_vars)]

        for raw_batch in train_loader:
            batch = {
                key: value
                for key, value in raw_batch.items()
                if torch.is_tensor(value)
            }
            target_mask = batch["target_mask"].to(dtype=torch.bool)
            if bool(target_mask.any()):
                x_last, _ = build_persistence_batch(batch, prior)
                target_var = batch["target_var"].to(dtype=torch.long)
                target_value = batch["target_value"].to(dtype=torch.float32)
                v = target_var[target_mask].cpu().numpy()
                y = target_value[target_mask].cpu().numpy()
                x = x_last[target_mask].cpu().numpy()
                p = prior[target_var[target_mask]].cpu().numpy()
                np.add.at(persist_err, v, np.abs(x - y))
                np.add.at(prior_err, v, np.abs(p - y))
                np.add.at(count, v, 1.0)

            obs_mask = batch["obs_mask"].to(dtype=torch.bool)
            obs_var = batch["obs_var"].to(dtype=torch.long)
            obs_value = batch["obs_value"].to(dtype=torch.float32)
            obs_delta = batch["obs_delta_t"].to(dtype=torch.float32)
            for row in range(obs_var.shape[0]):
                valid = obs_mask[row]
                for variable in torch.unique(obs_var[row][valid]).tolist():
                    selected = valid & (obs_var[row] == int(variable))
                    values = obs_value[row][selected].detach().cpu().numpy()
                    if len(values) >= 2:
                        within_std[int(variable)].append(float(np.std(values)))
                    # ``delta_t`` is already per-variable and leakage-safe.
                    gaps[int(variable)].extend(
                        float(value)
                        for value in obs_delta[row][selected].detach().cpu().tolist()
                        if np.isfinite(value) and value >= 0.0
                    )

        eps = 1e-9
        ratio = persist_err / np.maximum(prior_err, eps)
        raw = np.stack(
            [
                ratio,
                np.log1p(count),
                np.asarray(
                    [np.median(values) if values else 0.0 for values in within_std],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        (np.subtract(*np.quantile(values, [0.75, 0.25]))
                         / max(float(np.median(values)), eps))
                        if values and float(np.median(values)) > eps
                        else 0.0
                        for values in gaps
                    ],
                    dtype=np.float64,
                ),
            ],
            axis=1,
        ).astype(np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0) + eps
        values = (raw - mean) / std
        return cls(
            torch.from_numpy(values),
            torch.from_numpy(mean),
            torch.from_numpy(std),
            torch.from_numpy(raw),
        )

    def select(self, fields: Sequence[str]) -> torch.Tensor:
        indices = self._indices(fields)
        if not indices:
            raise ValueError("at least one descriptor field is required")
        return self.values[:, indices]

    def stats(self, fields: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._indices(fields)
        return self.mean[indices], self.std[indices]

    def _indices(self, fields: Sequence[str]) -> list[int]:
        indices = []
        for field in fields:
            if field not in self.FIELDS:
                raise KeyError(field)
            indices.append(self.FIELDS.index(field))
        if not indices:
            raise ValueError("at least one descriptor field is required")
        return indices

    def as_dict(self) -> dict[str, object]:
        return {
            "fields": list(self.FIELDS),
            "values": self.values.tolist(),
            "raw": self.raw.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "source": self.source,
        }
