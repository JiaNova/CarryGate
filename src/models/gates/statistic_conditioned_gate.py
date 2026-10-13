"""Shared gates conditioned on training-only variable descriptors (Phase 4 G3)."""

from __future__ import annotations

import torch
from torch import nn


class StatisticConditionedGate(nn.Module):
    """Implement ``omega = sigmoid(psi([s_v, log1p(delta)]))``."""

    def __init__(
        self,
        descriptors: torch.Tensor,
        hidden: int = 32,
        use_staleness: bool = True,
        descriptor_mean: torch.Tensor | None = None,
        descriptor_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        if descriptors.ndim != 2 or descriptors.shape[1] < 1:
            raise ValueError("descriptors must have shape (n_vars, n_features)")
        if not torch.isfinite(descriptors).all():
            raise ValueError("descriptors must be finite")
        if int(hidden) < 1:
            raise ValueError("hidden must be positive")
        self.register_buffer("descriptors", descriptors.clone())
        mean = torch.zeros(descriptors.shape[1]) if descriptor_mean is None else torch.as_tensor(descriptor_mean, dtype=torch.float32)
        std = torch.ones(descriptors.shape[1]) if descriptor_std is None else torch.as_tensor(descriptor_std, dtype=torch.float32)
        if mean.numel() != descriptors.shape[1] or std.numel() != descriptors.shape[1]:
            raise ValueError("descriptor mean/std must match descriptor features")
        self.register_buffer("descriptor_mean", mean.reshape(-1).clone())
        self.register_buffer("descriptor_std", std.reshape(-1).clone())
        self.use_staleness = bool(use_staleness)
        d_in = int(descriptors.shape[1]) + (1 if self.use_staleness else 0)
        self.psi = nn.Sequential(
            nn.Linear(d_in, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )
        # The residual head is zero-initialised.  Starting near persistence
        # preserves the documented line-E starting point while retaining a
        # trainable shared map.
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.constant_(self.psi[-1].bias, 4.0)

    @property
    def n_vars(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def params_per_variable(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        var_id = var_id.to(dtype=torch.long)
        if torch.any((var_id < 0) | (var_id >= self.n_vars)):
            raise ValueError("var_id contains an invalid variable index")
        features = self.descriptors[var_id]
        if self.use_staleness:
            features = torch.cat([features, log_delta.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.psi(features).squeeze(-1))
