"""Non-trainable gates used by the Phase 4 mechanism ablation."""

from __future__ import annotations

import torch
from torch import nn


class FrozenGate(nn.Module):
    """Return a supplied per-variable persistence weight without parameters."""

    def __init__(self, omega_per_var: torch.Tensor) -> None:
        super().__init__()
        omega = torch.as_tensor(omega_per_var, dtype=torch.float32).reshape(-1)
        if omega.numel() < 1:
            raise ValueError("omega_per_var must contain at least one value")
        if not torch.isfinite(omega).all():
            raise ValueError("omega_per_var must be finite")
        self.register_buffer("omega", omega.clamp(0.0, 1.0).clone())

    @property
    def n_vars(self) -> int:
        return int(self.omega.numel())

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        del log_delta
        var_id = var_id.to(dtype=torch.long)
        if torch.any((var_id < 0) | (var_id >= self.n_vars)):
            raise ValueError("var_id contains an invalid variable index")
        return self.omega[var_id]
