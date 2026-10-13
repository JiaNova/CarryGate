"""Truncated Fourier edge functions for the Phase 2 basis comparison."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierKANLayer(nn.Module):
    """KAN layer with globally supported truncated Fourier edge functions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_freq: int = 6,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if min(int(in_features), int(out_features), int(n_freq)) < 1:
            raise ValueError("FourierKAN dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_freq = int(n_freq)
        low, high = self._validate_range(x_range)
        self.register_buffer("x_min", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(high, dtype=torch.float32))
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.coef = nn.Parameter(
            torch.randn(2, self.out_features, self.in_features, self.n_freq) * 0.01
        )
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        self.register_buffer("coverage_total", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage_in_range", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage", torch.ones((), dtype=torch.float32))

    @staticmethod
    def _validate_range(x_range: tuple[float, float] | list[float]) -> tuple[float, float]:
        if len(x_range) != 2:
            raise ValueError("x_range must contain two values")
        low, high = float(x_range[0]), float(x_range[1])
        if not torch.isfinite(torch.tensor([low, high])).all() or high <= low:
            raise ValueError("x_range must be finite and increasing")
        return low, high

    @property
    def x_range(self) -> tuple[float, float]:
        return float(self.x_min.item()), float(self.x_max.item())

    @property
    def grid_range(self) -> tuple[float, float]:
        return self.x_range

    def set_grid_range(
        self,
        x_min: float | tuple[float, float],
        x_max: float | None = None,
    ) -> None:
        if x_max is None:
            if not isinstance(x_min, (tuple, list)):
                raise TypeError("set_grid_range expects (min, max) or two scalars")
            new_range = self._validate_range(x_min)
        else:
            new_range = self._validate_range((float(x_min), float(x_max)))
        self.x_min.fill_(new_range[0])
        self.x_max.fill_(new_range[1])

    def coverage_stats(self) -> dict[str, float | int]:
        return {
            "coverage": float(self.coverage.item()),
            "n_total": int(self.coverage_total.item()),
            "n_in_range": int(self.coverage_in_range.item()),
        }

    def reset_coverage(self) -> None:
        self.coverage_total.zero_()
        self.coverage_in_range.zero_()
        self.coverage.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, received {x.shape[-1]}"
            )
        with torch.no_grad():
            self.coverage_total += torch.tensor(x.numel(), device=self.coverage_total.device)
            inside = (x >= self.x_min) & (x <= self.x_max)
            self.coverage_in_range += inside.sum().to(self.coverage_in_range.device)
            self.coverage.copy_(
                self.coverage_in_range.float()
                / self.coverage_total.float().clamp_min(1.0)
            )
        span = (self.x_max - self.x_min).to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
        u = math.pi * (x - self.x_min.to(device=x.device, dtype=x.dtype)) / span
        frequencies = torch.arange(1, self.n_freq + 1, device=x.device, dtype=x.dtype)
        angle = u.unsqueeze(-1) * frequencies
        coef = self.coef.to(device=x.device, dtype=x.dtype)
        out = torch.einsum("...ik,oik->...o", torch.cos(angle), coef[0])
        out = out + torch.einsum("...ik,oik->...o", torch.sin(angle), coef[1])
        base_weight = self.base_weight.to(device=x.device, dtype=x.dtype)
        return out + F.linear(F.silu(x), base_weight)
