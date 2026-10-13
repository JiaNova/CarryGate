"""Gaussian-RBF edge functions for the Phase 2 basis comparison."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RBFKANLayer(nn.Module):
    """One RBF-KAN layer with auditable input-range coverage.

    Centers and widths are learned per input feature, while the RBF
    coefficients are learned per edge.  The layer deliberately keeps global
    Gaussian support: values outside the calibrated interval still have a
    differentiable path through the basis and the SiLU base term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 8,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if min(int(in_features), int(out_features), int(num_centers)) < 1:
            raise ValueError("RBFKAN dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_centers = int(num_centers)
        self.x_range = self._validate_range(x_range)

        centers = torch.linspace(*self.x_range, self.num_centers)
        self.centers = nn.Parameter(centers.unsqueeze(0).repeat(self.in_features, 1))
        spacing = (self.x_range[1] - self.x_range[0]) / max(self.num_centers - 1, 1)
        self.log_sigma = nn.Parameter(
            torch.full((self.in_features, self.num_centers), math.log(max(spacing, 1e-2)))
        )
        self.weight = nn.Parameter(
            torch.randn(self.out_features, self.in_features, self.num_centers) * 0.05
        )
        self.base_weight = nn.Parameter(torch.randn(self.out_features, self.in_features) * 0.05)

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
    def grid_range(self) -> tuple[float, float]:
        return self.x_range

    def set_grid_range(
        self,
        x_min: float | tuple[float, float],
        x_max: float | None = None,
    ) -> None:
        """Set the calibrated range and place the initial RBF centers on it."""

        if x_max is None:
            if not isinstance(x_min, (tuple, list)):
                raise TypeError("set_grid_range expects (min, max) or two scalars")
            new_range = self._validate_range(x_min)
        else:
            new_range = self._validate_range((float(x_min), float(x_max)))
        self.x_range = new_range
        spacing = (new_range[1] - new_range[0]) / max(self.num_centers - 1, 1)
        with torch.no_grad():
            centers = torch.linspace(
                new_range[0],
                new_range[1],
                self.num_centers,
                device=self.centers.device,
                dtype=self.centers.dtype,
            )
            self.centers.copy_(centers.unsqueeze(0).expand_as(self.centers))
            self.log_sigma.fill_(math.log(max(spacing, 1e-2)))

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
            inside = (x >= self.x_range[0]) & (x <= self.x_range[1])
            self.coverage_in_range += inside.sum().to(self.coverage_in_range.device)
            self.coverage.copy_(
                self.coverage_in_range.float()
                / self.coverage_total.float().clamp_min(1.0)
            )
        centers = self.centers.to(device=x.device, dtype=x.dtype)
        sigma = torch.exp(self.log_sigma).to(device=x.device, dtype=x.dtype).clamp_min(1e-4)
        diff = x.unsqueeze(-1) - centers.unsqueeze(0)
        basis = torch.exp(-0.5 * (diff / sigma.unsqueeze(0)) ** 2)
        rbf = torch.einsum("...ik,oik->...o", basis, self.weight)
        return rbf + F.linear(F.silu(x), self.base_weight)
