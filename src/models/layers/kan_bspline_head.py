"""Pure-PyTorch B-spline KAN layers.

The implementation follows the Cox--de Boor recursion used by efficient-kan:
each edge has a SiLU base weight, a spline scale, and one coefficient per
B-spline basis.  Keeping the layer local makes Phase 0 runnable offline and
keeps parameter counting exact; the optional third-party package is not needed
for correctness or reproducibility.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BSplineKANLayer(nn.Module):
    """One KAN layer with uniform B-spline edge functions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if min(in_features, out_features, grid_size) < 1:
            raise ValueError("KAN dimensions and grid_size must be positive")
        if spline_order < 0:
            raise ValueError("spline_order must be non-negative")
        if x_range[1] <= x_range[0]:
            raise ValueError("x_range must be increasing")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        knots = self._make_knots(self.x_range)
        self.register_buffer("knots", knots)
        self.register_buffer("coverage_total", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage_in_range", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage", torch.ones((), dtype=torch.float32))
        self.num_basis = self.grid_size + self.spline_order

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_basis)
        )
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    def _make_knots(self, x_range: tuple[float, float]) -> torch.Tensor:
        step = (float(x_range[1]) - float(x_range[0])) / self.grid_size
        return torch.arange(
            -self.spline_order,
            self.grid_size + self.spline_order + 1,
            dtype=torch.float32,
        ) * step + float(x_range[0])

    def set_grid_range(
        self,
        x_min: float | tuple[float, float],
        x_max: float | None = None,
    ) -> None:
        """Update the uniform knot range without changing learned weights."""

        if x_max is None:
            if not isinstance(x_min, (tuple, list)) or len(x_min) != 2:
                raise TypeError("set_grid_range expects (min, max) or two scalars")
            x_min, x_max = x_min
        x_min, x_max = float(x_min), float(x_max)
        if not torch.isfinite(torch.tensor([x_min, x_max])).all() or x_max <= x_min:
            raise ValueError("grid range must be finite and increasing")
        self.x_range = (x_min, x_max)
        new_knots = self._make_knots(self.x_range).to(self.knots)
        self.knots.copy_(new_knots)

    @property
    def grid_range(self) -> tuple[float, float]:
        """Alias used by audit code and serialized experiment reports."""

        return self.x_range

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

    def _clip_with_straight_through(self, x: torch.Tensor) -> torch.Tensor:
        """Clip the forward value while retaining a usable outside gradient."""

        clipped = torch.clamp(x, min=self.x_range[0], max=self.x_range[1])
        # A literal clamp has zero derivative outside the calibrated range.
        # The straight-through form keeps the boundary-value forward behavior
        # required by Phase 1 while preventing dead gradients during training.
        return x + (clipped - x).detach()

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions, returning ``(..., in, num_basis)``."""

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, received {x.shape[-1]}"
            )
        x = self._clip_with_straight_through(x)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        value = x.unsqueeze(-1)
        # Degree-zero basis functions.  The final interval is closed on the
        # right so that x == max(knots) has a well-defined representation.
        basis = ((value >= knots[:-1]) & (value < knots[1:])).to(x.dtype)
        basis[..., -1] = basis[..., -1] + (value[..., 0] == knots[-1]).to(x.dtype)

        # Cox--de Boor recursion.  The knot grid is uniform and extended by
        # spline_order intervals at both ends, so denominators are non-zero.
        for order in range(1, self.spline_order + 1):
            current = basis.shape[-1]
            left_den = knots[order : order + current - 1] - knots[: current - 1]
            right_den = knots[order + 1 : order + current] - knots[1:current]
            left = (value - knots[: current - 1]) / left_den
            right = (knots[order + 1 : order + current] - value) / right_den
            basis = left * basis[..., : current - 1] + right * basis[..., 1:current]
        return basis

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
        basis = self.b_splines(x)
        spline = torch.einsum(
            "...ik,oik,oi->...o", basis, self.spline_weight, self.spline_scaler
        )
        base = F.linear(F.silu(self._clip_with_straight_through(x)), self.base_weight)
        return base + spline


class BSplineKANHead(nn.Module):
    """Two B-spline KAN layers used for H0."""

    backend = "internal_bspline"

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        self.layer1 = BSplineKANLayer(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            x_range=x_range,
        )
        self.layer2 = BSplineKANLayer(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            x_range=x_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))
