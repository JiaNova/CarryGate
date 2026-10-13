"""One-dimensional edge functions used by the Phase 5 recency gates.

The event-encoder KAN layers in this project deliberately keep their historical
parameterisation.  Recency gates are a smaller, separate problem: one scalar
edge per variable.  The classes below therefore use fixed basis locations and
one coefficient per basis function, making the parameter budget transparent:

``RBF``
    ``num_centers + 1`` parameters per edge;
``B-spline``
    ``grid_size + spline_order + 2`` parameters per edge;
``Fourier``
    ``2 * n_freq + 1`` parameters per edge;
``rational``
    ``num_num + num_den + 1`` parameters per edge.

All four functions have global or straight-through support outside their
configured interval.  Consequently an extrapolated staleness value retains a
gradient instead of silently becoming a constant.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn


def _validate_range(x_range: tuple[float, float] | list[float]) -> tuple[float, float]:
    if len(x_range) != 2:
        raise ValueError("x_range must contain two values")
    low, high = float(x_range[0]), float(x_range[1])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("x_range must be finite and increasing")
    return low, high


class RangeAwareEdge(nn.Module):
    """Base class for a bank of scalar, group-specific edge functions."""

    in_features = 1
    out_features = 1

    def __init__(self, n_groups: int, x_range: tuple[float, float] | list[float]) -> None:
        super().__init__()
        if int(n_groups) < 1:
            raise ValueError("n_groups must be positive")
        self.n_groups = int(n_groups)
        self.x_range = _validate_range(x_range)

        self.register_buffer("coverage_total", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage_in_range", torch.zeros((), dtype=torch.long))
        self.register_buffer("coverage", torch.ones((), dtype=torch.float32))
        # Diagnostic evaluations -- the monotone wrapper's integration grid,
        # the saved gate curves -- must not be counted as observed data.  See
        # :func:`suspend_coverage`.
        self._coverage_enabled = True

    @property
    def grid_range(self) -> tuple[float, float]:
        return self.x_range

    @property
    def params_per_edge(self) -> int:
        raise NotImplementedError

    def _prepare(
        self, group: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        if not torch.is_tensor(group) or not torch.is_tensor(x):
            raise TypeError("group and x must be tensors")
        group = group.to(dtype=torch.long)
        if x.ndim == group.ndim + 1 and x.shape[-1] == 1:
            x = x[..., 0]
        if group.shape != x.shape:
            raise ValueError("group and x must have identical shapes (up to x[..., 1])")
        if torch.any((group < 0) | (group >= self.n_groups)):
            raise ValueError("group contains an invalid group id")
        shape = tuple(x.shape)
        flat_group = group.reshape(-1)
        flat_x = x.to(dtype=torch.float32).reshape(-1)
        self.record_coverage(flat_x)
        return flat_group.to(device=x.device), flat_x, shape

    def record_coverage(self, x: torch.Tensor) -> None:
        """Account ``x`` against the calibrated interval.

        Called automatically by :meth:`_prepare`.  A caller that evaluates the
        edge on synthetic points (see :func:`suspend_coverage`) is responsible
        for feeding the real inputs here instead, so ``coverage`` keeps meaning
        "fraction of observed staleness values inside the calibrated grid".
        """

        if not self._coverage_enabled:
            return
        with torch.no_grad():
            flat_x = x.detach().to(dtype=torch.float32).reshape(-1)
            self.coverage_total += torch.tensor(
                flat_x.numel(), device=self.coverage_total.device
            )
            inside = (flat_x >= self.x_range[0]) & (flat_x <= self.x_range[1])
            self.coverage_in_range += inside.sum().to(self.coverage_in_range.device)
            self.coverage.copy_(
                self.coverage_in_range.float()
                / self.coverage_total.float().clamp_min(1.0)
            )

    def _restore(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return value.reshape(shape)

    def set_grid_range(
        self,
        x_min: float | tuple[float, float] | list[float],
        x_max: float | None = None,
    ) -> None:
        if x_max is None:
            if not isinstance(x_min, (tuple, list)):
                raise TypeError("set_grid_range expects (min, max) or two scalars")
            new_range = _validate_range(x_min)
        else:
            new_range = _validate_range((float(x_min), float(x_max)))
        self.x_range = new_range
        self._on_range_changed()

    def _on_range_changed(self) -> None:
        """Allow subclasses with fixed basis locations to refresh them."""

    def reset_coverage(self) -> None:
        self.coverage_total.zero_()
        self.coverage_in_range.zero_()
        self.coverage.fill_(1.0)

    @property
    def coverage_enabled(self) -> bool:
        return bool(self._coverage_enabled)

    def coverage_stats(self) -> dict[str, float | int]:
        return {
            "coverage": float(self.coverage.item()),
            "n_total": int(self.coverage_total.item()),
            "n_in_range": int(self.coverage_in_range.item()),
        }

    @staticmethod
    def _straight_through_clip(x: torch.Tensor, x_range: tuple[float, float]) -> torch.Tensor:
        clipped = torch.clamp(x, min=x_range[0], max=x_range[1])
        return x + (clipped - x).detach()


class RBFEdge(RangeAwareEdge):
    """A fixed-centre Gaussian RBF edge with ``C + 1`` parameters."""

    def __init__(
        self,
        n_groups: int,
        num_centers: int = 8,
        x_range: tuple[float, float] | list[float] = (-3.0, 3.0),
    ) -> None:
        super().__init__(n_groups, x_range)
        if int(num_centers) < 1:
            raise ValueError("num_centers must be positive")
        self.num_centers = int(num_centers)
        self.coefficient = nn.Parameter(torch.empty(self.n_groups, self.num_centers))
        nn.init.normal_(self.coefficient, mean=0.0, std=0.01)
        self.base = nn.Parameter(torch.full((self.n_groups,), 4.0))
        self.register_buffer("centers", torch.empty(self.num_centers))
        self.register_buffer("sigma", torch.empty(self.num_centers))
        self._on_range_changed()

    @property
    def params_per_edge(self) -> int:
        return self.num_centers + 1

    def _on_range_changed(self) -> None:
        if not hasattr(self, "centers"):
            return
        low, high = self.x_range
        centers = torch.linspace(low, high, self.num_centers, device=self.centers.device)
        spacing = (high - low) / max(self.num_centers - 1, 1)
        self.centers.copy_(centers)
        self.sigma.fill_(max(spacing, 1e-2))

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        flat_group, flat_x, shape = self._prepare(group, x)
        centers = self.centers.to(device=flat_x.device, dtype=flat_x.dtype)
        sigma = self.sigma.to(device=flat_x.device, dtype=flat_x.dtype).clamp_min(1e-4)
        basis = torch.exp(-0.5 * ((flat_x.unsqueeze(-1) - centers) / sigma) ** 2)
        value = (basis * self.coefficient[flat_group]).sum(dim=-1) + self.base[flat_group] * (
            1.0 + 1e-4 * flat_x
        )
        return self._restore(value, shape)


class BSplineEdge(RangeAwareEdge):
    """A Cox--de Boor B-spline edge with the documented budget."""

    def __init__(
        self,
        n_groups: int,
        grid_size: int = 8,
        spline_order: int = 3,
        x_range: tuple[float, float] | list[float] = (-3.0, 3.0),
    ) -> None:
        super().__init__(n_groups, x_range)
        if int(grid_size) < 1 or int(spline_order) < 0:
            raise ValueError("grid_size must be positive and spline_order non-negative")
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.num_basis = self.grid_size + self.spline_order
        self.coefficient = nn.Parameter(torch.empty(self.n_groups, self.num_basis))
        nn.init.normal_(self.coefficient, mean=0.0, std=0.01)
        self.scale = nn.Parameter(torch.ones(self.n_groups))
        self.base = nn.Parameter(torch.full((self.n_groups,), 4.0))
        self.register_buffer("knots", torch.empty(self.grid_size + 2 * self.spline_order + 1))
        self._on_range_changed()

    @property
    def params_per_edge(self) -> int:
        return self.grid_size + self.spline_order + 2

    def _make_knots(self) -> torch.Tensor:
        step = (self.x_range[1] - self.x_range[0]) / self.grid_size
        return (
            torch.arange(
                -self.spline_order,
                self.grid_size + self.spline_order + 1,
                device=self.knots.device,
                dtype=self.knots.dtype,
            )
            * step
            + self.x_range[0]
        )

    def _on_range_changed(self) -> None:
        if hasattr(self, "knots"):
            self.knots.copy_(self._make_knots())

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = self._straight_through_clip(x, self.x_range)
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        value = x.unsqueeze(-1)
        basis = ((value >= knots[:-1]) & (value < knots[1:])).to(x.dtype)
        basis[..., -1] = basis[..., -1] + (value[..., 0] == knots[-1]).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            current = basis.shape[-1]
            left_den = knots[order : order + current - 1] - knots[: current - 1]
            right_den = knots[order + 1 : order + current] - knots[1:current]
            left = (value - knots[: current - 1]) / left_den.clamp_min(1e-12)
            right = (knots[order + 1 : order + current] - value) / right_den.clamp_min(1e-12)
            basis = left * basis[..., : current - 1] + right * basis[..., 1:current]
        return basis

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        flat_group, flat_x, shape = self._prepare(group, x)
        basis = self._basis(flat_x)
        value = (
            (basis * self.coefficient[flat_group]).sum(dim=-1) * self.scale[flat_group]
            + self.base[flat_group] * (1.0 + 1e-4 * flat_x)
        )
        return self._restore(value, shape)


class FourierEdge(RangeAwareEdge):
    """A globally supported truncated Fourier edge."""

    def __init__(
        self,
        n_groups: int,
        n_freq: int = 6,
        x_range: tuple[float, float] | list[float] = (-3.0, 3.0),
    ) -> None:
        super().__init__(n_groups, x_range)
        if int(n_freq) < 1:
            raise ValueError("n_freq must be positive")
        self.n_freq = int(n_freq)
        self.cosine = nn.Parameter(torch.empty(self.n_groups, self.n_freq))
        self.sine = nn.Parameter(torch.empty(self.n_groups, self.n_freq))
        nn.init.normal_(self.cosine, mean=0.0, std=0.01)
        nn.init.normal_(self.sine, mean=0.0, std=0.01)
        self.base = nn.Parameter(torch.full((self.n_groups,), 4.0))

    @property
    def params_per_edge(self) -> int:
        return 2 * self.n_freq + 1

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        flat_group, flat_x, shape = self._prepare(group, x)
        low, high = self.x_range
        u = math.pi * (flat_x - low) / max(high - low, 1e-6)
        frequency = torch.arange(
            1, self.n_freq + 1, device=flat_x.device, dtype=flat_x.dtype
        )
        angle = u.unsqueeze(-1) * frequency
        value = (
            (torch.cos(angle) * self.cosine[flat_group]).sum(dim=-1)
            + (torch.sin(angle) * self.sine[flat_group]).sum(dim=-1)
            + self.base[flat_group] * (1.0 + 1e-4 * flat_x)
        )
        return self._restore(value, shape)


class RationalEdge(RangeAwareEdge):
    """A stable rational polynomial edge with a positive denominator."""

    def __init__(
        self,
        n_groups: int,
        num_num: int = 4,
        num_den: int = 4,
        x_range: tuple[float, float] | list[float] = (-3.0, 3.0),
    ) -> None:
        super().__init__(n_groups, x_range)
        if int(num_num) < 1 or int(num_den) < 1:
            raise ValueError("num_num and num_den must be positive")
        self.num_num = int(num_num)
        self.num_den = int(num_den)
        self.numerator = nn.Parameter(torch.empty(self.n_groups, self.num_num))
        self.denominator = nn.Parameter(torch.empty(self.n_groups, self.num_den))
        nn.init.normal_(self.numerator, mean=0.0, std=0.01)
        nn.init.normal_(self.denominator, mean=-4.0, std=0.01)
        self.base = nn.Parameter(torch.full((self.n_groups,), 4.0))

    @property
    def params_per_edge(self) -> int:
        return self.num_num + self.num_den + 1

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        flat_group, flat_x, shape = self._prepare(group, x)
        low, high = self.x_range
        t = 2.0 * (flat_x - low) / max(high - low, 1e-6) - 1.0
        num_power = torch.arange(self.num_num, device=flat_x.device, dtype=flat_x.dtype)
        den_power = torch.arange(1, self.num_den + 1, device=flat_x.device, dtype=flat_x.dtype)
        numerator = (t.unsqueeze(-1).pow(num_power) * self.numerator[flat_group]).sum(dim=-1)
        # Even powers and softplus keep the denominator strictly positive,
        # including outside the calibrated interval.
        denominator = 1.0 + (
            t.unsqueeze(-1).pow(2.0 * den_power)
            * torch.nn.functional.softplus(self.denominator[flat_group])
        ).sum(dim=-1)
        value = self.base[flat_group] * (1.0 + 1e-4 * flat_x) + numerator / denominator.clamp_min(1e-6)
        return self._restore(value, shape)


@contextlib.contextmanager
def suspend_coverage(module: nn.Module) -> Iterator[None]:
    """Evaluate ``module`` without polluting range statistics.

    Two callers evaluate an edge on points that are not data: the monotone
    wrapper integrates over a synthetic grid, and the reporting scripts render
    omega(delta) on a fixed plotting grid.  Both are in-range by construction,
    so counting them would drive ``coverage`` to one and silence the
    grid-underflow audit; during calibration they would also mix a uniform
    grid into the quantiles that define the interval itself.  Suspending also
    disables the calibration forward hook, which reads the same flag.
    """

    edges = [
        layer for layer in module.modules() if isinstance(layer, RangeAwareEdge)
    ]
    previous = [layer._coverage_enabled for layer in edges]
    for layer in edges:
        layer._coverage_enabled = False
    try:
        yield
    finally:
        for layer, was_enabled in zip(edges, previous):
            layer._coverage_enabled = was_enabled


def build_gate_basis(
    basis: str,
    n_groups: int,
    num_centers: int = 8,
    x_range: tuple[float, float] | list[float] = (-3.0, 3.0),
    **kwargs: Any,
) -> RangeAwareEdge:
    """Build the only-varying basis component of a recency gate.

    ``num_centers`` is retained as the common budget knob from the plan.  The
    optional explicit arguments are used by the exact parameter matcher when a
    basis needs a different native hyper-parameter (for example 20 B-spline
    intervals and 12 Fourier frequencies both give 25 parameters).
    """

    name = str(basis).lower().replace("-", "_")
    if name in {"rbf", "gaussian_rbf"}:
        centers = kwargs.get("num_centers")
        return RBFEdge(
            n_groups,
            num_centers=int(num_centers if centers is None else centers),
            x_range=x_range,
        )
    if name in {"bspline", "b_spline", "spline"}:
        configured_grid = kwargs.get("grid_size")
        grid_size = int(
            max(int(num_centers) - 3, 1)
            if configured_grid is None
            else configured_grid
        )
        order = int(kwargs.get("spline_order", 3))
        return BSplineEdge(n_groups, grid_size=grid_size, spline_order=order, x_range=x_range)
    if name in {"fourier", "fou", "truncated_fourier"}:
        configured_freq = kwargs.get("n_freq")
        n_freq = int(
            max((int(num_centers) - 1) // 2, 1)
            if configured_freq is None
            else configured_freq
        )
        return FourierEdge(n_groups, n_freq=n_freq, x_range=x_range)
    if name in {"rational", "kat"}:
        configured_num = kwargs.get("num_num")
        configured_den = kwargs.get("num_den")
        num_num = int(
            (int(num_centers) + 1) // 2 if configured_num is None else configured_num
        )
        num_den = int(int(num_centers) // 2 if configured_den is None else configured_den)
        return RationalEdge(n_groups, num_num=num_num, num_den=num_den, x_range=x_range)
    raise ValueError(f"unknown gate basis: {basis}")


__all__ = [
    "RangeAwareEdge",
    "suspend_coverage",
    "RBFEdge",
    "BSplineEdge",
    "FourierEdge",
    "RationalEdge",
    "build_gate_basis",
]
