"""Structural monotone-decreasing wrapper for recency gates."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .gate_bases import RangeAwareEdge, suspend_coverage


class MonotoneDecreasingGate(nn.Module):
    """Wrap an unconstrained raw edge function with a decreasing integral.

    The Phase 5 pseudocode expresses a gate as ``sigmoid(base - decrement)``.
    The raw score is parameterised as ``base`` minus a cumulative trapezoidal
    integral of a strictly positive rate ``softplus(inner)``.  It is then
    passed through the plan's exact pseudocode,
    ``decrement = softplus(base - raw)`` and
    ``omega = sigmoid(base - decrement)``.  This gives a structural guarantee
    that ``omega(delta_2) <= omega(delta_1)`` whenever
    ``delta_2 >= delta_1``.  It also extrapolates with the positive end rate,
    so the constraint is not lost outside the fitted range.

    The integral depends only on the group, never on the individual event, so
    the quadrature table is built once per forward pass at
    ``(n_groups, n_grid)`` and broadcast to events with a one-hot contraction.
    Evaluating it per event instead costs ``numel(x) * n_grid`` inner-network
    calls; with the strict grouped dispatch -- which evaluates every group's
    sub-network on every row -- that is ``numel(x) * n_grid * n_groups`` and
    exhausts device memory well before a realistic batch.
    """

    def __init__(
        self,
        inner: nn.Module,
        x_range: tuple[float, float] = (0.0, 4.0),
        n_grid: int = 128,
    ) -> None:
        super().__init__()
        if int(n_grid) < 2:
            raise ValueError("n_grid must be at least two")
        if len(x_range) != 2 or float(x_range[1]) <= float(x_range[0]):
            raise ValueError("x_range must be finite and increasing")
        self.inner = inner
        self.x_range = (max(0.0, float(x_range[0])), float(x_range[1]))
        self.n_grid = int(n_grid)
        self.n_groups = int(getattr(inner, "n_groups", 1))
        if self.n_groups < 1:
            raise ValueError("inner gate must expose a positive n_groups")

    @property
    def upper(self) -> float:
        return max(self.x_range[1], 1.0)

    @staticmethod
    def _squeeze_output(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        if value.shape == shape:
            return value
        if value.shape == shape + (1,):
            return value[..., 0]
        raise ValueError("inner gate must return the group shape or group shape + (1,)")

    def _inner(self, var_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # GroupedMLP/GroupedKAN use ``(..., 1)`` inputs, while the compact
        # Phase 5 edge classes also accept that form.  Keeping the singleton
        # here lets the wrapper cover both implementations.
        value = self.inner(var_id, x.unsqueeze(-1))
        return self._squeeze_output(value, tuple(x.shape))

    def _quadrature(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(grid, rate, cumulative, base)`` for every group."""

        grid = torch.linspace(0.0, self.upper, self.n_grid, device=device, dtype=dtype)
        groups = torch.arange(self.n_groups, device=device, dtype=torch.long)
        grid_groups = groups.reshape(-1, 1).expand(self.n_groups, self.n_grid)
        grid_values = grid.reshape(1, -1).expand(self.n_groups, self.n_grid)
        with suspend_coverage(self.inner):
            raw_rate = self._inner(
                grid_groups.reshape(-1), grid_values.reshape(-1)
            ).reshape(self.n_groups, self.n_grid)
            base = self._inner(
                groups, torch.zeros(self.n_groups, device=device, dtype=dtype)
            )
        rate = F.softplus(raw_rate)
        cumulative = torch.zeros_like(rate)
        cumulative[:, 1:] = torch.cumsum(
            0.5 * (rate[:, 1:] + rate[:, :-1]) * (grid[1] - grid[0]), dim=-1
        )
        return grid, rate, cumulative, base

    def _observe(self, var_id: torch.Tensor, x: torch.Tensor) -> None:
        """Let range calibration and the coverage audit see the real inputs.

        The inner edge is only ever evaluated on the integration grid, so
        without this pass the calibration hook would never observe a staleness
        value and ``coverage`` would be computed from synthetic points.
        """

        edges = [
            layer for layer in self.inner.modules() if isinstance(layer, RangeAwareEdge)
        ]
        if not edges:
            return
        with torch.no_grad():
            self._inner(var_id, x)

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        if var_id.shape != log_delta.shape:
            raise ValueError("var_id and log_delta must have identical shapes")
        x = log_delta.to(dtype=torch.float32).clamp_min(0.0)
        if x.numel() == 0:
            return x

        groups = var_id.to(dtype=torch.long)
        if torch.any((groups < 0) | (groups >= self.n_groups)):
            raise ValueError("var_id contains an invalid group id")
        self._observe(groups, x)

        shape = tuple(x.shape)
        flat_x = x.reshape(-1)
        flat_group = groups.reshape(-1)
        grid, rate, cumulative, base_table = self._quadrature(x.device, x.dtype)
        # A one-hot contraction rather than an index gather: it is the same
        # deterministic dispatch the grouped layers use, and it keeps the
        # backward free of scatter-add accumulation.
        onehot = F.one_hot(flat_group, num_classes=self.n_groups).to(dtype=x.dtype)
        base = onehot @ base_table
        expanded = onehot @ cumulative

        upper = self.upper
        clipped = flat_x.clamp(max=upper)
        index = torch.bucketize(clipped.detach(), grid[1:-1], right=False)
        index = index.clamp(min=0, max=self.n_grid - 2)
        lower_x = grid[index]
        upper_x_grid = grid[index + 1]
        lower_c = torch.gather(expanded, -1, index.unsqueeze(-1)).squeeze(-1)
        upper_c = torch.gather(expanded, -1, (index + 1).unsqueeze(-1)).squeeze(-1)
        fraction = (clipped - lower_x) / (upper_x_grid - lower_x).clamp_min(1e-8)
        integrated = lower_c + fraction * (upper_c - lower_c)

        outside = flat_x > upper
        if bool(outside.any()):
            end_rate = onehot @ rate[:, -1]
            integrated = torch.where(
                outside,
                (onehot @ cumulative[:, -1]) + (flat_x - upper) * end_rate,
                integrated,
            )
        raw = base - integrated
        decrement = F.softplus(base - raw)
        return torch.sigmoid(base - decrement).reshape(shape)


__all__ = ["MonotoneDecreasingGate"]
