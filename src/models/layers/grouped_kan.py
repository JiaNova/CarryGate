"""Small grouped function layers used by the Phase 1 event encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .fourier_kan_layer import FourierKANLayer
from .kan_bspline_head import BSplineKANLayer
from .rbf_kan_layer import RBFKANLayer


class _GroupedBase(nn.Module):
    def __init__(self, n_groups: int, d_in: int, d_out: int) -> None:
        super().__init__()
        if min(int(n_groups), int(d_in), int(d_out)) < 1:
            raise ValueError("grouped layer dimensions must be positive")
        self.n_groups = int(n_groups)
        self.d_in = int(d_in)
        self.d_out = int(d_out)

    def _check(self, group: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 2 or x.shape[-1] != self.d_in:
            raise ValueError(f"expected x (...,{self.d_in})")
        if group.shape != x.shape[:-1]:
            raise ValueError("group must match x shape without its last dimension")
        group = group.to(device=x.device, dtype=torch.long)
        if torch.any((group < 0) | (group >= self.n_groups)):
            raise ValueError("group contains an invalid group id")
        return group.reshape(-1), x.reshape(-1, self.d_in)

    def _apply_grouped_reference(
        self, flat_group: torch.Tensor, flat_x: torch.Tensor, layers
    ) -> torch.Tensor:
        """The historical dispatch, retained for the best-effort path.

        ``index_put``/masked assignment has a scatter-add backward on CUDA.
        It is useful as the fast path when strict determinism is disabled, but
        must never be used silently under the strict audit mode.
        """

        result = flat_x.new_zeros((flat_x.shape[0], self.d_out))
        for group_id, layer in enumerate(layers):
            selected = flat_group == group_id
            if bool(selected.any()):
                result[selected] = layer(flat_x[selected])
        return result

    def _apply_grouped_deterministic(
        self, flat_group: torch.Tensor, flat_x: torch.Tensor, layers
    ) -> torch.Tensor:
        """Dense one-hot dispatch with a deterministic backward."""

        outputs = torch.stack([layer(flat_x) for layer in layers], dim=0)
        onehot = F.one_hot(flat_group, num_classes=len(layers))
        weight = onehot.to(outputs.dtype).movedim(-1, 0).unsqueeze(-1)
        return (outputs * weight).sum(dim=0)

    def _apply_grouped(self, group: torch.Tensor, x: torch.Tensor, layers) -> torch.Tensor:
        flat_group, flat_x = self._check(group, x)
        # For small group counts the deterministic form is cheap enough to be
        # the default.  When strict PyTorch determinism is requested, use it
        # for larger counts too; this is slower but is the only honest way to
        # run the per-variable audit without a CUDA scatter backward.  With
        # strict mode off, larger groups are explicitly labelled best-effort.
        strict = bool(torch.are_deterministic_algorithms_enabled())
        use_dense = self.n_groups <= 8 or strict
        self.determinism = "strict" if use_dense else "best_effort"
        if use_dense:
            result = self._apply_grouped_deterministic(flat_group, flat_x, layers)
        else:
            result = self._apply_grouped_reference(flat_group, flat_x, layers)
        return result.reshape(*x.shape[:-1], self.d_out)


class GroupedKAN(_GroupedBase):
    """One basis-function KAN layer per group."""

    def __init__(
        self,
        n_groups: int,
        d_in: int,
        d_out: int,
        grid_size: int = 8,
        spline_order: int = 3,
        x_range: tuple[float, float] = (-3.0, 3.0),
        basis: str = "bspline",
        num_centers: int = 8,
        n_freq: int = 6,
    ) -> None:
        super().__init__(n_groups, d_in, d_out)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.basis = str(basis).lower().replace("-", "_")
        self.num_centers = int(num_centers)
        self.n_freq = int(n_freq)
        if self.basis in {"bspline", "b_spline", "spline"}:
            layer_factory = lambda: BSplineKANLayer(
                d_in,
                d_out,
                grid_size=grid_size,
                spline_order=spline_order,
                x_range=x_range,
            )
            self.basis = "bspline"
        elif self.basis in {"rbf", "gaussian_rbf"}:
            layer_factory = lambda: RBFKANLayer(
                d_in,
                d_out,
                num_centers=num_centers,
                x_range=x_range,
            )
            self.basis = "rbf"
        elif self.basis in {"fourier", "fou", "truncated_fourier"}:
            layer_factory = lambda: FourierKANLayer(
                d_in,
                d_out,
                n_freq=n_freq,
                x_range=x_range,
            )
            self.basis = "fourier"
        else:
            raise ValueError("basis must be bspline, rbf, or fourier")
        self.layers = nn.ModuleList([layer_factory() for _ in range(n_groups)])

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._apply_grouped(group, x, self.layers)

    def reset_coverage(self) -> None:
        for layer in self.layers:
            layer.reset_coverage()

    def coverage_stats(self) -> dict[str, object]:
        return {
            str(index): layer.coverage_stats()
            for index, layer in enumerate(self.layers)
        }


class GroupedLinear(_GroupedBase):
    """A group-specific affine function for the identity/linear probes."""

    def __init__(self, n_groups: int, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__(n_groups, d_in, d_out)
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_out, bias=bias) for _ in range(n_groups)]
        )

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._apply_grouped(group, x, self.layers)


class GroupedMLP(_GroupedBase):
    """A group-specific two-layer SiLU MLP."""

    def __init__(
        self,
        n_groups: int,
        d_in: int,
        d_out: int,
        hidden_features: int = 16,
    ) -> None:
        super().__init__(n_groups, d_in, d_out)
        if int(hidden_features) < 1:
            raise ValueError("hidden_features must be positive")
        self.hidden_features = int(hidden_features)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in, hidden_features),
                    nn.SiLU(),
                    nn.Linear(hidden_features, d_out),
                )
                for _ in range(n_groups)
            ]
        )

    def forward(self, group: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._apply_grouped(group, x, self.layers)
