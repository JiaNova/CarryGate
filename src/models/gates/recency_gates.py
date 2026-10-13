"""Recency gates for the persistence-residual head.

Every gate maps ``(var_id, log1p(staleness)) -> omega in [0, 1]``, the weight
placed on the carry-forward value against the training prior.

Two axes are varied independently and must never be confounded:

``scope``
    ``"shared"``    one function for all variables.
    ``"per_variable"``  one function per variable.

gate family
    constant (``scalar``), fixed exponential (``gru_d``), learnable smooth
    (``mlp``), learnable edge function (``kan``).

The original Phase 3 F3 matrix varied both axes at once -- ``scalar`` was the
only per-variable gate while every other family was shared -- so its 4.5x
advantage could not be attributed.  ``scope`` is therefore explicit here and
defaults to ``"shared"`` for every family, including ``scalar``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..layers.grouped_kan import GroupedKAN, GroupedMLP
from .capacity_sweep import CapacityGate
from .frozen_gate import FrozenGate
from .gate_bases import build_gate_basis
from .monotone_gate import MonotoneDecreasingGate
from .statistic_conditioned_gate import StatisticConditionedGate

_SCOPES = ("shared", "per_variable")


def _n_groups(scope: str, n_vars: int) -> int:
    scope = str(scope).lower().replace("-", "_")
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, received {scope!r}")
    return 1 if scope == "shared" else int(n_vars)


def _group_of(scope: str, var_id: torch.Tensor) -> torch.Tensor:
    var_id = var_id.to(dtype=torch.long)
    if str(scope).lower().replace("-", "_") == "shared":
        return torch.zeros_like(var_id)
    return var_id


class ScalarGate(nn.Module):
    """Constant trust weight, independent of staleness."""

    def __init__(self, n_vars: int, scope: str = "shared") -> None:
        super().__init__()
        self.scope = str(scope)
        self.logit = nn.Parameter(torch.zeros(_n_groups(scope, n_vars)))

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        del log_delta
        return torch.sigmoid(self.logit[_group_of(self.scope, var_id)])


class GRUDGate(nn.Module):
    """GRU-D exponential decay: ``omega = exp(-relu(w * log1p(delta) + b))``."""

    def __init__(self, n_vars: int, scope: str = "shared") -> None:
        super().__init__()
        self.scope = str(scope)
        groups = _n_groups(scope, n_vars)
        self.w = nn.Parameter(torch.ones(groups))
        self.b = nn.Parameter(torch.zeros(groups))

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        group = _group_of(self.scope, var_id)
        return torch.exp(-F.relu(self.w[group] * log_delta + self.b[group]))


class MLPGate(nn.Module):
    """Learnable smooth recency function."""

    def __init__(
        self,
        n_vars: int,
        scope: str = "shared",
        hidden: int = 16,
        monotone: bool = False,
        x_range: tuple[float, float] = (0.0, 4.0),
    ) -> None:
        super().__init__()
        self.scope = str(scope)
        self.mlp = GroupedMLP(
            _n_groups(scope, n_vars), 1, 1, hidden_features=int(hidden)
        )
        self.monotone = bool(monotone)
        self.function = (
            MonotoneDecreasingGate(self.mlp, x_range=x_range)
            if self.monotone
            else None
        )

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        group = _group_of(self.scope, var_id)
        if self.function is not None:
            return self.function(group, log_delta)
        return torch.sigmoid(self.mlp(group, log_delta.unsqueeze(-1)).squeeze(-1))


class KANGate(nn.Module):
    """Learnable edge-function recency gate."""

    def __init__(
        self,
        n_vars: int,
        scope: str = "shared",
        basis: str = "rbf",
        num_centers: int = 8,
        x_range: tuple[float, float] = (-3.0, 3.0),
        grid_size: int | None = None,
        spline_order: int = 3,
        n_freq: int | None = None,
        num_num: int | None = None,
        num_den: int | None = None,
        parameterization: str = "fixed",
        monotone: bool = False,
    ) -> None:
        super().__init__()
        self.scope = str(scope)
        self.basis = str(basis).lower().replace("-", "_")
        self.parameterization = str(parameterization).lower().replace("-", "_")
        if self.parameterization not in {"fixed", "adaptive"}:
            raise ValueError("parameterization must be fixed or adaptive")
        groups = _n_groups(scope, n_vars)
        if self.parameterization == "adaptive":
            if self.basis not in {"rbf", "gaussian_rbf"}:
                raise ValueError("adaptive parameterization is only supported for RBF")
            self.edge = GroupedKAN(
                groups,
                d_in=1,
                d_out=1,
                basis="rbf",
                num_centers=int(num_centers),
                x_range=x_range,
            )
        else:
            self.edge = build_gate_basis(
                self.basis,
                groups,
                num_centers=int(num_centers),
                x_range=x_range,
                grid_size=grid_size,
                spline_order=int(spline_order),
                n_freq=n_freq,
                num_num=num_num,
                num_den=num_den,
            )
        self.monotone = bool(monotone)
        self.function = (
            MonotoneDecreasingGate(self.edge, x_range=x_range)
            if self.monotone
            else None
        )

    @property
    def params_per_variable(self) -> int:
        if self.parameterization == "adaptive":
            return 3 * int(self.edge.num_centers) + 1
        return int(self.edge.params_per_edge)

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        group = _group_of(self.scope, var_id)
        if self.function is not None:
            raw = self.function(group, log_delta)
        elif self.parameterization == "adaptive":
            raw = self.edge(group, log_delta.unsqueeze(-1)).squeeze(-1)
        else:
            raw = self.edge(group, log_delta)
        return raw if self.function is not None else torch.sigmoid(raw)


def build_recency_gate(gate_type, n_vars: int, scope: str = "shared", **kw):
    """Construct one gate.  ``scope`` is orthogonal to ``gate_type``."""

    gate_type = str(gate_type).lower().replace("-", "_")
    if gate_type == "none":
        return None
    if gate_type == "scalar":
        return ScalarGate(n_vars, scope=scope)
    if gate_type == "linear":
        return CapacityGate(n_vars, scope=scope, kind="linear")
    if gate_type == "gru_d":
        return GRUDGate(n_vars, scope=scope)
    if gate_type == "mlp":
        return MLPGate(
            n_vars,
            scope=scope,
            hidden=int(kw.get("hidden", 16)),
            monotone=bool(kw.get("monotone", False)),
            x_range=tuple(kw.get("x_range", (0.0, 4.0))),
        )
    if gate_type in {"mlp_monotone", "monotone_mlp"}:
        return MLPGate(
            n_vars,
            scope=scope,
            hidden=int(kw.get("hidden", 16)),
            monotone=True,
            x_range=tuple(kw.get("x_range", (0.0, 4.0))),
        )
    if gate_type == "kan":
        return KANGate(
            n_vars,
            scope=scope,
            basis=str(kw.get("basis", "rbf")),
            num_centers=int(kw.get("num_centers", 8)),
            x_range=tuple(kw.get("x_range", (-3.0, 3.0))),
            grid_size=kw.get("grid_size"),
            spline_order=int(kw.get("spline_order", 3)),
            n_freq=kw.get("n_freq"),
            num_num=kw.get("num_num"),
            num_den=kw.get("num_den"),
            parameterization=str(kw.get("parameterization", "fixed")),
            monotone=bool(kw.get("monotone", False)),
        )
    if gate_type in {"kan_monotone", "monotone_kan"}:
        return KANGate(
            n_vars,
            scope=scope,
            basis=str(kw.get("basis", "rbf")),
            num_centers=int(kw.get("num_centers", 8)),
            x_range=tuple(kw.get("x_range", (-3.0, 3.0))),
            grid_size=kw.get("grid_size"),
            spline_order=int(kw.get("spline_order", 3)),
            n_freq=kw.get("n_freq"),
            num_num=kw.get("num_num"),
            num_den=kw.get("num_den"),
            parameterization=str(kw.get("parameterization", "fixed")),
            monotone=True,
        )
    if gate_type == "frozen":
        if kw.get("omega_per_var") is None:
            raise ValueError("frozen gate requires omega_per_var")
        return FrozenGate(torch.as_tensor(kw["omega_per_var"], dtype=torch.float32))
    if gate_type in {"statistic_conditioned", "descriptor", "descriptors"}:
        descriptors = kw.get("descriptors")
        if descriptors is None:
            raise ValueError("statistic_conditioned gate requires descriptors")
        return StatisticConditionedGate(
            torch.as_tensor(descriptors, dtype=torch.float32),
            hidden=int(kw.get("descriptor_hidden", kw.get("hidden", 32))),
            use_staleness=bool(kw.get("use_staleness", True)),
            descriptor_mean=kw.get("descriptor_mean"),
            descriptor_std=kw.get("descriptor_std"),
        )
    raise ValueError(gate_type)
