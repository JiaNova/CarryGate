"""Capacity-controlled recency gates for Phase 4 G2-a."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..layers.grouped_kan import GroupedLinear, GroupedMLP


def _n_groups(scope: str, n_vars: int) -> int:
    canonical = str(scope).lower().replace("-", "_")
    if canonical not in {"shared", "per_variable"}:
        raise ValueError("scope must be shared or per_variable")
    return 1 if canonical == "shared" else int(n_vars)


def _group_of(scope: str, var_id: torch.Tensor) -> torch.Tensor:
    if str(scope).lower().replace("-", "_") == "shared":
        return torch.zeros_like(var_id, dtype=torch.long)
    return var_id.to(dtype=torch.long)


@dataclass(frozen=True)
class CapacitySpec:
    """One registered per-variable parameter budget."""

    name: str
    kind: str
    hidden: int | None
    params_per_variable: int


CAPACITY_SPECS = (
    CapacitySpec("G2-P1", "scalar", None, 1),
    CapacitySpec("G2-P2", "linear", None, 2),
    CapacitySpec("G2-P3", "mlp", 2, 7),
    CapacitySpec("G2-P4", "mlp", 8, 25),
    CapacitySpec("G2-P5", "mlp", 16, 49),
    CapacitySpec("G2-P6", "mlp", 64, 193),
)


def capacity_spec(name: str) -> CapacitySpec:
    key = str(name).upper()
    for spec in CAPACITY_SPECS:
        if spec.name.upper() == key:
            return spec
    raise KeyError(f"unknown capacity spec {name!r}")


class CapacityGate(nn.Module):
    """A scalar, linear, or one-hidden-layer gate with explicit capacity."""

    def __init__(
        self,
        n_vars: int,
        scope: str = "shared",
        kind: str = "mlp",
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.scope = str(scope)
        self.kind = str(kind).lower().replace("-", "_")
        groups = _n_groups(scope, n_vars)
        if self.kind == "scalar":
            self.logit = nn.Parameter(torch.zeros(groups))
            self.params_per_group = 1
        elif self.kind == "linear":
            self.function = GroupedLinear(groups, 1, 1)
            self.params_per_group = 2
        elif self.kind == "mlp":
            if hidden is None or int(hidden) < 1:
                raise ValueError("MLP capacity gates require hidden >= 1")
            self.hidden = int(hidden)
            self.function = GroupedMLP(groups, 1, 1, hidden_features=self.hidden)
            self.params_per_group = 3 * self.hidden + 1
        else:
            raise ValueError("kind must be scalar, linear, or mlp")
        self.params_per_variable = int(self.params_per_group)

    def forward(self, var_id: torch.Tensor, log_delta: torch.Tensor) -> torch.Tensor:
        group = _group_of(self.scope, var_id)
        if self.kind == "scalar":
            return torch.sigmoid(self.logit[group])
        return torch.sigmoid(self.function(group, log_delta.unsqueeze(-1)).squeeze(-1))


def build_capacity_gate(
    name: str,
    n_vars: int,
    scope: str = "shared",
) -> CapacityGate:
    spec = capacity_spec(name)
    return CapacityGate(
        n_vars,
        scope=scope,
        kind=spec.kind,
        hidden=spec.hidden,
    )
