"""Functional value/time-gap event encoders for H1."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from ..layers.grouped_kan import GroupedKAN, GroupedLinear, GroupedMLP
from .shared_lowrank import SharedPlusLowRank


PHYSIOLOGICAL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "hemodynamic": (
        "HR",
        "SysABP",
        "DiasABP",
        "MAP",
        "NISysABP",
        "NIDiasABP",
        "NIMAP",
    ),
    "respiratory": ("SaO2", "PaO2", "FiO2", "PaCO2", "MechVent", "RespRate"),
    "renal": ("Creatinine", "BUN", "Urine"),
    "metabolic": (
        "Glucose",
        "Lactate",
        "pH",
        "HCO3",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
    ),
    "hematologic": ("WBC", "HCT", "Platelets", "Na", "K", "Mg"),
    # The official Set-A/B/C 37-variable vocabulary has no ``Cl`` column;
    # Weight is retained here rather than silently dropped.
    "neuro_other": (
        "GCS",
        "Temp",
        "Albumin",
        "TroponinI",
        "TroponinT",
        "Cholesterol",
        "Weight",
    ),
}


def build_group_mapping(
    variable_names: Sequence[str],
    grouping: str | Sequence[int] = "physiological",
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return a validated variable-to-group map for all 37 variables."""

    names = tuple(variable_names)
    if isinstance(grouping, (list, tuple, torch.Tensor)):
        group = torch.as_tensor(grouping, dtype=torch.long)
        if group.numel() != len(names):
            raise ValueError("explicit group mapping must cover every variable")
        return group.reshape(-1), tuple(f"group_{i}" for i in range(int(group.max()) + 1))
    grouping = str(grouping).lower().replace("-", "_")
    if grouping == "shared":
        return torch.zeros(len(names), dtype=torch.long), ("shared",)
    if grouping in {"per_variable", "pervariable", "variable"}:
        return torch.arange(len(names), dtype=torch.long), names
    if grouping != "physiological":
        raise ValueError("grouping must be shared, physiological, or per_variable")
    lookup = {}
    group_names = tuple(PHYSIOLOGICAL_GROUPS)
    for group_id, group_name in enumerate(group_names):
        for variable in PHYSIOLOGICAL_GROUPS[group_name]:
            lookup[variable] = group_id
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"physiological grouping has no mapping for {missing}")
    return torch.tensor([lookup[name] for name in names], dtype=torch.long), group_names


class FunctionalEventEncoder(nn.Module):
    """Encode ``(variable, value, time-gap)`` event triplets.

    ``encoder_type`` is the only changed module across the H1 ablations.  The
    five supported types cover the saturation probe and the documented A0-A5
    matrix: identity, linear, MLP, additive value-only/both KAN, and joint KAN.
    """

    _ALIASES = {
        "e_identity": "identity",
        "e_linear": "linear",
        "e_mlp": "mlp",
        "kan_value": "kan_value_only",
        "kan_gap_only": "kan_gap_only",
        "kan_recency": "kan_gap_only",
        "joint": "kan_joint",
        "lowrank": "shared_lowrank",
    }

    def __init__(
        self,
        n_vars: int,
        d_model: int = 64,
        group_of_var: Sequence[int] | torch.Tensor | None = None,
        encoder_type: str = "mlp",
        grouping: str = "physiological",
        grid_size: int = 8,
        spline_order: int = 3,
        spline_x_range: tuple[float, float] = (-3.0, 3.0),
        mlp_hidden: int = 16,
        decay_type: str = "additive",
        basis: str = "bspline",
        num_centers: int = 8,
        n_freq: int = 6,
        lowrank_rank: int = 0,
        lowrank_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.n_vars = int(n_vars)
        self.d_model = int(d_model)
        if min(self.n_vars, self.d_model) < 1:
            raise ValueError("n_vars and d_model must be positive")
        canonical = str(encoder_type).lower().replace("-", "_")
        self.encoder_type = self._ALIASES.get(canonical, canonical)
        decay = str(decay_type).lower().replace("-", "_")
        self.decay_type = {
            "r0": "none",
            "r1": "gru_d",
            "r2": "mlp",
            "r3": "kan",
        }.get(decay, decay)
        self.basis = str(basis).lower().replace("-", "_")
        self.lowrank_rank = int(lowrank_rank)
        self.lowrank_hidden = int(lowrank_hidden or mlp_hidden)
        if self.encoder_type == "shared_lowrank" and self.decay_type != "additive":
            raise ValueError("shared_lowrank currently requires additive decay")
        if self.decay_type not in {"additive", "none", "gru_d", "mlp", "kan"}:
            raise ValueError("decay_type must be additive, none, gru_d, mlp, or kan")
        allowed = {
            "identity",
            "linear",
            "mlp",
            "kan_value_only",
            "kan_gap_only",
            "kan_both",
            "kan_joint",
            "shared_lowrank",
        }
        if self.encoder_type not in allowed:
            raise ValueError(f"unknown event encoder type: {encoder_type}")
        if group_of_var is None:
            group_tensor, group_names = build_group_mapping(
                tuple(str(i) for i in range(self.n_vars)), grouping="shared"
            )
        else:
            group_tensor = torch.as_tensor(group_of_var, dtype=torch.long).reshape(-1)
            if group_tensor.numel() != self.n_vars:
                raise ValueError("group_of_var must have one entry per variable")
            group_names = tuple(
                f"group_{i}" for i in range(int(group_tensor.max().item()) + 1)
            )
        if torch.any(group_tensor < 0):
            raise ValueError("group_of_var cannot contain negative IDs")
        self.register_buffer("group_of_var", group_tensor)
        self.group_names = group_names
        n_groups = len(group_names)
        self.var_emb = nn.Embedding(self.n_vars, self.d_model)

        if self.encoder_type == "identity":
            self.f_identity = GroupedLinear(n_groups, 2, self.d_model)
        elif self.encoder_type == "linear":
            self.f_value = GroupedLinear(n_groups, 1, self.d_model)
            self.f_gap = GroupedLinear(n_groups, 1, self.d_model)
        elif self.encoder_type == "mlp":
            self.f_value = GroupedMLP(n_groups, 1, self.d_model, hidden_features=mlp_hidden)
            self.f_gap = GroupedMLP(n_groups, 1, self.d_model, hidden_features=mlp_hidden)
        elif self.encoder_type == "shared_lowrank":
            self.f_value = SharedPlusLowRank(
                self.n_vars,
                1,
                self.d_model,
                rank=self.lowrank_rank,
                hidden=self.lowrank_hidden,
            )
            self.f_gap = SharedPlusLowRank(
                self.n_vars,
                1,
                self.d_model,
                rank=self.lowrank_rank,
                hidden=self.lowrank_hidden,
            )
        elif self.encoder_type == "kan_value_only":
            self.f_value = GroupedKAN(
                n_groups, 1, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )
            self.f_gap = GroupedMLP(n_groups, 1, self.d_model, hidden_features=mlp_hidden)
        elif self.encoder_type == "kan_gap_only":
            self.f_value = GroupedMLP(n_groups, 1, self.d_model, hidden_features=mlp_hidden)
            self.f_gap = GroupedKAN(
                n_groups, 1, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )
        elif self.encoder_type == "kan_both":
            self.f_value = GroupedKAN(
                n_groups, 1, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )
            self.f_gap = GroupedKAN(
                n_groups, 1, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )
        elif self.encoder_type == "kan_joint":
            self.f_joint = GroupedKAN(
                n_groups, 2, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )

        if self.decay_type in {"none", "gru_d"} and hasattr(self, "f_gap"):
            # R0/R1 do not learn a second additive gap function.
            del self.f_gap
        if self.decay_type == "gru_d":
            self.decay_weight = nn.Parameter(torch.ones(n_groups))
            self.decay_bias = nn.Parameter(torch.zeros(n_groups))
        elif self.decay_type == "mlp":
            self.f_gap = GroupedMLP(n_groups, 1, self.d_model, hidden_features=mlp_hidden)
        elif self.decay_type == "kan":
            self.f_gap = GroupedKAN(
                n_groups, 1, self.d_model, grid_size, spline_order, spline_x_range,
                basis=self.basis, num_centers=num_centers, n_freq=n_freq,
            )

    def forward(
        self,
        var_id: torch.Tensor,
        value: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if var_id.shape != value.shape or var_id.shape != delta_t.shape:
            raise ValueError("var_id, value, and delta_t must have identical shapes")
        if var_id.ndim < 1:
            raise ValueError("event tensors must have at least one dimension")
        var_id = var_id.to(dtype=torch.long)
        if torch.any((var_id < 0) | (var_id >= self.n_vars)):
            raise ValueError("var_id contains an invalid variable index")
        value = value.to(dtype=torch.float32)
        gap = torch.log1p(torch.clamp(delta_t.to(dtype=torch.float32), min=0.0))
        group = self.group_of_var[var_id]
        if self.decay_type != "additive":
            # R0-R3 isolate the recency multiplier while holding the value
            # function and Transformer fixed.  R1 is the GRU-D exponential;
            # R2/R3 learn a sigmoid MLP/KAN gate.
            value_function = self.f_value(group, value.unsqueeze(-1))
            if self.decay_type == "none":
                function = value_function
            elif self.decay_type == "gru_d":
                weight = self.decay_weight[group]
                bias = self.decay_bias[group]
                factor = torch.exp(-torch.relu(weight * gap + bias)).unsqueeze(-1)
                function = value_function * factor
            else:
                factor = torch.sigmoid(self.f_gap(group, gap.unsqueeze(-1)))
                function = value_function * factor
            return self.var_emb(var_id) + function
        if self.encoder_type == "identity":
            function = self.f_identity(group, torch.stack([value, gap], dim=-1))
        elif self.encoder_type == "kan_joint":
            function = self.f_joint(group, torch.stack([value, gap], dim=-1))
        elif self.encoder_type == "shared_lowrank":
            function = self.f_value(var_id, value.unsqueeze(-1)) + self.f_gap(
                var_id, gap.unsqueeze(-1)
            )
        else:
            function = self.f_value(group, value.unsqueeze(-1)) + self.f_gap(
                group, gap.unsqueeze(-1)
            )
        return self.var_emb(var_id) + function
