"""Persistence-residual forecast head from the Phase 3 specification."""

from __future__ import annotations

import torch
from torch import nn

from ..gates.recency_gates import build_recency_gate
from .forecast_head import FixedSinusoidalTimeEmbedding


class PersistenceResidualHead(nn.Module):
    """Predict ``x_last + Delta`` with an optional recency baseline gate."""

    def __init__(
        self,
        d_model: int,
        n_vars: int,
        gate_type: str = "none",
        descriptors: torch.Tensor | None = None,
        omega_per_var: torch.Tensor | None = None,
        use_staleness: bool = True,
        descriptor_hidden: int = 32,
        descriptor_mean: torch.Tensor | None = None,
        descriptor_std: torch.Tensor | None = None,
        residual_scale_scope: str | None = None,
        gate_basis: str = "rbf",
        gate_grid_size: int | None = None,
        gate_spline_order: int = 3,
        gate_n_freq: int | None = None,
        gate_num_num: int | None = None,
        gate_num_den: int | None = None,
        gate_parameterization: str = "fixed",
        gate_monotone: bool = False,
        **gate_kwargs,
    ) -> None:
        super().__init__()
        self.query_var = nn.Embedding(int(n_vars), int(d_model))
        self.time_emb = FixedSinusoidalTimeEmbedding(int(d_model))
        self.cross_attn = nn.MultiheadAttention(
            int(d_model), 4, batch_first=True
        )
        self.delta_out = nn.Linear(int(d_model), 1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        self.gate = build_recency_gate(
            gate_type,
            int(n_vars),
            descriptors=descriptors,
            omega_per_var=omega_per_var,
            use_staleness=use_staleness,
            descriptor_hidden=descriptor_hidden,
            descriptor_mean=descriptor_mean,
            descriptor_std=descriptor_std,
            basis=gate_basis,
            grid_size=gate_grid_size,
            spline_order=gate_spline_order,
            n_freq=gate_n_freq,
            num_num=gate_num_num,
            num_den=gate_num_den,
            parameterization=gate_parameterization,
            monotone=gate_monotone,
            **gate_kwargs,
        )
        self.residual_scale_scope = residual_scale_scope
        if residual_scale_scope is None:
            self.residual_scale = None
        elif str(residual_scale_scope).lower().replace("-", "_") == "shared":
            self.residual_scale = nn.Parameter(torch.ones(1))
        elif str(residual_scale_scope).lower().replace("-", "_") == "per_variable":
            self.residual_scale = nn.Parameter(torch.ones(int(n_vars)))
        else:
            raise ValueError("residual_scale_scope must be None, shared, or per_variable")
        self.gate_type = str(gate_type)

    def forward(
        self,
        H: torch.Tensor,
        obs_mask: torch.Tensor,
        tgt_var: torch.Tensor,
        tgt_time: torch.Tensor,
        x_last: torch.Tensor,
        prior: torch.Tensor,
        staleness: torch.Tensor,
    ) -> torch.Tensor:
        if H.ndim != 3 or tgt_var.ndim != 2 or tgt_time.shape != tgt_var.shape:
            raise ValueError("invalid memory or residual forecast query shapes")
        if any(value.shape != tgt_var.shape for value in (x_last, prior, staleness)):
            raise ValueError("residual baseline tensors must match target shape")
        q = self.query_var(tgt_var.to(dtype=torch.long)) + self.time_emb(tgt_time)
        valid_mask = obs_mask.to(dtype=torch.bool)
        safe_mask = ~valid_mask
        safe_memory = H.masked_fill(safe_mask.unsqueeze(-1), 0.0)
        all_pad = safe_mask.all(dim=1)
        if bool(all_pad.any()):
            safe_mask = safe_mask.clone()
            safe_mask[all_pad, 0] = False
        context, _ = self.cross_attn(
            q,
            safe_memory,
            safe_memory,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        delta = self.delta_out(context).squeeze(-1)
        if self.residual_scale is not None:
            if self.residual_scale.numel() == 1:
                delta = delta * self.residual_scale
            else:
                delta = delta * self.residual_scale[tgt_var.to(dtype=torch.long)]
        if self.gate is None:
            baseline = x_last
        else:
            omega = self.gate(
                tgt_var,
                torch.log1p(torch.clamp(staleness, min=0.0)),
            )
            baseline = omega * x_last + (1.0 - omega) * prior
        # A conditioned residual head subclasses this module and needs both
        # halves before they are summed.  Keeping the residual itself lets the
        # child scale it exactly; recovering it as ``output - baseline`` would
        # only be correct up to floating-point round-off.
        self._last_baseline = baseline
        self._last_residual = delta
        return baseline + delta
