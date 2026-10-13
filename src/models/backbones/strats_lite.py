"""Minimal STraTS-style triplet Transformer for Phase 1."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..encoders.functional_event_encoder import FunctionalEventEncoder, build_group_mapping
from ..heads.forecast_head import ForecastHead


class STraTSLite(nn.Module):
    """Triplet encoder with a fixed two-layer Transformer.

    There is intentionally **no positional encoding** here.  Absolute time is
    supplied through the event time-gap feature and the forecast query's fixed
    sinusoidal time embedding, as required by the Phase 1 protocol.
    """

    def __init__(
        self,
        n_vars: int = 37,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        encoder_type: str = "mlp",
        group_of_var=None,
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
        if group_of_var is None:
            # The default vocabulary is the official Phase 1 P12 order.
            from ...data.p12_parse import TIME_SERIES_VARIABLES

            if int(n_vars) == len(TIME_SERIES_VARIABLES):
                group_of_var, _ = build_group_mapping(TIME_SERIES_VARIABLES, grouping)
            else:
                # Small synthetic shape tests do not have clinical names;
                # keep them on one shared group rather than inventing a
                # partial physiological mapping.
                group_of_var = torch.zeros(int(n_vars), dtype=torch.long)
        self.event_encoder = FunctionalEventEncoder(
            n_vars=n_vars,
            d_model=d_model,
            group_of_var=group_of_var,
            encoder_type=encoder_type,
            grid_size=grid_size,
            spline_order=spline_order,
            spline_x_range=spline_x_range,
            mlp_hidden=mlp_hidden,
            decay_type=decay_type,
            basis=basis,
            num_centers=num_centers,
            n_freq=n_freq,
            lowrank_rank=lowrank_rank,
            lowrank_hidden=lowrank_hidden,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = int(d_model)

    def encode(
        self,
        var_id: torch.Tensor,
        value: torch.Tensor,
        delta_t: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.event_encoder(var_id, value, delta_t)
        if pad_mask is None:
            pad_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            pad_mask = pad_mask.to(device=hidden.device, dtype=torch.bool)
        # Avoid all-padding NaNs in the underlying attention implementation.
        hidden = hidden.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        all_pad = pad_mask.all(dim=1)
        safe_mask = pad_mask
        if bool(all_pad.any()):
            safe_mask = pad_mask.clone()
            safe_mask[all_pad, 0] = False
        encoded = self.transformer(hidden, src_key_padding_mask=safe_mask)
        return encoded.masked_fill(pad_mask.unsqueeze(-1), 0.0)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.encode(*args, **kwargs)


class STraTSForecastModel(nn.Module):
    """STraTS-lite plus forecast query and secondary mortality output."""

    def __init__(
        self,
        *,
        head_type: str = "direct",
        gate_type: str = "none",
        gate_scope: str = "shared",
        gate_hidden: int = 16,
        recency_num_centers: int = 8,
        recency_x_range: tuple[float, float] = (-3.0, 3.0),
        gate_descriptors: torch.Tensor | None = None,
        gate_omega_per_var: torch.Tensor | None = None,
        gate_use_staleness: bool = True,
        gate_descriptor_hidden: int = 32,
        gate_descriptor_mean: torch.Tensor | None = None,
        gate_descriptor_std: torch.Tensor | None = None,
        residual_scale_scope: str | None = None,
        recency_basis: str = "rbf",
        recency_grid_size: int | None = None,
        recency_spline_order: int = 3,
        recency_n_freq: int | None = None,
        recency_num_num: int | None = None,
        recency_num_den: int | None = None,
        recency_parameterization: str = "fixed",
        gate_monotone: bool = False,
        conditioned_residual: bool = False,
        residual_descriptors: torch.Tensor | None = None,
        residual_descriptor_hidden: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = STraTSLite(**kwargs)
        self.head_type = str(head_type).lower().replace("-", "_")
        if self.head_type == "direct":
            self.forecast_head = ForecastHead(
                self.backbone.event_encoder.var_emb,
                d_model=self.backbone.d_model,
                nhead=int(kwargs.get("nhead", 4)),
                dropout=float(kwargs.get("dropout", 0.1)),
            )
        elif self.head_type == "residual":
            from ..heads.conditioned_residual_head import ConditionedResidualHead
            from ..heads.persistence_residual_head import PersistenceResidualHead

            head_class = ConditionedResidualHead if conditioned_residual else PersistenceResidualHead
            if conditioned_residual and residual_descriptors is None:
                raise ValueError("conditioned residual heads require residual_descriptors")

            head_kwargs = dict(
                d_model=self.backbone.d_model,
                n_vars=int(kwargs["n_vars"]),
                gate_type=gate_type,
                scope=str(gate_scope),
                hidden=int(gate_hidden),
                num_centers=int(recency_num_centers),
                x_range=tuple(recency_x_range),
                descriptors=gate_descriptors,
                omega_per_var=gate_omega_per_var,
                use_staleness=bool(gate_use_staleness),
                descriptor_hidden=int(gate_descriptor_hidden),
                descriptor_mean=gate_descriptor_mean,
                descriptor_std=gate_descriptor_std,
                residual_scale_scope=residual_scale_scope,
                gate_basis=str(recency_basis),
                gate_grid_size=recency_grid_size,
                gate_spline_order=int(recency_spline_order),
                gate_n_freq=recency_n_freq,
                gate_num_num=recency_num_num,
                gate_num_den=recency_num_den,
                gate_parameterization=str(recency_parameterization),
                gate_monotone=bool(gate_monotone),
            )
            if conditioned_residual:
                head_kwargs.update(
                    descriptors=residual_descriptors,
                    hidden=int(residual_descriptor_hidden),
                )
                if gate_descriptors is not None:
                    head_kwargs["gate_descriptors"] = gate_descriptors
            self.forecast_head = head_class(**head_kwargs)
        else:
            raise ValueError("head_type must be direct or residual")
        self.mortality_head = nn.Linear(self.backbone.d_model, 1)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone.encode(
            batch["obs_var"],
            batch["obs_value"],
            batch["obs_delta_t"],
            batch.get("obs_pad_mask"),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encode(batch)
        if self.head_type == "direct":
            forecast = self.forecast_head(
                hidden,
                batch["target_var"],
                batch["target_time"],
                batch.get("obs_pad_mask"),
            )
        else:
            required = {"x_last", "prior", "staleness"}
            missing = required.difference(batch)
            if missing:
                raise KeyError(
                    "residual forecast batches must include "
                    + ", ".join(sorted(missing))
                )
            forecast = self.forecast_head(
                hidden,
                batch["obs_mask"],
                batch["target_var"],
                batch["target_time"],
                batch["x_last"],
                batch["prior"],
                batch["staleness"],
            )
        valid = batch["obs_mask"].to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        mortality = self.mortality_head(pooled).squeeze(-1)
        return {"forecast": forecast, "mortality": mortality, "hidden": hidden}

    def forecast(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(batch)["forecast"]
