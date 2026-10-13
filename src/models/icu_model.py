"""H0 model assembly and prediction-head factory."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from ..training.param_matching import match_head_spec, mlp_target_params
from .backbones.gru_backbone import GRUBackbone
from .layers.kan_bspline_head import BSplineKANHead
from .layers.mlp_head import LinearHead, MLPHead
from .layers.rbf_kan_head import RBFKANHead


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def build_head(
    head_type: str,
    *,
    in_dim: int,
    target_params: int | None = None,
    hidden_dim: int = 64,
    grid_size: int = 5,
    spline_order: int = 3,
    num_centers: int = 8,
    x_range: tuple[float, float] = (-5.0, 5.0),
) -> tuple[nn.Module, Dict[str, Any]]:
    """Build a head and return its auditable parameter specification."""

    canonical = head_type.lower().replace("-", "_")
    if canonical == "linear":
        head = LinearHead(in_dim, 1)
        spec = {
            "layer_type": "linear",
            "hidden_dim": None,
            "n_params": count_parameters(head),
            "target_params": target_params,
            "param_diff": None if target_params is None else count_parameters(head) - target_params,
            "param_diff_pct": None
            if target_params is None
            else 100.0 * (count_parameters(head) - target_params) / target_params,
        }
        return head, spec

    if canonical == "mlp_large":
        hidden = int(hidden_dim * 2)
        head = MLPHead(in_dim, hidden, 1)
        return head, {
            "layer_type": "mlp_large",
            "hidden_dim": hidden,
            "n_params": count_parameters(head),
            "target_params": target_params,
            "param_diff": None if target_params is None else count_parameters(head) - target_params,
            "param_diff_pct": None
            if target_params is None
            else 100.0 * (count_parameters(head) - target_params) / target_params,
        }

    if target_params is None:
        target_params = mlp_target_params(in_dim, hidden_dim=hidden_dim)

    if canonical == "mlp":
        head = MLPHead(in_dim, hidden_dim, 1)
        spec = {
            "layer_type": "mlp",
            "hidden_dim": hidden_dim,
            "n_params": count_parameters(head),
            "target_params": target_params,
            "param_diff": count_parameters(head) - target_params,
            "param_diff_pct": 100.0 * (count_parameters(head) - target_params) / target_params,
        }
        return head, spec

    if canonical not in {"kan_bspline", "rbf_kan"}:
        raise ValueError(f"unknown head type: {head_type}")

    spec = match_head_spec(
        canonical,
        in_dim,
        1,
        target_params,
        reference_hidden=hidden_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        num_centers=num_centers,
        search_grid=(canonical == "kan_bspline"),
        x_range=x_range,
    )
    if canonical == "kan_bspline":
        head = BSplineKANHead(
            in_dim,
            int(spec["hidden_dim"]),
            1,
            grid_size=int(spec["grid_size"]),
            spline_order=int(spec["spline_order"]),
            x_range=x_range,
        )
    else:
        head = RBFKANHead(
            in_dim,
            int(spec["hidden_dim"]),
            1,
            num_centers=int(spec["num_centers"]),
            x_range=x_range,
        )
    actual = count_parameters(head)
    if actual != spec["n_params"]:
        raise AssertionError(f"analytic head count disagrees with PyTorch count: {actual} != {spec['n_params']}")
    return head, spec


class ICUHead0Model(nn.Module):
    """Shared GRU backbone plus one replaceable H0 prediction head."""

    def __init__(self, backbone: GRUBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.backbone(x, static))
        return logits.squeeze(-1)
