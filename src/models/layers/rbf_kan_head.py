"""RBF-KAN prediction head kept for the Phase 0 protocol."""

from __future__ import annotations

import torch
from torch import nn

from .rbf_kan_layer import RBFKANLayer


class RBFKANHead(nn.Module):
    """Two RBF-KAN layers for the H0 head comparison."""

    backend = "rbf"

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = 1,
        num_centers: int = 8,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        self.layer1 = RBFKANLayer(in_features, hidden_features, num_centers, x_range)
        self.layer2 = RBFKANLayer(hidden_features, out_features, num_centers, x_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))
