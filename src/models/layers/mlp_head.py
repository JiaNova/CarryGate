"""MLP and linear prediction heads used as baselines."""

from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """Single affine output layer (B0)."""

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPHead(nn.Module):
    """Two-layer SiLU MLP head (B1/B1-large and Stage A baseline)."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int = 1) -> None:
        super().__init__()
        if hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))
