"""Fixed GRU backbone for the Phase 0 H0 comparison."""

from __future__ import annotations

import torch
from torch import nn


class GRUBackbone(nn.Module):
    """Encode 48 hourly feature vectors and fuse static descriptors."""

    def __init__(
        self,
        input_dim: int = 111,
        hidden_size: int = 64,
        static_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.static_dim = int(static_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch ignores recurrent dropout for a single layer.  The
            # explicit output dropout below preserves the documented 0.1
            # regularization while keeping both heads identical.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_projection = nn.Linear(static_dim, hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected x shape (B,T,{self.input_dim})")
        if static.ndim != 2 or static.shape[-1] != self.static_dim:
            raise ValueError(f"expected static shape (B,{self.static_dim})")
        sequence, _ = self.gru(x)
        last_hidden = sequence[:, -1, :]
        fused = last_hidden + self.static_projection(static)
        return self.output_dropout(fused)
