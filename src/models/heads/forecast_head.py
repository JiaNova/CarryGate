"""Linear-output cross-attention query head for irregular forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn


class FixedSinusoidalTimeEmbedding(nn.Module):
    """Parameter-free absolute-time embedding used only in forecast queries."""

    def __init__(self, d_model: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_period = float(max_period)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.to(dtype=torch.float32)
        half = self.d_model // 2
        if half == 0:
            return time.unsqueeze(-1)
        frequency = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        phase = time.unsqueeze(-1) * frequency
        result = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if result.shape[-1] < self.d_model:
            result = torch.cat([result, torch.zeros_like(result[..., :1])], dim=-1)
        return result[..., : self.d_model]


class ForecastHead(nn.Module):
    """Cross-attention from ``(target variable, target time)`` to event memory."""

    def __init__(
        self,
        var_embedding: nn.Embedding,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.var_embedding = var_embedding
        self.d_model = int(d_model)
        self.time_embedding = FixedSinusoidalTimeEmbedding(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        # Deliberately linear: Phase 0 showed that a nonlinear prediction head
        # is a saturated placement, so H1 measures only the event encoder.
        self.output = nn.Linear(d_model, 1)

    def forward(
        self,
        memory: torch.Tensor,
        target_var: torch.Tensor,
        target_time: torch.Tensor,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory.ndim != 3 or target_var.ndim != 2 or target_time.shape != target_var.shape:
            raise ValueError("invalid memory or forecast query shapes")
        query = self.var_embedding(target_var.to(dtype=torch.long))
        query = query + self.time_embedding(target_time)
        safe_memory = memory
        safe_mask = memory_pad_mask
        if safe_mask is not None:
            safe_mask = safe_mask.to(dtype=torch.bool)
            safe_memory = memory.masked_fill(safe_mask.unsqueeze(-1), 0.0)
            # MultiheadAttention does not define an all-masked row.  Such a
            # row is only padding (normally impossible after preprocessing),
            # so expose one zero memory token to keep the forward finite.
            all_pad = safe_mask.all(dim=1)
            if bool(all_pad.any()):
                safe_mask = safe_mask.clone()
                safe_mask[all_pad, 0] = False
        attended, _ = self.cross_attention(
            query,
            safe_memory,
            safe_memory,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        return self.output(attended).squeeze(-1)
