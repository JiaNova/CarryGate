"""Shared functions with a low-rank variable-specific adapter (S3)."""

from __future__ import annotations

import torch
from torch import nn


class SharedPlusLowRank(nn.Module):
    """Implement the nested low-rank adapter specified for Phase 2.

    ``phi_v(x) = phi_shared(x) + alpha * U_v @ (V.T psi(x))``.
    The gate starts at zero, so positive-rank models start exactly at the
    shared solution when their shared weights are held fixed.
    """

    def __init__(
        self,
        n_vars: int,
        d_in: int,
        d_out: int,
        rank: int,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        if min(int(n_vars), int(d_in), int(d_out), int(hidden)) < 1:
            raise ValueError("SharedPlusLowRank dimensions must be positive")
        if int(rank) < 0:
            raise ValueError("rank must be non-negative")
        self.n_vars = int(n_vars)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.rank = int(rank)
        self.hidden = int(hidden)
        self.shared = nn.Sequential(
            nn.Linear(self.d_in, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.d_out),
        )
        if self.rank > 0:
            self.psi = nn.Sequential(nn.Linear(self.d_in, self.hidden), nn.SiLU())
            self.V = nn.Parameter(torch.randn(self.hidden, self.rank) * 0.02)
            self.U = nn.Parameter(torch.zeros(self.n_vars, self.d_out, self.rank))
            nn.init.normal_(self.U, std=0.02)
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, var_id: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_in:
            raise ValueError(f"expected x (...,{self.d_in})")
        if var_id.shape != x.shape[:-1]:
            raise ValueError("var_id must match x shape without its last dimension")
        var_id = var_id.to(device=x.device, dtype=torch.long)
        if torch.any((var_id < 0) | (var_id >= self.n_vars)):
            raise ValueError("var_id contains an invalid variable index")
        out = self.shared(x)
        if self.rank == 0:
            return out
        z = self.psi(x) @ self.V
        u = self.U[var_id]
        adapter = torch.einsum("...or,...r->...o", u, z)
        return out + self.alpha * adapter
