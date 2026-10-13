"""Variable-conditioned residual path for Phase 5 Line J."""

from __future__ import annotations

import torch
from torch import nn

from .persistence_residual_head import PersistenceResidualHead


class ConditionedResidualHead(PersistenceResidualHead):
    """Apply ``g(s_v)`` to the learned residual while keeping the baseline.

    The forecast is

    ``omega*x_last + (1-omega)*prior + sigmoid(psi(s_v))*Delta``.

    Descriptors are registered as a buffer, so they follow the model device but
    are never optimized.  They must be fitted on the training split by the
    caller, exactly like the statistic-conditioned recency gate.
    """

    def __init__(
        self,
        *args,
        descriptors: torch.Tensor,
        hidden: int = 32,
        **kwargs,
    ) -> None:
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        if descriptors.ndim != 2 or descriptors.shape[1] < 1:
            raise ValueError("descriptors must have shape (n_vars, n_features)")
        if not torch.isfinite(descriptors).all():
            raise ValueError("descriptors must be finite")
        if int(hidden) < 1:
            raise ValueError("hidden must be positive")
        # ``gate_descriptors`` is used only when the recency gate and the
        # residual gate deliberately receive different descriptor sets.
        gate_descriptors = kwargs.pop("gate_descriptors", None)
        if gate_descriptors is not None:
            kwargs["descriptors"] = gate_descriptors
        super().__init__(*args, **kwargs)
        self.register_buffer("residual_descriptors", descriptors.clone())
        self.residual_gate = nn.Sequential(
            nn.Linear(descriptors.shape[1], int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )
        # The residual path starts open.  The residual decoder itself is still
        # zero-initialised by the parent, so the initial forecast remains the
        # registered persistence baseline.
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, 4.0)

    @property
    def n_descriptor_variables(self) -> int:
        return int(self.residual_descriptors.shape[0])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # The parent stores ``_last_baseline`` and ``_last_residual`` just
        # before summing them.  Calling it first preserves every
        # attention/query detail, and reading the residual back directly --
        # rather than as ``output - baseline`` -- makes the scale=1 case
        # bitwise identical to the parent instead of identical up to eps.
        super().forward(*args, **kwargs)
        baseline = self._last_baseline
        delta = self._last_residual
        if len(args) >= 3:
            tgt_var = args[2]
        elif "tgt_var" in kwargs:
            tgt_var = kwargs["tgt_var"]
        else:
            raise TypeError("tgt_var must be supplied positionally or by keyword")
        tgt_var = tgt_var.to(dtype=torch.long)
        if torch.any((tgt_var < 0) | (tgt_var >= self.n_descriptor_variables)):
            raise ValueError("target variable is outside residual descriptors")
        scale = torch.sigmoid(
            self.residual_gate(self.residual_descriptors[tgt_var]).squeeze(-1)
        )
        return baseline + scale * delta


__all__ = ["ConditionedResidualHead"]
