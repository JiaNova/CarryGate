"""NAIVE-1 carry-forward as a PyOmniTS model (project-side file).

Zero learned content: the forecast for every target of variable ``v`` is the
last value of ``v`` observed strictly before the cut-off, or the standardised
training median (exactly zero, see ``src/training/forecast_train.py``) when the
variable was never observed in the window.

This is the ICU-KAN metric's zero point. Gate D1b of
``ICU_KAN_D1_external_baseline_spec.md`` requires its predictions to agree
bitwise with ``src/data/persistence_baseline.py``.
"""

import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        # Never read in the forward pass; present only so that the optimiser and
        # the checkpoint writer receive a non-empty parameter list.
        self.unused = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        **kwargs,
    ) -> dict:
        batch_size, seq_len, enc_in = x.shape
        if x_mask is None:
            x_mask = torch.ones_like(x)
        valid = x_mask > 0
        if x_mark is not None:
            # The baseline is defined on observations strictly before the
            # cut-off; x_mark is normalised by the total stay length.
            cutoff = float(self.configs.seq_len) / (
                float(self.configs.seq_len) + float(self.configs.pred_len)
            )
            valid = valid & (x_mark < cutoff)

        times = x_mark.expand(batch_size, seq_len, enc_in)
        neg_inf = torch.finfo(times.dtype).min
        latest = torch.where(valid, times, torch.full_like(times, neg_inf)).max(
            dim=1, keepdim=True
        ).values
        at_latest = valid & (times == latest)
        # When a variable was read more than once at that same instant, the
        # registered baseline keeps the first such reading (torch.argmax returns
        # the first maximal index, and the adapter's stable sort preserves cache
        # order within a tie), so take the lowest row, not the highest.
        positions = torch.arange(seq_len, device=x.device).view(1, seq_len, 1).expand_as(at_latest)
        chosen = torch.where(at_latest, positions, torch.full_like(positions, seq_len)).min(dim=1).values
        observed = chosen < seq_len
        gathered = x.gather(1, chosen.clamp(max=seq_len - 1).unsqueeze(1)).squeeze(1)
        x_last = torch.where(observed, gathered, torch.zeros_like(gathered))

        pred_len = y.shape[1] if y is not None else self.configs.pred_len
        pred = x_last.unsqueeze(1).expand(batch_size, pred_len, enc_in).contiguous()
        # Keep the dummy parameter in the graph without touching the value.
        pred = pred + self.unused * 0.0

        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            f_dim = -1 if self.configs.features == "MS" else 0
            return {
                "pred": pred[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        raise NotImplementedError
