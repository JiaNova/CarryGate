"""This project's forecasting model as a PyOmniTS model (project-side file).

D3 E2. The model is trained here on the field's standard protocol, with the
registered configuration and no benchmark-specific tuning, which is the same
latitude the external baselines received in D1.

The harness hands models a dense ``(batch, time, variable)`` tensor with a mask.
This project's model consumes observation triplets carrying a per-variable
inter-arrival time, so the wrapper rebuilds those triplets. Two properties of
the reconstruction are not incidental and gate D3b checks both against the
project's own pipeline:

* ``delta_t`` is the gap to the previous observation *of the same variable*, or
  the timestamp itself for the first observation of that variable
  (``src/data/p12_parse.py``);
* the carry-forward value of a variable is its latest observation, and among
  observations tied at that latest instant the *first* in grid order, which is
  what ``torch.argmax`` selects in ``src/data/persistence_baseline.py``.

``torch.nonzero`` returns indices in row-major order, so the sample index of the
flattened events is already non-decreasing; the padding below relies on that and
asserts it rather than sorting again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.backbones.strats_lite import STraTSForecastModel  # noqa: E402

# The registered arm, transcribed from scripts/run_eicu_matrix.py. Nothing here
# is tuned for the benchmark; see ICU_KAN_D3_clinical_endpoint_spec.md 2.2.
REGISTERED = dict(
    head_type="residual", grouping="shared",
    gate_type="kan", gate_scope="per_variable",
    recency_basis="rbf", recency_num_centers=16,
    recency_parameterization="adaptive", recency_x_range=(0.0, 4.0),
    d_model=64, nhead=4, num_layers=2, d_ff=128, dropout=0.1,
)


def _group_starts(keys: Tensor) -> Tensor:
    starts = torch.ones_like(keys, dtype=torch.bool)
    if keys.numel() > 1:
        starts[1:] = keys[1:] != keys[:-1]
    return starts


class Events:
    """Flattened events of one dense window, with their padded coordinates."""

    def __init__(self, mask: Tensor, batch_size: int):
        self.sample, self.row, self.var = (mask > 0).nonzero(as_tuple=True)
        if self.sample.numel() > 1 and bool((self.sample[1:] < self.sample[:-1]).any()):
            raise AssertionError("nonzero did not return row-major order")
        counts = torch.bincount(self.sample, minlength=batch_size)
        self.width = max(int(counts.max().item()) if counts.numel() else 0, 1)
        offsets = torch.cumsum(counts, 0) - counts
        self.column = torch.arange(self.sample.numel(), device=mask.device) - offsets[self.sample]
        self.batch_size = batch_size
        self.valid = torch.zeros(batch_size, self.width, dtype=torch.bool, device=mask.device)
        self.valid[self.sample, self.column] = True

    def pad(self, values: Tensor) -> Tensor:
        out = torch.zeros(self.batch_size, self.width, dtype=values.dtype, device=values.device)
        out[self.sample, self.column] = values
        return out

    def gather(self, padded: Tensor) -> Tensor:
        return padded[self.sample, self.column]


class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.n_vars = int(configs.enc_in)
        self.total_hours = float(configs.seq_len) + float(configs.pred_len)
        self.inner = STraTSForecastModel(n_vars=self.n_vars, **REGISTERED)

    def _times(self, marks: Tensor, sample: Tensor, row: Tensor) -> Tensor:
        marks = marks[..., 0] if marks.dim() == 3 else marks
        return marks[sample, row] * self.total_hours

    def _delta_t(self, sample: Tensor, var: Tensor, time: Tensor) -> Tensor:
        key = sample * self.n_vars + var
        order = torch.argsort(key * 0 + time, stable=True)          # time ascending
        order = order[torch.argsort(key[order], stable=True)]        # then by key
        key_s, time_s = key[order], time[order]
        previous = torch.zeros_like(time_s)
        if time_s.numel() > 1:
            previous[1:] = time_s[:-1]
        starts = _group_starts(key_s)
        delta_s = torch.where(starts, time_s, (time_s - previous).clamp_min(0.0))
        delta = torch.empty_like(delta_s)
        delta[order] = delta_s
        return delta

    def _carry_forward(self, sample: Tensor, var: Tensor, time: Tensor,
                       value: Tensor, batch_size: int):
        key = sample * self.n_vars + var
        order = torch.argsort(time, stable=True, descending=True)    # latest first,
        order = order[torch.argsort(key[order], stable=True)]        # ties keep grid order
        firsts = _group_starts(key[order])
        picked = key[order][firsts]
        size = batch_size * self.n_vars
        last_value = torch.zeros(size, dtype=value.dtype, device=value.device)
        last_time = torch.zeros(size, dtype=time.dtype, device=time.device)
        seen = torch.zeros(size, dtype=torch.bool, device=value.device)
        last_value[picked] = value[order][firsts]
        last_time[picked] = time[order][firsts]
        seen[picked] = True
        return last_value, last_time, seen

    def build_batch(self, x, x_mark, x_mask, y, y_mark, y_mask) -> dict:
        """Exposed so that gate D3b can compare it with the project's own batch."""
        batch_size = x.shape[0]
        obs = Events(x_mask, batch_size)
        tgt = Events(y_mask, batch_size)
        obs_time = self._times(x_mark, obs.sample, obs.row)
        obs_value = x[obs.sample, obs.row, obs.var]
        tgt_time = self._times(y_mark, tgt.sample, tgt.row)

        delta = self._delta_t(obs.sample, obs.var, obs_time)
        last_value, last_time, seen = self._carry_forward(
            obs.sample, obs.var, obs_time, obs_value, batch_size)
        flat = tgt.sample * self.n_vars + tgt.var
        # A variable never observed in the window falls back to the training
        # prior, which is exactly zero in this normalisation.
        x_last = torch.where(seen[flat], last_value[flat], torch.zeros_like(tgt_time))
        staleness = torch.where(seen[flat], (tgt_time - last_time[flat]).clamp_min(0.0),
                                torch.zeros_like(tgt_time))
        return {
            "obs_var": obs.pad(obs.var.to(torch.long)),
            "obs_value": obs.pad(obs_value),
            "obs_delta_t": obs.pad(delta),
            "obs_mask": obs.valid,
            "target_var": tgt.pad(tgt.var.to(torch.long)),
            "target_time": tgt.pad(tgt_time),
            "x_last": tgt.pad(x_last),
            "staleness": tgt.pad(staleness),
            "prior": tgt.pad(torch.zeros_like(x_last)),
            "_targets": tgt,
        }

    def forward(self, x: Tensor, x_mark=None, x_mask=None, y=None,
                y_mark=None, y_mask=None, **kwargs) -> dict:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        if y_mask is None:
            y_mask = torch.ones_like(y)
        batch = self.build_batch(x, x_mark, x_mask, y, y_mark, y_mask)
        targets = batch.pop("_targets")
        forecast = self.inner(batch)["forecast"]

        pred = torch.zeros_like(y)
        pred[targets.sample, targets.row, targets.var] = targets.gather(forecast)
        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            f_dim = -1 if self.configs.features == "MS" else 0
            return {"pred": pred[:, :, f_dim:], "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:]}
        raise NotImplementedError
