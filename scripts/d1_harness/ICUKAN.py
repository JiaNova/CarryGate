"""ICU-KAN cohort adapter for PyOmniTS (project-side file; upstream unmodified).

Exposes the three ICU-KAN caches (P12, MIMIC-IV, eICU) through the same
``tsdmDataset`` surface the upstream P12/MIMIC classes use, so that every
baseline in ``models/`` runs on our cohorts without a single change to
upstream model code.

Selected with::

    --dataset_name ICUKAN
    --dataset_root_path data/processed
    --dataset_id {p12,mimiciv,eicu}
    --seq_len 24 --pred_len 24 --enc_in 37 --dec_in 37 --c_out 37

Spec: ICU_KAN_D1_external_baseline_spec.md sections 4 and 5.2.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (  # noqa: F401
    collate_fn,
    collate_fn_fractal,
    collate_fn_patch,
    collate_fn_tpatch,
    tsdmDataset,
)

# Hours spanned by one stay in the ICU-KAN task: observe [0, 24], forecast (24, 48].
L_TOTAL_HOURS = 48.0
# Variable count is frozen across cohorts by construction; absent variables keep
# their index and emit no events (see manuscript section 3.1).
N_VARIABLES = 37

_CACHE_FILES = {
    "p12": ("p12_events.pkl", "split_indices_12k.json"),
    "mimiciv": ("mimiciv_events.pkl", "mimiciv_split_indices_12k.json"),
    "eicu": ("eicu_events.pkl", "eicu_split_indices_12k.json"),
}
_CACHE_VARIANTS = {
    "registered": _CACHE_FILES,
    "jbi_patient": {
        **_CACHE_FILES,
        "eicu": ("eicu_events_jbi_v1.pkl", "eicu_split_indices_jbi_v1.json"),
    },
}


def _resolve_cache_spec(cohort: str, variant: str) -> tuple[str, str]:
    variant = str(variant or "registered").lower()
    if variant not in _CACHE_VARIANTS:
        raise ValueError(
            f"data variant must be one of {sorted(_CACHE_VARIANTS)}, got {variant!r}"
        )
    if variant == "jbi_patient" and cohort != "eicu":
        raise ValueError("jbi_patient data variant is defined only for the eICU cohort")
    return _CACHE_VARIANTS[variant][cohort]


class Inputs(NamedTuple):
    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor] | None = None


def _densify(var: np.ndarray, time: np.ndarray, value: np.ndarray) -> tuple[Tensor, Tensor]:
    """Triplets -> (t, x) with NaN wherever a variable was not observed.

    Times are divided by ``L_TOTAL_HOURS`` because every upstream length scan
    compares ``x_mark`` against ``bound / L_TOTAL``.

    A dense row cannot hold two readings of the same variable at the same
    instant, and the caches do contain such repeats -- 0.226% of P12 events,
    0.004% of MIMIC-IV, 0.165% of eICU, with up to 198 repeats at one eICU
    timestamp. Rather than averaging them away, which would change the target
    set the metric is computed over, repeats are placed on additional rows that
    carry the same timestamp. Row order stays non-decreasing in time and no
    event is dropped.
    """
    if var.size == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty((0, N_VARIABLES), dtype=torch.float32)
    order = np.lexsort((var, time))
    var, time, value = var[order], time[order], value[order]

    # Rank of each event within its (time, variable) group. The lexsort above
    # makes those groups contiguous, so ``inverse`` is non-decreasing.
    _, inverse, counts = np.unique(
        np.stack([time, var]), axis=1, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    group_start = np.searchsorted(inverse, np.arange(counts.size))
    rank = np.arange(inverse.size) - group_start[inverse]

    uniq_time, time_index = np.unique(time, return_inverse=True)
    rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
    np.maximum.at(rows_per_time, time_index, rank + 1)
    row_start = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])
    row_of = row_start[time_index] + rank

    n_rows = int(rows_per_time.sum())
    x = np.full((n_rows, N_VARIABLES), np.nan, dtype=np.float32)
    x[row_of, var] = value
    if int(np.isfinite(x).sum()) != var.size:
        raise ValueError("dense conversion lost or merged events")
    t = np.repeat(uniq_time, rows_per_time)
    return (
        torch.as_tensor(t / L_TOTAL_HOURS, dtype=torch.float32),
        torch.as_tensor(x, dtype=torch.float32),
    )


class _TaskDataset(Dataset):
    """Minimal stand-in for tsdm's ``TaskDataset``: yields ``Sample``."""

    def __init__(self, examples: list, indices: np.ndarray):
        self.examples = examples
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, key: int) -> Sample:
        idx = int(self.indices[key])
        ex = self.examples[idx]
        obs, tgt = ex["obs"], ex["target"]
        t_obs, x_obs = _densify(
            np.asarray(obs["var"]), np.asarray(obs["time"]), np.asarray(obs["value"])
        )
        t_tgt, x_tgt = _densify(
            np.asarray(tgt["var"]), np.asarray(tgt["time"]), np.asarray(tgt["value"])
        )
        return Sample(
            key=idx,
            inputs=Inputs(t_obs, x_obs, t_tgt),
            targets=x_tgt,
            originals=(torch.cat([t_obs, t_tgt]), torch.cat([x_obs, x_tgt])),
        )


class _Task:
    """Minimal stand-in for a tsdm ``BaseTask``: only ``get_dataset`` is used."""

    def __init__(self, examples: list, splits: dict[str, np.ndarray]):
        self.examples = examples
        self.splits = splits

    def get_dataset(self, key):
        _, flag = key
        return _TaskDataset(self.examples, self.splits[flag])


class Data(tsdmDataset):
    def __init__(self, configs, flag: str = "train", **kwargs):
        super().__init__(configs=configs, flag=flag)
        self.L_TOTAL = L_TOTAL_HOURS
        self._load_cache()
        self._check_lengths()
        self._preprocess()
        self._get_sample_index()

    def _load_cache(self):
        cohort = str(self.configs.dataset_id).lower()
        if cohort not in _CACHE_FILES:
            raise ValueError(f"--dataset_id must be one of {sorted(_CACHE_FILES)}, got {cohort!r}")
        root = Path(self.configs.dataset_root_path)
        variant = str(getattr(self.configs, "dataset_subset_name", None) or "registered").lower()
        events_name, split_name = _resolve_cache_spec(cohort, variant)
        cache_path = (root / events_name).resolve()
        split_path = (root / split_name).resolve()
        if not cache_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                f"ICU-KAN data source is incomplete: cache={cache_path}, split={split_path}"
            )
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        self._examples = payload["examples"]
        if len(payload["variable_names"]) != N_VARIABLES:
            raise ValueError(
                f"cache declares {len(payload['variable_names'])} variables, expected {N_VARIABLES}"
            )
        with open(split_path) as handle:
            raw_splits = json.load(handle)
        self._splits = {
            flag: np.asarray(raw_splits[flag], dtype=np.int64)
            for flag in ("train", "val", "test")
        }
        self._cohort = cohort
        self._data_variant = variant
        self.data_provenance = {
            "data_source": f"ICU-KAN/{cohort}",
            "data_variant": variant,
            "cache_path": str(cache_path),
            "split_path": str(split_path),
            "split_unit": str(raw_splits.get("split_unit", "registered_unverifiable")),
            "group_column": raw_splits.get("group_column"),
            "n_samples": int(len(self._examples)),
            "split_sizes": {flag: int(len(values)) for flag, values in self._splits.items()},
        }
        if variant == "jbi_patient":
            if self.data_provenance["split_unit"] != "patient_grouped":
                raise ValueError("jbi_patient source must declare split_unit=patient_grouped")
            if not all("uniquepid" in example for example in self._examples):
                raise ValueError("jbi_patient source must retain uniquepid in every example")

    def _preprocess(self):
        self._preprocess_base(_Task(self._examples, self._splits))

    def _get_sample_index(self):
        if self.flag == "test_all":
            self.sample_index = torch.as_tensor(
                np.concatenate([self._splits[f] for f in ("train", "val", "test")])
            )
        else:
            self.sample_index = torch.as_tensor(self._splits[self.flag])
