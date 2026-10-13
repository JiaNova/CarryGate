"""Variable-length PyTorch dataset and padding collate for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .persistence_baseline import build_persistence_baseline
from .p12_events import EventStandardizer, load_forecast_split


def _tensor(array, dtype):
    return torch.as_tensor(np.asarray(array), dtype=dtype)


class P12ForecastDataset(Dataset):
    """Dataset whose item contains only real observation/target events."""

    def __init__(
        self,
        examples: Sequence[Mapping[str, object]],
        indices: Sequence[int],
        train_median: torch.Tensor | np.ndarray | None = None,
        held_out_variables: Sequence[int] | None = None,
        held_out_icu_types: Sequence[int] | None = None,
        include_icu_types: Sequence[int] | None = None,
    ):
        self.examples = examples
        candidate_indices = np.asarray(indices, dtype=np.int64)
        held_icu = frozenset(
            int(value) for value in (held_out_icu_types or ())
        )
        include_icu = None if include_icu_types is None else frozenset(
            int(value) for value in include_icu_types
        )
        if held_icu or include_icu is not None:
            selected = []
            for sample_index in candidate_indices.tolist():
                icu_type = _example_icu_type(self.examples[int(sample_index)])
                if held_icu and icu_type in held_icu:
                    continue
                if include_icu is not None and icu_type not in include_icu:
                    continue
                selected.append(int(sample_index))
            candidate_indices = np.asarray(selected, dtype=np.int64)
        self.indices = candidate_indices
        held_out = () if held_out_variables is None else held_out_variables
        self.held_out_variables = frozenset(
            int(value) for value in held_out
        )
        self._persistence: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if train_median is not None:
            median = torch.as_tensor(train_median, dtype=torch.float32).reshape(-1)
            for sample_index in self.indices.tolist():
                example = self.examples[int(sample_index)]
                obs, target = example["obs"], example["target"]
                x_last, staleness = build_persistence_baseline(
                    _tensor(obs["var"], torch.long),
                    _tensor(obs["time"], torch.float32),
                    _tensor(obs["value"], torch.float32),
                    _tensor(target["var"], torch.long),
                    _tensor(target["time"], torch.float32),
                    float(example["t_obs"]),
                    median,
                )
                self._persistence[int(sample_index)] = (
                    x_last.numpy(),
                    staleness.numpy(),
                )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Dict[str, object]:
        example_index = int(self.indices[index])
        example = self.examples[example_index]
        obs = example["obs"]
        target = example["target"]
        item = {
            "record_id": int(example["record_id"]),
            "sample_id": str(example["sample_id"]),
            "label": float(example["label"]),
            "icu_type": _example_icu_type(example),
            "t_obs": float(example["t_obs"]),
            "obs_var": _tensor(obs["var"], torch.long),
            "obs_time": _tensor(obs["time"], torch.float32),
            "obs_value": _tensor(obs["value"], torch.float32),
            "obs_delta_t": _tensor(obs["delta_t"], torch.float32),
            "obs_first": _tensor(obs["is_first"], torch.bool),
            "target_var": _tensor(target["var"], torch.long),
            "target_time": _tensor(target["time"], torch.float32),
            "target_value": _tensor(target["value"], torch.float32),
            "target_delta_t": _tensor(target["delta_t"], torch.float32),
            "target_first": _tensor(target["is_first"], torch.bool),
        }
        target_var = item["target_var"]
        item["target_loss_mask"] = ~torch.isin(
            target_var,
            torch.as_tensor(sorted(self.held_out_variables), dtype=torch.long),
        ) if self.held_out_variables else torch.ones_like(target_var, dtype=torch.bool)
        if example_index in self._persistence:
            x_last, staleness = self._persistence[example_index]
            item["persistence_x_last"] = torch.as_tensor(x_last, dtype=torch.float32)
            item["persistence_staleness"] = torch.as_tensor(
                staleness, dtype=torch.float32
            )
        return item


def _pad_1d(items, length: int, value, dtype):
    result = torch.full((len(items), length), value, dtype=dtype)
    for row, item in enumerate(items):
        if len(item):
            result[row, : len(item)] = item
    return result


def _example_icu_type(example: Mapping[str, object]) -> int:
    """Return the cached ICU type, with a backward-compatible missing marker."""

    value = example.get("icu_type", -1)
    try:
        value = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1
    return int(value) if np.isfinite(value) else -1


def forecast_collate(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Pad event sequences and expose masks in the convention used by PyTorch.

    ``obs_pad_mask`` is ``True`` at padding positions because that is the
    convention required by ``nn.TransformerEncoder`` and
    ``nn.MultiheadAttention``.  ``target_mask`` is the inverse: it is ``True``
    only where a real future observation supplies a training target.
    """

    if not batch:
        raise ValueError("cannot collate an empty batch")
    obs_lengths = [int(item["obs_var"].numel()) for item in batch]
    target_lengths = [int(item["target_var"].numel()) for item in batch]
    max_obs = max(max(obs_lengths), 1)
    max_target = max(max(target_lengths), 1)
    obs_valid = torch.zeros((len(batch), max_obs), dtype=torch.bool)
    target_valid = torch.zeros((len(batch), max_target), dtype=torch.bool)
    for row, (obs_length, target_length) in enumerate(zip(obs_lengths, target_lengths)):
        obs_valid[row, :obs_length] = True
        target_valid[row, :target_length] = True

    output: Dict[str, object] = {
        "record_id": torch.tensor([int(item["record_id"]) for item in batch], dtype=torch.long),
        "sample_id": [str(item["sample_id"]) for item in batch],
        "label": torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32),
        "icu_type": torch.tensor([int(item.get("icu_type", -1)) for item in batch], dtype=torch.long),
        "t_obs": torch.tensor([float(item["t_obs"]) for item in batch], dtype=torch.float32),
        "obs_var": _pad_1d([item["obs_var"] for item in batch], max_obs, 0, torch.long),
        "obs_time": _pad_1d([item["obs_time"] for item in batch], max_obs, 0.0, torch.float32),
        "obs_value": _pad_1d([item["obs_value"] for item in batch], max_obs, 0.0, torch.float32),
        "obs_delta_t": _pad_1d([item["obs_delta_t"] for item in batch], max_obs, 0.0, torch.float32),
        "obs_first": _pad_1d([item["obs_first"] for item in batch], max_obs, False, torch.bool),
        "target_var": _pad_1d([item["target_var"] for item in batch], max_target, 0, torch.long),
        "target_time": _pad_1d([item["target_time"] for item in batch], max_target, 0.0, torch.float32),
        "target_value": _pad_1d([item["target_value"] for item in batch], max_target, 0.0, torch.float32),
        "target_delta_t": _pad_1d([item["target_delta_t"] for item in batch], max_target, 0.0, torch.float32),
        "target_first": _pad_1d([item["target_first"] for item in batch], max_target, False, torch.bool),
        "obs_mask": obs_valid,
        "obs_pad_mask": ~obs_valid,
        "target_mask": target_valid,
    }
    output["target_loss_mask"] = _pad_1d(
        [
            item.get(
                "target_loss_mask",
                torch.ones_like(item["target_var"], dtype=torch.bool),
            )
            for item in batch
        ],
        max_target,
        False,
        torch.bool,
    ) & target_valid
    if "persistence_x_last" in batch[0]:
        output["x_last"] = _pad_1d(
            [item["persistence_x_last"] for item in batch],
            max_target,
            0.0,
            torch.float32,
        )
        output["staleness"] = _pad_1d(
            [item["persistence_staleness"] for item in batch],
            max_target,
            0.0,
            torch.float32,
        )
    # The assertion is part of the data contract and catches accidental
    # future leakage if the split or collate code is changed later.
    if torch.any(output["obs_time"][output["obs_mask"]] >= output["t_obs"].repeat_interleave(obs_valid.sum(1))):
        raise AssertionError("future event entered the padded observation tensor")
    return output


@dataclass
class ForecastSplitBundle:
    standardizer: EventStandardizer
    cache: Mapping[str, object]
    datasets: Dict[str, P12ForecastDataset]
    split_indices: Dict[str, Sequence[int]]


def load_forecast_splits(
    cache_path: str | Path,
    split_path: str | Path,
) -> ForecastSplitBundle:
    cache, split_indices = load_forecast_split(cache_path, split_path)
    examples = cache["examples"]
    train_median = np.zeros(cache["standardizer"].n_variables, dtype=np.float32)
    datasets = {
        name: P12ForecastDataset(examples, indices, train_median=train_median)
        for name, indices in split_indices.items()
    }
    return ForecastSplitBundle(
        standardizer=cache["standardizer"],
        cache=cache,
        datasets=datasets,
        split_indices=split_indices,
    )


def make_forecast_loaders(
    bundle: ForecastSplitBundle,
    *,
    batch_size: int = 32,
    seed: int = 0,
    num_workers: int = 0,
    train_fraction: float = 1.0,
    train_fraction_seed: int = 42,
    held_out_variables: Sequence[int] | None = None,
    held_out_icu_types: Sequence[int] | None = None,
) -> Dict[str, DataLoader]:
    """Build loaders, optionally using a deterministic fraction of training data.

    Only the training dataset is subsampled.  Validation/test examples and the
    train-only standardizer remain unchanged, so the S2-P2 comparison changes
    data quantity without changing the evaluation protocol.
    """

    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    held_icu = tuple(int(value) for value in (held_out_icu_types or ()))
    train_dataset = bundle.datasets["train"]
    if train_fraction < 1.0:
        train_indices = np.asarray(bundle.split_indices["train"], dtype=np.int64)
        n_selected = max(1, int(round(len(train_indices) * train_fraction)))
        rng = np.random.default_rng(int(train_fraction_seed))
        selected = np.sort(rng.choice(train_indices, size=n_selected, replace=False))
        train_dataset = P12ForecastDataset(
            bundle.cache["examples"],
            selected,
            train_median=np.zeros(bundle.standardizer.n_variables, dtype=np.float32),
            held_out_variables=held_out_variables,
            held_out_icu_types=held_icu,
        )
    elif (held_out_variables is not None and len(held_out_variables) > 0) or held_icu:
        train_dataset = P12ForecastDataset(
            bundle.cache["examples"],
            bundle.split_indices["train"],
            train_median=np.zeros(bundle.standardizer.n_variables, dtype=np.float32),
            held_out_variables=held_out_variables,
            held_out_icu_types=held_icu,
        )
    val_dataset = bundle.datasets["val"]
    test_dataset = bundle.datasets["test"]
    if held_icu:
        val_dataset = P12ForecastDataset(
            bundle.cache["examples"],
            bundle.split_indices["val"],
            train_median=np.zeros(bundle.standardizer.n_variables, dtype=np.float32),
            include_icu_types=tuple(
                value
                for value in sorted({
                    _example_icu_type(example)
                    for example in bundle.cache["examples"]
                })
                if value >= 0 and value not in held_icu
            ),
        )
        test_dataset = P12ForecastDataset(
            bundle.cache["examples"],
            bundle.split_indices["test"],
            train_median=np.zeros(bundle.standardizer.n_variables, dtype=np.float32),
            include_icu_types=held_icu,
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            collate_fn=forecast_collate,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=forecast_collate,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=forecast_collate,
            pin_memory=torch.cuda.is_available(),
        ),
    }
