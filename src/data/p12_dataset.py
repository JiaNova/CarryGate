"""PyTorch datasets backed by the leakage-safe P12 cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .p12_hourly_agg import P12Cache, cache_features, load_preprocessed, load_split_indices


class P12Dataset(Dataset):
    """One fixed P12 split with ``(hourly_features, static_features, label)``."""

    def __init__(
        self,
        x: np.ndarray,
        static: np.ndarray,
        y: np.ndarray,
        record_ids: np.ndarray | None = None,
    ) -> None:
        if x.ndim != 3 or static.ndim != 2 or y.ndim != 1:
            raise ValueError("invalid P12 array dimensions")
        if not (len(x) == len(static) == len(y)):
            raise ValueError("P12 arrays must have the same first dimension")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.static = torch.from_numpy(np.ascontiguousarray(static, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.record_ids = (
            np.asarray(record_ids, dtype=np.int64)
            if record_ids is not None
            else np.arange(len(y), dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.static[index], self.y[index]


@dataclass
class P12SplitBundle:
    cache: P12Cache
    split_indices: Dict[str, Sequence[int]]
    datasets: Dict[str, P12Dataset]


def load_p12_splits(
    processed_path: str | Path,
    split_path: str | Path,
) -> P12SplitBundle:
    """Load all fixed splits from disk."""

    cache = load_preprocessed(processed_path)
    split_indices = load_split_indices(split_path)
    datasets: Dict[str, P12Dataset] = {}
    for name in ("train", "val", "test"):
        if name not in split_indices:
            raise ValueError(f"split file is missing {name}")
        indices = np.asarray(split_indices[name], dtype=np.int64)
        x, static, y = cache_features(cache, indices)
        datasets[name] = P12Dataset(x, static, y, cache.record_ids[indices])
    return P12SplitBundle(cache=cache, split_indices=split_indices, datasets=datasets)


def make_loaders(
    bundle: P12SplitBundle,
    *,
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create deterministic train-shuffle and eval loaders."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "train": DataLoader(
            bundle.datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            bundle.datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            bundle.datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
