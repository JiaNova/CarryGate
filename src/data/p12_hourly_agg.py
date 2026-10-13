"""Hourly aggregation, leakage-safe normalization, and caching for P12 H0."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .p12_parse import (
    STATIC_FEATURES,
    TIME_SERIES_VARIABLES,
    iter_record_paths,
    load_outcomes,
    parse_records,
)


N_HOURS = 48
EPS = 1e-6


@dataclass
class P12Preprocessor:
    """Statistics fitted on the training split only."""

    variable_names: Tuple[str, ...]
    value_mean: np.ndarray
    value_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    icu_mode: int

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    @property
    def input_dim(self) -> int:
        return 3 * self.n_variables

    @property
    def static_dim(self) -> int:
        return 8


@dataclass
class P12Cache:
    """Raw aggregated arrays plus train-only statistics.

    Keeping the pre-forward-fill values is important: robustness evaluation
    must be able to drop an observation and recompute forward filling and
    ``delta_t`` rather than merely zeroing a feature after preprocessing.
    """

    observed_values: np.ndarray  # (N, 48, 37), zeros where mask is false
    mask: np.ndarray  # (N, 48, 37), bool
    static_raw: np.ndarray  # (N, 5), NaN for missing values
    record_ids: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)
    preprocessor: P12Preprocessor

    @property
    def n_samples(self) -> int:
        return int(self.labels.shape[0])


def aggregate_record(
    record: Mapping[str, object],
    *,
    variable_names: Sequence[str] = TIME_SERIES_VARIABLES,
    n_hours: int = N_HOURS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-aggregate one event-level record into hourly bins.

    Returns ``(observed_values, mask, static_raw)``.  Missing observations are
    represented by zero in ``observed_values`` and are identified by ``mask``;
    zero is therefore still a valid observed clinical value.
    """

    variable_names = tuple(variable_names)
    index = {name: idx for idx, name in enumerate(variable_names)}
    buckets: List[List[List[float]]] = [
        [[] for _ in variable_names] for _ in range(n_hours)
    ]
    events = record.get("events", [])
    for variable, hour, value in events:  # type: ignore[misc]
        if variable not in index:
            continue
        hour_float = float(hour)
        bucket = int(np.floor(hour_float))
        if 0 <= bucket < n_hours and float(value) >= 0:
            buckets[bucket][index[variable]].append(float(value))

    observed = np.zeros((n_hours, len(variable_names)), dtype=np.float32)
    mask = np.zeros((n_hours, len(variable_names)), dtype=bool)
    for bucket in range(n_hours):
        for variable_idx, values in enumerate(buckets[bucket]):
            if values:
                observed[bucket, variable_idx] = np.float32(np.mean(values))
                mask[bucket, variable_idx] = True

    static_mapping = record.get("static", {})
    static = np.full(len(STATIC_FEATURES), np.nan, dtype=np.float32)
    for idx, name in enumerate(STATIC_FEATURES):
        value = static_mapping.get(name) if isinstance(static_mapping, Mapping) else None
        if value is not None and float(value) >= 0:
            static[idx] = np.float32(value)
    return observed, mask, static


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """Compute hours since the latest observation for each variable.

    At an observed bin, delta is zero.  If no observation has occurred yet,
    delta is ``b + 1`` as specified by the Phase 0 document.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim not in (2, 3):
        raise ValueError("mask must have shape (T,D) or (N,T,D)")
    batched = mask.ndim == 3
    if not batched:
        mask = mask[None, ...]
    n_samples, n_hours, n_variables = mask.shape
    delta = np.empty(mask.shape, dtype=np.float32)
    for sample in range(n_samples):
        last = np.full(n_variables, -1, dtype=np.int64)
        for hour in range(n_hours):
            observed = mask[sample, hour]
            delta[sample, hour] = np.where(
                observed,
                0.0,
                np.where(last >= 0, hour - last, hour + 1),
            )
            last[observed] = hour
    return delta[0] if not batched else delta


def _forward_fill(
    observed_values: np.ndarray,
    mask: np.ndarray,
    fill_values: np.ndarray,
) -> np.ndarray:
    observed_values = np.asarray(observed_values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    fill_values = np.asarray(fill_values, dtype=np.float32)
    if observed_values.shape != mask.shape:
        raise ValueError("observed_values and mask must have the same shape")
    batched = observed_values.ndim == 3
    if not batched:
        observed_values = observed_values[None, ...]
        mask = mask[None, ...]
    n_samples, n_hours, n_variables = observed_values.shape
    if fill_values.shape != (n_variables,):
        raise ValueError("fill_values must have shape (n_variables,)")
    filled = np.empty_like(observed_values, dtype=np.float32)
    for sample in range(n_samples):
        previous = fill_values.copy()
        for hour in range(n_hours):
            previous = np.where(mask[sample, hour], observed_values[sample, hour], previous)
            filled[sample, hour] = previous
    return filled[0] if not batched else filled


def build_hourly_features(
    observed_values: np.ndarray,
    mask: np.ndarray,
    value_mean: np.ndarray,
    value_std: np.ndarray,
) -> np.ndarray:
    """Apply ffill, standardization, mask and log-delta encoding.

    The output ordering is exactly ``[z, mask, log1p(delta)]`` along the last
    dimension, yielding 111 features for the official 37-variable P12 data.
    """

    observed_values = np.asarray(observed_values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    value_mean = np.asarray(value_mean, dtype=np.float32)
    value_std = np.asarray(value_std, dtype=np.float32)
    n_variables = observed_values.shape[-1]
    if value_mean.shape != (n_variables,) or value_std.shape != (n_variables,):
        raise ValueError("value statistics have the wrong shape")
    filled = _forward_fill(observed_values, mask, value_mean)
    z = (filled - value_mean) / (value_std + EPS)
    delta = compute_delta(mask)
    gap = np.log1p(delta).astype(np.float32)
    result = np.concatenate([z, mask.astype(np.float32), gap], axis=-1)
    return result.astype(np.float32)


def _safe_mean_std(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > EPS), stds, 1.0).astype(np.float32)
    return means, stds


def fit_preprocessor(
    observed_values: np.ndarray,
    mask: np.ndarray,
    static_raw: np.ndarray,
    variable_names: Sequence[str] = TIME_SERIES_VARIABLES,
) -> P12Preprocessor:
    """Fit all fill/normalization statistics from training records only."""

    observed_values = np.asarray(observed_values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    static_raw = np.asarray(static_raw, dtype=np.float32)
    if observed_values.ndim != 3 or mask.shape != observed_values.shape:
        raise ValueError("training values/mask must have shape (N,T,D)")
    if static_raw.shape != (observed_values.shape[0], len(STATIC_FEATURES)):
        raise ValueError("static_raw has the wrong shape")

    observed_for_stats = np.where(mask, observed_values, np.nan)
    value_mean, value_std = _safe_mean_std(observed_for_stats.reshape(-1, observed_values.shape[-1]))
    static_mean, static_std = _safe_mean_std(static_raw)
    # Gender is already a binary descriptor; retain its original [0,1] scale.
    static_std[1] = 1.0
    known_icu = static_raw[:, 3][np.isfinite(static_raw[:, 3])]
    icu_mode = int(np.bincount(known_icu.astype(np.int64)).argmax()) if known_icu.size else 1
    return P12Preprocessor(
        variable_names=tuple(variable_names),
        value_mean=value_mean,
        value_std=value_std,
        static_mean=static_mean.astype(np.float32),
        static_std=static_std.astype(np.float32),
        icu_mode=icu_mode,
    )


def build_static_features(static_raw: np.ndarray, preprocessor: P12Preprocessor) -> np.ndarray:
    """Fill and encode the five static descriptors into eight features."""

    static_raw = np.asarray(static_raw, dtype=np.float32)
    if static_raw.ndim == 1:
        static_raw = static_raw[None, :]
        unbatch = True
    else:
        unbatch = False
    if static_raw.shape[1] != len(STATIC_FEATURES):
        raise ValueError("static_raw must have five columns")
    filled = np.where(np.isfinite(static_raw), static_raw, preprocessor.static_mean)
    # ICUType is categorical; missing values use the training-set mode.
    icu = static_raw[:, 3].copy()
    icu[~np.isfinite(icu)] = preprocessor.icu_mode
    result = np.empty((filled.shape[0], 8), dtype=np.float32)
    result[:, 0] = (filled[:, 0] - preprocessor.static_mean[0]) / (
        preprocessor.static_std[0] + EPS
    )
    result[:, 1] = filled[:, 1]
    result[:, 2] = (filled[:, 2] - preprocessor.static_mean[2]) / (
        preprocessor.static_std[2] + EPS
    )
    result[:, 3] = (filled[:, 4] - preprocessor.static_mean[4]) / (
        preprocessor.static_std[4] + EPS
    )
    for icu_type in range(1, 5):
        result[:, 3 + icu_type] = (icu == icu_type).astype(np.float32)
    return result[0] if unbatch else result


def stratified_split_indices(
    labels: Sequence[int],
    *,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict:
    """Create one fixed, deterministic stratified 70/15/15 split."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("labels must be a one-dimensional array with at least 3 items")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("split fractions must be in (0,1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below 1")
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        n_train = int(round(len(class_indices) * train_fraction))
        n_val = int(round(len(class_indices) * val_fraction))
        if n_train < 1 or n_val < 1 or len(class_indices) - n_train - n_val < 1:
            raise ValueError(f"class {label} is too small for requested split")
        split["train"].extend(class_indices[:n_train].tolist())
        split["val"].extend(class_indices[n_train : n_train + n_val].tolist())
        split["test"].extend(class_indices[n_train + n_val :].tolist())
    for key in split:
        split[key] = sorted(split[key])
    return split


def save_preprocessed(
    path: str | Path,
    cache: P12Cache,
    split_indices: dict,
    split_path: str | Path | None = None,
) -> None:
    """Write the cache and split metadata atomically enough for local use."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "variable_names": list(cache.preprocessor.variable_names),
        "static_features": list(STATIC_FEATURES),
        "n_hours": int(cache.observed_values.shape[1]),
        "input_dim": cache.preprocessor.input_dim,
        "static_dim": cache.preprocessor.static_dim,
    }
    np.savez_compressed(
        path,
        observed_values=cache.observed_values.astype(np.float32),
        mask=cache.mask.astype(np.uint8),
        static_raw=cache.static_raw.astype(np.float32),
        record_ids=cache.record_ids.astype(np.int64),
        labels=cache.labels.astype(np.int64),
        value_mean=cache.preprocessor.value_mean,
        value_std=cache.preprocessor.value_std,
        static_mean=cache.preprocessor.static_mean,
        static_std=cache.preprocessor.static_std,
        icu_mode=np.asarray(cache.preprocessor.icu_mode, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata), dtype="U"),
    )
    if split_path is not None:
        split_path = Path(split_path)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_payload = {
            "split_seed": split_indices.get("split_seed", 42),
            "record_ids": cache.record_ids.tolist(),
            "indices": {
                key: list(map(int, value))
                for key, value in split_indices.items()
                if key in {"train", "val", "test"}
            },
        }
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(split_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def load_preprocessed(path: str | Path) -> P12Cache:
    """Load a cache produced by :func:`save_preprocessed`."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        preprocessor = P12Preprocessor(
            variable_names=tuple(metadata["variable_names"]),
            value_mean=data["value_mean"].astype(np.float32),
            value_std=data["value_std"].astype(np.float32),
            static_mean=data["static_mean"].astype(np.float32),
            static_std=data["static_std"].astype(np.float32),
            icu_mode=int(data["icu_mode"].item()),
        )
        return P12Cache(
            observed_values=data["observed_values"].astype(np.float32),
            mask=data["mask"].astype(bool),
            static_raw=data["static_raw"].astype(np.float32),
            record_ids=data["record_ids"].astype(np.int64),
            labels=data["labels"].astype(np.int64),
            preprocessor=preprocessor,
        )


def load_split_indices(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "indices" in payload:
        return {key: list(map(int, value)) for key, value in payload["indices"].items()}
    return {key: list(map(int, value)) for key, value in payload.items() if key in {"train", "val", "test"}}


def cache_features(cache: P12Cache, indices: Sequence[int], mask_override: np.ndarray | None = None):
    """Return torch-independent numpy arrays for a selected split."""

    indices = np.asarray(indices, dtype=np.int64)
    observed = cache.observed_values[indices]
    mask = cache.mask[indices] if mask_override is None else np.asarray(mask_override, dtype=bool)
    if mask.shape != observed.shape:
        raise ValueError("mask_override has the wrong shape")
    x = build_hourly_features(
        observed,
        mask,
        cache.preprocessor.value_mean,
        cache.preprocessor.value_std,
    )
    static = build_static_features(cache.static_raw[indices], cache.preprocessor)
    y = cache.labels[indices].astype(np.float32)
    return x, static, y


def build_cache_from_raw(
    raw_dir: str | Path,
    *,
    split_seed: int = 42,
    split_path: str | Path | None = None,
) -> Tuple[P12Cache, dict]:
    """Parse official Set A, fit train-only statistics, and build a cache."""

    paths = iter_record_paths(raw_dir, "set-a")
    records = parse_records(paths)
    outcomes = load_outcomes(Path(raw_dir) / "Outcomes-a.txt")
    if len(records) != len(outcomes):
        raise ValueError("record/outcome count mismatch")
    unknown = sorted({parameter for record in records for parameter in record["unknown_parameters"]})
    if unknown:
        raise ValueError(f"unexpected P12 parameters found: {unknown}")

    record_ids = np.asarray([record["record_id"] for record in records], dtype=np.int64)
    labels = np.asarray([outcomes[int(record_id)] for record_id in record_ids], dtype=np.int64)
    aggregate = [aggregate_record(record) for record in records]
    observed = np.stack([item[0] for item in aggregate])
    mask = np.stack([item[1] for item in aggregate])
    static_raw = np.stack([item[2] for item in aggregate])
    split = stratified_split_indices(labels, seed=split_seed)
    split["split_seed"] = split_seed
    preprocessor = fit_preprocessor(observed[split["train"]], mask[split["train"]], static_raw[split["train"]])
    cache = P12Cache(observed, mask, static_raw, record_ids, labels, preprocessor)
    if split_path is not None:
        # save_preprocessed writes this in the same structure used by the CLI;
        # the caller can still save the NPZ separately.
        pass
    return cache, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed/p12_hourly.npz")
    parser.add_argument("--split-out", default="data/processed/split_indices.json")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    cache, split = build_cache_from_raw(args.raw, split_seed=args.split_seed)
    save_preprocessed(args.out, cache, split, args.split_out)
    report = {
        "path": str(args.out),
        "n_samples": cache.n_samples,
        "record_id_min": int(cache.record_ids.min()),
        "record_id_max": int(cache.record_ids.max()),
        "deaths": int(cache.labels.sum()),
        "split_sizes": {key: len(value) for key, value in split.items() if key in {"train", "val", "test"}},
        "input_dim": cache.preprocessor.input_dim,
        "static_dim": cache.preprocessor.static_dim,
        "statistics_fit_on": "train observed bins only",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
