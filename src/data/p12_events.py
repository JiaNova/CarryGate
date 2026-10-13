"""Leakage-safe event-level data preparation for Phase 1 forecasting.

The module deliberately keeps the representation small: a record is converted
to sorted NumPy arrays, the train split fits one robust per-variable
standardizer, and a dataset item contains only observation-window events plus
real prediction-window targets.  No interpolation or forward filling is used.
"""

from __future__ import annotations

import json
import pickle
import warnings
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .p12_parse import (
    TIME_SERIES_VARIABLES,
    annotate_events,
    iter_record_paths,
    load_outcomes,
    parse_records,
)


EPS = 1e-6
DEFAULT_T_OBS = 24.0
N_HOURS = 48.0


@dataclass
class EventStandardizer:
    """Training-only robust statistics and clipping diagnostics."""

    variable_names: Tuple[str, ...]
    median: np.ndarray
    iqr: np.ndarray
    lower_clip: np.ndarray
    upper_clip: np.ndarray
    train_mean: np.ndarray
    clipping_ratio: np.ndarray

    @classmethod
    def fit(
        cls,
        records: Sequence[Mapping[str, object]],
        variable_names: Sequence[str] = TIME_SERIES_VARIABLES,
    ) -> "EventStandardizer":
        names = tuple(variable_names)
        index = {name: i for i, name in enumerate(names)}
        values: List[List[float]] = [[] for _ in names]
        for record in records:
            for event in event_metadata(record):
                variable = str(event["variable"])
                if variable in index:
                    values[index[variable]].append(float(event["value"]))

        median = np.zeros(len(names), dtype=np.float32)
        iqr = np.ones(len(names), dtype=np.float32)
        lower = np.full(len(names), -1.0, dtype=np.float32)
        upper = np.full(len(names), 1.0, dtype=np.float32)
        mean = np.zeros(len(names), dtype=np.float32)
        ratio = np.zeros(len(names), dtype=np.float32)
        for variable_idx, variable_values in enumerate(values):
            if not variable_values:
                continue
            raw = np.asarray(variable_values, dtype=np.float64)
            med = float(np.median(raw))
            q25, q75 = np.quantile(raw, [0.25, 0.75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale < EPS:
                scale = 1.0
            standardized = (raw - med) / (scale + EPS)
            low, high = np.quantile(standardized, [0.001, 0.999])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.min(standardized)), float(np.max(standardized))
                if high <= low:
                    low, high = low - 1.0, high + 1.0
            clipped = np.clip(standardized, low, high)
            median[variable_idx] = np.float32(med)
            iqr[variable_idx] = np.float32(scale)
            lower[variable_idx] = np.float32(low)
            upper[variable_idx] = np.float32(high)
            mean[variable_idx] = np.float32(np.mean(clipped))
            ratio[variable_idx] = np.float32(
                np.mean((standardized < low) | (standardized > high))
            )

        problematic = [
            f"{names[i]}={ratio[i]:.2%}"
            for i in range(len(names))
            if ratio[i] > 0.01
        ]
        if problematic:
            warnings.warn(
                "Phase 1 training-value clipping exceeds 1% for: "
                + ", ".join(problematic),
                RuntimeWarning,
                stacklevel=2,
            )
        return cls(names, median, iqr, lower, upper, mean, ratio)

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    def transform(self, variable_id: int | str, value: float) -> float:
        index = self._index(variable_id)
        standardized = (float(value) - float(self.median[index])) / (
            float(self.iqr[index]) + EPS
        )
        return float(np.clip(standardized, self.lower_clip[index], self.upper_clip[index]))

    def transform_arrays(
        self,
        variable_ids: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        variable_ids = np.asarray(variable_ids, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if variable_ids.shape != values.shape:
            raise ValueError("variable_ids and values must have the same shape")
        result = (values - self.median[variable_ids]) / (self.iqr[variable_ids] + EPS)
        return np.clip(
            result,
            self.lower_clip[variable_ids],
            self.upper_clip[variable_ids],
        ).astype(np.float32)

    def _index(self, variable_id: int | str) -> int:
        if isinstance(variable_id, str):
            try:
                return self.variable_names.index(variable_id)
            except ValueError as exc:
                raise KeyError(variable_id) from exc
        index = int(variable_id)
        if index < 0 or index >= self.n_variables:
            raise IndexError(index)
        return index

    def as_dict(self) -> Dict[str, object]:
        return {
            "variable_names": list(self.variable_names),
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "lower_clip": self.lower_clip.tolist(),
            "upper_clip": self.upper_clip.tolist(),
            "train_mean": self.train_mean.tolist(),
            "clipping_ratio": self.clipping_ratio.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EventStandardizer":
        return cls(
            tuple(str(item) for item in payload["variable_names"]),
            np.asarray(payload["median"], dtype=np.float32),
            np.asarray(payload["iqr"], dtype=np.float32),
            np.asarray(payload["lower_clip"], dtype=np.float32),
            np.asarray(payload["upper_clip"], dtype=np.float32),
            np.asarray(payload["train_mean"], dtype=np.float32),
            np.asarray(payload["clipping_ratio"], dtype=np.float32),
        )


def event_metadata(record: Mapping[str, object]) -> List[Dict[str, object]]:
    """Return chronologically ordered events with gap and first flags."""

    supplied = record.get("events_with_delta")
    if supplied is not None:
        events = [dict(item) for item in supplied]  # type: ignore[arg-type]
        required = {"variable", "time", "value", "delta_t", "is_first_observation"}
        if all(required.issubset(item) for item in events):
            return sorted(events, key=lambda item: float(item["time"]))
    raw_events = record.get("events", [])
    return annotate_events(raw_events)  # type: ignore[arg-type]


def split_events(
    record: Mapping[str, object],
    t_obs: float = DEFAULT_T_OBS,
    horizon: float = N_HOURS,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split events into ``t < T_obs`` observations and real future targets."""

    if not 0.0 < float(t_obs) < float(horizon):
        raise ValueError("t_obs must lie strictly inside the 48-hour horizon")
    events = event_metadata(record)
    observation = [event for event in events if float(event["time"]) < t_obs]
    target = [
        event
        for event in events
        if t_obs <= float(event["time"]) < float(horizon)
    ]
    # This assertion is intentionally local to the data boundary: the model
    # can only receive the first list, so future events cannot leak into it.
    if any(float(event["time"]) >= t_obs for event in observation):
        raise AssertionError("future event entered the observation-window encoder")
    return observation, target


def make_event_example(
    record: Mapping[str, object],
    standardizer: EventStandardizer,
    *,
    t_obs: float = DEFAULT_T_OBS,
    horizon: float = N_HOURS,
    variable_names: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Convert one parsed record to a normalized forecasting example."""

    names = tuple(variable_names or standardizer.variable_names)
    index = {name: i for i, name in enumerate(names)}
    observation, target = split_events(record, t_obs=t_obs, horizon=horizon)

    def arrays(events: Sequence[Mapping[str, object]]) -> Dict[str, np.ndarray]:
        selected = [event for event in events if str(event["variable"]) in index]
        variable_ids = np.asarray(
            [index[str(event["variable"])] for event in selected], dtype=np.int64
        )
        times = np.asarray([float(event["time"]) for event in selected], dtype=np.float32)
        values_raw = np.asarray(
            [float(event["value"]) for event in selected], dtype=np.float32
        )
        delta = np.asarray(
            [float(event["delta_t"]) for event in selected], dtype=np.float32
        )
        first = np.asarray(
            [bool(event["is_first_observation"]) for event in selected], dtype=bool
        )
        values = standardizer.transform_arrays(variable_ids, values_raw)
        return {
            "var": variable_ids,
            "time": times,
            "value": values,
            "value_raw": values_raw,
            "delta_t": delta,
            "is_first": first,
        }

    obs = arrays(observation)
    tgt = arrays(target)
    return {
        "record_id": int(record["record_id"]),
        "sample_id": str(record.get("sample_id", record["record_id"])),
        "label": int(record.get("label", 0)),
        "set_name": str(record.get("set_name", "a")),
        # Keep the domain label beside the event arrays so Line K can perform
        # leave-one-ICU-type-out splits without reparsing raw records.
        "icu_type": int(
            float(record.get("static", {}).get("ICUType", -1))
            if isinstance(record.get("static", {}), Mapping)
            and record.get("static", {}).get("ICUType") is not None
            else -1
        ),
        "t_obs": float(t_obs),
        "obs": obs,
        "target": tgt,
    }


def stratified_event_split(
    labels: Sequence[int],
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> Dict[str, List[int]]:
    """Create a deterministic stratified 70/15/15 split over all sets."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 3:
        raise ValueError("labels must contain at least three records")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("split fractions must be positive")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be below one")
    rng = np.random.default_rng(int(seed))
    buckets = {"train": [], "val": [], "test": []}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_train = int(round(len(indices) * train_fraction))
        n_val = int(round(len(indices) * val_fraction))
        n_train = min(max(n_train, 1), len(indices) - 2)
        n_val = min(max(n_val, 1), len(indices) - n_train - 1)
        buckets["train"].extend(indices[:n_train].tolist())
        buckets["val"].extend(indices[n_train : n_train + n_val].tolist())
        buckets["test"].extend(indices[n_train + n_val :].tolist())
    for values in buckets.values():
        values.sort()
    if sum(len(values) for values in buckets.values()) != len(labels):
        raise AssertionError("stratified split lost records")
    return buckets


def load_all_p12_records(
    raw_dir: str | Path,
    sets: Sequence[str] = ("a", "b", "c"),
) -> List[dict]:
    """Parse and label the requested official sets with unique sample IDs."""

    raw_dir = Path(raw_dir)
    records: List[dict] = []
    for set_name in sets:
        set_name = str(set_name).lower().replace("set-", "")
        paths = iter_record_paths(raw_dir, f"set-{set_name}")
        parsed = parse_records(paths)
        outcomes = load_outcomes(raw_dir / f"Outcomes-{set_name}.txt")
        parsed_ids = {int(record["record_id"]) for record in parsed}
        if parsed_ids != set(outcomes):
            raise ValueError(f"record/outcome mismatch for set-{set_name}")
        for record in parsed:
            record["set_name"] = set_name
            record["sample_id"] = f"{set_name}-{int(record['record_id']):04d}"
            record["label"] = int(outcomes[int(record["record_id"])])
            records.append(record)
    return records


def prepare_p12_forecast_data(
    raw_dir: str | Path,
    processed_dir: str | Path = "data/processed",
    *,
    sets: Sequence[str] = ("a", "b", "c"),
    t_obs: float = DEFAULT_T_OBS,
    split_seed: int = 42,
    cache_name: str = "p12_events.pkl",
    split_name: str = "split_indices_12k.json",
) -> Dict[str, object]:
    """Parse all selected sets, fit train-only stats, and write a cache."""

    records = load_all_p12_records(raw_dir, sets=sets)
    labels = [int(record["label"]) for record in records]
    split_indices = stratified_event_split(labels, seed=split_seed)
    train_records = [records[i] for i in split_indices["train"]]
    standardizer = EventStandardizer.fit(train_records)
    examples = [make_event_example(record, standardizer, t_obs=t_obs) for record in records]

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / cache_name
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "examples": examples,
                # Store plain arrays rather than the class object.  This keeps
                # caches made via ``python -m src.data.p12_events`` loadable;
                # otherwise pickle would record the temporary ``__main__``
                # module name.
                "standardizer": standardizer.as_dict(),
                "variable_names": list(standardizer.variable_names),
                "sets": list(sets),
                "t_obs": float(t_obs),
                "split_seed": int(split_seed),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    split_payload = {
        "split_seed": int(split_seed),
        "t_obs": float(t_obs),
        "sets": list(sets),
        "n_samples": len(examples),
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
        "sample_ids": [str(example["sample_id"]) for example in examples],
        "label_counts": {
            name: int(sum(labels[index] for index in indices))
            for name, indices in split_indices.items()
        },
    }
    split_path = processed_dir / split_name
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {
        "cache_path": str(cache_path),
        "split_path": str(split_path),
        "n_samples": len(examples),
        "split_sizes": {name: len(value) for name, value in split_indices.items()},
        "clipping_ratio": standardizer.clipping_ratio.tolist(),
    }


def load_p12_forecast_cache(path: str | Path) -> Dict[str, object]:
    """Load a cache created by :func:`prepare_p12_forecast_data`."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload.get("standardizer"), Mapping):
        payload["standardizer"] = EventStandardizer.from_dict(payload["standardizer"])
    return payload


def load_forecast_split(
    cache_path: str | Path,
    split_path: str | Path,
):
    """Return the cache and validated split indices."""

    cache = load_p12_forecast_cache(cache_path)
    with Path(split_path).open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    n_samples = len(cache["examples"])
    if int(split.get("n_samples", n_samples)) != n_samples:
        raise ValueError("forecast cache and split file contain different sample counts")
    all_indices = [int(i) for name in ("train", "val", "test") for i in split[name]]
    if sorted(all_indices) != list(range(n_samples)) or len(set(all_indices)) != n_samples:
        raise ValueError("forecast split is not a partition of the cache")
    return cache, {name: [int(i) for i in split[name]] for name in ("train", "val", "test")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--sets", default="a,b,c", help="comma-separated official sets")
    parser.add_argument("--t-obs", type=float, default=DEFAULT_T_OBS)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_p12_forecast_data(
        args.raw,
        args.processed_dir,
        sets=tuple(item.strip() for item in args.sets.split(",") if item.strip()),
        t_obs=args.t_obs,
        split_seed=args.split_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
