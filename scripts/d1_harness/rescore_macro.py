#!/usr/bin/env python
"""Re-score the external runs with the registered macro-averaged skill.

Why this exists. ``score_run.py`` computed persistence skill as one pooled ratio
over all scored events. The registered ICU-KAN metric is not that: it is the
*unweighted mean over variables* of each variable's own skill ratio, with
variables excluded when their persistence denominator is degenerate or they
carry fewer than 50 test events (``src/utils/skill_metrics.persistence_skill``).
A pooled ratio and a mean of ratios are different estimators, so the original
head-to-head table compared external models to this project under two different
metrics.

Gate D1b did not catch it: under either estimator carry-forward scores exactly
zero, so the gate was insensitive to the very difference it appeared to license.

This script re-reads the saved predictions and scores them with the project's
own metric function, so there is no second implementation of the metric.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.persistence_baseline import build_persistence_baseline  # noqa: E402
from src.utils.skill_metrics import persistence_skill  # noqa: E402

N_VARIABLES = 37
UNSCORED = frozenset({6, 19, 32})
CACHES = {
    "p12": ("p12_events.pkl", "split_indices_12k.json"),
    "mimiciv": ("mimiciv_events.pkl", "mimiciv_split_indices_12k.json"),
    "eicu": ("eicu_events.pkl", "eicu_split_indices_12k.json"),
}
VARIANT_CACHES = {
    "registered": CACHES,
    "jbi_patient": {
        **CACHES,
        "eicu": ("eicu_events_jbi_v1.pkl", "eicu_split_indices_jbi_v1.json"),
        "mimiciv": (
            "mimiciv_events_jbi_v1.pkl",
            "mimiciv_split_indices_jbi_v1.json",
        ),
    },
}
_CACHE: dict[str, dict] = {}


def cohort_events(cohort: str, data_variant: str = "registered") -> dict:
    """Per-event variable index, truth and carry-forward value for the test split.

    Events are enumerated in the same dense (row, variable) order the adapter
    writes, so they line up position-for-position with the saved prediction
    tensors.
    """
    variant = str(data_variant or "registered").lower()
    if variant not in VARIANT_CACHES:
        raise ValueError(f"unknown data variant: {variant}")
    cache_key = f"{cohort}:{variant}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    events_name, split_name = VARIANT_CACHES[variant][cohort]
    with open(ROOT / "data" / "processed" / events_name, "rb") as handle:
        examples = pickle.load(handle)["examples"]
    with open(ROOT / "data" / "processed" / split_name) as handle:
        split_payload = json.load(handle)
    test_index = split_payload["test"]

    median = torch.zeros(N_VARIABLES, dtype=torch.float32)
    var_all, truth_all, last_all = [], [], []
    for index in test_index:
        example = examples[index]
        obs, tgt = example["obs"], example["target"]
        var = np.asarray(tgt["var"])
        time = np.asarray(tgt["time"])
        value = np.asarray(tgt["value"])
        x_last, _ = build_persistence_baseline(
            torch.as_tensor(np.asarray(obs["var"]), dtype=torch.long),
            torch.as_tensor(np.asarray(obs["time"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(obs["value"]), dtype=torch.float32),
            torch.as_tensor(var, dtype=torch.long),
            torch.as_tensor(time, dtype=torch.float32),
            float(example["t_obs"]),
            median,
        )
        # Reproduce the adapter's dense layout exactly. Repeated (time, variable)
        # events are placed on additional rows carrying the same timestamp, so a
        # plain (time, variable) sort is NOT the traversal order of the dense
        # grid: C-order iteration visits (row, variable).
        order = np.lexsort((var, time))
        var_s, time_s, value_s = var[order], time[order], x_last.numpy()[order]
        truth_s = value[order]
        _, inverse, counts = np.unique(np.stack([time_s, var_s]), axis=1,
                                       return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        group_start = np.searchsorted(inverse, np.arange(counts.size))
        rank = np.arange(inverse.size) - group_start[inverse]
        uniq_time, time_index = np.unique(time_s, return_inverse=True)
        rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
        np.maximum.at(rows_per_time, time_index, rank + 1)
        row_start = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])
        row_of = row_start[time_index] + rank
        dense = np.lexsort((var_s, row_of))     # C-order traversal of (row, var)
        var_all.append(var_s[dense])
        truth_all.append(truth_s[dense])
        last_all.append(value_s[dense])
    cached = {
        "test_index": list(test_index),
        "var": np.concatenate(var_all),
        "truth": np.concatenate(truth_all),
        "x_last": np.concatenate(last_all),
        "data_variant": variant,
        "data_source": f"ICU-KAN/{cohort}",
        "cache_path": str((ROOT / "data" / "processed" / events_name).resolve()),
        "split_path": str((ROOT / "data" / "processed" / split_name).resolve()),
        "split_unit": str(split_payload.get("split_unit", "registered_unverifiable")),
        "group_column": split_payload.get("group_column"),
    }
    _CACHE[cache_key] = cached
    return cached


def score(eval_dir: Path, cohort: str, data_variant: str = "registered") -> dict:
    ref = cohort_events(cohort, data_variant=data_variant)
    pred = np.load(eval_dir / "output_pred.npy")
    y = np.load(eval_dir / "input_y.npy")
    y_mask = np.load(eval_dir / "input_y_mask.npy").astype(bool)
    sample_id = np.load(eval_dir / "input_sample_ID.npy").astype(np.int64)
    if sample_id.tolist() != ref["test_index"]:
        raise AssertionError("run did not iterate the registered test split in order")

    # Dense (sample, row, variable) traversal in C order reproduces the
    # (time, variable) lexsort per sample, which is how the reference list above
    # was built.
    flat_truth = y[y_mask]
    flat_pred = pred[y_mask]
    if flat_truth.size != ref["truth"].size:
        raise AssertionError(
            f"dense grid holds {flat_truth.size} events, cache holds {ref['truth'].size}")
    if not np.allclose(flat_truth, ref["truth"], atol=1e-5, rtol=0):
        raise AssertionError("dense targets do not match the registered target values")

    scored = ~np.isin(ref["var"], sorted(UNSCORED))
    macro, frame, excluded = persistence_skill(
        flat_pred[scored], ref["truth"][scored], ref["var"][scored], ref["x_last"][scored])
    abs_err = float(np.abs(flat_pred[scored] - ref["truth"][scored]).sum())
    base_err = float(np.abs(ref["x_last"][scored] - ref["truth"][scored]).sum())
    return {
        "cohort": cohort,
        "n_scored_events": int(scored.sum()),
        "persistence_skill_macro": float(macro),
        "persistence_skill_pooled": float(1.0 - abs_err / base_err),
        "n_variables_scored": int(len(frame)),
        "excluded_variables": excluded,
        "MAE": abs_err / int(scored.sum()),
        "data_variant": ref["data_variant"],
        "data_source": ref["data_source"],
        "cache_path": ref["cache_path"],
        "split_path": ref["split_path"],
        "split_unit": ref["split_unit"],
        "group_column": ref["group_column"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "results" / "d1_external" / "d1c_matrix.csv")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "d1_external" / "d1c_matrix_macro.csv")
    args = parser.parse_args()

    rows = []
    frame = pd.read_csv(args.matrix)
    for r in frame.itertuples():
        data_variant = str(getattr(r, "data_variant", "registered"))
        result = score(Path(r.eval_dir), r.cohort, data_variant=data_variant)
        rows.append({"model": r.model, "cohort": r.cohort, "seed": r.seed,
                     "persistence_skill_pooled_original": r.persistence_skill,
                     **{k: v for k, v in result.items() if k != "cohort"}})
        print(f"{r.model:14}{r.cohort:9}seed={r.seed}  "
              f"macro={result['persistence_skill_macro']:+.4f}  "
              f"pooled={result['persistence_skill_pooled']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.17g")
    print(f"\nwrote {len(out)} rows to {args.out}")
    delta = (out.persistence_skill_pooled - out.persistence_skill_pooled_original).abs().max()
    print(f"max |pooled(new) - pooled(original)| = {delta:.2e}  (should be ~0)")


if __name__ == "__main__":
    main()
