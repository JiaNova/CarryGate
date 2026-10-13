"""Gate D1b: the ICU-KAN adapter must not change the task.

Compares the NAIVE-1 predictions produced inside the PyOmniTS harness against
``src.data.persistence_baseline``, which is the registered carry-forward used
by every ICU-KAN result to date. Three assertions, per
ICU_KAN_D1_external_baseline_spec.md section 5.2:

1. every target event's prediction equals the registered ``x_last`` bitwise;
2. persistence skill computed from those predictions is exactly zero;
3. event, variable and split counts match the registered cache.

Usage:
    python scripts/d1_harness/check_adapter_fidelity.py <eval_dir> <cohort>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.persistence_baseline import build_persistence_baseline  # noqa: E402

L_TOTAL_HOURS = 48.0
N_VARIABLES = 37
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
    },
}


def _cache_spec(cohort: str, data_variant: str) -> tuple[str, str]:
    variant = str(data_variant or "registered").lower()
    if variant not in VARIANT_CACHES:
        raise ValueError(f"unknown data variant: {variant}")
    if variant == "jbi_patient" and cohort != "eicu":
        raise ValueError("jbi_patient adapter checks are defined only for eICU")
    return VARIANT_CACHES[variant][cohort]


def main(eval_dir: Path, cohort: str, data_variant: str = "registered") -> None:
    variant = str(data_variant or "registered").lower()
    events_name, split_name = _cache_spec(cohort, variant)
    cache_path = (ROOT / "data" / "processed" / events_name).resolve()
    split_path = (ROOT / "data" / "processed" / split_name).resolve()
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    examples = payload["examples"]
    with open(split_path) as handle:
        splits = json.load(handle)

    pred = np.load(eval_dir / "output_pred.npy")
    y = np.load(eval_dir / "input_y.npy")
    y_mask = np.load(eval_dir / "input_y_mask.npy")
    sample_id = np.load(eval_dir / "input_sample_ID.npy").astype(np.int64)

    if sample_id.tolist() != list(splits["test"]):
        raise AssertionError("harness did not iterate the registered test split in order")

    median = torch.zeros(N_VARIABLES, dtype=torch.float32)
    n_events = 0
    max_abs_diff = 0.0
    mismatched = 0
    num = den = 0.0

    for row, index in enumerate(sample_id.tolist()):
        example = examples[index]
        obs, tgt = example["obs"], example["target"]
        x_last, _ = build_persistence_baseline(
            torch.as_tensor(np.asarray(obs["var"]), dtype=torch.long),
            torch.as_tensor(np.asarray(obs["time"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(obs["value"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(tgt["var"]), dtype=torch.long),
            torch.as_tensor(np.asarray(tgt["time"]), dtype=torch.float32),
            float(example["t_obs"]),
            median,
        )
        reference = x_last.numpy()
        truth = np.asarray(tgt["value"], dtype=np.float32)

        # Locate each target event in the dense grid: same lexsort and same
        # duplicate-aware row assignment the adapter uses.
        var = np.asarray(tgt["var"])
        time = np.asarray(tgt["time"])
        order = np.lexsort((var, time))
        _, inverse, counts = np.unique(
            np.stack([time[order], var[order]]), axis=1,
            return_inverse=True, return_counts=True,
        )
        inverse = inverse.ravel()
        rank = np.arange(inverse.size) - np.searchsorted(inverse, np.arange(counts.size))[inverse]
        uniq_time, time_index = np.unique(time[order], return_inverse=True)
        rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
        np.maximum.at(rows_per_time, time_index, rank + 1)
        row_of = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])[time_index] + rank

        got = pred[row, row_of, var[order]]
        want = reference[order]
        diff = np.abs(got - want)
        max_abs_diff = max(max_abs_diff, float(diff.max(initial=0.0)))
        mismatched += int((got != want).sum())
        n_events += var.size

        # Skill denominator and numerator are both the carry-forward error here.
        num += float(np.abs(got - truth[order]).sum())
        den += float(np.abs(want - truth[order]).sum())

    valid_cells = int(y_mask.sum())
    report = {
        "cohort": cohort,
        "data_source": f"ICU-KAN/{cohort}",
        "data_variant": variant,
        "cache_path": str(cache_path),
        "split_path": str(split_path),
        "split_unit": str(splits.get("split_unit", "registered_stay_level")),
        "group_column": splits.get("group_column"),
        "eval_dir": str(eval_dir),
        "test_stays": len(sample_id),
        "target_events": n_events,
        "dense_valid_cells": valid_cells,
        "events_equal_cells": n_events == valid_cells,
        "mismatched_predictions": mismatched,
        "max_abs_deviation": max_abs_diff,
        "persistence_skill": 1.0 - num / den,
        "n_variables": len(payload["variable_names"]),
        "split_sizes": {k: len(splits[k]) for k in ("train", "val", "test")},
    }
    print(json.dumps(report, indent=2))

    assert mismatched == 0, f"{mismatched} predictions differ from the registered baseline"
    assert max_abs_diff == 0.0, "predictions are not bitwise identical"
    assert n_events == valid_cells, "dense grid does not carry exactly the target events"
    assert report["persistence_skill"] == 0.0, "carry-forward does not score exactly zero skill"
    print("D1b PASS")

    out = ROOT / "results" / "d1_external" / "d1b_adapter_fidelity.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[cohort] = report
    out.write_text(json.dumps(existing, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("cohort", choices=tuple(CACHES))
    parser.add_argument("--data-variant", choices=tuple(VARIANT_CACHES), default="registered")
    args = parser.parse_args()
    main(args.eval_dir, args.cohort, args.data_variant)
