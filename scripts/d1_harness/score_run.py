"""Score one harness run in both the field's units and the ICU-KAN metric.

Reads the arrays PyOmniTS saves under an ``eval_*`` directory and reports MSE,
MAE and persistence skill. The skill denominator is the registered
carry-forward error, which gate D1b showed the harness reproduces bitwise, so
skill computed here is on the same scale as every ICU-KAN result to date.

Usage:
    python scripts/d1_harness/score_run.py <eval_dir> <cohort> [--json]
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
# Excluded from scoring in every cohort because their persistence denominator is
# degenerate or near-empty in P12 (manuscript section 3.1).
UNSCORED = frozenset({6, 19, 32})


_BASELINE_CACHE: dict[tuple[str, str], dict] = {}


def cohort_baseline(cohort: str, data_variant: str = "registered") -> dict:
    """Per-cohort carry-forward error, computed once and reused.

    The denominator depends only on the cohort, so recomputing it for every
    model and seed repeats identical work 45 times per cohort.
    """
    variant = str(data_variant or "registered").lower()
    if variant not in VARIANT_CACHES:
        raise ValueError(f"unknown data variant: {variant}")
    if variant == "jbi_patient" and cohort != "eicu":
        raise ValueError("jbi_patient scoring is defined only for eICU")
    cache_key = (cohort, variant)
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]
    events_name, split_name = VARIANT_CACHES[variant][cohort]
    with open(ROOT / "data" / "processed" / events_name, "rb") as handle:
        examples = pickle.load(handle)["examples"]
    with open(ROOT / "data" / "processed" / split_name) as handle:
        split_payload = json.load(handle)
    test_index = split_payload["test"]

    median = torch.zeros(N_VARIABLES, dtype=torch.float32)
    unscored = sorted(UNSCORED)
    base_err = 0.0
    for index in test_index:
        example = examples[index]
        obs, tgt = example["obs"], example["target"]
        var = np.asarray(tgt["var"])
        x_last, _ = build_persistence_baseline(
            torch.as_tensor(np.asarray(obs["var"]), dtype=torch.long),
            torch.as_tensor(np.asarray(obs["time"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(obs["value"]), dtype=torch.float32),
            torch.as_tensor(var, dtype=torch.long),
            torch.as_tensor(np.asarray(tgt["time"]), dtype=torch.float32),
            float(example["t_obs"]),
            median,
        )
        keep = ~np.isin(var, unscored)
        base_err += float(
            np.abs(x_last.numpy()[keep] - np.asarray(tgt["value"])[keep]).sum()
        )
    cached = {
        "test_index": list(test_index),
        "base_err": base_err,
        "unscored": unscored,
        "data_variant": variant,
        "data_source": f"ICU-KAN/{cohort}",
        "cache_path": str((ROOT / "data" / "processed" / events_name).resolve()),
        "split_path": str((ROOT / "data" / "processed" / split_name).resolve()),
        "split_unit": str(split_payload.get("split_unit", "registered_unverifiable")),
        "group_column": split_payload.get("group_column"),
    }
    _BASELINE_CACHE[cache_key] = cached
    return cached


def score(eval_dir: Path, cohort: str, data_variant: str = "registered") -> dict:
    cached = cohort_baseline(cohort, data_variant=data_variant)
    pred = np.load(eval_dir / "output_pred.npy")
    y = np.load(eval_dir / "input_y.npy")
    y_mask = np.load(eval_dir / "input_y_mask.npy").astype(bool)
    sample_id = np.load(eval_dir / "input_sample_ID.npy").astype(np.int64)
    if sample_id.tolist() != cached["test_index"]:
        raise AssertionError("run did not iterate the registered test split in order")

    keep = y_mask.copy()
    keep[:, :, cached["unscored"]] = False
    truth = y[keep]
    guess = pred[keep]
    n = int(keep.sum())
    abs_err = float(np.abs(guess - truth).sum())
    return {
        "cohort": cohort,
        "n_scored_events": n,
        "MSE": float(((guess - truth) ** 2).sum()) / n,
        "MAE": abs_err / n,
        "data_variant": cached["data_variant"],
        "data_source": cached["data_source"],
        "cache_path": cached["cache_path"],
        "split_path": cached["split_path"],
        "split_unit": cached["split_unit"],
        "group_column": cached["group_column"],
        "persistence_skill": 1.0 - abs_err / cached["base_err"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("cohort", choices=sorted(CACHES))
    parser.add_argument("--data-variant", choices=sorted(VARIANT_CACHES), default="registered")
    args = parser.parse_args()
    print(json.dumps(score(args.eval_dir, args.cohort, args.data_variant), indent=2))
