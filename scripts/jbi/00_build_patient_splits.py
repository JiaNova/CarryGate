#!/usr/bin/env python
"""Build and audit patient-grouped JBI extension splits.

The original 12k eICU split is stay-level.  This entry point creates a new
split without modifying the registered file.  MIMIC-IV is deliberately
blocked when its cache does not contain a patient identifier; guessing a
mapping from an incomplete auxiliary file would be worse than leaving the
issue visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "jbi"
SEED = 42

COHORTS = {
    "p12": ("p12_events.pkl", "split_indices_12k.json"),
    "mimiciv": ("mimiciv_events.pkl", "mimiciv_split_indices_12k.json"),
    "eicu": ("eicu_events.pkl", "eicu_split_indices_12k.json"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def _stratified_group_split(
    examples: list[dict[str, Any]], group_column: str, seed: int = SEED
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, int] = {}
    for index, example in enumerate(examples):
        key = str(example[group_column])
        groups[key].append(index)
        labels[key] = max(labels.get(key, 0), int(example.get("label", 0)))

    rng = np.random.default_rng(int(seed))
    assigned: dict[str, str] = {}
    by_label: dict[int, list[str]] = defaultdict(list)
    for group, label in labels.items():
        by_label[int(label)].append(group)
    for label, group_list in sorted(by_label.items()):
        ordered = np.asarray(group_list, dtype=object)
        ordered = ordered[rng.permutation(len(ordered))]
        n = len(ordered)
        n_val = int(round(n * 0.15))
        n_test = int(round(n * 0.15))
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - n_val - 1))
        for group in ordered[:n_test]:
            assigned[str(group)] = "test"
        for group in ordered[n_test : n_test + n_val]:
            assigned[str(group)] = "val"
        for group in ordered[n_test + n_val :]:
            assigned[str(group)] = "train"

    result = {name: [] for name in ("train", "val", "test")}
    for group, indices in groups.items():
        result[assigned[group]].extend(int(index) for index in indices)
    for name in result:
        result[name].sort()
    all_indices = [index for name in result for index in result[name]]
    if sorted(all_indices) != list(range(len(examples))):
        raise AssertionError("group split is not a partition")
    return result


def _identity_column(examples: list[dict[str, Any]], cohort: str) -> str | None:
    for candidate in ("uniquepid", "subject_id", "patient_id"):
        if examples and all(candidate in example and example[candidate] is not None for example in examples):
            return candidate
    if cohort == "p12":
        return "record_id"
    return None


def _audit(
    cohort: str,
    examples: list[dict[str, Any]],
    split: dict[str, list[int]],
    column: str,
    *,
    identity_verifiable: bool = True,
) -> dict[str, Any]:
    groups = {name: {str(examples[index][column]) for index in indices} for name, indices in split.items()}
    overlaps = {
        f"{left}&{right}": len(groups[left] & groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "cohort": cohort,
        "group_column": column,
        "n_stays": len(examples),
        "identity_verifiable": identity_verifiable,
        "n_patients": len({str(example[column]) for example in examples})
        if identity_verifiable else None,
        "split_sizes_stays": {name: len(indices) for name, indices in split.items()},
        "split_sizes_patients": {name: len(groups[name]) for name in groups}
        if identity_verifiable else None,
        "overlaps": overlaps,
        "group_disjoint": all(value == 0 for value in overlaps.values()),
        "patient_disjoint": all(value == 0 for value in overlaps.values())
        if identity_verifiable else None,
        "label_counts": {
            name: int(sum(int(examples[index].get("label", 0)) for index in indices))
            for name, indices in split.items()
        },
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(str(example.get("sample_id")) for example in examples).encode()
        ).hexdigest(),
    }


def build_one(cohort: str) -> dict[str, Any]:
    cache_name, _ = COHORTS[cohort]
    # Once the MIMIC-IV patient-level cache exists, the audit must report the
    # repaired artifact rather than continuing to describe the registered
    # stay-level cache as the current state.
    if cohort == "mimiciv" and (PROCESSED / "mimiciv_events_jbi_v1.pkl").exists():
        cache_path = PROCESSED / "mimiciv_events_jbi_v1.pkl"
    else:
        cache_path = PROCESSED / cache_name
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    examples = list(cache["examples"])
    column = _identity_column(examples, cohort)
    if column is None:
        return {
            "cohort": cohort,
            "status": "BLOCKED_MISSING_PATIENT_ID",
            "cache": str(cache_path),
            "example_keys": sorted(examples[0].keys()) if examples else [],
        }

    # PhysioNet-2012 has no cross-record patient identifier in the cache.
    # Keep the registered record-level partition auditable, but never label it
    # as a patient-level split or write a new pseudo-patient split artifact.
    if cohort == "p12" and column == "record_id":
        registered_path = PROCESSED / COHORTS[cohort][1]
        registered = json.loads(registered_path.read_text())
        audit = _audit(
            cohort,
            examples,
            {name: list(registered[name]) for name in ("train", "val", "test")},
            column,
            identity_verifiable=False,
        )
        return {
            "cohort": cohort,
            "status": "NOT_VERIFIABLE_NO_PATIENT_ID",
            "cache": str(cache_path),
            "registered_split": str(registered_path),
            "group_column": column,
            "audit": audit,
        }

    split = _stratified_group_split(examples, column)
    audit = _audit(cohort, examples, split, column)
    if not audit["group_disjoint"]:
        raise AssertionError(f"{cohort}: grouped split has overlap {audit['overlaps']}")
    payload = {
        "analysis_plan_version": "JBI_analysis_plan_v1",
        "split_seed": SEED,
        "t_obs": float(cache.get("t_obs", 24.0)),
        "sets": [cohort],
        "n_samples": len(examples),
        "n_groups": int(audit["n_patients"]),
        "group_column": column,
        "split_unit": "patient_grouped",
        "status": "ok",
        "split_strategy": "stratified_by_patient_level_any_positive_label_70_15_15",
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "sample_ids": [str(example.get("sample_id")) for example in examples],
        "audit": audit,
    }
    if cohort == "mimiciv" and cache_path.name == "mimiciv_events_jbi_v1.pkl":
        payload.update(
            {
                "source_cache": str(cache_path.resolve()),
                "registered_source_cache": str((PROCESSED / "mimiciv_events.pkl").resolve()),
                "identity_map": str((PROCESSED / "mimiciv_subject_map.csv").resolve()),
                "standardizer_fit": "training_patients_only",
            }
        )
    split_path = PROCESSED / f"{cohort}_split_indices_jbi_v1.json"
    split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    return {"status": "ok", "split_path": str(split_path), **audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["all", *COHORTS], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    reports = [build_one(cohort) for cohort in cohorts]
    report = {
        "analysis_plan_version": "JBI_analysis_plan_v1",
        "seed": SEED,
        "reports": reports,
    }
    path = OUT / "e0_split_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
