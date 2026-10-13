#!/usr/bin/env python
"""Audit LOHO hospital/patient separation and draw frozen P2 train/val samples.

This entry point is permitted only after the independent P2 plan commit.
It reads the existing LOHO cache and fold file, removes any held-out test
patient who also appears in the other four hospital folds, then draws exact
patient-grouped 8,400 / 1,800 train/validation stay samples per fold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN_PATH = ROOT / "paper" / "跨医院验证分析计划_P2.md"
CACHE_PATH = ROOT / "data" / "processed" / "eicu_loho_events.pkl"
FOLDS_PATH = ROOT / "results" / "phase3_eicu" / "e_loho" / "loho_folds.csv"
OUT = ROOT / "results" / "jbi"
AUDIT_PATH = OUT / "p2_hospital_overlap_audit.json"
SAMPLE_PATH = OUT / "p2_training_sample_manifest.json"
TRAIN_STAYS = 8_400
VAL_STAYS = 1_800
BASE_SEED = 20261002
N_FOLDS = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def plan_first_commit() -> str:
    return subprocess.check_output(
        ["git", "log", "--reverse", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).splitlines()[0]


def load_fold_map() -> tuple[dict[int, int], dict[int, int]]:
    hospitals: dict[int, int] = {}
    declared_stays: dict[int, int] = defaultdict(int)
    with FOLDS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"hospital_id", "n_stays", "size_layer", "fold"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected LOHO fold columns: {reader.fieldnames}")
        for row in reader:
            hospital = int(row["hospital_id"])
            fold = int(row["fold"])
            if hospital in hospitals:
                raise ValueError(f"hospital {hospital} appears more than once in fold file")
            if fold not in range(N_FOLDS):
                raise ValueError(f"hospital {hospital} has invalid fold {fold}")
            hospitals[hospital] = fold
            declared_stays[hospital] = int(row["n_stays"])
    if len(hospitals) != 145 or set(hospitals.values()) != set(range(N_FOLDS)):
        raise ValueError(f"expected five folds over 145 hospitals, got {len(hospitals)}")
    return hospitals, dict(declared_stays)


def grouped_sample(
    candidate_indices: np.ndarray,
    examples: list[dict[str, Any]],
    rng: np.random.Generator,
    target_stays: int,
    unavailable_patients: set[str] | None = None,
) -> tuple[np.ndarray, set[str], int]:
    """Choose whole patient groups in seeded order without exceeding target."""

    unavailable = unavailable_patients or set()
    groups: dict[str, list[int]] = defaultdict(list)
    for index in candidate_indices.tolist():
        patient = str(examples[int(index)]["uniquepid"])
        if patient not in unavailable:
            groups[patient].append(int(index))
    patient_ids = sorted(groups)
    if patient_ids:
        order = rng.permutation(np.asarray(patient_ids, dtype=object)).tolist()
    else:
        order = []
    selected: list[int] = []
    selected_patients: set[str] = set()
    skipped_groups = 0
    count = 0
    for patient in order:
        rows = groups[str(patient)]
        if count + len(rows) > target_stays:
            skipped_groups += 1
            continue
        selected.extend(rows)
        selected_patients.add(str(patient))
        count += len(rows)
        if count == target_stays:
            break
    if count != target_stays:
        raise RuntimeError(
            f"could not draw exactly {target_stays} stays without splitting patients; "
            f"reached {count}, skipped {skipped_groups} oversize groups"
        )
    selected.sort()
    return np.asarray(selected, dtype=np.int64), selected_patients, skipped_groups


def _patient_names(examples: list[dict[str, Any]], indices: np.ndarray) -> set[str]:
    return {str(examples[int(index)]["uniquepid"]) for index in indices.tolist()}


def _stay_ids(examples: list[dict[str, Any]], indices: np.ndarray) -> list[int]:
    return [int(examples[int(index)]["record_id"]) for index in indices.tolist()]


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(PLAN_PATH)
    first_commit = plan_first_commit()
    current_plan_sha = file_sha256(PLAN_PATH)
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    examples = payload["examples"]
    if len(examples) != 61_678:
        raise ValueError(f"expected 61,678 LOHO stays, got {len(examples)}")

    fold_map, declared_hospital_stays = load_fold_map()
    indices_by_fold: dict[int, list[int]] = {fold: [] for fold in range(N_FOLDS)}
    hospital_counts: dict[int, int] = defaultdict(int)
    unknown_hospitals: list[dict[str, int]] = []
    patient_fold_hospitals: dict[str, set[int]] = defaultdict(set)
    for index, example in enumerate(examples):
        hospital = int(example["hospital_id"])
        if hospital not in fold_map:
            unknown_hospitals.append({"stay_index": int(index), "hospital_id": hospital})
            continue
        fold = fold_map[hospital]
        indices_by_fold[fold].append(index)
        hospital_counts[hospital] += 1
        patient_fold_hospitals[str(example["uniquepid"])].add(fold)

    actual_hospitals = set(hospital_counts)
    hospital_count_match = actual_hospitals == set(fold_map)
    per_hospital_count_match = {
        str(hospital): int(hospital_counts.get(hospital, 0)) == int(expected)
        for hospital, expected in declared_hospital_stays.items()
    }
    cache_hospital_fold_counts = {
        str(fold): int(len(indices_by_fold[fold])) for fold in range(N_FOLDS)
    }
    unique_cross_fold_patients = {
        patient: sorted(folds)
        for patient, folds in patient_fold_hospitals.items()
        if len(folds) > 1
    }

    fold_audits: list[dict[str, Any]] = []
    fold_samples: dict[str, dict[str, Any]] = {}
    sampling_errors: list[str] = []
    overlap_gate = not unknown_hospitals and hospital_count_match and all(per_hospital_count_match.values())

    for fold in range(N_FOLDS):
        heldout_idx = np.asarray(indices_by_fold[fold], dtype=np.int64)
        train_pool_idx = np.asarray(
            [index for other in range(N_FOLDS) if other != fold for index in indices_by_fold[other]],
            dtype=np.int64,
        )
        heldout_patients = _patient_names(examples, heldout_idx)
        train_pool_patients = _patient_names(examples, train_pool_idx)
        cross_patients = heldout_patients & train_pool_patients
        test_idx = np.asarray(
            [
                index
                for index in heldout_idx.tolist()
                if str(examples[int(index)]["uniquepid"]) not in cross_patients
            ],
            dtype=np.int64,
        )
        kept_test_patients = _patient_names(examples, test_idx)
        test_overlap_after_removal = kept_test_patients & train_pool_patients
        heldout_hospitals = {hospital for hospital, assigned in fold_map.items() if assigned == fold}
        train_hospitals = set(fold_map) - heldout_hospitals
        hospital_overlap = heldout_hospitals & train_hospitals

        audit = {
            "fold": int(fold),
            "heldout_hospital_count": len(heldout_hospitals),
            "training_pool_hospital_count": len(train_hospitals),
            "hospital_overlap_count": len(hospital_overlap),
            "training_pool_stay_count": int(len(train_pool_idx)),
            "heldout_stay_count_before_patient_filter": int(len(heldout_idx)),
            "cross_side_patient_count": len(cross_patients),
            "heldout_stays_removed_for_cross_side_patients": int(len(heldout_idx) - len(test_idx)),
            "heldout_test_stay_count": int(len(test_idx)),
            "heldout_test_patient_count": len(kept_test_patients),
            "test_patient_overlap_after_removal": len(test_overlap_after_removal),
            "hospital_disjoint": len(hospital_overlap) == 0,
            "test_patient_disjoint_after_filter": len(test_overlap_after_removal) == 0,
        }
        fold_audits.append(audit)
        if not audit["hospital_disjoint"] or not audit["test_patient_disjoint_after_filter"]:
            overlap_gate = False

        fold_seed = BASE_SEED + fold
        rng = np.random.Generator(np.random.PCG64(fold_seed))
        try:
            train_idx, train_patients, train_skipped = grouped_sample(
                train_pool_idx, examples, rng, TRAIN_STAYS
            )
            train_index_set = set(train_idx.tolist())
            remaining_idx = np.asarray(
                [index for index in train_pool_idx.tolist() if index not in train_index_set],
                dtype=np.int64,
            )
            val_idx, val_patients, val_skipped = grouped_sample(
                remaining_idx,
                examples,
                rng,
                VAL_STAYS,
                unavailable_patients=train_patients,
            )
            train_val_patient_overlap = train_patients & val_patients
            if train_val_patient_overlap:
                raise RuntimeError("patient-grouped train/validation overlap after draw")
            if set(train_idx.tolist()) & set(val_idx.tolist()):
                raise RuntimeError("train/validation stay overlap after draw")
            fold_samples[str(fold)] = {
                "fold": fold,
                "seed": fold_seed,
                "train_stay_indices": train_idx.tolist(),
                "validation_stay_indices": val_idx.tolist(),
                "test_stay_indices": test_idx.tolist(),
                "train_stay_ids": _stay_ids(examples, train_idx),
                "validation_stay_ids": _stay_ids(examples, val_idx),
                "test_stay_ids": _stay_ids(examples, test_idx),
                "train_patient_ids": sorted(train_patients),
                "validation_patient_ids": sorted(val_patients),
                "test_patient_ids": sorted(kept_test_patients),
                "train_count": int(len(train_idx)),
                "validation_count": int(len(val_idx)),
                "test_count": int(len(test_idx)),
                "train_skipped_patient_groups": int(train_skipped),
                "validation_skipped_patient_groups": int(val_skipped),
                "train_patient_overlap_count": 0,
            }
            audit["sample_counts_exact"] = len(train_idx) == TRAIN_STAYS and len(val_idx) == VAL_STAYS
            audit["train_validation_patient_disjoint"] = len(train_val_patient_overlap) == 0
            audit["sample_seed"] = fold_seed
        except Exception as exc:
            sampling_errors.append(f"fold {fold}: {type(exc).__name__}: {exc}")
            audit["sample_counts_exact"] = False
            audit["train_validation_patient_disjoint"] = False
            audit["sample_error"] = f"{type(exc).__name__}: {exc}"

    all_overlap_pass = bool(
        overlap_gate
        and all(fold["hospital_disjoint"] and fold["test_patient_disjoint_after_filter"] for fold in fold_audits)
    )
    all_sampling_pass = bool(
        len(fold_samples) == N_FOLDS
        and not sampling_errors
        and all(
            fold_samples[str(fold)]["train_count"] == TRAIN_STAYS
            and fold_samples[str(fold)]["validation_count"] == VAL_STAYS
            and fold_samples[str(fold)]["train_patient_ids"]
            and fold_samples[str(fold)]["validation_patient_ids"]
            for fold in range(N_FOLDS)
        )
    )

    audit_report = {
        "schema_version": "p2_hospital_overlap_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_first_commit": first_commit,
        "analysis_plan_sha256": current_plan_sha,
        "cache_path": str(CACHE_PATH.relative_to(ROOT)),
        "cache_sha256": file_sha256(CACHE_PATH),
        "fold_file_path": str(FOLDS_PATH.relative_to(ROOT)),
        "fold_file_sha256": file_sha256(FOLDS_PATH),
        "n_stays": len(examples),
        "n_hospitals_in_fold_map": len(fold_map),
        "n_hospitals_in_cache": len(actual_hospitals),
        "unknown_hospital_stays": unknown_hospitals,
        "fold_file_hospital_stay_counts_match_cache": all(per_hospital_count_match.values()),
        "cross_fold_patient_count_before_test_filter": len(unique_cross_fold_patients),
        "cross_fold_patient_id_sha256": stable_json_sha(sorted(unique_cross_fold_patients.items())),
        "folds": fold_audits,
        "overlap_gate_pass": all_overlap_pass,
        "sampling_gate_pass": all_sampling_pass,
        "sampling_errors": sampling_errors,
        "training_gate": "PASS" if all_overlap_pass and all_sampling_pass else "STOP",
    }
    sample_manifest = {
        "schema_version": "p2_training_sample_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_first_commit": first_commit,
        "analysis_plan_sha256": current_plan_sha,
        "source_cache_path": str(CACHE_PATH.relative_to(ROOT)),
        "source_cache_sha256": audit_report["cache_sha256"],
        "source_fold_file_path": str(FOLDS_PATH.relative_to(ROOT)),
        "source_fold_file_sha256": audit_report["fold_file_sha256"],
        "base_seed": BASE_SEED,
        "train_stays_per_fold": TRAIN_STAYS,
        "validation_stays_per_fold": VAL_STAYS,
        "heldout_test_rule": "all held-out hospital stays after removing patients present in training hospitals",
        "folds": fold_samples,
        "status": "PASS" if all_overlap_pass and all_sampling_pass else "STOP",
        "sampling_errors": sampling_errors,
    }
    sample_manifest["sample_ids_sha256"] = stable_json_sha(
        {
            fold: {
                key: sample_manifest["folds"][fold][key]
                for key in ("train_stay_ids", "validation_stay_ids", "test_stay_ids")
            }
            for fold in sample_manifest["folds"]
        }
    )
    return audit_report, sample_manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for existing in (AUDIT_PATH, SAMPLE_PATH):
        if existing.exists():
            raise FileExistsError(f"refusing to overwrite P2 prep artifact: {existing}")
    audit, samples = run()
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SAMPLE_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"hospital overlap audit: {audit['training_gate']} → {AUDIT_PATH}")
    print(f"sample manifest: {samples['status']} → {SAMPLE_PATH}")
    for fold in audit["folds"]:
        print(
            f"fold {fold['fold']}: train pool={fold['training_pool_stay_count']}; "
            f"held-out={fold['heldout_stay_count_before_patient_filter']}; "
            f"cross-side patients={fold['cross_side_patient_count']}; "
            f"removed test stays={fold['heldout_stays_removed_for_cross_side_patients']}; "
            f"test={fold['heldout_test_stay_count']}; exact sample={fold.get('sample_counts_exact', False)}"
        )
    return 0 if audit["training_gate"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
