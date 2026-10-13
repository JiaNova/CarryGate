#!/usr/bin/env python
"""Registered post hoc objective-by-metric sensitivity analysis (R4).

The preflight phase checks all frozen macro-checkpoint Track B outputs and all
90 P2 prediction hashes before any new metric is calculated.  The analysis
phase refuses to run until that preflight record exists and still matches the
current inputs and source code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi" / "revision_sensitivity"
PAPER_OUT = ROOT / "results" / "paper"
PREFLIGHT_PATH = OUT / "r4_preflight_manifest.json"
PLAN_PATH = ROOT / "paper" / "JAMIA修订补充分析计划.md"
PLAN_MANIFEST_PATH = OUT / "analysis_plan_manifest.json"
TRACKB_READER_PATH = ROOT / "scripts" / "jbi" / "23_adapter_vs_external_paired.py"
TRACKB_MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")
OBJECTIVES = ("MSE", "VariableBalancedMSE", "MAE")
OBJECTIVE_LABEL = {
    "MSE": "MSE",
    "VariableBalancedMSE": "Variable-balanced MSE",
    "MAE": "masked L1",
}
OBJECTIVE_CELL = {
    "MSE": "mse_macro",
    "VariableBalancedMSE": "balanced_macro",
    "MAE": "mae_macro",
}
P2_MODELS = TRACKB_MODELS[:3]
P2_OBJECTIVES = ("MSE", "MAE")
P2_SEEDS = (2024, 2025, 2026)
TRACKB_COHORTS = ("eicu", "mimiciv")
UNSCORED = frozenset({6, 19, 32})
N_VARIABLES = 37
MIN_TEST_EVENTS = 50
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261012
DISPLAY_MODEL = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi",
                 "HyperIMTS": "HyperIMTS", "Warpformer": "Warpformer"}
COMPARATORS = ("GraFITi", "HyperIMTS")
GENERATED_RELATIVE = (
    "results/jbi/revision_sensitivity/r4_trackb_metrics.csv",
    "results/jbi/revision_sensitivity/r4_p2_metrics.csv",
    "results/jbi/revision_sensitivity/r4_paired_differences.csv",
    "results/jbi/revision_sensitivity/r4_p2_c3_summary.csv",
    "results/jbi/revision_sensitivity/r4_per_variable_raw_rmse.csv",
    "results/jbi/revision_sensitivity/r4_pooled_checkpoint_evidence.csv",
    "results/jbi/revision_sensitivity/r4_report.json",
    "results/jbi/revision_sensitivity/r4_manifest.json",
    "results/jbi/revision_sensitivity/r4_S13_objective_metric_matrix.tex",
    "results/paper/S13_objective_metric_matrix.pdf",
    "results/paper/S13_objective_metric_matrix.png",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hash_entry(path: Path, role: str, **context: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)), "role": role,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
            **context}


def reader_module():
    spec = importlib.util.spec_from_file_location("trackb_paired_reader", TRACKB_READER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load registered Track B reader: {TRACKB_READER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_raw_dense(cohort: str, test_index: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build raw targets in exactly the same C-order event layout as reader._reference."""
    from scripts.d1_harness.rescore_macro import VARIANT_CACHES
    from src.data.p12_events import EPS

    cache_name, _ = VARIANT_CACHES["jbi_patient"][cohort]
    cache_path = ROOT / "data" / "processed" / cache_name
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    examples = cache["examples"]
    raw_all: list[np.ndarray] = []
    var_all: list[np.ndarray] = []
    for index in test_index:
        target = examples[int(index)]["target"]
        var = np.asarray(target["var"], dtype=np.int64)
        time_values = np.asarray(target["time"])
        raw = np.asarray(target["value_raw"], dtype=np.float64)
        order = np.lexsort((var, time_values))
        var_s, time_s, raw_s = var[order], time_values[order], raw[order]
        _, inverse, counts = np.unique(
            np.stack([time_s, var_s]), axis=1, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()
        group_start = np.searchsorted(inverse, np.arange(counts.size))
        rank = np.arange(inverse.size) - group_start[inverse]
        uniq_time, time_index = np.unique(time_s, return_inverse=True)
        rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
        np.maximum.at(rows_per_time, time_index, rank + 1)
        row_start = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])
        row_of = row_start[time_index] + rank
        dense = np.lexsort((var_s, row_of))
        raw_all.append(raw_s[dense])
        var_all.append(var_s[dense])
    raw_truth = np.concatenate(raw_all)
    raw_var = np.concatenate(var_all)
    standardizer = cache["standardizer"]
    medians = np.asarray(standardizer["median"], dtype=np.float64)
    iqrs = np.asarray(standardizer["iqr"], dtype=np.float64)
    lower = np.asarray(standardizer["lower_clip"], dtype=np.float64)
    upper = np.asarray(standardizer["upper_clip"], dtype=np.float64)
    raw32 = raw_truth.astype(np.float32).astype(np.float64)
    transformed = np.clip((raw32 - medians[raw_var]) / (iqrs[raw_var] + EPS),
                          lower[raw_var], upper[raw_var]).astype(np.float32)
    return raw_truth, {
        "raw_variable": raw_var,
        "median": medians,
        "iqr": iqrs,
        "variable_names": list(cache["variable_names"]),
        "standardizer_sha256": sha256_json(standardizer),
        "transformed_target": transformed,
    }


def _record_input(path: Path, role: str, records: list[dict[str, Any]], **context: Any) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"R4 input escapes project root: {resolved}")
    entry = source_hash_entry(resolved, role, **context)
    records.append(entry)
    return entry


def _npz_member_shape(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read only an NPY member header from an NPZ, without inflating its payload."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(n) for n in shape), np.dtype(dtype)


def preflight() -> dict[str, Any]:
    """Validate all frozen inputs and return a hash-complete preflight record."""
    started = time.time()
    failures: list[str] = []
    input_hashes: list[dict[str, Any]] = []
    cell_counts = {"trackb_cells": 0, "trackb_eval_directories": 0,
                   "p2_prediction_files": 0, "p2_completed_runs": 0}
    expected_plan_sha = "fade749e20c54e9340dbbb01ec9a6c8e12a8ea912eb34baec7ccced30e38e320"
    plan_manifest = load_json(PLAN_MANIFEST_PATH)
    actual_plan_sha = sha256_file(PLAN_PATH)
    if (plan_manifest.get("status") != "RULES_FROZEN_BEFORE_NEW_COMPUTATION"
            or plan_manifest.get("new_computation_started") is not False
            or plan_manifest.get("plan_sha256") != expected_plan_sha
            or actual_plan_sha != expected_plan_sha
            or "R4" not in plan_manifest.get("frozen_tasks", [])):
        failures.append("R4 registration document/manifest hash or status does not match")
    _record_input(PLAN_PATH, "frozen_analysis_plan", input_hashes)
    _record_input(PLAN_MANIFEST_PATH, "registration_manifest", input_hashes)
    _record_input(TRACKB_READER_PATH, "registered_trackb_input_reader", input_hashes)

    reader = reader_module()
    trackb_manifest_paths = {
        "eicu": ROOT / "results/jbi/trackb_controls/run_manifest.json",
        "mimiciv": ROOT / "results/jbi/trackb_controls_mimiciv/run_manifest.json",
    }
    for cohort in TRACKB_COHORTS:
        manifest_path = trackb_manifest_paths[cohort]
        manifest = load_json(manifest_path)
        manifest_dir = manifest_path.parent
        summary_path = manifest_dir / "e2_trackb_controls_summary.csv"
        audit_path = manifest_dir / "e2_trackb_controls_audit.json"
        _record_input(manifest_path, "trackb_training_run_manifest", input_hashes,
                      cohort=cohort)
        _record_input(summary_path, "trackb_frozen_summary", input_hashes,
                      cohort=cohort)
        _record_input(audit_path, "trackb_frozen_audit", input_hashes,
                      cohort=cohort)
        runs = [r for r in manifest.get("runs", [])
                if r.get("phase") == "factorial"
                and r.get("selection") == "macro_persistence"
                and r.get("model") in TRACKB_MODELS
                and r.get("loss") in OBJECTIVES]
        keyed = {(r.get("model"), r.get("loss")): r for r in runs}
        expected_keys = {(model, objective) for model in TRACKB_MODELS
                         for objective in OBJECTIVES}
        if set(keyed) != expected_keys or len(runs) != 12:
            failures.append(f"{cohort}: expected 12 unique factorial macro runs; got {len(runs)}")
        expected_run_seeds = [2024, 2025, 2026, 2027, 2028]
        if manifest.get("seeds") != expected_run_seeds:
            failures.append(f"{cohort}: frozen Track B seed roster differs")
        if manifest.get("data_variant") != "jbi_patient":
            failures.append(f"{cohort}: Track B manifest is not on jbi_patient cache")

        ref = reader._reference(cohort)
        cache_name, split_name = reader.VARIANT_CACHES["jbi_patient"][cohort]
        _record_input(ROOT / "data/processed" / cache_name, "trackb_event_cache",
                      input_hashes, cohort=cohort)
        _record_input(ROOT / "data/processed" / split_name, "trackb_patient_split",
                      input_hashes, cohort=cohort)
        split_payload = load_json(ROOT / "data/processed" / split_name)
        if (split_payload.get("split_unit") != "patient_grouped"
                or [int(i) for i in split_payload["test"]] != ref["test_index"]):
            failures.append(f"{cohort}: frozen Track B split/cache identity mismatch")

        eval_dir_identity: tuple[list[int], str, str, tuple[int, ...], tuple[int, ...]] | None = None
        target_cache_verified = False
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                rec = keyed.get((model, objective))
                if rec is None:
                    continue
                cell_counts["trackb_cells"] += 1
                if (rec.get("status") != "completed" or rec.get("returncode") != 0
                        or rec.get("itr") != 5 or rec.get("data_variant") != "jbi_patient"
                        or rec.get("selection") != "macro_persistence"):
                    failures.append(f"{cohort}/{model}/{objective}: macro run record is incomplete")
                tag = str(rec.get("tag", ""))
                tag_cell = {"MSE": "mse", "VariableBalancedMSE": "balanced", "MAE": "mae"}[objective]
                if (not tag.endswith(f"_{tag_cell}_macro_v1")
                        or model not in tag):
                    failures.append(f"{cohort}/{model}/{objective}: unexpected macro tag {tag}")
                dirs = reader._eval_dirs(model, tag, cohort)
                if len(dirs) != 5:
                    failures.append(f"{cohort}/{model}/{objective}: expected 5 eval dirs, found {len(dirs)}")
                    continue
                output_root = (ROOT / str(rec["output_root"])).resolve()
                iter_ids = []
                for eval_dir in dirs:
                    cell_counts["trackb_eval_directories"] += 1
                    if not eval_dir.resolve().is_relative_to(output_root):
                        failures.append(f"{cohort}/{model}/{objective}: eval dir escapes recorded output root")
                    try:
                        iter_id = int(eval_dir.parent.name.removeprefix("iter"))
                    except ValueError:
                        iter_id = -1
                    iter_ids.append(iter_id)
                    required = ("configs.yaml", "input_sample_ID.npy", "input_y.npy",
                                "input_y_mask.npy", "output_pred.npy")
                    paths = {name: (eval_dir.parent / name if name == "configs.yaml"
                                    else eval_dir / name) for name in required}
                    missing = [str(path) for path in paths.values() if not path.is_file()]
                    if missing:
                        failures.append(f"{cohort}/{model}/{objective}/{eval_dir.name}: missing {missing}")
                        continue
                    hashed = {}
                    for name, path in paths.items():
                        hashed[name] = _record_input(
                            path, "trackb_macro_eval_array" if name != "configs.yaml"
                            else "trackb_macro_eval_config", input_hashes,
                            cohort=cohort, model=model, objective=objective,
                            iteration=iter_id, array=name)
                    try:
                        config = yaml.safe_load(paths["configs.yaml"].read_text(encoding="utf-8")) or {}
                        if (config.get("dataset_subset_name") != "jbi_patient"
                                or config.get("loss") != objective):
                            failures.append(f"{cohort}/{model}/{objective}: config cache/loss mismatch")
                        sample_id = np.load(paths["input_sample_ID.npy"], mmap_mode="r").astype(np.int64)
                        y_header = np.load(paths["input_y.npy"], mmap_mode="r")
                        mask_header = np.load(paths["input_y_mask.npy"], mmap_mode="r")
                        pred_header = np.load(paths["output_pred.npy"], mmap_mode="r")
                        if (sample_id.tolist() != ref["test_index"]
                                or y_header.shape != mask_header.shape or y_header.shape != pred_header.shape
                                or y_header.ndim != 3 or y_header.shape[-1] != N_VARIABLES):
                            failures.append(f"{cohort}/{model}/{objective}: eval shapes/split fail checks")
                        current = (sample_id.tolist(), hashed["input_y.npy"]["sha256"],
                                   hashed["input_y_mask.npy"]["sha256"],
                                   tuple(y_header.shape), tuple(mask_header.shape))
                        if eval_dir_identity is None:
                            eval_dir_identity = current
                        elif current != eval_dir_identity:
                            failures.append(f"{cohort}: eval split or cached target/mask identity differs across factorial cells")
                        if not target_cache_verified:
                            y = np.asarray(y_header)
                            mask = np.asarray(mask_header, dtype=bool)
                            if (int(mask.sum()) != ref["n_events_dense"]
                                    or not np.allclose(y[mask], ref["dense_truth"], atol=1e-5, rtol=0)):
                                failures.append(f"{cohort}: macro eval target does not match frozen cache")
                            target_cache_verified = True
                    except Exception as exc:
                        failures.append(f"{cohort}/{model}/{objective}/{eval_dir.name}: {type(exc).__name__}: {exc}")
                if sorted(iter_ids) != list(range(5)):
                    failures.append(f"{cohort}/{model}/{objective}: expected iter0..iter4, found {sorted(iter_ids)}")

        frozen_audit = load_json(audit_path)
        if (frozen_audit.get("status") != "PASS"
                or frozen_audit.get("final_seeds") != 5
                or frozen_audit.get("split_unit") != "patient_grouped"
                or frozen_audit.get("selection_source") != "validation_only_training_summary"):
            failures.append(f"{cohort}: Track B frozen source audit is not PASS")

    # Pooled-checkpoint evidence is read only from the frozen source summary.
    for cohort in TRACKB_COHORTS:
        evidence_path = trackb_manifest_paths[cohort].parent / "e2_trackb_controls_summary.csv"
        if not evidence_path.is_file():
            failures.append(f"{cohort}: missing pooled-checkpoint factorial summary")

    p2_ledger_path = ROOT / "results/jbi/p2_loho_test_inference_ledger.json"
    p2_manifest_path = ROOT / "results/jbi/p2_loho_test_batch_manifest.json"
    p2_freeze_path = ROOT / "results/jbi/p2_loho_checkpoint_freeze.json"
    ledger = load_json(p2_ledger_path)
    ledger_input = _record_input(p2_ledger_path, "p2_test_inference_ledger", input_hashes)
    if not p2_manifest_path.is_file():
        failures.append("P2 frozen test batch manifest is missing")
    else:
        if sha256_file(p2_manifest_path) != ledger.get("test_batch_manifest_sha256"):
            failures.append("P2 test batch manifest SHA256 differs from ledger")
        _record_input(p2_manifest_path, "p2_test_batch_manifest", input_hashes)
    if p2_freeze_path.is_file():
        freeze_doc = load_json(p2_freeze_path)
        _record_input(p2_freeze_path, "p2_checkpoint_freeze", input_hashes)
        if (freeze_doc.get("checkpoint_count") != 90
                or freeze_doc.get("test_inference_ledger_sha256") != ledger_input["sha256"]):
            failures.append("P2 checkpoint freeze does not bind the current 90-run inference ledger")
    else:
        failures.append("P2 checkpoint freeze document is missing")
    if (ledger.get("status") != "COMPLETE" or ledger.get("checkpoint_count") != 90
            or ledger.get("completed_run_count") != 90):
        failures.append("P2 test inference ledger is not a complete 90-run freeze")
    run_results = ledger.get("run_results", {})
    expected_p2_keys = {
        f"fold{fold}/{model}/{objective}/seed{seed}"
        for fold in range(5) for model in P2_MODELS
        for objective in P2_OBJECTIVES for seed in P2_SEEDS
    }
    if set(run_results) != expected_p2_keys:
        failures.append(f"P2 ledger has {len(run_results)} runs, expected exactly 90")
    batch_manifest = load_json(p2_manifest_path) if p2_manifest_path.is_file() else {}
    fold_batches = batch_manifest.get("folds", batch_manifest.get("fold_batch_manifests", {}))
    ledger_fold_batches = ledger.get("fold_batch_manifests", {})
    checked_fold_identity: set[str] = set()
    for key in expected_p2_keys:
        record = run_results.get(key)
        if record is None:
            continue
        cell_counts["p2_completed_runs"] += int(record.get("status") == "completed")
        relpath = record.get("path")
        path = ROOT / str(relpath)
        if record.get("status") != "completed" or not path.is_file():
            failures.append(f"P2 frozen prediction missing or incomplete: {key}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            failures.append(f"P2 prediction SHA256 mismatch: {key}")
        _record_input(path, "p2_frozen_prediction_npz", input_hashes, run_key=key)
        cell_counts["p2_prediction_files"] += 1
        fold_text = key.split("/", 1)[0].removeprefix("fold")
        fold_doc = ledger_fold_batches.get(fold_text, {})
        fold_sha = record.get("fold_batch_manifest_sha256")
        batch_fold = fold_batches.get(fold_text, {}) if isinstance(fold_batches, dict) else {}
        if (fold_doc.get("sha256") is None or fold_sha != fold_doc.get("sha256")
                or batch_fold.get("sha256") != fold_sha):
            failures.append(f"P2 fold batch identity mismatch: {key}")
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if (str(arrays["run_key"].item()) != key
                        or int(arrays["fold"].item()) != int(fold_text)
                        or str(arrays["model"].item()) != key.split("/")[1]
                        or str(arrays["objective"].item()) != key.split("/")[2]
                        or int(arrays["seed"].item()) != int(key.rsplit("seed", 1)[1])
                        or str(arrays["fold_batch_manifest_sha256"].item()) != fold_sha):
                    failures.append(f"P2 NPZ identity metadata mismatch: {key}")
                sample_ids = arrays["test_sample_index"].astype(np.int64)
                event_count = int(record.get("test_event_count", -1))
                stay_count = int(record.get("test_stay_count", -1))
                batches = batch_fold.get("batches", []) if isinstance(batch_fold, dict) else []
                if sample_ids.size != stay_count:
                    failures.append(f"P2 NPZ test stay count mismatch: {key}")
                if batches:
                    expected_samples = [int(index) for batch in batches
                                         for index in batch.get("sample_indices", [])]
                    if expected_samples != sample_ids.tolist():
                        failures.append(f"P2 NPZ test sample order differs from frozen batch manifest: {key}")
                if len(arrays["variable_names"]) != N_VARIABLES:
                    failures.append(f"P2 NPZ variable roster mismatch: {key}")
                event_members = ("event_sample_index", "event_patient_id", "event_variable_index",
                                 "truth_standardized", "prediction_standardized",
                                 "carry_forward_standardized", "truth_raw", "prediction_raw",
                                 "carry_forward_raw")
                for member in event_members:
                    shape, _dtype = _npz_member_shape(path, member)
                    if shape != (event_count,):
                        failures.append(f"P2 NPZ {member} shape/count mismatch: {key}")
                if fold_text not in checked_fold_identity:
                    event_ids = arrays["event_sample_index"].astype(np.int64)
                    patient_ids = arrays["event_patient_id"]
                    variables = arrays["event_variable_index"].astype(np.int64)
                    if (event_ids.size != event_count or patient_ids.size != event_count
                            or variables.size != event_count
                            or variables.min(initial=0) < 0 or variables.max(initial=-1) >= N_VARIABLES
                            or not np.isin(event_ids, sample_ids).all()):
                        failures.append(f"P2 fold{fold_text}: event/sample identity mismatch")
                    checked_fold_identity.add(fold_text)
        except Exception as exc:
            failures.append(f"P2 prediction unreadable {key}: {type(exc).__name__}: {exc}")

    if cell_counts["trackb_cells"] != 24 or cell_counts["trackb_eval_directories"] != 120:
        failures.append("Track B macro input cardinality is not 24 cells / 120 eval directories")
    if cell_counts["p2_prediction_files"] != 90 or cell_counts["p2_completed_runs"] != 90:
        failures.append("P2 input cardinality is not 90 verified frozen predictions")
    deduped: dict[str, dict[str, Any]] = {}
    for row in input_hashes:
        key = row["path"]
        if key in deduped and deduped[key]["sha256"] != row["sha256"]:
            failures.append(f"input changed during preflight hashing: {key}")
        deduped[key] = row
    return {
        "schema_version": "jamia_r4_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "preflighted_at_utc": utc_now(),
        "analysis_plan_sha256": actual_plan_sha,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "counts": cell_counts,
        "input_hash_count": len(deduped),
        "input_hashes": list(deduped.values()),
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def grid_sum(patient: np.ndarray, variable: np.ndarray, values: np.ndarray,
             n_patients: int) -> np.ndarray:
    result = np.zeros((n_patients, N_VARIABLES), dtype=np.float64)
    np.add.at(result, (patient, variable), values)
    return result


def score_grid(error: np.ndarray, baseline: np.ndarray, counts: np.ndarray,
               *, metric: str, aggregation: str, eligible_override: np.ndarray | None = None) -> float:
    denom = baseline.sum(axis=0)
    numerator = error.sum(axis=0)
    if aggregation == "macro":
        eligible = ((counts.sum(axis=0) >= MIN_TEST_EVENTS)
                    & (denom > 1e-9)
                    & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED)))
        if eligible_override is not None:
            eligible &= eligible_override
        if not eligible.any():
            return float("nan")
        return float(np.mean(1.0 - numerator[eligible] / denom[eligible]))
    keep = ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))
    pooled_base = float(denom[keep].sum())
    if pooled_base <= 1e-9:
        return float("nan")
    return float(1.0 - numerator[keep].sum() / pooled_base)


def build_trackb_reference(cohort: str, reader: Any) -> dict[str, Any]:
    ref = reader._reference(cohort)
    raw_truth, raw_meta = _reference_raw_dense(cohort, ref["test_index"])
    if (raw_truth.size != ref["dense_truth"].size
            or not np.array_equal(raw_meta["raw_variable"], ref["dense_var"])
            or not np.allclose(raw_meta["transformed_target"], ref["dense_truth"],
                               atol=1e-5, rtol=0)):
        raise AssertionError(f"{cohort}: raw target cannot be reconciled to frozen standardized cache")
    keep = np.asarray(ref["keep"], dtype=bool)
    return {**ref, "raw_truth": raw_truth, "raw_truth_scored": raw_truth[keep],
            "raw_meta": raw_meta}


def load_trackb_cell(cohort: str, model: str, objective: str, tag: str,
                     ref: dict[str, Any], reader: Any) -> dict[str, Any]:
    directories = reader._eval_dirs(model, tag, cohort)
    if len(directories) != 5:
        raise AssertionError(f"{cohort}/{model}/{objective}: expected 5 eval dirs")
    n_patients = ref["n_patients"]
    patient = np.asarray(ref["patient"], dtype=np.int64)
    variable = np.asarray(ref["variable"], dtype=np.int64)
    truth = np.asarray(ref["truth"], dtype=np.float64)
    baseline_abs_event = np.abs(np.asarray(ref["x_last"], dtype=np.float64) - truth)
    baseline_sq_event = (np.asarray(ref["x_last"], dtype=np.float64) - truth) ** 2
    baseline_abs = grid_sum(patient, variable, baseline_abs_event, n_patients)
    baseline_sq = grid_sum(patient, variable, baseline_sq_event, n_patients)
    counts = grid_sum(patient, variable, np.ones(truth.size), n_patients)
    abs_seeds: list[np.ndarray] = []
    sq_seeds: list[np.ndarray] = []
    rmse_seed_rows: list[dict[int, tuple[float, int]]] = []
    score_keep = np.asarray(ref["keep"], dtype=bool)
    raw_truth = np.asarray(ref["raw_truth_scored"], dtype=np.float64)
    raw_var = variable
    median = np.asarray(ref["raw_meta"]["median"], dtype=np.float64)
    iqr = np.asarray(ref["raw_meta"]["iqr"], dtype=np.float64)
    for eval_dir in directories:
        mask = np.load(eval_dir / "input_y_mask.npy").astype(bool)
        prediction = np.asarray(np.load(eval_dir / "output_pred.npy")[mask], dtype=np.float64)
        if prediction.size != ref["n_events_dense"]:
            raise AssertionError(f"{eval_dir}: target count differs from frozen reference")
        prediction = prediction[score_keep]
        if not np.isfinite(prediction).all():
            raise AssertionError(f"{eval_dir}: non-finite prediction")
        abs_seeds.append(grid_sum(patient, variable, np.abs(prediction - truth), n_patients))
        sq_seeds.append(grid_sum(patient, variable, (prediction - truth) ** 2, n_patients))
        prediction_raw = prediction * iqr[raw_var] + median[raw_var]
        squared_raw = (prediction_raw - raw_truth) ** 2
        rows = {}
        for var in np.unique(raw_var):
            selected = raw_var == var
            rows[int(var)] = (float(squared_raw[selected].sum()), int(selected.sum()))
        rmse_seed_rows.append(rows)
    return {
        "abs_error": np.mean(np.stack(abs_seeds), axis=0),
        "sq_error": np.mean(np.stack(sq_seeds), axis=0),
        "baseline_abs": baseline_abs,
        "baseline_sq": baseline_sq,
        "counts": counts,
        "rmse_sse_by_seed": rmse_seed_rows,
        "n_seeds": len(directories),
    }


def trackb_bootstrap(ref: dict[str, Any], cells: dict[tuple[str, str], dict[str, Any]],
                     reps: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patient bootstrap all Track B cells using common draws within cohort."""
    n = int(ref["n_patients"])
    rng = np.random.default_rng(seed)
    draws = np.zeros((reps, n), dtype=np.int16)
    for b in range(reps):
        draws[b] = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.int16)
    key_order = [(model, obj) for model in TRACKB_MODELS for obj in OBJECTIVES]
    abs_cube = np.stack([cells[key]["abs_error"] for key in key_order], axis=-1)
    sq_cube = np.stack([cells[key]["sq_error"] for key in key_order], axis=-1)
    base_abs = next(iter(cells.values()))["baseline_abs"]
    base_sq = next(iter(cells.values()))["baseline_sq"]
    count_grid = next(iter(cells.values()))["counts"]
    b_counts = draws @ count_grid
    b_base_abs = draws @ base_abs
    b_base_sq = draws @ base_sq
    b_abs = (draws @ abs_cube.reshape(n, -1)).reshape(reps, N_VARIABLES, -1)
    b_sq = (draws @ sq_cube.reshape(n, -1)).reshape(reps, N_VARIABLES, -1)
    skills: dict[tuple[str, str, str, str], np.ndarray] = {}
    full_skills: dict[tuple[str, str, str, str], float] = {}
    for i, (model, obj) in enumerate(key_order):
        for metric, errors, baseline in (("L1", cells[(model, obj)]["abs_error"], base_abs),
                                         ("S2", cells[(model, obj)]["sq_error"], base_sq)):
            error_boot = b_abs[:, :, i] if metric == "L1" else b_sq[:, :, i]
            base_boot = b_base_abs if metric == "L1" else b_base_sq
            valid = ((b_counts >= MIN_TEST_EVENTS) & (base_boot > 1e-9)
                     & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))[None, :])
            ratio = np.divide(error_boot, base_boot, out=np.zeros_like(error_boot),
                              where=base_boot > 1e-9)
            macro = np.full(reps, np.nan, dtype=np.float64)
            for b in range(reps):
                if valid[b].any():
                    macro[b] = float(np.mean(1.0 - ratio[b, valid[b]]))
            keep = ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))
            pooled_base = base_boot[:, keep].sum(axis=1)
            pooled_err = error_boot[:, keep].sum(axis=1)
            pooled = np.where(pooled_base > 1e-9, 1.0 - pooled_err / pooled_base, np.nan)
            for aggregation, values in (("macro", macro), ("pooled", pooled)):
                key = (model, obj, metric, aggregation)
                skills[key] = values
                full_skills[key] = score_grid(errors, baseline, count_grid,
                                               metric=metric, aggregation=aggregation)
    return skills, full_skills


def p2_bootstrap_fold(fold: int, cells: dict[tuple[str, str], dict[str, Any]],
                      baseline_abs: np.ndarray, baseline_sq: np.ndarray,
                      counts: np.ndarray, reps: int, rng: np.random.Generator
                      ) -> tuple[dict[tuple[str, str, str, str], np.ndarray],
                                 dict[tuple[str, str, str, str], float]]:
    key_order = [(model, obj) for model in P2_MODELS for obj in P2_OBJECTIVES]
    n = next(iter(cells.values()))["abs_error"].shape[0]
    draws = np.zeros((reps, n), dtype=np.int16)
    for b in range(reps):
        draws[b] = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.int16)
    b_counts = draws @ counts
    b_base_abs = draws @ baseline_abs
    b_base_sq = draws @ baseline_sq
    abs_cube = np.stack([cells[key]["abs_error"] for key in key_order], axis=-1)
    sq_cube = np.stack([cells[key]["sq_error"] for key in key_order], axis=-1)
    b_abs = (draws @ abs_cube.reshape(n, -1)).reshape(reps, N_VARIABLES, -1)
    b_sq = (draws @ sq_cube.reshape(n, -1)).reshape(reps, N_VARIABLES, -1)
    skills: dict[tuple[str, str, str, str], np.ndarray] = {}
    full: dict[tuple[str, str, str, str], float] = {}
    for i, (model, obj) in enumerate(key_order):
        for metric, errors, base, b_error, b_base in (
                ("L1", cells[(model, obj)]["abs_error"], baseline_abs, b_abs[:, :, i], b_base_abs),
                ("S2", cells[(model, obj)]["sq_error"], baseline_sq, b_sq[:, :, i], b_base_sq)):
            eligible = ((b_counts >= MIN_TEST_EVENTS) & (b_base > 1e-9)
                        & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))[None, :])
            ratio = np.divide(b_error, b_base, out=np.zeros_like(b_error), where=b_base > 1e-9)
            macro = np.full(reps, np.nan)
            for b in range(reps):
                if eligible[b].any():
                    macro[b] = np.mean(1.0 - ratio[b, eligible[b]])
            keep = ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))
            den = b_base[:, keep].sum(axis=1)
            pooled = np.where(den > 1e-9, 1.0 - b_error[:, keep].sum(axis=1) / den, np.nan)
            for aggregation, values in (("macro", macro), ("pooled", pooled)):
                key = (model, obj, metric, aggregation)
                skills[key] = values
                full[key] = score_grid(errors, base, counts, metric=metric,
                                       aggregation=aggregation)
    return skills, full


def load_p2_fold(fold: int, ledger: dict[str, Any], records_by_fold: dict[int, Any]) -> tuple[
        dict[tuple[str, str], dict[str, Any]], np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    patient_index: dict[str, int] = {}
    patient_labels: list[str] = []
    base_abs: np.ndarray | None = None
    base_sq: np.ndarray | None = None
    counts: np.ndarray | None = None
    patient_event: np.ndarray | None = None
    variable_event: np.ndarray | None = None
    raw_truth_event: np.ndarray | None = None
    cf_raw_event: np.ndarray | None = None
    event_count = -1
    for model in P2_MODELS:
        for objective in P2_OBJECTIVES:
            abs_seeds = []
            sq_seeds = []
            rmse_seeds: list[dict[int, tuple[float, int]]] = []
            for seed in P2_SEEDS:
                key = f"fold{fold}/{model}/{objective}/seed{seed}"
                record = ledger["run_results"][key]
                path = ROOT / record["path"]
                with np.load(path, allow_pickle=False) as z:
                    pred_std = z["prediction_standardized"].astype(np.float64)
                    pred_raw = z["prediction_raw"].astype(np.float64)
                    if not patient_labels:
                        event_patient_ids = z["event_patient_id"].astype(str)
                        unique = sorted(set(event_patient_ids.tolist()))
                        patient_labels.extend(unique)
                        patient_index.update({pid: i for i, pid in enumerate(unique)})
                        patient = np.fromiter((patient_index[x] for x in event_patient_ids),
                                              dtype=np.int64, count=event_patient_ids.size)
                        variable = z["event_variable_index"].astype(np.int64)
                        truth_std = z["truth_standardized"].astype(np.float64)
                        cf_std = z["carry_forward_standardized"].astype(np.float64)
                        truth_raw = z["truth_raw"].astype(np.float64)
                        patient_event, variable_event = patient, variable
                        raw_truth_event, cf_raw_event = truth_raw, z["carry_forward_raw"].astype(np.float64)
                        event_count = int(variable.size)
                        base_abs = grid_sum(patient, variable, np.abs(cf_std - truth_std), len(patient_labels))
                        base_sq = grid_sum(patient, variable, (cf_std - truth_std) ** 2, len(patient_labels))
                        counts = grid_sum(patient, variable, np.ones(variable.size), len(patient_labels))
                    elif pred_std.size != event_count or pred_raw.size != event_count:
                        raise AssertionError(f"P2 fold{fold}: prediction event count differs at {key}")
                    abs_seeds.append(grid_sum(patient, variable, np.abs(pred_std - truth_std), len(patient_labels)))
                    sq_seeds.append(grid_sum(patient, variable, (pred_std - truth_std) ** 2, len(patient_labels)))
                    sums: dict[int, tuple[float, int]] = {}
                    for var in np.unique(variable):
                        sel = variable == var
                        sums[int(var)] = (float(np.square(pred_raw[sel] - truth_raw[sel]).sum()), int(sel.sum()))
                    rmse_seeds.append(sums)
            cells[(model, objective)] = {
                "abs_error": np.mean(np.stack(abs_seeds), axis=0),
                "sq_error": np.mean(np.stack(sq_seeds), axis=0),
                "rmse_sse_by_seed": rmse_seeds,
                "n_seeds": len(P2_SEEDS),
            }
    assert (base_abs is not None and base_sq is not None and counts is not None
            and patient_event is not None and variable_event is not None
            and raw_truth_event is not None and cf_raw_event is not None)
    return (cells, base_abs, base_sq, counts, patient_event,
            {"variable_event": variable_event, "patient_labels": np.asarray(patient_labels),
             "truth_raw": raw_truth_event, "carry_forward_raw": cf_raw_event,
             "event_count": event_count})


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), int(valid.size)


def fmt(value: Any, digits: int = 4) -> str:
    return "--" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def figure_matrix(trackb_rows: list[dict[str, Any]], path_pdf: Path, path_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    cohort_labels = {"eicu": "eICU-CRD", "mimiciv": "MIMIC-IV"}
    panel_specs = [
        ("macro", "L1", "Macro absolute-error skill"),
        ("pooled", "L1", "Pooled absolute-error skill"),
        ("macro", "S2", "Macro squared-error skill"),
        ("pooled", "S2", "Pooled squared-error skill"),
    ]
    lookup = {(r["cohort"], r["model"], r["objective"], r["metric"], r["aggregation"]): r["skill"]
              for r in trackb_rows}
    values = [float(value) for value in lookup.values() if np.isfinite(value)]
    bound = max(0.05, max(abs(min(values)), abs(max(values)))) if values else 0.05
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 11.0), constrained_layout=False)
    plt.subplots_adjust(left=0.13, right=0.89, top=0.92, bottom=0.12,
                        hspace=0.48, wspace=0.27)
    image = None
    for row_i, (aggregation, metric, title) in enumerate(panel_specs):
        for col_i, cohort in enumerate(TRACKB_COHORTS):
            ax = axes[row_i, col_i]
            matrix = np.asarray([
                [lookup[(cohort, model, obj, metric, aggregation)] for obj in OBJECTIVES]
                for model in TRACKB_MODELS
            ], dtype=np.float64)
            image = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            ax.set_xticks(range(len(OBJECTIVES)), ["MSE", "Balanced\nMSE", "Masked L1"], fontsize=8)
            ax.set_yticks(range(len(TRACKB_MODELS)),
                          [DISPLAY_MODEL[m] for m in TRACKB_MODELS], fontsize=8)
            ax.set_title(f"{title} | {cohort_labels[cohort]}", fontsize=9, pad=6)
            for yi in range(len(TRACKB_MODELS)):
                for xi in range(len(OBJECTIVES)):
                    val = matrix[yi, xi]
                    color = "white" if abs(val) > bound * 0.57 else "black"
                    ax.text(xi, yi, fmt(val), ha="center", va="center", fontsize=8,
                            fontweight="semibold", color=color)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if image is not None:
        cax = fig.add_axes([0.25, 0.055, 0.50, 0.018])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.set_label("Persistence skill (positive = better than carry-forward)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Training objective × evaluation metric", fontsize=14, y=0.975,
                 fontweight="semibold")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def tex_s13(trackb_rows: list[dict[str, Any]], difference_rows: list[dict[str, Any]],
            p2_rows: list[dict[str, Any]], p2_c3_rows: list[dict[str, Any]],
            path: Path) -> None:
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}p{1.35cm}p{2.0cm}p{2.8cm}rrrr@{}}",
        r"\caption{Objective-by-metric matrix for the frozen patient-grouped Track B test predictions. Values are five-seed means. Macro scores equally weight variables with at least 50 events and a non-degenerate carry-forward denominator; pooled scores pool squared or absolute errors over the fixed 34 scoreable variables. On continuation pages, Result 1--4 refer to the current panel's columns. All newly computed cross-metric analyses are post hoc sensitivity analyses with reporting rules fixed before computation.}\label{tab:s13-objective-metric}\\",
        r"\toprule",
        r"\multicolumn{7}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{Panel A. Five-seed persistence skill by cohort, architecture, and training objective}\\",
        r"\midrule",
        r"Cohort & Model & Objective & Macro L1 & Pooled L1 & Macro $S^2$ & Pooled $S^2$ \\ ",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule Cohort & Model/metric & Objective/contrast & Result 1 & Result 2 & Result 3 & Result 4 \\ \midrule",
        r"\endhead",
    ]
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in trackb_rows:
        grouped.setdefault((row["cohort"], row["model"], row["objective"]), {})[
            f"{row['metric']}_{row['aggregation']}"] = row["skill"]
    cohort_label = {"eicu": "eICU-CRD", "mimiciv": "MIMIC-IV"}
    for cohort in TRACKB_COHORTS:
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                data = grouped[(cohort, model, objective)]
                nums = [fmt(data[key]) for key in ("L1_macro", "L1_pooled", "S2_macro", "S2_pooled")]
                lines.append(f"{cohort_label[cohort]} & {DISPLAY_MODEL[model]} & "
                             f"{OBJECTIVE_LABEL[objective]} & " + " & ".join(nums) + r" \\")
            lines.append(r"\addlinespace")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{Panel B. Paired patient-cluster bootstrap contrasts; entries are estimate and percentile 95\% CI}\\",
        r"\midrule",
        r"Cohort & Metric/scale & Paired contrast & Estimate & CI lower & CI upper & Valid draws \\ ",
        r"\midrule",
    ]
    cohort_label = {"eicu": "eICU-CRD", "mimiciv": "MIMIC-IV"}
    for row in difference_rows:
        metric = "L1" if row["metric"] == "absolute-error skill" else "$S^2$"
        scale = "macro" if row["aggregation"] == "macro" else "pooled"
        if row["comparison_type"] == "objective_effect":
            contrast = f"L1 $-$ MSE; {DISPLAY_MODEL[row['model_or_left']]}"
        else:
            contrast = (f"{DISPLAY_MODEL[row['model_or_left']]} $-$ "
                        f"{DISPLAY_MODEL[row['right']]}; {row['objective_or_contrast']}")
        lines.append(
            f"{cohort_label[row['cohort']]} & {metric} {scale} & {contrast} & "
            f"{fmt(row['estimate'])} & {fmt(row['ci_low'])} & {fmt(row['ci_high'])} & "
            f"{row['valid_replicates']}" + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{Panel C. P2 per-fold, per-model, per-objective persistence skill}\\",
        r"\midrule",
        r"Cohort & Model/fold & Objective & Macro L1 & Pooled L1 & Macro $S^2$ & Pooled $S^2$ \\ ",
        r"\midrule",
    ]
    p2_grouped: dict[tuple[int, str, str], dict[str, float]] = {}
    for row in p2_rows:
        p2_grouped.setdefault((row["fold"], row["model"], row["objective"]), {})[
            f"{row['metric']}_{row['aggregation']}"] = row["skill"]
    for fold in range(5):
        for model in P2_MODELS:
            for objective in P2_OBJECTIVES:
                data = p2_grouped[(fold, model, objective)]
                nums = [fmt(data[key]) for key in ("L1_macro", "L1_pooled", "S2_macro", "S2_pooled")]
                lines.append(f"P2 eICU & fold{fold} {DISPLAY_MODEL[model]} & "
                             f"{OBJECTIVE_LABEL[objective]} & " + " & ".join(nums) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{p{\dimexpr\textwidth-2\tabcolsep\relax}}{Panel D. P2 C3 summaries; fold-specific medians followed by the median across five held-hospital folds}\\",
        r"\midrule",
        r"Cohort & Metric/scale & C3 contrast & Estimate & CI lower & CI upper & Valid draws \\ ",
        r"\midrule",
    ]
    for row in p2_c3_rows:
        metric = "L1" if row["metric"] == "L1" else "$S^2$"
        scale = "macro" if row["aggregation"] == "macro" else "pooled"
        lines.append(
            f"P2 eICU & {metric} {scale} & {row['objective']} & "
            f"{fmt(row['estimate_median_across_folds'])} & {fmt(row['ci_low'])} & "
            f"{fmt(row['ci_high'])} & {row['valid_replicates']}" + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\normalsize",
        r"\begin{figure}[htbp]\centering",
        r"\includegraphics[width=0.98\linewidth]{S13_objective_metric_matrix.pdf}",
        r"\caption{Training objective by evaluation metric on frozen, patient-grouped Track B test predictions. Each heatmap cell is the five-seed mean persistence skill, with color centered at zero. The squared-error score is the MSE persistence skill $S^2_v=1-\sum(\hat y-y)^2/\sum(x_{\mathrm{last}}-y)^2$. Analyses are post hoc sensitivity analyses with reporting rules fixed before computation.}",
        r"\label{fig:s13-objective-metric}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_existing(refresh: bool, *, include_preflight: bool = True) -> None:
    targets = [ROOT / path for path in GENERATED_RELATIVE]
    if include_preflight:
        targets.append(PREFLIGHT_PATH)
    existing = [path for path in targets if path.exists()]
    if not existing:
        return
    if not refresh:
        raise FileExistsError(
            "R4 outputs already exist; pass --refresh-generated to archive them before regenerating"
        )
    archive = ROOT / "results" / "jbi" / "_archive" / (
        "r4_objective_metric_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    for path in existing:
        dest = archive / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    write_json(archive / "archive_manifest.json", {
        "archived_at_utc": utc_now(),
        "reason": "--refresh-generated requested for R4 objective-by-metric analysis",
        "files": [str(path.relative_to(ROOT)) for path in existing],
    })


def _bootstrap_trackb_differences(cohort: str,
                                  bootstrap: dict[tuple[str, str, str, str], np.ndarray],
                                  full: dict[tuple[str, str, str, str], float]) -> list[dict[str, Any]]:
    rows = []
    metrics = ("L1", "S2")
    aggregations = ("macro", "pooled")
    for metric in metrics:
        for aggregation in aggregations:
            for model in TRACKB_MODELS:
                masked = bootstrap[(model, "MAE", metric, aggregation)]
                mse = bootstrap[(model, "MSE", metric, aggregation)]
                delta = masked - mse
                low, high, n_valid = percentile_ci(delta)
                rows.append({
                    "cohort": cohort, "analysis": "Track B macro-checkpoint",
                    "metric": "absolute-error skill" if metric == "L1" else "squared-error skill S2",
                    "aggregation": aggregation, "comparison_type": "objective_effect",
                    "objective_or_contrast": "masked L1 minus MSE",
                    "model_or_left": model, "right": "MSE-trained same architecture",
                    "estimate": full[(model, "MAE", metric, aggregation)]
                    - full[(model, "MSE", metric, aggregation)],
                    "ci_low": low, "ci_high": high,
                    "bootstrap_reps": N_BOOTSTRAP, "valid_replicates": n_valid,
                    "bootstrap_unit": "patient", "bootstrap_seed": BOOTSTRAP_SEED,
                })
            for objective in OBJECTIVES:
                for comparator in COMPARATORS:
                    delta = (bootstrap[("ICUKANForecaster", objective, metric, aggregation)]
                             - bootstrap[(comparator, objective, metric, aggregation)])
                    low, high, n_valid = percentile_ci(delta)
                    rows.append({
                        "cohort": cohort, "analysis": "Track B macro-checkpoint",
                        "metric": "absolute-error skill" if metric == "L1" else "squared-error skill S2",
                        "aggregation": aggregation, "comparison_type": "architecture_difference",
                        "objective_or_contrast": OBJECTIVE_LABEL[objective],
                        "model_or_left": "ICUKANForecaster",
                        "right": comparator, "estimate": full[("ICUKANForecaster", objective, metric, aggregation)]
                        - full[(comparator, objective, metric, aggregation)],
                        "ci_low": low, "ci_high": high,
                        "bootstrap_reps": N_BOOTSTRAP, "valid_replicates": n_valid,
                        "bootstrap_unit": "patient", "bootstrap_seed": BOOTSTRAP_SEED,
                    })
    return rows


def _assemble(args: argparse.Namespace) -> None:
    started = time.time()
    reader = reader_module()
    trackb_cells: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    trackb_refs: dict[str, dict[str, Any]] = {}
    trackb_rows: list[dict[str, Any]] = []
    trackb_diff_rows: list[dict[str, Any]] = []
    raw_rmse_rows: list[dict[str, Any]] = []
    pooled_checkpoint_rows: list[dict[str, Any]] = []
    for cohort in TRACKB_COHORTS:
        ref = build_trackb_reference(cohort, reader)
        trackb_refs[cohort] = ref
        summary_path = ROOT / ("results/jbi/trackb_controls/e2_trackb_controls_summary.csv"
                               if cohort == "eicu" else
                               "results/jbi/trackb_controls_mimiciv/e2_trackb_controls_summary.csv")
        frozen_summary = pd.read_csv(summary_path)
        frozen_summary = frozen_summary[frozen_summary.phase == "factorial"].copy()
        cells: dict[tuple[str, str], dict[str, Any]] = {}
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                row = next(r for r in load_json(ROOT / ("results/jbi/trackb_controls/run_manifest.json"
                    if cohort == "eicu" else "results/jbi/trackb_controls_mimiciv/run_manifest.json"))["runs"]
                    if r.get("phase") == "factorial" and r.get("model") == model
                    and r.get("loss") == objective and r.get("selection") == "macro_persistence")
                cell = load_trackb_cell(cohort, model, objective, row["tag"], ref, reader)
                cells[(model, objective)] = cell
                for metric in ("L1", "S2"):
                    errors = cell["abs_error"] if metric == "L1" else cell["sq_error"]
                    baseline = cell["baseline_abs"] if metric == "L1" else cell["baseline_sq"]
                    for aggregation in ("macro", "pooled"):
                        value = score_grid(errors, baseline, cell["counts"], metric=metric,
                                           aggregation=aggregation)
                        trackb_rows.append({
                            "cohort": cohort, "model": model, "model_display": DISPLAY_MODEL[model],
                            "objective": objective, "objective_label": OBJECTIVE_LABEL[objective],
                            "metric": metric, "metric_label": "absolute-error skill" if metric == "L1"
                            else "squared-error persistence skill", "aggregation": aggregation,
                            "skill": value, "event_count": int(cell["counts"].sum()),
                            "patient_count": int(ref["n_patients"]),
                            "variables_in_macro": int(((cell["counts"].sum(axis=0) >= MIN_TEST_EVENTS)
                                & (baseline.sum(axis=0) > 1e-9)
                                & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))).sum()),
                            "seed_count": cell["n_seeds"], "checkpoint_selection": "macro_persistence",
                        })
                # Per-variable raw RMSE is descriptive; no cross-variable mean or inference.
                for var in range(N_VARIABLES):
                    if var in UNSCORED:
                        continue
                    sums = [entry.get(var, (0.0, 0))[0] for entry in cell["rmse_sse_by_seed"]]
                    counts = [entry.get(var, (0.0, 0))[1] for entry in cell["rmse_sse_by_seed"]]
                    count = counts[0] if counts else 0
                    per_seed = [math.sqrt(s / c) if c else float("nan")
                                for s, c in zip(sums, counts)]
                    finite = np.asarray([v for v in per_seed if np.isfinite(v)], dtype=float)
                    raw_rmse_rows.append({
                        "source": "Track B", "cohort_or_fold": cohort, "model": model,
                        "objective": objective, "variable_index": var,
                        "variable_name": ref["raw_meta"].get("variable_names", [])[var]
                        if ref["raw_meta"].get("variable_names") else f"variable_{var}",
                        "event_count": count, "seed_count": int(finite.size),
                        "raw_rmse_seed_mean": float(finite.mean()) if finite.size else float("nan"),
                        "raw_rmse_seed_sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                        "unit": "original clinical unit", "inferential_comparison": False,
                    })
        # Required L1 macro checkpoint reproduction against Table S1 source.
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                cell = cells[(model, objective)]
                macro = score_grid(cell["abs_error"], cell["baseline_abs"], cell["counts"],
                                   metric="L1", aggregation="macro")
                pooled = score_grid(cell["abs_error"], cell["baseline_abs"], cell["counts"],
                                    metric="L1", aggregation="pooled")
                source = frozen_summary[(frozen_summary.model == model)
                                        & (frozen_summary.cell == OBJECTIVE_CELL[objective])]
                if len(source) != 1:
                    raise AssertionError(f"{cohort}/{model}/{objective}: Table S1 source row missing/duplicated")
                ref_macro = float(source.iloc[0].macro_mean)
                if abs(macro - ref_macro) >= 1e-6:
                    raise AssertionError(f"{cohort}/{model}/{objective}: L1 macro differs from Table S1 by {macro-ref_macro}")
                trackb_cells[cohort] = cells
        # Pooled-checkpoint sensitivity is an existing frozen summary, not a new selection.
        for model in TRACKB_MODELS[:3]:
            mse_cell = "mse_pooled"
            l1_cell = "mae_pooled"
            mse_rows = frozen_summary[(frozen_summary.model == model) & (frozen_summary.cell == mse_cell)]
            l1_rows = frozen_summary[(frozen_summary.model == model) & (frozen_summary.cell == l1_cell)]
            if len(mse_rows) != 1 or len(l1_rows) != 1:
                raise AssertionError(f"{cohort}/{model}: frozen pooled checkpoint evidence missing")
            pooled_checkpoint_rows.append({
                "cohort": cohort, "model": model,
                "checkpoint_selection": "pooled_mse (frozen validation-only selection)",
                "masked_L1_macro_skill": float(l1_rows.iloc[0].macro_mean),
                "MSE_macro_skill": float(mse_rows.iloc[0].macro_mean),
                "masked_L1_minus_MSE_macro_skill": float(l1_rows.iloc[0].macro_mean - mse_rows.iloc[0].macro_mean),
                "masked_L1_pooled_skill": float(l1_rows.iloc[0].pooled_mean),
                "MSE_pooled_skill": float(mse_rows.iloc[0].pooled_mean),
                "masked_L1_minus_MSE_pooled_skill": float(l1_rows.iloc[0].pooled_mean - mse_rows.iloc[0].pooled_mean),
                "source": str(summary_path.relative_to(ROOT)),
            })
        boot, full = trackb_bootstrap(ref, cells, N_BOOTSTRAP, BOOTSTRAP_SEED)
        trackb_diff_rows.extend(_bootstrap_trackb_differences(cohort, boot, full))

    # P2's complete 90-run freeze is re-read only from the verified NPZ set.
    p2_ledger = load_json(ROOT / "results/jbi/p2_loho_test_inference_ledger.json")
    p2_rows: list[dict[str, Any]] = []
    p2_c3_rows: list[dict[str, Any]] = []
    p2_folds: dict[int, dict[tuple[str, str, str, str], float]] = {}
    p2_boot_draws: dict[int, dict[tuple[str, str, str, str], np.ndarray]] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for fold in range(5):
        cells, base_abs, base_sq, counts, patient_event, meta = load_p2_fold(
            fold, p2_ledger, {}
        )
        boot, full = p2_bootstrap_fold(fold, cells, base_abs, base_sq, counts,
                                       N_BOOTSTRAP, rng)
        p2_folds[fold] = full
        p2_boot_draws[fold] = boot
        for (model, objective, metric, aggregation), value in full.items():
            p2_rows.append({
                "fold": fold, "model": model, "model_display": DISPLAY_MODEL[model],
                "objective": objective, "objective_label": OBJECTIVE_LABEL[objective],
                "metric": metric, "aggregation": aggregation, "skill": value,
                "event_count": int(counts.sum()),
                "patient_count": int(len(meta["patient_labels"])),
                "variables_in_macro": int(((counts.sum(axis=0) >= MIN_TEST_EVENTS)
                    & (base_sq.sum(axis=0) > 1e-9)
                    & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))).sum()),
                "seed_count": 3, "source": "frozen P2 eICU internal held-hospital test fold",
            })
        # Raw RMSE is reported variable-by-variable and never averaged across variables.
        for (model, objective), cell in cells.items():
            for var in range(N_VARIABLES):
                if var in UNSCORED:
                    continue
                sums = [entry.get(var, (0.0, 0))[0] for entry in cell["rmse_sse_by_seed"]]
                counts_var = [entry.get(var, (0.0, 0))[1] for entry in cell["rmse_sse_by_seed"]]
                per_seed = [math.sqrt(s / c) if c else float("nan")
                            for s, c in zip(sums, counts_var)]
                finite = np.asarray([v for v in per_seed if np.isfinite(v)], dtype=float)
                p2_rows.append({}) if False else None
                raw_rmse_rows.append({
                    "source": "P2 eICU held-hospital", "cohort_or_fold": f"fold{fold}",
                    "model": model, "objective": objective, "variable_index": var,
                    "variable_name": "", "event_count": counts_var[0] if counts_var else 0,
                    "seed_count": int(finite.size),
                    "raw_rmse_seed_mean": float(finite.mean()) if finite.size else float("nan"),
                    "raw_rmse_seed_sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                    "unit": "original clinical unit", "inferential_comparison": False,
                })

    # P2 C3 aggregates within each fold first, then takes the median across folds.
    c3_point: dict[tuple[str, str, str], list[float]] = {}
    c3_boot: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for metric in ("L1", "S2"):
        for aggregation in ("macro", "pooled"):
            key = (metric, aggregation, "objective_effect")
            c3_point[key] = []
            c3_boot[key] = []
            for fold in range(5):
                full = p2_folds[fold]
                vals = [abs(full[(model, "MAE", metric, aggregation)]
                            - full[(model, "MSE", metric, aggregation)]) for model in P2_MODELS]
                c3_point[key].append(float(np.median(vals)))
                boot = p2_boot_draws[fold]
                rep_vals = []
                for model in P2_MODELS:
                    rep_vals.append(np.abs(boot[(model, "MAE", metric, aggregation)]
                                           - boot[(model, "MSE", metric, aggregation)]))
                c3_boot[key].append(np.median(np.stack(rep_vals, axis=1), axis=1))
            joined = np.median(np.stack(c3_boot[key], axis=1), axis=1)
            lo, hi, n_valid = percentile_ci(joined)
            p2_c3_rows.append({
                "metric": metric, "aggregation": aggregation,
                "comparison_type": "objective_effect", "objective": "median absolute masked L1 minus MSE",
                "estimate_median_across_folds": float(np.median(c3_point[key])),
                "ci_low": lo, "ci_high": hi, "bootstrap_reps": N_BOOTSTRAP,
                "valid_replicates": n_valid, "bootstrap_unit": "patient within held-hospital fold",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "fold_level_values": json.dumps(c3_point[key]),
            })
            for objective in P2_OBJECTIVES:
                key = (metric, aggregation, f"architecture_gap_{objective}")
                c3_point[key] = []
                c3_boot[key] = []
                pairs = [(P2_MODELS[i], P2_MODELS[j]) for i in range(len(P2_MODELS))
                         for j in range(i + 1, len(P2_MODELS))]
                for fold in range(5):
                    full = p2_folds[fold]
                    values = [abs(full[(left, objective, metric, aggregation)]
                                  - full[(right, objective, metric, aggregation)])
                              for left, right in pairs]
                    c3_point[key].append(float(np.median(values)))
                    boot = p2_boot_draws[fold]
                    rep_vals = [np.abs(boot[(left, objective, metric, aggregation)]
                                       - boot[(right, objective, metric, aggregation)])
                                for left, right in pairs]
                    c3_boot[key].append(np.median(np.stack(rep_vals, axis=1), axis=1))
                joined = np.median(np.stack(c3_boot[key], axis=1), axis=1)
                lo, hi, n_valid = percentile_ci(joined)
                p2_c3_rows.append({
                    "metric": metric, "aggregation": aggregation,
                    "comparison_type": "architecture_gap", "objective": OBJECTIVE_LABEL[objective],
                    "estimate_median_across_folds": float(np.median(c3_point[key])),
                    "ci_low": lo, "ci_high": hi, "bootstrap_reps": N_BOOTSTRAP,
                    "valid_replicates": n_valid, "bootstrap_unit": "patient within held-hospital fold",
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "fold_level_values": json.dumps(c3_point[key]),
                })

    # Attach variable labels from the frozen P2 prediction files.
    with np.load(ROOT / "results/jbi/p2_loho_predictions/fold0/ICUKANForecaster/MAE/seed2024.npz",
                 allow_pickle=False) as z:
        p2_variable_names = z["variable_names"].astype(str).tolist()
    trackb_variable_names = trackb_refs["eicu"]["raw_meta"]["variable_names"]
    for row in raw_rmse_rows:
        if row["source"].startswith("P2"):
            row["variable_name"] = p2_variable_names[int(row["variable_index"])]
        else:
            row["variable_name"] = trackb_variable_names[int(row["variable_index"])]

    trackb_df = pd.DataFrame(trackb_rows)
    p2_df = pd.DataFrame(p2_rows)
    differences_df = pd.DataFrame(trackb_diff_rows)
    c3_df = pd.DataFrame(p2_c3_rows)
    rmse_df = pd.DataFrame(raw_rmse_rows)
    pooled_df = pd.DataFrame(pooled_checkpoint_rows)

    # A short, deterministic report drives the registered manuscript wording.
    key_s2 = differences_df[(differences_df.metric == "squared-error skill S2")
                            & (differences_df.aggregation == "macro")
                            & (differences_df.comparison_type == "objective_effect")]
    mse_arch = differences_df[(differences_df.metric == "squared-error skill S2")
                              & (differences_df.aggregation == "macro")
                              & (differences_df.comparison_type == "architecture_difference")
                              & (differences_df.objective_or_contrast == "MSE")]
    summary = {
        "registered_interpretation": "objective-metric alignment is evaluated against squared-error persistence skill; macro L1 remains the primary endpoint",
        "trackb_s2_objective_effect_by_cohort": {},
        "trackb_s2_mse_architecture_differences_by_cohort": {},
        "pooled_checkpoint_masked_l1_minus_mse_macro_ranges": {},
        "p2_c3": c3_df.to_dict(orient="records"),
    }
    for cohort in TRACKB_COHORTS:
        rows = key_s2[key_s2.cohort == cohort]
        summary["trackb_s2_objective_effect_by_cohort"][cohort] = rows[[
            "model_or_left", "estimate", "ci_low", "ci_high"]].to_dict(orient="records")
        rows = mse_arch[mse_arch.cohort == cohort]
        summary["trackb_s2_mse_architecture_differences_by_cohort"][cohort] = rows[[
            "model_or_left", "right", "estimate", "ci_low", "ci_high"]].to_dict(orient="records")
        rows = pooled_df[pooled_df.cohort == cohort]
        summary["pooled_checkpoint_masked_l1_minus_mse_macro_ranges"][cohort] = {
            "min": float(rows.masked_L1_minus_MSE_macro_skill.min()),
            "max": float(rows.masked_L1_minus_MSE_macro_skill.max()),
            "models": rows.model.tolist(),
        }
    summary["counts"] = {
        "trackb_macro_cells": int(len(trackb_df) / 4),
        "trackb_metric_rows": int(len(trackb_df)),
        "p2_metric_rows": int(len(p2_df)),
        "trackb_paired_difference_rows": int(len(differences_df)),
        "p2_c3_rows": int(len(c3_df)),
        "per_variable_rmse_rows": int(len(rmse_df)),
        "trackb_macro_checkpoint_seeds": 5,
        "p2_runs": 90,
        "bootstrap_reps": N_BOOTSTRAP,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="r4_stage_", dir=OUT))
    try:
        files = {
            "trackb": stage / "r4_trackb_metrics.csv",
            "p2": stage / "r4_p2_metrics.csv",
            "differences": stage / "r4_paired_differences.csv",
            "p2_c3": stage / "r4_p2_c3_summary.csv",
            "rmse": stage / "r4_per_variable_raw_rmse.csv",
            "pooled": stage / "r4_pooled_checkpoint_evidence.csv",
            "report": stage / "r4_report.json",
            "tex": stage / "r4_S13_objective_metric_matrix.tex",
            "figure_pdf": stage / "S13_objective_metric_matrix.pdf",
            "figure_png": stage / "S13_objective_metric_matrix.png",
        }
        trackb_df.to_csv(files["trackb"], index=False, float_format="%.17g")
        p2_df.to_csv(files["p2"], index=False, float_format="%.17g")
        differences_df.to_csv(files["differences"], index=False, float_format="%.17g")
        c3_df.to_csv(files["p2_c3"], index=False, float_format="%.17g")
        rmse_df.to_csv(files["rmse"], index=False, float_format="%.17g")
        pooled_df.to_csv(files["pooled"], index=False, float_format="%.17g")
        write_json(files["report"], summary)
        figure_matrix(trackb_rows, files["figure_pdf"], files["figure_png"])
        tex_s13(trackb_rows, trackb_diff_rows, p2_rows, p2_c3_rows, files["tex"])
        manifest = {
            "schema_version": "jamia_r4_objective_metric_manifest_v1",
            "status": "COMPLETE",
            "generated_at_utc": utc_now(),
            "analysis_plan_sha256": sha256_file(PLAN_PATH),
            "preflight_manifest_sha256": sha256_file(PREFLIGHT_PATH),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "plan_status": "post hoc sensitivity analysis; reporting rules fixed before computation",
            "estimands": {
                "square_skill": "1 - sum((prediction-truth)^2) / sum((carry-forward-truth)^2)",
                "absolute_skill": "1 - sum(abs(prediction-truth)) / sum(abs(carry-forward-truth))",
                "macro": "equal mean over fixed scored variables with >=50 events and non-degenerate carry-forward denominator",
                "pooled": "pooled error sums over fixed 34 scored variables",
                "trackb_seed_aggregation": "equal mean of five seed error numerators then skill; common fixed denominator",
                "raw_rmse": "per-variable, per-seed raw-unit RMSE then seed mean; descriptive only",
                "p2_c3": "within-fold medians of model-level absolute effects, followed by median across five folds",
            },
            "bootstrap": {"reps": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                          "cluster": "patient", "ci": "percentile 95%"},
            "input_files": load_json(PREFLIGHT_PATH)["input_hashes"],
            "preflight_counts": load_json(PREFLIGHT_PATH)["counts"],
            "outputs": {},
        }
        for name, path in files.items():
            manifest["outputs"][name] = {"staged_filename": path.name,
                                         "size_bytes": path.stat().st_size,
                                         "sha256": sha256_file(path)}
        write_json(stage / "r4_manifest.json", manifest)
        # Every output is published only after all computations and serialization succeed.
        final_map = {
            files["trackb"]: OUT / "r4_trackb_metrics.csv",
            files["p2"]: OUT / "r4_p2_metrics.csv",
            files["differences"]: OUT / "r4_paired_differences.csv",
            files["p2_c3"]: OUT / "r4_p2_c3_summary.csv",
            files["rmse"]: OUT / "r4_per_variable_raw_rmse.csv",
            files["pooled"]: OUT / "r4_pooled_checkpoint_evidence.csv",
            files["report"]: OUT / "r4_report.json",
            files["tex"]: OUT / "r4_S13_objective_metric_matrix.tex",
            files["figure_pdf"]: PAPER_OUT / "S13_objective_metric_matrix.pdf",
            files["figure_png"]: PAPER_OUT / "S13_objective_metric_matrix.png",
            stage / "r4_manifest.json": OUT / "r4_manifest.json",
        }
        for source, target in final_map.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps({"status": "COMPLETE", "report": summary,
                      "outputs": [str(ROOT / p) for p in GENERATED_RELATIVE]},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate/hash all frozen inputs without computing any metric")
    parser.add_argument("--run", action="store_true", help="compute registered R4 results")
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive existing R4 outputs before regeneration")
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("choose exactly one of --preflight-only or --run")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        if PREFLIGHT_PATH.exists() and not args.refresh_generated:
            parser.error("preflight manifest exists; use --refresh-generated to archive old R4 outputs")
        if args.refresh_generated:
            archive_existing(True)
        result = preflight()
        write_json(PREFLIGHT_PATH, result)
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "input_hash_count": result["input_hash_count"],
                          "failures": result["failures"],
                          "preflight_manifest": str(PREFLIGHT_PATH.relative_to(ROOT))},
                         ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    if not PREFLIGHT_PATH.is_file():
        raise SystemExit("R4 analysis requires a successful --preflight-only run first")
    saved = load_json(PREFLIGHT_PATH)
    stale_inputs = []
    for entry in saved.get("input_hashes", []):
        path = ROOT / entry["path"]
        if (not path.is_file() or path.stat().st_size != entry.get("size_bytes")
                or path.stat().st_mtime_ns != entry.get("mtime_ns")):
            stale_inputs.append(entry["path"])
    if (saved.get("status") != "PASS"
            or saved.get("script_sha256") != sha256_file(Path(__file__).resolve())
            or saved.get("analysis_plan_sha256") != sha256_file(PLAN_PATH)
            or stale_inputs):
        raise SystemExit("R4 preflight is stale or failed; no metric calculation was started")
    archive_existing(args.refresh_generated, include_preflight=False)
    _assemble(args)


if __name__ == "__main__":
    main()
