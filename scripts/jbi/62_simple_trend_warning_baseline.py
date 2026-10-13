#!/usr/bin/env python
"""Registered P1 simple-trend warning baseline (revision task R7).

The preflight binds the frozen revision plan, P1 plan, one-pass test ledger,
prediction artifacts, event caches, and existing P1 summaries before trend
grades are calculated. The trend comparator is descriptive and does not alter
the registered P1 I1/I2 decisions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from output_safety import prepare_generation


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi" / "revision_sensitivity"
PAPER_OUT = ROOT / "results" / "paper"
SCRIPT_PATH = Path(__file__).resolve()
REVISION_PLAN_PATH = ROOT / "paper" / "JAMIA修订补充分析计划.md"
REVISION_PLAN_MANIFEST_PATH = OUT / "analysis_plan_manifest.json"
P1_PLAN_PATH = ROOT / "paper" / "预设用例分析计划_P1.md"
P1_EXTRACTION_MANIFEST_PATH = ROOT / "results" / "jbi" / "p1_landmark_extraction_manifest.json"
P1_FIDELITY_PATH = ROOT / "results" / "jbi" / "p1_landmark_cache_fidelity.json"
P1_FIDELITY_SUMMARY_PATH = ROOT / "results" / "jbi" / "p1_landmark_cache_fidelity_summary.json"
P1_LEDGER_PATH = ROOT / "results" / "jbi" / "p1_landmark_test_inference_ledger.json"
P1_BATCH_PATH = ROOT / "results" / "jbi" / "p1_landmark_test_batch_manifest.json"
P1_FREEZE_PATH = ROOT / "results" / "jbi" / "p1_landmark_checkpoint_freeze.json"
P1_VERDICT_PATH = ROOT / "results" / "jbi" / "p1_landmark_verdict.json"
P1_METRICS_PATH = ROOT / "results" / "jbi" / "p1_landmark_metrics.csv"
P1_NB_PATH = ROOT / "results" / "jbi" / "p1_landmark_decision_curve.csv"
P1_GRADE_SCRIPT_PATH = ROOT / "scripts" / "jbi" / "41_landmark_use_case.py"
P1_EXTRACTION_SCRIPT_PATH = ROOT / "scripts" / "jbi" / "39_extract_landmark_events.py"
P1_INFERENCE_SCRIPT_PATH = ROOT / "scripts" / "jbi" / "51_run_p1_test_once.py"
P1_FAILED_TEST_ATTEMPT = Path(os.environ.get(
    "ICU_KAN_R7_P1_FAILED_ATTEMPT",
    str(ROOT / "results/jbi/_work/p1_landmark_test_failed_attempts/attempt01"),
))
R7_FAILED_ATTEMPT_ARCHIVE = Path(os.environ.get(
    "ICU_KAN_R7_FAILED_ATTEMPT_ARCHIVE",
    str(OUT / "_archive/r7_attempt/archive_manifest.json"),
))
PREFLIGHT_PATH = OUT / "r7_preflight_manifest.json"
EXPECTED_REVISION_PLAN_SHA256 = "fade749e20c54e9340dbbb01ec9a6c8e12a8ea912eb34baec7ccced30e38e320"
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
DISPLAY_MODEL = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi", "HyperIMTS": "HyperIMTS"}
MODEL_COLORS = {"ICUKANForecaster": "#176B87", "GraFITi": "#9A5B13", "HyperIMTS": "#626F47"}
TREND_COLOR = "#B23A48"
STRATEGY_ORDER = ("no-alarm", "alert-all", "trend", *MODELS)
MIN_INTERVAL_HOURS = 1.0
MAX_EXTRAPOLATION_HOURS = 24.0
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260914
MIN_SUPERIOR_THRESHOLDS = 14

OUTPUTS = {
    "window_audit": OUT / "r7_window_audit_counts.csv",
    "summary": OUT / "r7_trend_summary.csv",
    "curves": OUT / "r7_trend_net_benefit_curves.csv",
    "report": OUT / "r7_report.json",
    "table_tex": OUT / "r7_S15_trend_summary.tex",
    "figure_pdf": PAPER_OUT / "S15_r7_simple_trend_baseline.pdf",
    "figure_png": PAPER_OUT / "S15_r7_simple_trend_baseline.png",
    "manifest": OUT / "r7_manifest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_p1_plan_sha256() -> str:
    raw = P1_PLAN_PATH.read_bytes()
    marker = "\n## 偏离与结果解释说明".encode("utf-8")
    if raw.count(marker) != 1:
        raise RuntimeError("P1 plan must retain its single append-only deviation section")
    frozen_body = raw.split(marker, 1)[0]
    return hashlib.sha256(frozen_body).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_record(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def expected_run_keys() -> set[str]:
    return {
        f"{cohort}/{model}/seed{seed}"
        for cohort in COHORTS
        for model in MODELS
        for seed in (2024, 2025, 2026, 2027, 2028)
    }


def prediction_path(record: dict[str, Any]) -> Path:
    path = (ROOT / str(record.get("path", ""))).resolve()
    if ROOT not in path.parents:
        raise RuntimeError(f"P1 prediction path escapes project root: {path}")
    return path


def import_p1_grader():
    spec = importlib.util.spec_from_file_location("p1_landmark_use_case_r7", P1_GRADE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen P1 grade and outcome implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_input_paths() -> list[tuple[Path, str]]:
    paths = [
        (REVISION_PLAN_PATH, "frozen_revision_analysis_plan"),
        (REVISION_PLAN_MANIFEST_PATH, "revision_plan_registration"),
        (P1_PLAN_PATH, "frozen_P1_analysis_plan"),
        (P1_EXTRACTION_MANIFEST_PATH, "P1_event_extraction_manifest"),
        (P1_FIDELITY_PATH, "P1_cache_fidelity_audit"),
        (P1_FIDELITY_SUMMARY_PATH, "P1_cache_fidelity_summary"),
        (P1_LEDGER_PATH, "P1_test_inference_ledger"),
        (P1_BATCH_PATH, "P1_test_batch_manifest"),
        (P1_FREEZE_PATH, "P1_checkpoint_freeze"),
        (P1_VERDICT_PATH, "P1_frozen_verdict"),
        (P1_METRICS_PATH, "P1_metrics"),
        (P1_NB_PATH, "P1_decision_curve"),
        (P1_GRADE_SCRIPT_PATH, "frozen_P1_grade_and_outcome_code"),
        (P1_EXTRACTION_SCRIPT_PATH, "P1_event_extraction_code"),
        (P1_INFERENCE_SCRIPT_PATH, "P1_test_inference_code"),
        (R7_FAILED_ATTEMPT_ARCHIVE, "R7_failed_prepublication_attempt_archive"),
        (P1_FAILED_TEST_ATTEMPT / "archive_metadata.json", "P1_prior_failed_test_attempt_metadata"),
        (P1_FAILED_TEST_ATTEMPT / "failed_ledger.json", "P1_prior_failed_test_attempt_ledger"),
        (ROOT / "data/processed/eicu_events_landmark_v1.pkl", "eicu_raw_event_cache"),
        (ROOT / "data/processed/mimiciv_events_landmark_v1.pkl", "mimiciv_raw_event_cache"),
    ]
    return paths


def preflight() -> dict[str, Any]:
    started = time.time()
    failures: list[str] = []
    input_hashes: list[dict[str, Any]] = []
    counts: dict[str, Any] = {"cohorts": 0, "completed_runs": 0, "planned_runs": 30, "prediction_files": 0}
    try:
        p1 = import_p1_grader()
        revision_sha = sha256_file(REVISION_PLAN_PATH)
        plan_registration = load_json(REVISION_PLAN_MANIFEST_PATH)
        if (
            revision_sha != EXPECTED_REVISION_PLAN_SHA256
            or plan_registration.get("status") != "RULES_FROZEN_BEFORE_NEW_COMPUTATION"
            or plan_registration.get("plan_sha256") != EXPECTED_REVISION_PLAN_SHA256
            or "R7" not in plan_registration.get("frozen_tasks", [])
        ):
            failures.append("R7 is not bound to the registered frozen revision-plan hash/task")
        p1_plan_sha = frozen_p1_plan_sha256()
        extraction = load_json(P1_EXTRACTION_MANIFEST_PATH)
        fidelity = load_json(P1_FIDELITY_PATH)
        fidelity_summary = load_json(P1_FIDELITY_SUMMARY_PATH)
        ledger = load_json(P1_LEDGER_PATH)
        batch = load_json(P1_BATCH_PATH)
        freeze = load_json(P1_FREEZE_PATH)
        verdict = load_json(P1_VERDICT_PATH)
        if extraction.get("analysis_plan_sha256") != p1_plan_sha:
            failures.append("P1 extraction manifest is bound to a different P1 plan")
        if (
            ledger.get("analysis_plan_sha256") != p1_plan_sha
            or batch.get("analysis_plan_sha256") != p1_plan_sha
            or freeze.get("analysis_plan_sha256") != p1_plan_sha
        ):
            failures.append("P1 test ledger/batch does not match the frozen P1 plan hash")
        if ledger.get("status") != "COMPLETE" or ledger.get("test_inference_consumed") is not True:
            failures.append("P1 one-pass test inference ledger is not complete")
        if ledger.get("test_inference_passes_authorized") != 1:
            failures.append("P1 ledger does not attest exactly one authorized inference pass")
        if (
            freeze.get("status") != "FROZEN_WITH_NOT_COMPLETED_CELLS"
            or freeze.get("planned_run_count") != 30
            or freeze.get("run_count") != 28
            or freeze.get("checkpoint_set_sha256") != ledger.get("checkpoint_set_sha256")
        ):
            failures.append("P1 freeze does not match the registered 30-cell / 28-run status")
        expected_missing = {"mimiciv/GraFITi/seed2026", "mimiciv/GraFITi/seed2027"}
        completed = set(ledger.get("run_results", {}))
        missing = set(freeze.get("not_completed_run_keys", []))
        if (
            missing != expected_missing
            or completed != expected_run_keys() - expected_missing
            or ledger.get("completed_run_count") != 28
            or ledger.get("planned_run_count") != 30
            or ledger.get("completed_checkpoint_count") != 28
            or set(freeze.get("test_inference_run_keys", [])) != completed
        ):
            failures.append("P1 completed/missing run coverage differs from the frozen denominator")
        if (
            batch.get("checkpoint_set_sha256") != ledger.get("checkpoint_set_sha256")
            or ledger.get("test_batch_manifest_sha256") != sha256_file(P1_BATCH_PATH)
        ):
            failures.append("P1 batch manifest hash does not match the completed ledger")
        if (
            fidelity.get("overall_pass") is not True
            or fidelity_summary.get("overall_pass") is not True
            or fidelity.get("plan_sha256") != p1_plan_sha
            or fidelity_summary.get("analysis_plan_sha256") != p1_plan_sha
            or fidelity_summary.get("detailed_report_sha256") != sha256_file(P1_FIDELITY_PATH)
            or fidelity_summary.get("extraction_manifest_sha256") != sha256_file(P1_EXTRACTION_MANIFEST_PATH)
        ):
            failures.append("P1 event-cache fidelity audit is not passing or hash-bound")
        if extraction.get("cache_fidelity_report") != str(P1_FIDELITY_PATH.relative_to(ROOT)):
            failures.append("P1 extraction manifest points to a different cache fidelity audit")
        if extraction.get("cache_fidelity_report_sha256") != sha256_file(P1_FIDELITY_PATH):
            failures.append("P1 extraction manifest fidelity report SHA256 mismatch")
        for cohort in COHORTS:
            cache_path = ROOT / f"data/processed/{cohort}_events_landmark_v1.pkl"
            if extraction.get("cohorts", {}).get(cohort, {}).get("cache_sha256") != sha256_file(cache_path):
                failures.append(f"{cohort} P1 event cache SHA256 differs from its extraction manifest")
        if verdict.get("analysis_plan_sha256") != p1_plan_sha:
            failures.append("P1 verdict is bound to a different P1 plan")
        if verdict.get("test_inference_ledger_sha256") != sha256_file(P1_LEDGER_PATH):
            failures.append("P1 verdict does not bind the current inference ledger")
        test_use_interpretation = p1._test_use_interpretation(ledger, is_validation_rehearsal=False)

        for path, role in _required_input_paths():
            input_hashes.append(hash_record(path, role))
        evidence_rel = freeze.get("not_completed_evidence_path")
        if evidence_rel:
            evidence_path = (ROOT / str(evidence_rel)).resolve()
            input_hashes.append(hash_record(evidence_path, "P1_not_completed_cell_evidence"))

        # Check the P1 point summaries cover all eight fixed cells, all fixed
        # thresholds, and the original observed-seed estimator.
        import pandas as pd

        metrics = pd.read_csv(P1_METRICS_PATH)
        curves = pd.read_csv(P1_NB_PATH)
        metrics_all = metrics.loc[metrics["scope"] == "all_reference_windows"]
        curves_all = curves.loc[curves["scope"] == "all_reference_windows"]
        cells = set(zip(metrics_all["cohort"], metrics_all["component"]))
        if len(cells) != 8:
            failures.append(f"P1 metrics have {len(cells)} all-reference cells instead of 8")
        expected_thresholds = set(np.round(np.arange(0.05, 0.301, 0.01), 2))
        model_curves = curves_all.loc[curves_all["model_or_strategy"].isin(MODELS)]
        if (
            set(np.round(model_curves["threshold_probability"].astype(float), 2)) != expected_thresholds
            or len(model_curves) != 8 * len(MODELS) * len(expected_thresholds)
        ):
            failures.append("P1 all-reference model decision curves do not cover the fixed 8 x 3 x 26 grid")
        if verdict.get("fixed_denominator", {}).get("planned_run_count") not in (None, 30):
            failures.append("P1 verdict fixed denominator differs from 30 planned runs")

        for key in sorted(completed):
            record = ledger["run_results"][key]
            path = prediction_path(record)
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                failures.append(f"P1 prediction missing or SHA256 mismatch: {key}")
                continue
            input_hashes.append(hash_record(path, "P1_frozen_prediction_npz", run_key=key))
            counts["prediction_files"] += 1
            try:
                with np.load(path, allow_pickle=False) as data:
                    required = {
                        "run_key", "checkpoint_sha256", "batch_manifest_sha256", "sample_id",
                        "stay_id", "patient_id", "landmark_hour", "outcome_exposure_hours",
                        "component_variables", "reference_raw", "reference_seen", "grid_sample_id",
                        "grid_component_index", "grid_hour", "grid_pred_raw", "event_sample_id",
                        "event_component_index", "event_true_raw",
                    }
                    if not required.issubset(set(data.files)):
                        failures.append(f"P1 prediction lacks required arrays: {key}")
                    cohort, model, seed = key.split("/")
                    if str(data["run_key"].item()) != key:
                        failures.append(f"P1 prediction run key mismatch: {key}")
                    checkpoint_hashes = {
                        str(item.get("run_key")): str(item.get("sha256"))
                        for item in freeze.get("checkpoints", [])
                    }
                    if str(data["checkpoint_sha256"].item()) != checkpoint_hashes.get(key):
                        failures.append(f"P1 frozen checkpoint hash mismatch: {key}")
                    expected_batch = batch.get("cohorts", {}).get(cohort, {}).get("batch_manifest_sha256")
                    if str(data["batch_manifest_sha256"].item()) != expected_batch:
                        failures.append(f"P1 batch identity mismatch: {key}")
                    if str(data["cohort"].item()) != cohort or str(data["model"].item()) != model:
                        failures.append(f"P1 prediction cohort/model metadata mismatch: {key}")
                    if int(data["seed"].item()) != int(seed.replace("seed", "")):
                        failures.append(f"P1 prediction seed metadata mismatch: {key}")
            except Exception as exc:  # produce a complete failure manifest rather than aborting early
                failures.append(f"P1 prediction unreadable ({key}): {type(exc).__name__}: {exc}")
        counts["completed_runs"] = len(completed)
        counts["cohorts"] = len(COHORTS)
    except Exception as exc:
        failures.append(f"preflight exception: {type(exc).__name__}: {exc}")

    payload = {
        "schema_version": "jamia_r7_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "preflighted_at_utc": utc_now(),
        "analysis_plan_sha256": EXPECTED_REVISION_PLAN_SHA256,
        "p1_analysis_plan_sha256": frozen_p1_plan_sha256() if P1_PLAN_PATH.is_file() else None,
        "current_p1_plan_file_sha256": sha256_file(P1_PLAN_PATH) if P1_PLAN_PATH.is_file() else None,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "python_version": sys.version,
        "counts": counts,
        "test_use_interpretation": test_use_interpretation if "test_use_interpretation" in locals() else None,
        "input_hashes": input_hashes,
        "input_hash_count": len(input_hashes),
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("R7 preflight failed; no trend calculations were run")
    return payload


def verify_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("R7 preflight manifest is required before analysis")
    record = load_json(PREFLIGHT_PATH)
    if (
        record.get("status") != "PASS"
        or record.get("analysis_plan_sha256") != EXPECTED_REVISION_PLAN_SHA256
        or record.get("script_sha256") != sha256_file(SCRIPT_PATH)
    ):
        raise RuntimeError("R7 preflight manifest status, plan, or script hash mismatch")
    for item in record.get("input_hashes", []):
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"R7 input changed after preflight: {item.get('path')}")
    return record


def safe_ratio(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.asarray(num, dtype=np.float64) / np.asarray(den, dtype=np.float64)
    value = np.where(np.asarray(den) > 0, value, np.nan)
    return float(value) if value.ndim == 0 else value


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high), int(values.size)


def nanmean_axis0(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    count = valid.sum(axis=0)
    total = np.where(valid, array, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def patient_bootstrap(patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(patient_ids.astype(str))
    index = np.searchsorted(unique, patient_ids.astype(str))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    multiplicity = np.empty((BOOTSTRAP_REPS, len(unique)), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPS):
        draw = rng.integers(0, len(unique), size=len(unique))
        multiplicity[replicate] = np.bincount(draw, minlength=len(unique))
    return index.astype(np.int64), multiplicity


def count_by_patient(group_index: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(group_index, weights=np.asarray(values, dtype=np.float64), minlength=n_groups)


def sufficient_counts(labels: dict[str, np.ndarray], alarm: np.ndarray, group_index: np.ndarray, n_groups: int) -> dict[str, np.ndarray]:
    reference = labels["reference_seen"].astype(bool)
    outcome = reference & labels["outcome"].astype(bool)
    alarm_ref = reference & np.asarray(alarm, dtype=bool)
    true_positive = alarm_ref & outcome
    false_positive = alarm_ref & ~outcome
    return {
        "N": count_by_patient(group_index, reference, n_groups),
        "O": count_by_patient(group_index, outcome, n_groups),
        "A": count_by_patient(group_index, alarm_ref, n_groups),
        "TP": count_by_patient(group_index, true_positive, n_groups),
        "FP": count_by_patient(group_index, false_positive, n_groups),
        "exposure_hours": count_by_patient(
            group_index,
            np.where(reference, labels["exposure_hours"], 0.0),
            n_groups,
        ),
    }


def point_metrics(counts: dict[str, np.ndarray]) -> dict[str, float]:
    get = lambda key: float(np.sum(counts[key]))
    n, o, alarms, tp, fp = (get(key) for key in ("N", "O", "A", "TP", "FP"))
    exposure = get("exposure_hours")
    return {
        "n_reference_windows": n,
        "measurement_ascertained_deterioration_count": o,
        "trend_alarm_count": alarms,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "window_sensitivity": float(safe_ratio(tp, o)),
        "ppv_lower_bound": float(safe_ratio(tp, alarms)),
        "alarms_per_reference_eligible_patient_day": float(safe_ratio(alarms, exposure / 24.0)),
    }


def nb_point(counts: dict[str, np.ndarray], thresholds: np.ndarray) -> np.ndarray:
    n = float(np.sum(counts["N"]))
    tp = float(np.sum(counts["TP"]))
    fp = float(np.sum(counts["FP"]))
    odds = thresholds / (1.0 - thresholds)
    return tp / n - fp / n * odds


def nb_bootstrap(counts: dict[str, np.ndarray], multiplicity: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    sample = {name: multiplicity @ values for name, values in counts.items() if name in {"N", "TP", "FP"}}
    odds = thresholds / (1.0 - thresholds)
    n = sample["N"]
    tp = sample["TP"]
    fp = sample["FP"]
    return safe_ratio(tp[:, None], n[:, None]) - safe_ratio(fp[:, None], n[:, None]) * odds[None, :]


def metric_bootstrap(counts: dict[str, np.ndarray], multiplicity: np.ndarray) -> dict[str, np.ndarray]:
    sample = {name: multiplicity @ values for name, values in counts.items()}
    sensitivity = safe_ratio(sample["TP"], sample["O"])
    burden = safe_ratio(sample["A"], sample["exposure_hours"] / 24.0)
    ppv = safe_ratio(sample["TP"], sample["A"])
    return {"window_sensitivity": sensitivity, "alarms_per_reference_eligible_patient_day": burden, "ppv_lower_bound": ppv}


def read_event_cache(
    path: Path,
    component_variables: list[str],
    component_variable_indices: list[int],
    stay_ids: set[int],
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    variable_names = list(cache.get("variable_names", []))
    if len(component_variables) != 4 or len(component_variable_indices) != 4:
        raise RuntimeError("P1 event cache artifact must identify exactly four SOFA components")
    variable_to_component = {}
    for component_index, (name, variable_index) in enumerate(zip(component_variables, component_variable_indices)):
        if variable_index < 0 or variable_index >= len(variable_names) or variable_names[variable_index] != name:
            raise RuntimeError(f"raw event cache component variable mapping mismatch: {name}/{variable_index}")
        variable_to_component[variable_index] = component_index
    examples = cache.get("examples", [])
    if len(examples) != 12000:
        raise RuntimeError(f"{path.name} expected 12,000 full event stays, found {len(examples)}")
    selected: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    seen_stays: set[int] = set()
    for example in examples:
        stay_id = int(example["record_id"])
        if stay_id not in stay_ids:
            continue
        if stay_id in seen_stays:
            raise RuntimeError(f"duplicate stay ID in {path.name}: {stay_id}")
        seen_stays.add(stay_id)
        events = example["events"]
        var = np.asarray(events["var"], dtype=np.int64)
        time_raw = np.asarray(events["time_raw"], dtype=np.float64)
        value_raw = np.asarray(events["value_raw"], dtype=np.float64)
        event_order = np.asarray(events["event_order"], dtype=np.int64)
        if not (var.shape == time_raw.shape == value_raw.shape == event_order.shape):
            raise RuntimeError(f"event arrays have different lengths for a P1 test stay in {path.name}")
        by_component: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for variable_index, component_index in variable_to_component.items():
            keep = (var == variable_index) & np.isfinite(time_raw) & np.isfinite(value_raw)
            times = time_raw[keep]
            values = value_raw[keep]
            orders = event_order[keep]
            # Stable ordering is part of the frozen event stream; exact-time
            # ties remain separate measurements and therefore have zero gap.
            order = np.lexsort((orders, times))
            by_component[component_index] = (times[order], values[order], orders[order])
        selected[stay_id] = by_component
    if seen_stays != stay_ids:
        missing = len(stay_ids - seen_stays)
        raise RuntimeError(f"{path.name} missing {missing} P1 test stays in the raw event cache")
    del cache, examples
    gc.collect()
    return selected


def trend_alarms(
    artifact: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    component_index: int,
    component: dict[str, Any],
    event_data: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    p1_module: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    sample_ids = artifact["sample_id"].astype(np.int64)
    stay_ids = artifact["stay_id"].astype(np.int64)
    landmarks = artifact["landmark_hour"].astype(np.float64)
    exposures = artifact["outcome_exposure_hours"].astype(np.float64)
    n = sample_ids.size
    alarm = np.zeros(n, dtype=bool)
    status = {
        "all_landmark_windows": n,
        "no_reference_windows": int((~labels["reference_seen"]).sum()),
        "reference_windows": int(labels["reference_seen"].sum()),
        "reference_windows_with_0_input_measurements": 0,
        "reference_windows_with_1_input_measurement": 0,
        "reference_windows_with_2plus_input_measurements": 0,
        "reference_windows_last_two_gap_lt_1h": 0,
        "reference_windows_trend_eligible": 0,
        "reference_windows_without_valid_grid_query": 0,
        "reference_windows_with_valid_grid_query": 0,
        "reference_windows_trend_alarm": 0,
        "reference_windows_trend_no_alarm_with_usable_slope": 0,
        "reference_windows_trend_no_alarm_due_to_unusable_slope_or_empty_grid": 0,
        "valid_grid_query_count": 0,
    }
    grid_keep = artifact["grid_component_index"].astype(np.int64) == component_index
    grid_samples = artifact["grid_sample_id"][grid_keep].astype(np.int64)
    grid_hours = artifact["grid_hour"][grid_keep].astype(np.float64)
    query_times: dict[int, list[float]] = {}
    for sample_id, hour in zip(grid_samples, grid_hours):
        query_times.setdefault(int(sample_id), []).append(float(hour))
    ref_grade = labels["reference_grade"]
    reference_seen = labels["reference_seen"].astype(bool)
    for row in range(n):
        if not reference_seen[row]:
            continue
        stay_id = int(stay_ids[row])
        times, values, _orders = event_data[stay_id][component_index]
        landmark = landmarks[row]
        lower = max(0.0, landmark - 48.0)
        input_mask = (times >= lower) & (times < landmark)
        input_times = times[input_mask]
        input_values = values[input_mask]
        if input_times.size < 2:
            status[
                "reference_windows_with_0_input_measurements"
                if input_times.size == 0
                else "reference_windows_with_1_input_measurement"
            ] += 1
            status["reference_windows_trend_no_alarm_due_to_unusable_slope_or_empty_grid"] += 1
            continue
        status["reference_windows_with_2plus_input_measurements"] += 1
        t0, t1 = float(input_times[-2]), float(input_times[-1])
        v0, v1 = float(input_values[-2]), float(input_values[-1])
        gap = t1 - t0
        if gap < MIN_INTERVAL_HOURS:
            status["reference_windows_last_two_gap_lt_1h"] += 1
            status["reference_windows_trend_no_alarm_due_to_unusable_slope_or_empty_grid"] += 1
            continue
        status["reference_windows_trend_eligible"] += 1
        queries = np.asarray(query_times.get(int(sample_ids[row]), []), dtype=np.float64)
        expected_queries = landmark + np.arange(0.5, 24.0, 1.0)
        expected_queries = expected_queries[expected_queries < landmark + exposures[row]]
        if not np.array_equal(np.sort(queries), expected_queries):
            raise RuntimeError(
                f"P1 grid does not match the frozen hourly midpoint window "
                f"(sample row {row}, component {component_index})"
            )
        if queries.size == 0:
            status["reference_windows_without_valid_grid_query"] += 1
            status["reference_windows_trend_no_alarm_due_to_unusable_slope_or_empty_grid"] += 1
            continue
        status["reference_windows_with_valid_grid_query"] += 1
        status["valid_grid_query_count"] += int(queries.size)
        slope = (v1 - v0) / gap
        spans = np.clip(queries - t1, 0.0, MAX_EXTRAPOLATION_HOURS)
        projected = v1 + slope * spans
        projected_grade = p1_module.grade(projected, component)
        if np.any(projected_grade > ref_grade[row]):
            alarm[row] = True
            status["reference_windows_trend_alarm"] += 1
        else:
            status["reference_windows_trend_no_alarm_with_usable_slope"] += 1
    return alarm, status


def _assert_same_labels(left: dict[str, np.ndarray], right: dict[str, np.ndarray], cohort: str, key: str) -> None:
    for field in ("sample_id", "reference_grade", "reference_seen", "measured", "outcome", "stay_id", "patient_id", "exposure_hours"):
        a, b = left[field], right[field]
        if not np.array_equal(a, b):
            raise RuntimeError(f"P1 outcome/denominator inputs differ across artifacts: {cohort}/{key}/{field}")


def _read_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _p1_curve_map(path: Path) -> dict[tuple[str, str, str, float], dict[str, float]]:
    import pandas as pd

    frame = pd.read_csv(path)
    frame = frame.loc[frame["scope"] == "all_reference_windows"].copy()
    result: dict[tuple[str, str, str, float], dict[str, float]] = {}
    for _, row in frame.iterrows():
        key = (str(row["cohort"]), str(row["component"]), str(row["model_or_strategy"]), round(float(row["threshold_probability"]), 2))
        current = {"net_benefit": float(row["net_benefit"]), "ci_low": float(row["net_benefit_ci_low"]), "ci_high": float(row["net_benefit_ci_high"])}
        if key in result:
            prior = result[key]
            for name in current:
                if not (np.isnan(prior[name]) and np.isnan(current[name])) and not math.isclose(prior[name], current[name], rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"P1 duplicated reference strategy rows disagree: {key}/{name}")
        else:
            result[key] = current
    return result


def _p1_metric_map(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    import pandas as pd

    frame = pd.read_csv(path)
    frame = frame.loc[frame["scope"] == "all_reference_windows"].copy()
    result: dict[tuple[str, str, str], dict[str, float]] = {}
    for _, row in frame.iterrows():
        if str(row["model_or_strategy"]) not in MODELS:
            continue
        key = (str(row["cohort"]), str(row["component"]), str(row["model_or_strategy"]))
        current = {
            "window_sensitivity": float(row["window_sensitivity"]),
            "window_sensitivity_ci_low": float(row["window_sensitivity_ci_low"]),
            "window_sensitivity_ci_high": float(row["window_sensitivity_ci_high"]),
            "alarms_per_reference_eligible_patient_day": float(row["alarms_per_reference_eligible_patient_day"]),
            "alarms_per_reference_eligible_patient_day_ci_low": float(row["alarms_per_reference_eligible_patient_day_ci_low"]),
            "alarms_per_reference_eligible_patient_day_ci_high": float(row["alarms_per_reference_eligible_patient_day_ci_high"]),
            "observed_seed_count": float(row["observed_seed_count"]),
        }
        if key in result:
            prior = result[key]
            for name in current:
                if not (np.isnan(prior[name]) and np.isnan(current[name])) and not math.isclose(prior[name], current[name], rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"P1 duplicated metric strategy rows disagree: {key}/{name}")
        else:
            result[key] = current
    return result


def _metrics_from_bootstrap(counts_by_seed: list[dict[str, np.ndarray]], multiplicity: np.ndarray) -> dict[str, Any]:
    point_by_seed = [point_metrics(counts) for counts in counts_by_seed]
    metric_names = point_by_seed[0].keys()
    point = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in point_by_seed], dtype=np.float64)
        point[name] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    boot_by_metric: dict[str, list[np.ndarray]] = {}
    for counts in counts_by_seed:
        for name, draws in metric_bootstrap(counts, multiplicity).items():
            boot_by_metric.setdefault(name, []).append(draws)
    ci = {
        name: percentile_ci(nanmean_axis0(np.stack(draws, axis=0)))
        for name, draws in boot_by_metric.items()
    }
    return {"point": point, "ci": ci}


def _model_nb(counts_by_seed: list[dict[str, np.ndarray]], multiplicity: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, int]]:
    points = np.stack([nb_point(counts, thresholds) for counts in counts_by_seed])
    point = nanmean_axis0(points)
    draws = np.stack([nb_bootstrap(counts, multiplicity, thresholds) for counts in counts_by_seed])
    averaged = nanmean_axis0(draws)
    cis = np.asarray([percentile_ci(averaged[:, idx]) for idx in range(len(thresholds))], dtype=np.float64)
    return point, averaged, cis


def _trend_summary_tex(rows: list[dict[str, Any]]) -> str:
    def n(value: Any, digits: int = 3) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return "--"
        return f"{float(value):.{digits}f}"

    def ci(value: Any, low: Any, high: Any) -> str:
        return f"{n(value)} ({n(low)}--{n(high)})"

    lines = [
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\caption{Post hoc comparison with a fixed two-point linear trend warning rule. The trend is descriptive and does not enter P1 I1/I2.}",
        r"\label{tab:r7-simple-trend}",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Cohort & Component & Ref. windows & Trend alarms & Sensitivity (95\% CI) & Alarms / patient-day (95\% CI) & CarryGate $>$ trend & GraFITi $>$ trend & HyperIMTS $>$ trend " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        cohort = "eICU" if row["cohort"] == "eicu" else "MIMIC-IV"
        lines.append(
            f"{cohort} & {row['component']} & {int(row['reference_windows'])} & "
            f"{int(row['trend_alarm_count'])} & {ci(row['trend_window_sensitivity'], row['trend_sensitivity_ci_low'], row['trend_sensitivity_ci_high'])} & "
            f"{ci(row['trend_alarms_per_patient_day'], row['trend_alarm_burden_ci_low'], row['trend_alarm_burden_ci_high'])} & "
            f"{int(row['carrygate_thresholds_above_trend'])}/26 & "
            f"{int(row['grafiti_thresholds_above_trend'])}/26 & "
            f"{int(row['hyperimts_thresholds_above_trend'])}/26 " + r"\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}", r"}",
        r"\parbox{0.98\linewidth}{\footnotesize Sensitivity is for measurement-ascertained deterioration among reference-eligible windows. Patient-day burden uses the P1 exposure denominator. A model is counted as exceeding trend in a cell when its seed-mean net benefit is strictly higher at at least 14 of the 26 fixed thresholds.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _plot_curves(curve_rows: list[dict[str, Any]], path_pdf: Path, path_png: Path, threshold_values: list[float]) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    thresholds = [float(value) for value in threshold_values]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.0), sharex=True)
    component_order = ["Platelets", "Bilirubin", "GCS", "Creatinine"]
    component_labels = {"Platelets": "Coagulation / Platelets", "Bilirubin": "Liver / Bilirubin", "GCS": "CNS / GCS", "Creatinine": "Renal / Creatinine"}
    cohort_order = ["eicu", "mimiciv"]
    for row_idx, cohort in enumerate(cohort_order):
        for col_idx, component in enumerate(component_order):
            ax = axes[row_idx, col_idx]
            subset = [r for r in curve_rows if r["cohort"] == cohort and r["component"] == component and r["curve_type"] == "strategy"]
            lookup = {r["strategy"]: r for r in subset}
            x = np.asarray(thresholds, dtype=float)
            for strategy in ("no-alarm", "alert-all"):
                values = np.asarray(lookup[strategy]["net_benefit_values"], dtype=float)
                ax.plot(x, values, color="#777777" if strategy == "alert-all" else "#202020", linestyle=":" if strategy == "alert-all" else "--", linewidth=1.15, label=strategy)
            trend = lookup["trend"]
            y = np.asarray(trend["net_benefit_values"], dtype=float)
            lo = np.asarray(trend["ci_low_values"], dtype=float)
            hi = np.asarray(trend["ci_high_values"], dtype=float)
            ax.fill_between(x, lo, hi, color=TREND_COLOR, alpha=0.13, linewidth=0)
            ax.plot(x, y, color=TREND_COLOR, linestyle="-.", linewidth=1.8, label="Simple trend")
            for model in MODELS:
                row = lookup[model]
                ax.plot(x, np.asarray(row["net_benefit_values"], dtype=float), color=MODEL_COLORS[model], linewidth=1.35, label=DISPLAY_MODEL[model])
            ax.axhline(0.0, color="#9b9b9b", linewidth=0.65)
            ax.set_title(f"{'eICU' if cohort == 'eicu' else 'MIMIC-IV'} — {component_labels[component]}", fontsize=9)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.tick_params(axis="both", labelsize=7)
            if col_idx == 0:
                ax.set_ylabel("Net benefit", fontsize=8)
            if row_idx == 1:
                ax.set_xlabel("Threshold probability", fontsize=8)
            ax.set_xlim(min(x), max(x))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Simple two-point trend warning versus P1 model alerts", fontsize=12, y=0.99)
    fig.text(0.5, 0.055, "Curves use the all-reference-window P1 denominator; shaded band shows patient-cluster bootstrap 95% CI for the trend rule.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.09, 0.99, 0.96), w_pad=0.8, h_pad=0.9)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> dict[str, Any]:
    preflight_manifest = verify_preflight()
    existing = [str(path) for path in [*OUTPUTS.values()] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing R7 outputs: {existing}")
    p1 = import_p1_grader()
    thresholds = np.asarray(p1.THRESHOLDS, dtype=np.float64)
    p1_metric_rows = _p1_metric_map(P1_METRICS_PATH)
    p1_curve_rows = _p1_curve_map(P1_NB_PATH)
    ledger = load_json(P1_LEDGER_PATH)
    extraction = load_json(P1_EXTRACTION_MANIFEST_PATH)
    report_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    report_cells: list[dict[str, Any]] = []
    seeds_by_cohort: dict[str, dict[str, list[int]]] = {}
    expected_variables: dict[str, list[str]] = {}

    for cohort in COHORTS:
        run_keys = sorted(key for key in ledger["run_results"] if key.startswith(f"{cohort}/"))
        model_seeds = {
            model: sorted(int(key.rsplit("seed", 1)[1]) for key in run_keys if key.split("/")[1] == model)
            for model in MODELS
        }
        seeds_by_cohort[cohort] = model_seeds
        if not model_seeds["GraFITi"] and cohort == "mimiciv":
            raise RuntimeError("MIMIC-IV GraFITi has no completed runs")
        first_key = run_keys[0]
        reference_artifact = _read_prediction(prediction_path(ledger["run_results"][first_key]))
        sample_artifact = reference_artifact
        sample_ids = sample_artifact["sample_id"].astype(np.int64)
        patient_ids = sample_artifact["patient_id"].astype(str)
        stay_ids = sample_artifact["stay_id"].astype(np.int64)
        landmark_hours = sample_artifact["landmark_hour"].astype(np.float64)
        exposure_hours = sample_artifact["outcome_exposure_hours"].astype(np.float64)
        if len(set(sample_ids.tolist())) != sample_ids.size:
            raise RuntimeError(f"duplicate P1 sample IDs in {cohort} reference artifact")
        if np.any((exposure_hours < 0) | (exposure_hours > 24.0)):
            raise RuntimeError(f"invalid P1 outcome exposure in {cohort}")
        variables = [str(x) for x in sample_artifact["component_variables"].tolist()]
        if variables != ["Platelets", "Bilirubin", "GCS", "Creatinine"]:
            raise RuntimeError(f"P1 component variable order changed for {cohort}: {variables}")
        all_test_stays = set(stay_ids.tolist())
        component_variable_indices = sample_artifact["component_variable_indices"].astype(np.int64).tolist()
        event_data = read_event_cache(
            ROOT / f"data/processed/{cohort}_events_landmark_v1.pkl",
            variables,
            component_variable_indices,
            all_test_stays,
        )
        group_index, multiplicity = patient_bootstrap(patient_ids)
        n_groups = multiplicity.shape[1]

        artifacts_by_model: dict[str, list[dict[str, np.ndarray]]] = {model: [] for model in MODELS}
        for key in run_keys:
            artifact = _read_prediction(prediction_path(ledger["run_results"][key]))
            for field, baseline in (("sample_id", sample_ids), ("stay_id", stay_ids), ("patient_id", patient_ids), ("landmark_hour", landmark_hours), ("outcome_exposure_hours", exposure_hours), ("reference_raw", sample_artifact["reference_raw"]), ("reference_seen", sample_artifact["reference_seen"])):
                if not np.array_equal(artifact[field], baseline):
                    raise RuntimeError(f"P1 window roster/reference differs across runs: {key}/{field}")
            if not np.array_equal(artifact["grid_sample_id"], sample_artifact["grid_sample_id"]) or not np.array_equal(artifact["grid_component_index"], sample_artifact["grid_component_index"]) or not np.array_equal(artifact["grid_hour"], sample_artifact["grid_hour"]):
                raise RuntimeError(f"P1 query grid differs across runs: {key}")
            model = key.split("/")[1]
            artifacts_by_model[model].append(artifact)

        for component_index, component in enumerate(p1.COMPONENTS):
            name = str(component["variable"])
            labels = p1._window_labels(sample_artifact, component_index, component)
            trend_alarm, audit = trend_alarms(sample_artifact, labels, component_index, component, event_data, p1)
            reference = labels["reference_seen"].astype(bool)
            if audit["reference_windows"] != int(reference.sum()):
                raise RuntimeError(f"R7 reference count mismatch: {cohort}/{name}")
            trend_counts = sufficient_counts(labels, trend_alarm, group_index, n_groups)
            trend_metrics = point_metrics(trend_counts)
            trend_metric_boot = metric_bootstrap(trend_counts, multiplicity)
            trend_metric_ci = {metric: percentile_ci(values) for metric, values in trend_metric_boot.items()}
            trend_nb = nb_point(trend_counts, thresholds)
            trend_nb_draws = nb_bootstrap(trend_counts, multiplicity, thresholds)
            trend_nb_ci = np.asarray([percentile_ci(trend_nb_draws[:, i]) for i in range(len(thresholds))], dtype=np.float64)
            model_counts: dict[str, list[dict[str, np.ndarray]]] = {}
            model_metrics: dict[str, dict[str, Any]] = {}
            model_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for model in MODELS:
                counts_for_seeds: list[dict[str, np.ndarray]] = []
                label_for_model = None
                for artifact in artifacts_by_model[model]:
                    seed_labels = p1._window_labels(artifact, component_index, component)
                    if label_for_model is None:
                        label_for_model = seed_labels
                    else:
                        _assert_same_labels(label_for_model, seed_labels, cohort, model)
                    counts_for_seeds.append(sufficient_counts(seed_labels, seed_labels["alarm"], group_index, n_groups))
                model_counts[model] = counts_for_seeds
                model_metrics[model] = _metrics_from_bootstrap(counts_for_seeds, multiplicity)
                model_curves[model] = _model_nb(counts_for_seeds, multiplicity, thresholds)
                # The independent calculation must reproduce the already
                # frozen P1 point curve and all-reference metric point.
                saved_metric = p1_metric_rows[(cohort, name, model)]
                computed_metric = model_metrics[model]["point"]
                for output_name in ("window_sensitivity", "alarms_per_reference_eligible_patient_day"):
                    if not math.isclose(computed_metric[output_name], saved_metric[output_name], rel_tol=1e-9, abs_tol=1e-10):
                        raise RuntimeError(f"R7 recomputed P1 metric differs from frozen result: {cohort}/{name}/{model}/{output_name}")
                for threshold_index, threshold in enumerate(thresholds):
                    saved = p1_curve_rows[(cohort, name, model, round(float(threshold), 2))]["net_benefit"]
                    point = float(model_curves[model][0][threshold_index])
                    if not math.isclose(point, saved, rel_tol=1e-9, abs_tol=1e-11):
                        raise RuntimeError(f"R7 recomputed P1 net benefit differs from frozen result: {cohort}/{name}/{model}/{threshold:.2f}")

            # Patient-cluster paired difference curves use the same bootstrap
            # multiplicities for model alerts and the deterministic trend rule.
            paired_differences: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for model in MODELS:
                model_nb_draws = np.stack([nb_bootstrap(c, multiplicity, thresholds) for c in model_counts[model]])
                model_nb_mean_draws = nanmean_axis0(model_nb_draws)
                difference_draws = model_nb_mean_draws - trend_nb_draws
                difference_point = model_curves[model][0] - trend_nb
                difference_ci = np.asarray([percentile_ci(difference_draws[:, j]) for j in range(len(thresholds))], dtype=np.float64)
                paired_differences[model] = (difference_point, difference_draws, difference_ci)

            trend_gt_no = int(np.sum(trend_nb > 0.0))
            trend_gt_all = 0
            threshold_counts = {}
            for model in MODELS:
                diff = paired_differences[model][0]
                n_better = int(np.sum(diff > 0.0))
                threshold_counts[model] = n_better
            # Alert-all is a fixed reference on the same event prevalence.
            o = float(np.sum(trend_counts["O"]))
            n_ref = float(np.sum(trend_counts["N"]))
            prevalence = o / n_ref if n_ref else float("nan")
            alert_all_curve = prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))
            trend_gt_all = int(np.sum(trend_nb > alert_all_curve))

            summary_row = {
                "cohort": cohort,
                "component": name,
                "organ": str(component["organ"]),
                "all_landmark_windows": audit["all_landmark_windows"],
                "reference_windows": trend_metrics["n_reference_windows"],
                "no_reference_windows": audit["no_reference_windows"],
                "measurement_ascertained_deterioration_windows": trend_metrics["measurement_ascertained_deterioration_count"],
                "trend_input_windows_0_measurements": audit["reference_windows_with_0_input_measurements"],
                "trend_input_windows_1_measurement": audit["reference_windows_with_1_input_measurement"],
                "trend_input_windows_2plus_measurements": audit["reference_windows_with_2plus_input_measurements"],
                "trend_last_two_gap_lt_1h": audit["reference_windows_last_two_gap_lt_1h"],
                "trend_eligible_windows": audit["reference_windows_trend_eligible"],
                "trend_windows_without_valid_query": audit["reference_windows_without_valid_grid_query"],
                "trend_alarm_count": trend_metrics["trend_alarm_count"],
                "trend_window_sensitivity": trend_metrics["window_sensitivity"],
                "trend_sensitivity_ci_low": trend_metric_ci["window_sensitivity"][0],
                "trend_sensitivity_ci_high": trend_metric_ci["window_sensitivity"][1],
                "trend_ppv_lower_bound": trend_metrics["ppv_lower_bound"],
                "trend_alarms_per_patient_day": trend_metrics["alarms_per_reference_eligible_patient_day"],
                "trend_alarm_burden_ci_low": trend_metric_ci["alarms_per_reference_eligible_patient_day"][0],
                "trend_alarm_burden_ci_high": trend_metric_ci["alarms_per_reference_eligible_patient_day"][1],
                "trend_gt_no_alarm_thresholds": trend_gt_no,
                "trend_gt_alert_all_thresholds": trend_gt_all,
                "carrygate_thresholds_above_trend": threshold_counts["ICUKANForecaster"],
                "carrygate_exceeds_trend_in_cell": threshold_counts["ICUKANForecaster"] >= MIN_SUPERIOR_THRESHOLDS,
                "grafiti_thresholds_above_trend": threshold_counts["GraFITi"],
                "grafiti_exceeds_trend_in_cell": threshold_counts["GraFITi"] >= MIN_SUPERIOR_THRESHOLDS,
                "hyperimts_thresholds_above_trend": threshold_counts["HyperIMTS"],
                "hyperimts_exceeds_trend_in_cell": threshold_counts["HyperIMTS"] >= MIN_SUPERIOR_THRESHOLDS,
                "trend_nb_mean_over_26_thresholds": float(np.mean(trend_nb)),
                "model_seed_coverage": json.dumps({model: model_seeds[model] for model in MODELS}, separators=(",", ":")),
            }
            for model in MODELS:
                prefix = DISPLAY_MODEL[model].lower().replace("-", "")
                summary_row[f"{prefix}_window_sensitivity"] = model_metrics[model]["point"]["window_sensitivity"]
                summary_row[f"{prefix}_window_sensitivity_ci_low"] = model_metrics[model]["ci"]["window_sensitivity"][0]
                summary_row[f"{prefix}_window_sensitivity_ci_high"] = model_metrics[model]["ci"]["window_sensitivity"][1]
                summary_row[f"{prefix}_alarms_per_patient_day"] = model_metrics[model]["point"]["alarms_per_reference_eligible_patient_day"]
                summary_row[f"{prefix}_alarm_burden_ci_low"] = model_metrics[model]["ci"]["alarms_per_reference_eligible_patient_day"][0]
                summary_row[f"{prefix}_alarm_burden_ci_high"] = model_metrics[model]["ci"]["alarms_per_reference_eligible_patient_day"][1]
            report_rows.append(summary_row)
            audit_rows.append({"cohort": cohort, "component": name, "organ": str(component["organ"]), **audit})

            for strategy in ("trend", "no-alarm", "alert-all", *MODELS):
                if strategy == "trend":
                    point_curve = trend_nb
                    lo_curve = trend_nb_ci[:, 0]
                    hi_curve = trend_nb_ci[:, 1]
                    valid = trend_nb_ci[:, 2]
                elif strategy == "no-alarm":
                    point_curve = np.zeros(len(thresholds), dtype=float)
                    lo_curve = np.zeros(len(thresholds), dtype=float)
                    hi_curve = np.zeros(len(thresholds), dtype=float)
                    valid = np.full(len(thresholds), BOOTSTRAP_REPS)
                elif strategy == "alert-all":
                    # Use the same P1 reference formula and patient bootstrap.
                    point_curve = alert_all_curve
                    prevalence_draws = safe_ratio(multiplicity @ trend_counts["O"], multiplicity @ trend_counts["N"])
                    all_draws = prevalence_draws[:, None] - (1.0 - prevalence_draws[:, None]) * (thresholds[None, :] / (1.0 - thresholds[None, :]))
                    cis = np.asarray([percentile_ci(all_draws[:, j]) for j in range(len(thresholds))], dtype=float)
                    lo_curve, hi_curve, valid = cis[:, 0], cis[:, 1], cis[:, 2]
                else:
                    point_curve, _draws, cis = model_curves[strategy]
                    lo_curve, hi_curve, valid = cis[:, 0], cis[:, 1], cis[:, 2]
                for threshold_index, threshold in enumerate(thresholds):
                    row = {
                        "cohort": cohort,
                        "component": name,
                        "organ": str(component["organ"]),
                        "threshold_probability": float(threshold),
                        "strategy": strategy,
                        "curve_type": "strategy",
                        "net_benefit": float(point_curve[threshold_index]),
                        "net_benefit_ci_low": float(lo_curve[threshold_index]),
                        "net_benefit_ci_high": float(hi_curve[threshold_index]),
                        "valid_bootstrap_replicates": int(valid[threshold_index]),
                    }
                    curve_rows.append(row)
            for model in MODELS:
                point_diff, draw_diff, diff_ci = paired_differences[model]
                for threshold_index, threshold in enumerate(thresholds):
                    curve_rows.append({
                        "cohort": cohort,
                        "component": name,
                        "organ": str(component["organ"]),
                        "threshold_probability": float(threshold),
                        "strategy": f"{DISPLAY_MODEL[model]} minus trend",
                        "curve_type": "paired_difference",
                        "net_benefit": float(point_diff[threshold_index]),
                        "net_benefit_ci_low": float(diff_ci[threshold_index, 0]),
                        "net_benefit_ci_high": float(diff_ci[threshold_index, 1]),
                        "valid_bootstrap_replicates": int(diff_ci[threshold_index, 2]),
                    })

        del event_data, artifacts_by_model, sample_artifact, reference_artifact
        gc.collect()

    # Fixed no-alarm / alert-all checks are explicit and threshold-complete.
    if len(report_rows) != 8 or len(audit_rows) != 8:
        raise RuntimeError(f"R7 expected exactly eight cohort-component cells, got {len(report_rows)}")
    model_cell_counts = {
        DISPLAY_MODEL[model]: sum(bool(row[f"{DISPLAY_MODEL[model].lower().replace('-', '')}_exceeds_trend_in_cell"]) for row in report_rows)
        for model in MODELS
    }
    overall_model_cells = sum(
        any(bool(row[f"{DISPLAY_MODEL[model].lower().replace('-', '')}_exceeds_trend_in_cell"]) for model in MODELS)
        for row in report_rows
    )
    if len({(row["cohort"], row["component"]) for row in report_rows}) != 8:
        raise RuntimeError("R7 output contains duplicate or missing cohort-component cells")

    curve_payload_rows = []
    for cell in sorted({(row["cohort"], row["component"]) for row in curve_rows}):
        for strategy in STRATEGY_ORDER:
            rows = [r for r in curve_rows if (r["cohort"], r["component"]) == cell and r["curve_type"] == "strategy" and r["strategy"] == strategy]
            rows = sorted(rows, key=lambda r: r["threshold_probability"])
            curve_payload_rows.append({
                "cohort": cell[0], "component": cell[1], "curve_type": "strategy", "strategy": strategy,
                "threshold_values": sorted({float(r["threshold_probability"]) for r in rows}),
                "net_benefit_values": [r["net_benefit"] for r in rows],
                "ci_low_values": [r["net_benefit_ci_low"] for r in rows],
                "ci_high_values": [r["net_benefit_ci_high"] for r in rows],
            })
    report = {
        "schema_version": "jamia_r7_report_v1",
        "analysis_plan_sha256": EXPECTED_REVISION_PLAN_SHA256,
        "p1_analysis_plan_sha256": frozen_p1_plan_sha256(),
        "current_p1_plan_file_sha256": sha256_file(P1_PLAN_PATH),
        "preflight_status": "PASS",
        "analysis_scope": "post hoc descriptive comparator on frozen P1 test landmarks; does not alter I1/I2",
        "historical_test_access": preflight_manifest.get("test_use_interpretation"),
        "trend_rule": {
            "input_window": "max(0, landmark-48h) <= measurement_time < landmark",
            "selection": "last two raw measurements for the component, ordered by time and stable event order",
            "minimum_interval_hours": MIN_INTERVAL_HOURS,
            "queries": "exact P1 hourly midpoint queries retained after outcome-window discharge censoring",
            "projection": "raw-unit linear slope; query horizon clipped to [0,24h]",
            "alarm": "any projected P1 grade strictly worse than the fixed P1 reference grade",
            "ineligible_or_empty_query": "no trend alarm; retained in the same all-reference-window denominator",
        },
        "estimation": {
            "models": "frozen P1 model seed-mean point estimates over observed completed runs",
            "planned_runs": 30,
            "completed_runs": 28,
            "missing_runs": ["mimiciv/GraFITi/seed2026", "mimiciv/GraFITi/seed2027"],
            "model_beats_trend_rule": f"strictly higher seed-mean net benefit at >= {MIN_SUPERIOR_THRESHOLDS}/26 fixed thresholds in a cell; mirrors P1's fixed 14/26 comparison count",
            "bootstrap": {"unit": "patient cluster within cohort", "replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "ci": "percentile 95%"},
        },
        "model_cells_exceeding_trend": model_cell_counts,
        "cells_with_any_model_exceeding_trend": overall_model_cells,
        "cells_with_trend_exceeding_no_alarm_any_threshold": sum(int(row["trend_gt_no_alarm_thresholds"] > 0) for row in report_rows),
        "cells_with_trend_exceeding_alert_all_any_threshold": sum(int(row["trend_gt_alert_all_thresholds"] > 0) for row in report_rows),
        "cell_results": report_rows,
        "window_audit": audit_rows,
        "input_hashes": preflight_manifest["input_hashes"],
    }
    with tempfile.TemporaryDirectory(prefix="r7_stage_", dir=str(OUT)) as temporary_dir:
        stage_root = Path(temporary_dir)
        staged = {name: stage_root / path.name for name, path in OUTPUTS.items()}
        _plot_curves(curve_payload_rows, staged["figure_pdf"], staged["figure_png"], list(thresholds))
        _write_table_tex(report_rows, staged["table_tex"])
        _write_csv(staged["window_audit"], audit_rows)
        _write_csv(staged["summary"], report_rows)
        _write_csv(staged["curves"], curve_rows)
        staged["report"].write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output_hashes = [
            {
                "path": str(OUTPUTS[name].relative_to(ROOT)),
                "role": "R7_output",
                "size_bytes": staged[name].stat().st_size,
                "sha256": sha256_file(staged[name]),
            }
            for name in OUTPUTS
            if name != "manifest"
        ]
        manifest = {
            "schema_version": "jamia_r7_manifest_v1",
            "status": "COMPLETE",
            "created_at_utc": utc_now(),
            "analysis_plan_sha256": EXPECTED_REVISION_PLAN_SHA256,
            "p1_analysis_plan_sha256": frozen_p1_plan_sha256(),
            "current_p1_plan_file_sha256": sha256_file(P1_PLAN_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "preflight_manifest_sha256": sha256_file(PREFLIGHT_PATH),
            "python_version": sys.version,
            "bootstrap": report["estimation"]["bootstrap"],
            "seed_coverage_by_cohort": seeds_by_cohort,
            "model_cells_exceeding_trend": model_cell_counts,
            "cells_with_any_model_exceeding_trend": overall_model_cells,
            "input_hashes": preflight_manifest["input_hashes"],
            "counts": {
                "cells": 8,
                "completed_model_runs": 28,
                "planned_model_runs": 30,
                "window_audit_rows": len(audit_rows),
                "summary_rows": len(report_rows),
                "threshold_rows": sum(1 for row in curve_rows if row["curve_type"] == "strategy"),
                "paired_difference_rows": sum(1 for row in curve_rows if row["curve_type"] == "paired_difference"),
            },
            "outputs": output_hashes,
        }
        staged["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # All analysis and figure outputs are complete in the staging directory
        # before any file is moved to its official path. Publish the manifest last.
        for name in OUTPUTS:
            if name != "manifest":
                staged[name].replace(OUTPUTS[name])
        staged["manifest"].replace(OUTPUTS["manifest"])
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty R7 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
        handle.flush()
    temp.replace(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NA"
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_table_tex(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _trend_summary_tex(rows)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive prior R7 preflight/results before replacing them")
    args = parser.parse_args()
    if args.command == "preflight":
        prepare_generation(ROOT, (PREFLIGHT_PATH, *OUTPUTS.values()),
                           refresh=args.refresh_generated, label="r7_generated")
        result = preflight()
        print(json.dumps({"status": result["status"], "failures": result["failures"], "input_hash_count": result["input_hash_count"]}, ensure_ascii=False, indent=2))
    else:
        prepare_generation(ROOT, OUTPUTS.values(), refresh=args.refresh_generated,
                           label="r7_generated")
        result = run_analysis()
        print(json.dumps({"status": result["status"], "outputs": result["counts"], "model_cells_exceeding_trend": result["model_cells_exceeding_trend"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
