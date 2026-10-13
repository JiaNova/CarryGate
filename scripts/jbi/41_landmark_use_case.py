#!/usr/bin/env python
"""Run frozen P1 test-set analyses or an isolated validation-only rehearsal.

This script consumes only the one-pass, frozen-checkpoint prediction artifacts
written by ``51_run_p1_test_once.py``. Its main endpoint is always named
measurement-ascertained deterioration; unmeasured outcome windows remain
known zeros under that operational definition, with PPV interpreted as a
lower bound on latent clinical PPV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/jbi"
PLAN = ROOT / "paper/预设用例分析计划_P1.md"
LEDGER = RESULTS / "p1_landmark_test_inference_ledger.json"
BATCH_MANIFEST = RESULTS / "p1_landmark_test_batch_manifest.json"
PREDICTIONS = RESULTS / "p1_landmark_predictions"
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260914
THRESHOLDS = np.round(np.arange(0.05, 0.301, 0.01), 2)
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
SEEDS = (2024, 2025, 2026, 2027, 2028)
COMPONENTS = (
    {"organ": "Coagulation", "variable": "Platelets", "direction": "low", "bounds": (150.0, 100.0, 50.0, 20.0)},
    {"organ": "Liver", "variable": "Bilirubin", "direction": "high", "bounds": (1.2, 2.0, 6.0, 12.0)},
    {"organ": "CNS", "variable": "GCS", "direction": "low", "bounds": (15.0, 13.0, 10.0, 6.0), "support": (3.0, 15.0)},
    {"organ": "Renal", "variable": "Creatinine", "direction": "high", "bounds": (1.2, 2.0, 3.5, 5.0)},
)
STRATEGIES = (*MODELS, "no-alarm", "alert-all")
SCOPES = ("all_reference_windows", "measured_windows_only")
OUT_METRICS = RESULTS / "p1_landmark_metrics.csv"
OUT_NB = RESULTS / "p1_landmark_decision_curve.csv"
OUT_AGREEMENT = RESULTS / "p1_landmark_agreement.csv"
OUT_ASCERTAINMENT = RESULTS / "p1_landmark_ascertainment.csv"
OUT_VERDICT = RESULTS / "p1_landmark_verdict.json"
OUT_FIG_PDF = ROOT / "results/paper/F8_p1_landmark_use_case.pdf"
OUT_FIG_PNG = ROOT / "results/paper/F8_p1_landmark_use_case.png"
FAILED_TEST_ATTEMPT = RESULTS / "_work/p1_landmark_test_failed_attempts/attempt01_20261010T071740Z"
VALIDATION_REHEARSAL_MANIFEST: Path | None = None
ACTIVE_LEDGER_PATH = LEDGER
ACTIVE_BATCH_MANIFEST_PATH = BATCH_MANIFEST


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
    temp.replace(path)


def grade(values: np.ndarray, component: dict[str, Any]) -> np.ndarray:
    """Return SOFA-derived 0--4 grades using the frozen component cut-points."""
    values = np.asarray(values, dtype=np.float64)
    if "support" in component:
        low, high = component["support"]
        values = np.clip(np.round(values), low, high)
    bounds = np.asarray(component["bounds"], dtype=np.float64)
    if component["direction"] == "low":
        return np.sum(values[..., None] < bounds, axis=-1).astype(np.int8)
    return np.sum(values[..., None] >= bounds, axis=-1).astype(np.int8)


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.asarray(numerator, dtype=np.float64) / np.asarray(denominator, dtype=np.float64)
    value = np.where(np.asarray(denominator) > 0, value, np.nan)
    return float(value) if value.ndim == 0 else value


def _percentile_ci(samples: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high), int(values.size)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty analysis table: {path}")
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
    if isinstance(value, np.generic):
        return value.item()
    return value


def _test_use_interpretation(ledger: dict[str, Any], *, is_validation_rehearsal: bool) -> str:
    """Describe P1 test access from the completed ledger and archived first failure."""
    if is_validation_rehearsal:
        return "no test data accessed; validation-only code rehearsal, not a study result"

    metadata_path = FAILED_TEST_ATTEMPT / "archive_metadata.json"
    failed_ledger_path = FAILED_TEST_ATTEMPT / "failed_ledger.json"
    if not metadata_path.is_file() or not failed_ledger_path.is_file():
        raise FileNotFoundError("archived failed first P1 test attempt is required for test-use wording")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failed = json.loads(failed_ledger_path.read_text(encoding="utf-8"))
    if (
        metadata.get("sha256") != _sha256(failed_ledger_path)
        or metadata.get("outputs_written") is not False
        or metadata.get("run_results_count") != 0
        or failed.get("status") != "FAILED"
        or failed.get("run_results") != {}
        or failed.get("failed_run_key") != metadata.get("failed_run_key")
        or failed.get("checkpoint_set_sha256") != ledger.get("checkpoint_set_sha256")
    ):
        raise RuntimeError("archived failed P1 test attempt does not match its recorded no-output failure")
    if (
        ledger.get("status") != "COMPLETE"
        or ledger.get("test_inference_consumed") is not True
        or ledger.get("completed_run_count") != ledger.get("completed_checkpoint_count")
        or ledger.get("completed_run_count") != len(ledger.get("run_results", {}))
        or ledger.get("test_inference_passes_authorized") != 1
    ):
        raise RuntimeError("P1 test inference ledger does not attest one complete frozen-checkpoint pass")

    return (
        f"one failed first test access ({metadata['failed_run_key']}; no run outputs written; "
        f"not used for selection), followed by one complete inference pass over "
        f"{ledger['completed_checkpoint_count']} frozen checkpoints of "
        f"{ledger['planned_run_count']} planned cells; no test-based model or threshold selection"
    )


def _expected_runs() -> set[str]:
    return {f"{cohort}/{model}/seed{seed}" for cohort in COHORTS for model in MODELS for seed in SEEDS}


def _load_and_validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    if VALIDATION_REHEARSAL_MANIFEST is not None:
        return _load_validation_rehearsal_inputs(VALIDATION_REHEARSAL_MANIFEST)
    global ACTIVE_LEDGER_PATH, ACTIVE_BATCH_MANIFEST_PATH
    ACTIVE_LEDGER_PATH, ACTIVE_BATCH_MANIFEST_PATH = LEDGER, BATCH_MANIFEST
    if not LEDGER.is_file() or not BATCH_MANIFEST.is_file():
        raise FileNotFoundError("P1 one-pass test inference ledger and batch manifest are required")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    batch_manifest = json.loads(BATCH_MANIFEST.read_text(encoding="utf-8"))
    if ledger.get("status") != "COMPLETE" or ledger.get("test_inference_consumed") is not True:
        raise RuntimeError("P1 test inference pass is incomplete")
    if ledger.get("analysis_plan_sha256") != _sha256(PLAN):
        raise RuntimeError("test inference ledger is bound to a different P1 plan")
    if batch_manifest.get("checkpoint_set_sha256") != ledger.get("checkpoint_set_sha256"):
        raise RuntimeError("test batch manifest is bound to a different checkpoint set")
    if batch_manifest.get("analysis_plan_sha256") != _sha256(PLAN):
        raise RuntimeError("test batch manifest is bound to a different P1 plan")
    freeze_path = RESULTS / "p1_landmark_checkpoint_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_status = freeze.get("status")
    if freeze_status not in {"ALL_30_CHECKPOINTS_FROZEN", "FROZEN_WITH_NOT_COMPLETED_CELLS"}:
        raise RuntimeError("the test checkpoint freeze is absent or incomplete")
    if freeze.get("checkpoint_set_sha256") != ledger.get("checkpoint_set_sha256"):
        raise RuntimeError("prediction ledger checkpoint set differs from the frozen set")
    completed_keys = set(str(key) for key in freeze.get("test_inference_run_keys", []))
    not_completed_keys = set(str(key) for key in freeze.get("not_completed_run_keys", []))
    if (
        freeze.get("planned_run_count") != 30
        or freeze.get("run_count") != len(completed_keys)
        or ledger.get("completed_run_count") != len(completed_keys)
        or set(ledger.get("run_results", {})) != completed_keys
        or completed_keys | not_completed_keys != _expected_runs()
        or completed_keys & not_completed_keys
    ):
        raise RuntimeError("P1 test inference does not match the resolved 30-cell fixed denominator")
    if freeze_status == "ALL_30_CHECKPOINTS_FROZEN" and not not_completed_keys:
        if completed_keys != _expected_runs():
            raise RuntimeError("all-complete freeze does not cover all 30 fixed-denominator cells")
    elif freeze_status == "FROZEN_WITH_NOT_COMPLETED_CELLS":
        if set(freeze.get("not_runnable_run_keys", [])):
            raise RuntimeError("not_completed analysis cannot be mixed with not_runnable checkpoint coverage")

    files: dict[str, dict[str, dict[str, Any]]] = {cohort: {} for cohort in COHORTS}
    for key in sorted(completed_keys):
        result = ledger["run_results"][key]
        path = (ROOT / result["path"]).resolve()
        if ROOT not in path.parents or not path.is_file() or _sha256(path) != result.get("sha256"):
            raise RuntimeError(f"prediction artifact missing or hash-mismatched for {key}")
        cohort, model, seed_text = key.split("/")
        with np.load(path, allow_pickle=False) as data:
            artifact = {name: data[name] for name in data.files}
        if str(artifact["run_key"].item()) != key:
            raise RuntimeError(f"prediction artifact key mismatch for {key}")
        if str(artifact["checkpoint_sha256"].item()) != freeze_hash(freeze, key):
            raise RuntimeError(f"prediction artifact checkpoint hash mismatch for {key}")
        cohort_batch = batch_manifest.get("cohorts", {}).get(cohort)
        if not cohort_batch or artifact["batch_manifest_sha256"].item() != cohort_batch.get("batch_manifest_sha256"):
            raise RuntimeError(f"prediction artifact batch manifest mismatch for {key}")
        files[cohort][f"{model}/{seed_text}"] = artifact
    return ledger, batch_manifest, files


def _load_validation_rehearsal_inputs(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Load explicitly marked validation artifacts without touching test paths."""
    global ACTIVE_LEDGER_PATH, ACTIVE_BATCH_MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    ledger = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {f"{cohort}/{model}/seed2024" for cohort in COHORTS for model in MODELS}
    if (
        ledger.get("status") != "COMPLETE"
        or ledger.get("split_used") != "validation"
        or ledger.get("test_inference_consumed") is not False
        or ledger.get("analysis_plan_sha256") != _sha256(PLAN)
        or set(ledger.get("run_results", {})) != expected_keys
    ):
        raise RuntimeError("validation rehearsal bundle is incomplete or not bound to the P1 plan")
    batch_path = (ROOT / str(ledger.get("batch_manifest_path", ""))).resolve()
    if ROOT not in batch_path.parents or not batch_path.is_file():
        raise RuntimeError("validation rehearsal batch manifest is missing or outside the project")
    batch_manifest = json.loads(batch_path.read_text(encoding="utf-8"))
    if (
        batch_manifest.get("split_used") != "validation"
        or batch_manifest.get("analysis_plan_sha256") != _sha256(PLAN)
        or batch_manifest.get("cohorts", {}).keys() != set(COHORTS)
    ):
        raise RuntimeError("validation rehearsal batch manifest is not a validation-only bundle")

    files: dict[str, dict[str, dict[str, Any]]] = {cohort: {} for cohort in COHORTS}
    for key in sorted(expected_keys):
        record = ledger["run_results"][key]
        path = (ROOT / str(record.get("path", ""))).resolve()
        if ROOT not in path.parents or not path.is_file() or _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"validation prediction artifact missing or hash-mismatched for {key}")
        cohort, model, seed_text = key.split("/")
        with np.load(path, allow_pickle=False) as data:
            artifact = {name: data[name] for name in data.files}
        if (
            str(artifact.get("run_key", np.asarray("")).item()) != key
            or str(artifact.get("split_used", np.asarray("")).item()) != "validation"
            or str(artifact["checkpoint_sha256"].item()) != record.get("checkpoint_sha256")
            or str(artifact["batch_manifest_sha256"].item()) != batch_manifest["cohorts"][cohort]["batch_manifest_sha256"]
        ):
            raise RuntimeError(f"validation prediction provenance mismatch for {key}")
        files[cohort][f"{model}/{seed_text}"] = artifact
    ACTIVE_LEDGER_PATH, ACTIVE_BATCH_MANIFEST_PATH = manifest_path, batch_path
    return ledger, batch_manifest, files


def freeze_hash(freeze: dict[str, Any], key: str) -> str:
    matches = [item["sha256"] for item in freeze.get("checkpoints", []) if item.get("run_key") == key]
    if len(matches) != 1:
        raise RuntimeError(f"freeze record lacks a unique checkpoint hash for {key}")
    return str(matches[0])


def _patient_bootstrap(patient_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(patient_id.astype(str))
    index = np.searchsorted(unique, patient_id.astype(str))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    multiplicity = np.empty((BOOTSTRAP_REPS, len(unique)), dtype=np.int32)
    for replicate in range(BOOTSTRAP_REPS):
        draw = rng.integers(0, len(unique), size=len(unique))
        multiplicity[replicate] = np.bincount(draw, minlength=len(unique))
    return index.astype(np.int64), multiplicity


def _group_sum(group_index: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(group_index, weights=np.asarray(values, dtype=np.float64), minlength=n_groups)


def _window_labels(artifact: dict[str, np.ndarray], component_index: int, component: dict[str, Any]) -> dict[str, np.ndarray]:
    sample_ids = artifact["sample_id"].astype(np.int64)
    n = sample_ids.size
    id_to_row = {int(sample_id): row for row, sample_id in enumerate(sample_ids.tolist())}
    reference = artifact["reference_raw"][:, component_index]
    seen = artifact["reference_seen"][:, component_index].astype(bool)
    ref_grade = grade(reference, component)
    measured = np.zeros(n, dtype=bool)
    outcome = np.zeros(n, dtype=bool)
    event_sample = artifact["event_sample_id"].astype(np.int64)
    event_component = artifact["event_component_index"].astype(np.int64)
    keep = event_component == component_index
    if np.any(keep):
        event_ids = event_sample[keep]
        rows = np.asarray([id_to_row[int(value)] for value in event_ids], dtype=np.int64)
        measured[rows] = True
        true_grade = grade(artifact["event_true_raw"][keep], component)
        # Repeated measurements can map to the same landmark. Buffered
        # advanced indexing can lose earlier true values for duplicate rows.
        np.logical_or.at(outcome, rows, true_grade > ref_grade[rows])

    alarm = np.zeros(n, dtype=bool)
    grid_component = artifact["grid_component_index"].astype(np.int64)
    keep_grid = grid_component == component_index
    if np.any(keep_grid):
        grid_rows = np.asarray(
            [id_to_row[int(value)] for value in artifact["grid_sample_id"][keep_grid].astype(np.int64)],
            dtype=np.int64,
        )
        pred_grade = grade(artifact["grid_pred_raw"][keep_grid], component)
        # Each component has multiple hourly grid queries per window; preserve
        # an alarm raised by any one of them.
        np.logical_or.at(alarm, grid_rows, pred_grade > ref_grade[grid_rows])
    return {
        "sample_id": sample_ids,
        "reference_grade": ref_grade,
        "reference_seen": seen,
        "measured": measured,
        "outcome": outcome,
        "alarm": alarm,
        "stay_id": artifact["stay_id"].astype(np.int64),
        "patient_id": artifact["patient_id"].astype(str),
        "exposure_hours": artifact["outcome_exposure_hours"].astype(np.float64),
    }


def _patient_bits(group_index: np.ndarray, outcome: np.ndarray, alarm: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.bincount(
        group_index, weights=np.asarray(outcome, dtype=np.float64), minlength=n_groups
    ) > 0
    captured = np.bincount(
        group_index, weights=np.asarray(outcome & alarm, dtype=np.float64), minlength=n_groups
    ) > 0
    return positive, captured


def _count_vector(labels: dict[str, np.ndarray], alarm: np.ndarray, scope_mask: np.ndarray, group_index: np.ndarray, n_groups: int) -> dict[str, np.ndarray]:
    y = labels["outcome"]
    measured = labels["measured"]
    not_alarm = ~alarm
    masks = {
        "N": scope_mask,
        "O": scope_mask & y,
        "A": scope_mask & alarm,
        "TP": scope_mask & alarm & y,
        "FP": scope_mask & alarm & ~y,
        "TN": scope_mask & not_alarm & ~y,
        "FN": scope_mask & not_alarm & y,
        "M": scope_mask & measured,
        "alarm_measured": scope_mask & alarm & measured,
        "noalarm_measured": scope_mask & not_alarm & measured,
        "unmeasured_alarm": scope_mask & alarm & ~measured,
        "measured_alarm_without_outcome": scope_mask & alarm & measured & ~y,
    }
    result = {name: _group_sum(group_index, mask.astype(np.float64), n_groups) for name, mask in masks.items()}
    result["exposure_hours"] = _group_sum(
        group_index,
        np.where(scope_mask, labels["exposure_hours"], 0.0),
        n_groups,
    )
    patient_positive, patient_captured = _patient_bits(group_index, scope_mask & y, alarm, n_groups)
    result["patient_positive"] = patient_positive.astype(np.float64)
    result["patient_captured"] = (patient_positive & patient_captured).astype(np.float64)
    return result


def _metric_values(counts: dict[str, np.ndarray] | dict[str, float]) -> dict[str, Any]:
    get = lambda name: np.asarray(counts[name], dtype=np.float64)
    N, O, A = get("N"), get("O"), get("A")
    TP, FP, TN, FN = get("TP"), get("FP"), get("TN"), get("FN")
    measured_rate_alarm = _safe_ratio(get("alarm_measured"), A)
    measured_rate_noalarm = _safe_ratio(get("noalarm_measured"), N - A)
    return {
        "n_reference_windows": N,
        "measurement_ascertained_deterioration_count": O,
        "measurement_ascertained_deterioration_rate": _safe_ratio(O, N),
        "window_sensitivity": _safe_ratio(TP, TP + FN),
        "ppv_lower_bound": _safe_ratio(TP, TP + FP),
        "specificity": _safe_ratio(TN, TN + FP),
        "false_positive_count": FP,
        "true_positive_count": TP,
        "false_negative_count": FN,
        "true_negative_count": TN,
        "alarm_count": A,
        "alarm_measured_window_count": get("alarm_measured"),
        "alarmed_window_count": A,
        "non_alarm_measured_window_count": get("noalarm_measured"),
        "non_alarmed_window_count": N - A,
        "alarms_per_reference_eligible_patient_day": _safe_ratio(A, get("exposure_hours") / 24.0),
        "patient_capture_rate": _safe_ratio(get("patient_captured"), get("patient_positive")),
        "measurement_rate_alarmed": measured_rate_alarm,
        "measurement_rate_non_alarmed": measured_rate_noalarm,
        "measurement_rate_difference_alarm_minus_noalarm": measured_rate_alarm - measured_rate_noalarm,
        "unmeasured_alarm_count": get("unmeasured_alarm"),
        "measured_alarm_without_measurement_ascertained_deterioration_count": get("measured_alarm_without_outcome"),
    }


def _point_counts(group_counts_by_seed: list[dict[str, np.ndarray]]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    names = group_counts_by_seed[0].keys()
    points_by_seed = []
    boot_by_name: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for counts in group_counts_by_seed:
        points = {name: np.sum(value) for name, value in counts.items()}
        points_by_seed.append(_metric_values(points))
        for name, group_value in counts.items():
            boot_by_name[name].append(group_value)
    point = {}
    for key in points_by_seed[0]:
        vals = np.asarray([row[key] for row in points_by_seed], dtype=np.float64)
        point[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return point, {name: np.stack(values) for name, values in boot_by_name.items()}


def _bootstrap_metric_rows(
    group_counts_by_seed: list[dict[str, np.ndarray]],
    multiplicity: np.ndarray,
) -> tuple[dict[str, tuple[float, float, int]], np.ndarray]:
    metric_draws: dict[str, list[np.ndarray]] = {}
    seed_nb = []
    for counts in group_counts_by_seed:
        sampled = {name: multiplicity @ vector for name, vector in counts.items()}
        metrics = _metric_values(sampled)
        for name, values in metrics.items():
            metric_draws.setdefault(name, []).append(np.asarray(values, dtype=np.float64))
        N = sampled["N"]
        TP, FP = sampled["TP"], sampled["FP"]
        seed_nb.append(np.stack([
            _safe_ratio(TP, N) - _safe_ratio(FP, N) * (pt / (1.0 - pt))
            for pt in THRESHOLDS
        ], axis=1))
    averaged = {name: np.nanmean(np.stack(values), axis=0) for name, values in metric_draws.items()}
    nb = np.nanmean(np.stack(seed_nb), axis=0)
    cis = {name: _percentile_ci(values) for name, values in averaged.items()}
    return cis, nb


def _nb_from_counts(group_counts_by_seed: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    per_seed = []
    for counts in group_counts_by_seed:
        total = {name: float(np.sum(value)) for name, value in counts.items()}
        N = total["N"]
        if N <= 0:
            per_seed.append(np.full(len(THRESHOLDS), np.nan))
            continue
        odds = THRESHOLDS / (1.0 - THRESHOLDS)
        per_seed.append(total["TP"] / N - total["FP"] / N * odds)
    model_nb = np.nanmean(np.stack(per_seed), axis=0)
    prevalence_by_seed = [float(np.sum(row["O"]) / np.sum(row["N"])) if np.sum(row["N"]) else np.nan for row in group_counts_by_seed]
    treat_all_by_seed = [p - (1 - p) * (THRESHOLDS / (1 - THRESHOLDS)) for p in prevalence_by_seed]
    return {
        "model": model_nb,
        "no_alarm": np.zeros(len(THRESHOLDS), dtype=np.float64),
        "alert_all": np.nanmean(np.stack(treat_all_by_seed), axis=0),
    }


def _metric_boot_arrays(
    group_counts_by_seed: list[dict[str, np.ndarray]],
    multiplicity: np.ndarray,
) -> dict[str, np.ndarray]:
    per_seed = []
    for counts in group_counts_by_seed:
        names = list(counts)
        group_matrix = np.stack([counts[name] for name in names], axis=1)
        sampled_matrix = multiplicity @ group_matrix
        sampled = {name: sampled_matrix[:, column] for column, name in enumerate(names)}
        per_seed.append(_metric_values(sampled))
    result = {}
    for name in per_seed[0]:
        result[name] = np.nanmean(np.stack([np.asarray(row[name]) for row in per_seed]), axis=0)
    return result


def _bootstrap_net_benefit(
    group_counts_by_seed: list[dict[str, np.ndarray]], multiplicity: np.ndarray
) -> np.ndarray:
    odds = THRESHOLDS / (1.0 - THRESHOLDS)
    seed_curves = []
    for counts in group_counts_by_seed:
        group_matrix = np.stack([counts[name] for name in ("N", "TP", "FP")], axis=1)
        sampled = multiplicity @ group_matrix
        denominator, true_positive, false_positive = sampled.T
        prevalence = _safe_ratio(true_positive, denominator)
        false_positive_rate = _safe_ratio(false_positive, denominator)
        seed_curves.append(prevalence[:, None] - false_positive_rate[:, None] * odds[None, :])
    return np.nanmean(np.stack(seed_curves), axis=0)


def _bootstrap_stats_for_run(
    labels: dict[str, np.ndarray], alarm: np.ndarray, scope_mask: np.ndarray,
    group_index: np.ndarray, n_groups: int,
) -> dict[str, np.ndarray]:
    # one run's cluster-level sufficient statistics, ready for paired resampling
    return _count_vector(labels, alarm, scope_mask, group_index, n_groups)


def _agreement_data(artifact: dict[str, np.ndarray], labels: dict[str, np.ndarray], component_index: int, component: dict[str, Any], group_index: np.ndarray, n_groups: int) -> dict[str, np.ndarray]:
    id_to_row = {int(sample_id): row for row, sample_id in enumerate(artifact["sample_id"].astype(np.int64).tolist())}
    keep = artifact["event_component_index"].astype(np.int64) == component_index
    sample_ids = artifact["event_sample_id"][keep].astype(np.int64)
    rows = np.asarray([id_to_row[int(sample_id)] for sample_id in sample_ids], dtype=np.int64)
    keep_ref = labels["reference_seen"][rows]
    rows = rows[keep_ref]
    event_rows = np.flatnonzero(keep)[keep_ref]
    true_grade = grade(artifact["event_true_raw"][event_rows], component)
    pred_grade = grade(artifact["event_pred_raw"][event_rows], component)
    ref_grade = labels["reference_grade"][rows]
    sample_group = group_index[rows]
    n_measurements = np.bincount(sample_group, minlength=n_groups).astype(np.float64)
    model_error = np.bincount(sample_group, weights=(pred_grade != true_grade).astype(np.float64), minlength=n_groups)
    carry_error = np.bincount(sample_group, weights=(ref_grade != true_grade).astype(np.float64), minlength=n_groups)
    return {
        "n": n_measurements,
        "model_error": model_error,
        "carry_error": carry_error,
        "event_stay_ids": labels["stay_id"][rows].astype(np.int64),
        "event_patient_ids": labels["patient_id"][rows].astype(str),
    }


def _run_analysis() -> dict[str, Any]:
    ledger, batch_manifest, files = _load_and_validate_inputs()
    outputs = [OUT_METRICS, OUT_NB, OUT_AGREEMENT, OUT_ASCERTAINMENT, OUT_VERDICT, OUT_FIG_PDF, OUT_FIG_PNG]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing P1 analysis outputs: {existing}")

    metrics_rows: list[dict[str, Any]] = []
    nb_rows: list[dict[str, Any]] = []
    ascertainment_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    verdict_cells = []
    input_hashes = {
        ACTIVE_LEDGER_PATH.name: _sha256(ACTIVE_LEDGER_PATH),
        ACTIVE_BATCH_MANIFEST_PATH.name: _sha256(ACTIVE_BATCH_MANIFEST_PATH),
    }
    is_validation_rehearsal = VALIDATION_REHEARSAL_MANIFEST is not None
    analysis_required_seeds = [2024] if is_validation_rehearsal else list(SEEDS)
    seeds_by_cohort: dict[str, dict[str, list[int]]] = {}
    not_completed_keys = set(
        str(key) for key in ledger.get("not_completed_run_keys", [])
    ) if not is_validation_rehearsal else set()
    seed_coverage_by_cohort: dict[str, dict[str, dict[str, Any]]] = {}

    for cohort in COHORTS:
        seeds_by_model: dict[str, list[int]] = {}
        for model in MODELS:
            observed = sorted(
                int(key.rsplit("seed", 1)[1])
                for key in files[cohort]
                if key.startswith(f"{model}/seed")
            )
            if not observed or any(seed not in SEEDS for seed in observed):
                raise RuntimeError(f"invalid prediction seed coverage for {cohort}/{model}: {observed}")
            missing_from_fixed_set = {
                int(key.rsplit("seed", 1)[1])
                for key in not_completed_keys
                if key.startswith(f"{cohort}/{model}/seed")
            }
            expected_observed = sorted(set(SEEDS) - missing_from_fixed_set)
            if is_validation_rehearsal:
                expected_observed = analysis_required_seeds
            if observed != expected_observed:
                raise RuntimeError(
                    f"prediction seed coverage for {cohort}/{model} does not match the resolved fixed denominator: "
                    f"observed={observed}, expected={expected_observed}"
                )
            seeds_by_model[model] = observed
        seed_coverage_by_cohort[cohort] = {
            model: {
                "observed_seeds": list(seeds_by_model[model]),
                "required_seeds": list(analysis_required_seeds),
                "observed_seed_count": len(seeds_by_model[model]),
                "required_seed_count": len(analysis_required_seeds),
                "complete": seeds_by_model[model] == analysis_required_seeds,
            }
            for model in MODELS
        }
        union_seeds = set().union(*(set(values) for values in seeds_by_model.values()))
        if VALIDATION_REHEARSAL_MANIFEST is not None:
            if any(values != [2024] for values in seeds_by_model.values()):
                raise RuntimeError(f"validation rehearsal requires exactly seed2024 for each model in {cohort}")
        elif union_seeds != set(SEEDS):
            raise RuntimeError(f"test analysis seed union is not the frozen set {SEEDS} for {cohort}")
        seeds_by_cohort[cohort] = seeds_by_model
        reference_seed = seeds_by_model[MODELS[0]][0]
        sample_artifact = files[cohort][f"{MODELS[0]}/seed{reference_seed}"]
        sample_ids = sample_artifact["sample_id"].astype(np.int64)
        n_samples = len(sample_ids)
        for model in MODELS:
            for seed in seeds_by_model[model]:
                artifact = files[cohort][f"{model}/seed{seed}"]
                input_hashes[f"{cohort}/{model}/seed{seed}"] = str(ledger["run_results"][f"{cohort}/{model}/seed{seed}"]["sha256"])
                for field in ("sample_id", "batch_index", "stay_id", "patient_id", "landmark_hour", "outcome_exposure_hours"):
                    if not np.array_equal(sample_artifact[field], artifact[field]):
                        raise RuntimeError(f"sample order or metadata differs across runs: {cohort}/{model}/seed{seed}/{field}")
        patient_ids = sample_artifact["patient_id"].astype(str)
        group_index, multiplicity = _patient_bootstrap(patient_ids)
        n_groups = multiplicity.shape[1]

        for component_index, component in enumerate(COMPONENTS):
            model_store: dict[str, dict[str, Any]] = {}
            for model in MODELS:
                seeds = seeds_by_cohort[cohort][model]
                seed_coverage_complete = seeds == analysis_required_seeds
                labels_by_seed = []
                alarms_by_seed = []
                for seed in seeds:
                    artifact = files[cohort][f"{model}/seed{seed}"]
                    labels = _window_labels(artifact, component_index, component)
                    if not np.array_equal(labels["sample_id"], sample_ids):
                        raise RuntimeError("sample IDs changed while building component outcomes")
                    labels_by_seed.append(labels)
                    alarms_by_seed.append(labels["alarm"])

                for scope in SCOPES:
                    scope_counts_by_strategy: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in STRATEGIES}
                    scope_labels_by_seed = labels_by_seed
                    for seed_index, labels in enumerate(scope_labels_by_seed):
                        scope_mask = labels["reference_seen"].copy()
                        if scope == "measured_windows_only":
                            scope_mask &= labels["measured"]
                        for strategy in STRATEGIES:
                            alarm = (
                                alarms_by_seed[seed_index] if strategy == model
                                else np.zeros(n_samples, dtype=bool) if strategy == "no-alarm"
                                else np.ones(n_samples, dtype=bool) if strategy == "alert-all"
                                else alarms_by_seed[seed_index]
                            )
                            # The three model strategies are handled when the matching model is processed.
                            if strategy in MODELS and strategy != model:
                                continue
                            counts = _bootstrap_stats_for_run(labels, alarm, scope_mask, group_index, n_groups)
                            scope_counts_by_strategy[strategy].append(counts)

                    strategy_set = [model, "no-alarm", "alert-all"]
                    for strategy in strategy_set:
                        group_counts = scope_counts_by_strategy[strategy]
                        if len(group_counts) != len(seeds):
                            raise RuntimeError(f"missing seed-level sufficient statistics for {cohort}/{component['variable']}/{strategy}")
                        point_by_seed = []
                        for count in group_counts:
                            total = {name: float(np.sum(values)) for name, values in count.items()}
                            point_by_seed.append(_metric_values(total))
                        point = {
                            key: float(np.nanmean([row[key] for row in point_by_seed]))
                            if np.isfinite(np.asarray([row[key] for row in point_by_seed], dtype=np.float64)).any()
                            else float("nan")
                            for key in point_by_seed[0]
                        }
                        boot_metrics = _metric_boot_arrays(group_counts, multiplicity)
                        ci_metrics = {name: _percentile_ci(values) for name, values in boot_metrics.items()}
                        strategy_rows = {
                            "analysis_plan_version": "p1_landmark_v1",
                            "cohort": cohort,
                            "organ": component["organ"],
                            "component": component["variable"],
                            "model_or_strategy": strategy,
                            "scope": scope,
                            "seed_aggregation": "equal-weight mean of seed-specific metrics",
                            "observed_seeds": ",".join(str(seed) for seed in seeds),
                            "observed_seed_count": len(seeds),
                            "required_seed_count": len(analysis_required_seeds),
                            "seed_coverage_complete": seed_coverage_complete,
                            "bootstrap_n": BOOTSTRAP_REPS,
                            "bootstrap_seed": BOOTSTRAP_SEED,
                            "bootstrap_unit": "patient cluster, stratified within cohort",
                            "measurement_ascertained_deterioration_definition": "measured outcome-window component grade strictly worse than last input grade; otherwise 0",
                            "ppv_interpretation": "lower bound on latent clinical PPV; unmeasured alarm windows may contain unrecorded events",
                        }
                        for metric_name, value in point.items():
                            strategy_rows[metric_name] = value
                            lo, hi, valid = ci_metrics.get(metric_name, (float("nan"), float("nan"), 0))
                            strategy_rows[f"{metric_name}_ci_low"] = lo
                            strategy_rows[f"{metric_name}_ci_high"] = hi
                            strategy_rows[f"{metric_name}_bootstrap_valid_replicates"] = valid
                        if scope == "all_reference_windows":
                            stay_alarm = {}
                            for seed_index, labels in enumerate(labels_by_seed):
                                alarm = (
                                    labels["alarm"] if strategy == model
                                    else np.zeros(n_samples, dtype=bool) if strategy == "no-alarm"
                                    else np.ones(n_samples, dtype=bool)
                                )
                                fp_window = labels["reference_seen"] & alarm & ~labels["outcome"]
                                unmeasured_fp = fp_window & ~labels["measured"]
                                measured_fp = fp_window & labels["measured"]
                                for name, mask in (("o0", fp_window), ("unmeasured", unmeasured_fp), ("measured_without_o1", measured_fp)):
                                    by_stay = {}
                                    for stay_id, count in zip(labels["stay_id"][mask], np.ones(mask.sum(), dtype=np.int64)):
                                        by_stay[int(stay_id)] = by_stay.get(int(stay_id), 0) + int(count)
                                    all_stays = np.unique(labels["stay_id"][labels["reference_seen"]])
                                    values = np.asarray([by_stay.get(int(stay), 0) for stay in all_stays], dtype=np.float64)
                                    stay_alarm.setdefault(name, []).append(values)
                            for name, arrays in stay_alarm.items():
                                common = np.mean(np.stack(arrays), axis=0)
                                strategy_rows[f"per_stay_{name}_alarm_median"] = float(np.median(common))
                                strategy_rows[f"per_stay_{name}_alarm_iqr"] = float(np.percentile(common, 75) - np.percentile(common, 25))
                                strategy_rows[f"per_stay_{name}_alarm_p90"] = float(np.percentile(common, 90))
                            patient_rates = []
                            for seed_index, labels in enumerate(labels_by_seed):
                                alarm = (
                                    labels["alarm"] if strategy == model
                                    else np.zeros(n_samples, dtype=bool) if strategy == "no-alarm"
                                    else np.ones(n_samples, dtype=bool)
                                )
                                eligible = labels["reference_seen"]
                                patient_alarms = _group_sum(
                                    group_index, (eligible & alarm).astype(np.float64), n_groups
                                )
                                patient_days = _group_sum(
                                    group_index,
                                    np.where(eligible, labels["exposure_hours"], 0.0) / 24.0,
                                    n_groups,
                                )
                                rates = np.full(n_groups, np.nan, dtype=np.float64)
                                np.divide(patient_alarms, patient_days, out=rates, where=patient_days > 0)
                                patient_rates.append(rates)
                            mean_patient_rates = np.nanmean(np.stack(patient_rates), axis=0)
                            finite_patient_rates = mean_patient_rates[np.isfinite(mean_patient_rates)]
                            strategy_rows["per_patient_alarm_rate_n"] = int(finite_patient_rates.size)
                            strategy_rows["per_patient_alarms_per_reference_eligible_patient_day_median"] = (
                                float(np.median(finite_patient_rates)) if finite_patient_rates.size else float("nan")
                            )
                            strategy_rows["per_patient_alarms_per_reference_eligible_patient_day_iqr"] = (
                                float(np.percentile(finite_patient_rates, 75) - np.percentile(finite_patient_rates, 25))
                                if finite_patient_rates.size else float("nan")
                            )
                            strategy_rows["per_patient_alarms_per_reference_eligible_patient_day_p90"] = (
                                float(np.percentile(finite_patient_rates, 90)) if finite_patient_rates.size else float("nan")
                            )
                        metrics_rows.append(strategy_rows)

                        if scope == "all_reference_windows":
                            ascertainment_rows.append({
                                "analysis_plan_version": "p1_landmark_v1",
                                "cohort": cohort,
                                "organ": component["organ"],
                                "component": component["variable"],
                                "model_or_strategy": strategy,
                                "seed_aggregation": "equal-weight mean of seed-specific counts and rates",
                                "observed_seeds": ",".join(str(seed) for seed in seeds),
                                "observed_seed_count": len(seeds),
                                "required_seed_count": len(analysis_required_seeds),
                                "seed_coverage_complete": seed_coverage_complete,
                                "bootstrap_n": BOOTSTRAP_REPS,
                                "bootstrap_seed": BOOTSTRAP_SEED,
                                "bootstrap_unit": "patient cluster, stratified within cohort",
                                "alarm_measurement_count_mean_across_seeds": point["alarm_measured_window_count"],
                                "alarmed_window_count_mean_across_seeds": point["alarmed_window_count"],
                                "measurement_rate_alarmed": point["measurement_rate_alarmed"],
                                "measurement_rate_alarmed_ci_low": ci_metrics["measurement_rate_alarmed"][0],
                                "measurement_rate_alarmed_ci_high": ci_metrics["measurement_rate_alarmed"][1],
                                "measurement_rate_alarmed_valid_bootstrap_replicates": ci_metrics["measurement_rate_alarmed"][2],
                                "non_alarm_measurement_count_mean_across_seeds": point["non_alarm_measured_window_count"],
                                "non_alarmed_window_count_mean_across_seeds": point["non_alarmed_window_count"],
                                "measurement_rate_non_alarmed": point["measurement_rate_non_alarmed"],
                                "measurement_rate_non_alarmed_ci_low": ci_metrics["measurement_rate_non_alarmed"][0],
                                "measurement_rate_non_alarmed_ci_high": ci_metrics["measurement_rate_non_alarmed"][1],
                                "measurement_rate_non_alarmed_valid_bootstrap_replicates": ci_metrics["measurement_rate_non_alarmed"][2],
                                "measurement_rate_difference_alarm_minus_noalarm": point["measurement_rate_difference_alarm_minus_noalarm"],
                                "measurement_rate_difference_ci_low": ci_metrics["measurement_rate_difference_alarm_minus_noalarm"][0],
                                "measurement_rate_difference_ci_high": ci_metrics["measurement_rate_difference_alarm_minus_noalarm"][1],
                                "measurement_rate_difference_valid_bootstrap_replicates": ci_metrics["measurement_rate_difference_alarm_minus_noalarm"][2],
                                "unmeasured_alarm_count_mean_across_seeds": point["unmeasured_alarm_count"],
                                "unmeasured_alarm_count_ci_low": ci_metrics["unmeasured_alarm_count"][0],
                                "unmeasured_alarm_count_ci_high": ci_metrics["unmeasured_alarm_count"][1],
                                "unmeasured_alarm_count_valid_bootstrap_replicates": ci_metrics["unmeasured_alarm_count"][2],
                                "measured_alarm_without_measurement_ascertained_deterioration_count_mean_across_seeds": point["measured_alarm_without_measurement_ascertained_deterioration_count"],
                                "measured_alarm_without_measurement_ascertained_deterioration_count_ci_low": ci_metrics["measured_alarm_without_measurement_ascertained_deterioration_count"][0],
                                "measured_alarm_without_measurement_ascertained_deterioration_count_ci_high": ci_metrics["measured_alarm_without_measurement_ascertained_deterioration_count"][1],
                                "measured_alarm_without_measurement_ascertained_deterioration_count_valid_bootstrap_replicates": ci_metrics["measured_alarm_without_measurement_ascertained_deterioration_count"][2],
                            })

                        boot_nb_curve = _bootstrap_net_benefit(group_counts, multiplicity)
                        for threshold_index, threshold in enumerate(THRESHOLDS):
                            net_by_seed = []
                            treat_all_by_seed = []
                            for count in group_counts:
                                total = {name: float(np.sum(values)) for name, values in count.items()}
                                N = total["N"]
                                prevalence = total["O"] / N if N else np.nan
                                odds = float(threshold / (1.0 - threshold))
                                net = total["TP"] / N - total["FP"] / N * odds if N else np.nan
                                net_by_seed.append(net)
                                treat_all_by_seed.append(prevalence - (1.0 - prevalence) * odds if N else np.nan)
                            point_nb = float(np.nanmean(net_by_seed))
                            lo, hi, n_valid = _percentile_ci(boot_nb_curve[:, threshold_index])
                            nb_rows.append({
                                **{key: strategy_rows[key] for key in ("analysis_plan_version", "cohort", "organ", "component", "model_or_strategy", "scope", "bootstrap_n", "bootstrap_seed", "bootstrap_unit")},
                                "threshold_probability": float(threshold),
                                "net_benefit": point_nb,
                                "net_benefit_ci_low": lo,
                                "net_benefit_ci_high": hi,
                                "treat_none_net_benefit": 0.0,
                                "treat_all_net_benefit": float(np.nanmean(treat_all_by_seed)),
                                "valid_bootstrap_replicates": n_valid,
                            })

                        if scope == "all_reference_windows":
                            nb_curves = _nb_from_counts(group_counts)
                            model_store[strategy] = {
                                "metrics": point,
                                "net_benefit": nb_curves["model"],
                                "alert_all_net_benefit": nb_curves["alert_all"],
                                "counts": group_counts,
                            }

                # Event-level I1 uses the same bootstrap patient draws and is paired across seeds.
                agreement_by_seed = []
                for seed_index, seed in enumerate(seeds):
                    artifact = files[cohort][f"{model}/seed{seed}"]
                    agreement_by_seed.append(_agreement_data(artifact, labels_by_seed[seed_index], component_index, component, group_index, n_groups))
                point_errors = []
                boot_diff_by_seed = []
                boot_model_by_seed = []
                boot_carry_by_seed = []
                for agreement in agreement_by_seed:
                    n_event = float(np.sum(agreement["n"]))
                    point_errors.append((
                        float(np.sum(agreement["model_error"]) / n_event) if n_event else np.nan,
                        float(np.sum(agreement["carry_error"]) / n_event) if n_event else np.nan,
                        n_event,
                    ))
                    boot_n = multiplicity @ agreement["n"]
                    boot_m = multiplicity @ agreement["model_error"]
                    boot_c = multiplicity @ agreement["carry_error"]
                    boot_model_by_seed.append(_safe_ratio(boot_m, boot_n))
                    boot_carry_by_seed.append(_safe_ratio(boot_c, boot_n))
                    boot_diff_by_seed.append(_safe_ratio(boot_m, boot_n) - _safe_ratio(boot_c, boot_n))
                model_error_mean = float(np.nanmean([item[0] for item in point_errors]))
                carry_error_mean = float(np.nanmean([item[1] for item in point_errors]))
                difference_mean = model_error_mean - carry_error_mean
                model_err_ci = _percentile_ci(np.nanmean(np.stack(boot_model_by_seed), axis=0))
                carry_err_ci = _percentile_ci(np.nanmean(np.stack(boot_carry_by_seed), axis=0))
                diff_ci = _percentile_ci(np.nanmean(np.stack(boot_diff_by_seed), axis=0))
                n_events = int(round(float(np.nanmean([item[2] for item in point_errors]))))
                i1_events = agreement_by_seed[0]
                n_stays = int(np.unique(i1_events["event_stay_ids"]).size)
                n_patients = int(np.unique(i1_events["event_patient_ids"]).size)
                agreement_rows.append({
                    "analysis_plan_version": "p1_landmark_v1",
                    "cohort": cohort,
                    "organ": component["organ"],
                    "component": component["variable"],
                    "model": model,
                    "seed_aggregation": "equal-weight mean of seed-specific event misclassification rates",
                    "observed_seeds": ",".join(str(seed) for seed in seeds),
                    "observed_seed_count": len(seeds),
                    "required_seed_count": len(analysis_required_seeds),
                    "seed_coverage_complete": seed_coverage_complete,
                    "model_event_misclassification_rate": model_error_mean,
                    "model_event_misclassification_ci_low": model_err_ci[0],
                    "model_event_misclassification_ci_high": model_err_ci[1],
                    "carry_forward_event_misclassification_rate": carry_error_mean,
                    "carry_forward_event_misclassification_ci_low": carry_err_ci[0],
                    "carry_forward_event_misclassification_ci_high": carry_err_ci[1],
                    "model_minus_carry_forward_difference": difference_mean,
                    "difference_ci_low": diff_ci[0],
                    "difference_ci_high": diff_ci[1],
                    "n_measured_events_mean_across_seeds": n_events,
                    "n_stays": n_stays,
                    "n_patients": n_patients,
                    "carry_forward_strictly_better": bool(
                        seed_coverage_complete and carry_error_mean < model_error_mean
                    ),
                    "bootstrap_n": BOOTSTRAP_REPS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "patient cluster",
                })

            # I2 uses the prespecified main-scope, seed-averaged net benefits.
            cell_models = [model_store.get(model) for model in MODELS]
            passed = []
            for model, values in zip(MODELS, cell_models):
                if values is None:
                    passed.append(False)
                    continue
                model_nb = values["net_benefit"]
                alert_all_nb = values["alert_all_net_benefit"]
                wins = (model_nb > 0.0) & (model_nb > alert_all_nb)
                coverage_complete = seed_coverage_by_cohort[cohort][model]["complete"]
                passed.append(bool(coverage_complete and np.sum(wins) >= 14))
            n_model_pass = int(sum(passed))
            verdict_cells.append({
                "cohort": cohort,
                "organ": component["organ"],
                "component": component["variable"],
                "model_passes": dict(zip(MODELS, passed)),
                "model_seed_coverage_complete": {
                    model: seed_coverage_by_cohort[cohort][model]["complete"] for model in MODELS
                },
                "model_observed_seed_counts": {
                    model: seed_coverage_by_cohort[cohort][model]["observed_seed_count"] for model in MODELS
                },
                "models_passing_14_of_26_thresholds": n_model_pass,
                "cell_pass": bool(n_model_pass >= 2),
            })

    i1_count = sum(bool(row["carry_forward_strictly_better"]) for row in agreement_rows)
    i2_count = sum(bool(row["cell_pass"]) for row in verdict_cells)
    i1_pass = i1_count >= 18
    i2_pass = i2_count >= 4
    branch = "A" if i1_pass and i2_pass else "B" if i1_pass else "C"

    _write_csv(OUT_METRICS, metrics_rows)
    _write_csv(OUT_NB, nb_rows)
    _write_csv(OUT_AGREEMENT, agreement_rows)
    _write_csv(OUT_ASCERTAINMENT, ascertainment_rows)
    observed_runs = sum(len(files[cohort]) for cohort in COHORTS)
    verdict = {
        "schema_version": "p1_landmark_verdict_v1",
        "analysis_plan_version": "p1_landmark_v1",
        "analysis_scope": "validation_rehearsal_only" if is_validation_rehearsal else "frozen_test_analysis",
        "split_used": "validation" if is_validation_rehearsal else "test",
        "analysis_plan_sha256": _sha256(PLAN),
        "analysis_input_manifest_sha256": _sha256(ACTIVE_LEDGER_PATH),
        "analysis_batch_manifest_sha256": _sha256(ACTIVE_BATCH_MANIFEST_PATH),
        **({
            "test_inference_ledger_sha256": _sha256(LEDGER),
            "test_batch_manifest_sha256": _sha256(BATCH_MANIFEST),
        } if not is_validation_rehearsal else {}),
        "prediction_input_sha256": input_hashes,
        "fixed_denominator": {
            "planned_runs": 30 if not is_validation_rehearsal else None,
            "observed_runs": observed_runs,
            "completed_runs": observed_runs if not is_validation_rehearsal else None,
            "not_completed_run_keys": sorted(not_completed_keys),
            "missing_runs_remain_nonpassing_in_denominator": not is_validation_rehearsal,
            "rehearsal_input_run_count": observed_runs if is_validation_rehearsal else None,
            "patient_resampling": True,
        },
        "seed_coverage_by_cohort": seed_coverage_by_cohort,
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "patient cluster, stratified within cohort"},
        "I1": {"favorable_cells": i1_count, "required": 18, "pass": i1_pass, "cell_count": 24, "rehearsal_only": is_validation_rehearsal},
        "I2": {"passing_cells": i2_count, "required": 4, "pass": i2_pass, "cell_count": 8, "scope": "all_reference_windows only", "rehearsal_only": is_validation_rehearsal},
        "selected_branch": None if is_validation_rehearsal else branch,
        "candidate_branch_for_rehearsal": branch if is_validation_rehearsal else None,
        "I2_cells": verdict_cells,
        "interpretation": {
            "primary_outcome": "measurement-ascertained deterioration",
            "unmeasured_windows": "coded 0 under the operational endpoint; this does not mean no latent clinical deterioration",
            "ppv": "lower bound on latent clinical PPV because unmeasured alarms may contain unrecorded events",
            "measurement_process": "report observed alarm-stratified measurement rates; do not impute or weight",
            "test_use": _test_use_interpretation(
                ledger, is_validation_rehearsal=is_validation_rehearsal
            ),
        },
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    _atomic_json(OUT_VERDICT, verdict)
    _plot(metrics_rows)
    print(json.dumps({
        "analysis_scope": verdict["analysis_scope"],
        "I1": verdict["I1"], "I2": verdict["I2"],
        "candidate_branch_for_rehearsal" if is_validation_rehearsal else "branch": branch,
    }, indent=2))
    return verdict


def _plot(metrics_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategy_order = (*MODELS, "no-alarm", "alert-all")
    colors = {"ICUKANForecaster": "#3A6EA5", "GraFITi": "#2D8C63", "HyperIMTS": "#8055A5", "no-alarm": "#5C6770", "alert-all": "#D97A38"}
    component_order = [item["variable"] for item in COMPONENTS]
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.2), sharex="col")
    for col, component in enumerate(component_order):
        for row, metric_name in enumerate(("window_sensitivity", "alarms_per_reference_eligible_patient_day")):
            ax = axes[row, col]
            ax.grid(axis="y", color="#dce2e8", linewidth=0.65)
            ax.set_axisbelow(True)
            for strategy_index, strategy in enumerate(strategy_order):
                for cohort_index, cohort in enumerate(COHORTS):
                    entry = next((item for item in metrics_rows if item["cohort"] == cohort and item["component"] == component and item["model_or_strategy"] == strategy and item["scope"] == "all_reference_windows"), None)
                    if entry is None:
                        continue
                    value = entry.get(metric_name, float("nan"))
                    if not np.isfinite(value):
                        continue
                    low = entry.get(f"{metric_name}_ci_low", value)
                    high = entry.get(f"{metric_name}_ci_high", value)
                    x = strategy_index + (-0.14 if cohort_index == 0 else 0.14)
                    ax.errorbar(x, value, yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                                fmt="o" if cohort_index == 0 else "s", color=colors[strategy],
                                markersize=4.2, linewidth=0.8, capsize=1.5, alpha=0.92)
            ax.set_title(component, fontsize=9)
            if row == 0:
                ax.set_ylim(-0.04, 1.04)
                if col == 0:
                    ax.set_ylabel("Sensitivity for measurement-\nascertained deterioration")
            elif col == 0:
                ax.set_ylabel("Alarms per reference-eligible patient-day")
            ax.set_xticks(range(len(strategy_order)))
            ax.set_xticklabels(["CarryGate", "GraFITi", "HyperIMTS", "No alarm", "Alert all"], rotation=32, ha="right", fontsize=7)
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", linestyle="", color="#333", label="eICU"),
                        plt.Line2D([0], [0], marker="s", linestyle="", color="#333", label="MIMIC-IV")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    title = (
        "P1 validation-only I1/I2 rehearsal · measurement-ascertained deterioration"
        if VALIDATION_REHEARSAL_MANIFEST is not None
        else "P1 held-out ICU use case · measurement-ascertained deterioration"
    )
    fig.suptitle(title, y=1.055, fontsize=12)
    fig.text(
        0.5, 0.005,
        (
            "Validation subset only; this rehearsal is not a study result. "
            "Unmeasured windows are coded 0; PPV is a lower bound."
            if VALIDATION_REHEARSAL_MANIFEST is not None else
            "eICU and MIMIC-IV fixed test splits; patient-cluster bootstrap 95% CIs. "
            "Unmeasured windows are coded 0; PPV is a lower bound. Retrospective measurement may differ by alarm status."
        ),
        ha="center", va="bottom", fontsize=7, color="#4f5b66",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global VALIDATION_REHEARSAL_MANIFEST
    global OUT_METRICS, OUT_NB, OUT_AGREEMENT, OUT_ASCERTAINMENT, OUT_VERDICT, OUT_FIG_PDF, OUT_FIG_PNG
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="validate frozen test artifacts without recomputing the analyses")
    parser.add_argument(
        "--validation-rehearsal-manifest", type=Path, default=None,
        help="run the same I1/I2 computation on a separate validation-only rehearsal bundle",
    )
    args = parser.parse_args()
    if args.validation_rehearsal_manifest is not None:
        VALIDATION_REHEARSAL_MANIFEST = args.validation_rehearsal_manifest.resolve()
        rehearsal_root = VALIDATION_REHEARSAL_MANIFEST.parent / "analysis"
        OUT_METRICS = rehearsal_root / "p1_validation_rehearsal_metrics.csv"
        OUT_NB = rehearsal_root / "p1_validation_rehearsal_decision_curve.csv"
        OUT_AGREEMENT = rehearsal_root / "p1_validation_rehearsal_agreement.csv"
        OUT_ASCERTAINMENT = rehearsal_root / "p1_validation_rehearsal_ascertainment.csv"
        OUT_VERDICT = rehearsal_root / "p1_validation_rehearsal_verdict.json"
        OUT_FIG_PDF = rehearsal_root / "p1_validation_rehearsal.pdf"
        OUT_FIG_PNG = rehearsal_root / "p1_validation_rehearsal.png"
    if args.audit_only:
        if VALIDATION_REHEARSAL_MANIFEST is not None:
            raise ValueError("--audit-only is reserved for the frozen test inference ledger")
        ledger, batch_manifest, files = _load_and_validate_inputs()
        print(json.dumps({"status": "PASS", "checkpoint_runs": len(ledger["run_results"]), "cohorts": sorted(batch_manifest["cohorts"]), "prediction_counts": {cohort: len(files[cohort]) for cohort in COHORTS}}, indent=2))
        return 0
    _run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
