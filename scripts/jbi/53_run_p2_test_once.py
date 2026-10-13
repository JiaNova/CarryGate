#!/usr/bin/env python
"""Freeze all P2 checkpoints, then perform the single authorized test pass.

The ``freeze`` command only inspects run manifests and checkpoint files.  The
``test`` command refuses to construct a test loader until the full 90-run
checkpoint set is frozen and every artifact hash has been rechecked.  A lock
and write-once prediction paths make the test pass fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("ICU_KAN_PYTHON", sys.executable)
TRAINER_PATH = ROOT / "scripts/jbi/52_train_p2_loho.py"
FREEZE_PATH = ROOT / "results/jbi/p2_loho_checkpoint_freeze.json"
TRAIN_MANIFEST_PATH = ROOT / "results/jbi/p2_loho_training_manifest.json"
QUEUE_STATUS_PATH = ROOT / "results/jbi/p2_loho_queue_status.json"
SAMPLE_MANIFEST_PATH = ROOT / "results/jbi/p2_training_sample_manifest.json"
OVERLAP_AUDIT_PATH = ROOT / "results/jbi/p2_hospital_overlap_audit.json"
LOHO_CACHE_PATH = ROOT / "data/processed/eicu_loho_events.pkl"
VARIABLE_MAP_PATH = ROOT / "configs/mimiciv_variable_map.yaml"
PREDICTION_ROOT = ROOT / "results/jbi/p2_loho_predictions"
BATCH_MANIFEST_PATH = ROOT / "results/jbi/p2_loho_test_batch_manifest.json"
LEDGER_PATH = ROOT / "results/jbi/p2_loho_test_inference_ledger.json"
LOCK_PATH = ROOT / "results/jbi/.p2_loho_test_inference_once.lock"
FAILED_PREINFERENCE_LEDGER_PATH = (
    ROOT / "results/jbi/_work/p2_loho_test_failed_attempts/"
    "preinference_attempt/p2_loho_test_inference_ledger.json"
)
OBJECTIVES = ("MAE", "MSE")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
SEEDS = (2024, 2025, 2026)
N_FOLDS = 5
N_VARIABLES = 37
L_TOTAL_HOURS = 48.0
BATCH_SIZE = 128
GPU_MARGIN_MIB = 2048
REGISTERED_RETRY_REASONS = {
    "fold0/GraFITi/MAE/seed2024": "cuda_oom_before_p2_microbatch_path",
    **{
        f"fold4/HyperIMTS/{objective}/seed{seed}": "validation_oom_before_split_validation"
        for objective in OBJECTIVES for seed in SEEDS
    },
}
VALIDATION_SPLIT_RUN_KEYS = {
    f"fold4/HyperIMTS/{objective}/seed{seed}"
    for objective in OBJECTIVES for seed in SEEDS
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _load_trainer():
    spec = importlib.util.spec_from_file_location("p2_trainer_for_test_once", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_run_keys(trainer) -> set[str]:
    return {str(job["run_key"]) for job in trainer._jobs()}


def _event_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"queue event has no ISO timestamp: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"queue event has invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"queue event timestamp has no timezone: {value!r}")
    return parsed


def _validate_queue_attempt_history(
    queue: dict[str, Any], by_key: dict[str, dict[str, Any]], expected: set[str]
) -> dict[str, Any]:
    """Accept only the seven archived, registered failures followed by success."""
    started_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
    finished_by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
    for event in queue.get("runs_started", []):
        key = str(event.get("run_key"))
        if key not in started_by_key:
            raise RuntimeError(f"P2 queue has an unexpected started run: {key}")
        started_by_key[key].append(event)
    for event in queue.get("runs_finished", []):
        key = str(event.get("run_key"))
        if key not in finished_by_key:
            raise RuntimeError(f"P2 queue has an unexpected finished run: {key}")
        finished_by_key[key].append(event)

    if set(REGISTERED_RETRY_REASONS) - expected:
        raise RuntimeError("registered retry allowlist contains a run outside the fixed P2 set")
    for key in expected:
        row = by_key[key]
        history = row.get("attempt_history", [])
        if not isinstance(history, list):
            raise RuntimeError(f"{key}: attempt_history is not a list")
        allowed_reason = REGISTERED_RETRY_REASONS.get(key)
        if allowed_reason is None:
            if history:
                raise RuntimeError(f"{key}: unregistered retry history is not freeze-eligible")
        else:
            if len(history) != 1:
                raise RuntimeError(f"{key}: expected exactly one registered retry record")
            item = history[0]
            snapshot = item.get("failed_attempt_snapshot", {})
            if (
                item.get("attempt_number") != 1
                or item.get("reason") != allowed_reason
                or snapshot.get("run_key") != key
                or snapshot.get("status") != "failed"
            ):
                raise RuntimeError(f"{key}: registered retry does not match its archived failure")
            archived = item.get("archived", {})
            destinations = [entry.get("to") for entry in archived.values()
                            if isinstance(entry, dict) and entry.get("to")]
            if len(destinations) < 3:
                raise RuntimeError(f"{key}: retry does not record the checkpoint/log/epoch archives")
            for relative in destinations:
                path = (ROOT / str(relative)).resolve()
                if ROOT not in path.parents or not path.exists():
                    raise RuntimeError(f"{key}: archived retry artifact is missing or outside the project: {relative}")

        starts = sorted(started_by_key[key], key=lambda event: _event_time(event.get("started_at_utc")))
        finishes = sorted(finished_by_key[key], key=lambda event: _event_time(event.get("finished_at_utc")))
        expected_attempts = 1 + len(history)
        if len(starts) != expected_attempts or len(finishes) != expected_attempts:
            raise RuntimeError(
                f"{key}: queue history has {len(starts)} starts/{len(finishes)} finishes, "
                f"but manifest records {expected_attempts} attempts"
            )
        previous_finish: datetime | None = None
        for index, (start, finish) in enumerate(zip(starts, finishes)):
            start_time = _event_time(start.get("started_at_utc"))
            finish_time = _event_time(finish.get("finished_at_utc"))
            if finish_time < start_time or (previous_finish is not None and start_time < previous_finish):
                raise RuntimeError(f"{key}: queue attempts are not sequentially paired")
            previous_finish = finish_time
            final_attempt = index == expected_attempts - 1
            if final_attempt:
                recorded_status = finish.get("manifest_status")
                if (
                    int(finish.get("returncode", -1)) != 0
                    or recorded_status not in (None, "completed")
                    or row.get("status") != "completed"
                ):
                    raise RuntimeError(f"{key}: final queue attempt did not complete successfully")
            elif (
                int(finish.get("returncode", 0)) == 0
                or finish.get("manifest_status") not in (None, "failed")
            ):
                raise RuntimeError(f"{key}: prior attempt is not a recorded failed attempt")

    if (
        set(started_by_key) != expected or set(finished_by_key) != expected
        or sum(map(len, started_by_key.values())) != sum(1 + len(by_key[key].get("attempt_history", [])) for key in expected)
    ):
        raise RuntimeError("P2 queue attempt history does not cover the fixed run set")
    retry_keys = sorted(key for key in expected if by_key[key].get("attempt_history"))
    if set(retry_keys) != set(REGISTERED_RETRY_REASONS):
        raise RuntimeError(f"P2 registered retry set differs from the frozen allowlist: {retry_keys}")
    return {
        "status": "PASS",
        "unique_run_count": len(expected),
        "started_attempt_count": sum(map(len, started_by_key.values())),
        "finished_attempt_count": sum(map(len, finished_by_key.values())),
        "registered_retry_count": len(retry_keys),
        "registered_retries": {key: REGISTERED_RETRY_REASONS[key] for key in retry_keys},
        "final_successful_attempt_count": len(expected),
    }


def _validate_validation_split_manifest(trainer, by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    proof = json.loads(trainer.VALIDATION_SPLIT_PROOF.read_text(encoding="utf-8"))
    evidence = trainer._validation_split_evidence()
    if proof.get("status") != "PASS":
        raise RuntimeError("P2 validation-split equivalence/calibration gate is not PASS")
    for key in sorted(VALIDATION_SPLIT_RUN_KEYS):
        record = by_key[key].get("p2_padded_microbatch")
        if not isinstance(record, dict) or record.get("validation_split") != evidence:
            raise RuntimeError(f"{key}: manifest validation_split evidence is missing or stale")
    return {
        "status": "PASS",
        "validated_run_count": len(VALIDATION_SPLIT_RUN_KEYS),
        "validated_run_keys": sorted(VALIDATION_SPLIT_RUN_KEYS),
        "evidence": evidence,
    }


def _completed_run_records(
    trainer,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    for path in (TRAIN_MANIFEST_PATH, QUEUE_STATUS_PATH, SAMPLE_MANIFEST_PATH,
                 OVERLAP_AUDIT_PATH, trainer.SELF_CHECK, trainer.PLAN,
                 LOHO_CACHE_PATH, trainer.FOLD_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    queue = json.loads(QUEUE_STATUS_PATH.read_text(encoding="utf-8"))
    expected = _expected_run_keys(trainer)
    rows = manifest.get("runs", [])
    by_key = {str(row.get("run_key")): row for row in rows}
    if (
        len(rows) != 90 or len(by_key) != 90 or set(by_key) != expected
        or any(row.get("status") != "completed" for row in rows)
    ):
        counts: dict[str, int] = {}
        for row in rows:
            state = str(row.get("status"))
            counts[state] = counts.get(state, 0) + 1
        raise RuntimeError(f"P2 checkpoint freeze requires all 90 completed runs, got {counts}")
    if (
        queue.get("status") != "ALL_90_RUNS_COMPLETE_CHECKPOINT_FREEZE_REQUIRED"
        or queue.get("test_inference") != "not_authorized_by_this_queue"
    ):
        raise RuntimeError("P2 queue status does not authorize checkpoint freeze")
    attempt_audit = _validate_queue_attempt_history(queue, by_key, expected)
    validation_split_audit = _validate_validation_split_manifest(trainer, by_key)
    if manifest.get("test_inference_authorized") is not False:
        raise RuntimeError("training manifest unexpectedly authorizes test inference")
    plan_identity = trainer.analysis_plan_identity()
    plan_sha = str(plan_identity["analysis_plan_sha256"])
    samples = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = json.loads(OVERLAP_AUDIT_PATH.read_text(encoding="utf-8"))
    self_check = json.loads(trainer.SELF_CHECK.read_text(encoding="utf-8"))
    source_checks = {
        "analysis_plan_sha256": (plan_sha, manifest.get("analysis_plan_sha256")),
        "sample_manifest_sha256": (_sha256(SAMPLE_MANIFEST_PATH), manifest.get("sample_manifest_sha256")),
        "loho_cache_sha256": (_sha256(LOHO_CACHE_PATH), manifest.get("loho_cache_sha256")),
        "fold_map_sha256": (_sha256(trainer.FOLD_MAP), manifest.get("fold_map_sha256")),
    }
    if any(actual != recorded for actual, recorded in source_checks.values()):
        raise RuntimeError(f"P2 frozen source hashes differ from training manifest: {source_checks}")
    if (
        samples.get("status") != "PASS" or audit.get("training_gate") != "PASS"
        or samples.get("analysis_plan_sha256") != plan_sha
        or audit.get("analysis_plan_sha256") != plan_sha
        or self_check.get("status") != "PASS"
        or manifest.get("self_check_sha256") != _sha256(trainer.SELF_CHECK)
    ):
        raise RuntimeError("P2 plan, overlap audit, sample manifest or self-check is not frozen PASS")
    run_records: list[dict[str, Any]] = []
    for key in sorted(expected):
        row = by_key[key]
        if row.get("test_inference") != "not_run" or row.get("s5_wallclock_cap_triggered") is True:
            raise RuntimeError(f"{key}: training row has an unexpected test/S5 terminal state")
        duration = float(row.get("duration_sec", float("nan")))
        if (
            int(row.get("s5_wallclock_cap_seconds", -1)) != trainer.P2_RUN_WALLCLOCK_CAP_SEC
            or not np.isfinite(duration) or duration <= 0
            or duration > trainer.P2_RUN_WALLCLOCK_CAP_SEC
        ):
            raise RuntimeError(f"{key}: run duration does not satisfy the frozen 10,800-second S5 cap")
        for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                                   ("training_summary_path", "training_summary_sha256"),
                                   ("epoch_validation_log", "epoch_validation_log_sha256")):
            rel = row.get(path_key)
            if not rel:
                raise RuntimeError(f"{key}: missing {path_key}")
            path = (ROOT / str(rel)).resolve()
            if ROOT not in path.parents or not path.is_file() or _sha256(path) != row.get(hash_key):
                raise RuntimeError(f"{key}: {path_key} hash/path validation failed")
        if int(row.get("best_epoch", 0)) < 0 or not np.isfinite(float(row.get("best_validation_macro_persistence_skill", np.nan))):
            raise RuntimeError(f"{key}: invalid best validation selection record")
        run_records.append({
            "run_key": key,
            "fold": int(row["fold"]),
            "model": str(row["model"]),
            "objective": str(row["objective"]),
            "seed": int(row["seed"]),
            "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "training_summary_path": str(row["training_summary_path"]),
            "training_summary_sha256": str(row["training_summary_sha256"]),
            "epoch_validation_log": str(row["epoch_validation_log"]),
            "epoch_validation_log_sha256": str(row["epoch_validation_log_sha256"]),
            "best_epoch": int(row["best_epoch"]),
            "best_validation_macro_persistence_skill": float(row["best_validation_macro_persistence_skill"]),
            "duration_sec": duration,
            "cuda_peak_reserved_mib": int(row["cuda_peak_reserved_mib"]),
        })
    return manifest, queue, run_records, attempt_audit, validation_split_audit


def freeze() -> int:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"refusing to replace existing P2 freeze: {FREEZE_PATH}")
    trainer = _load_trainer()
    manifest, queue, records, attempt_audit, validation_split_audit = _completed_run_records(trainer)
    checkpoints_digest = _stable_sha(records)
    freeze_doc = {
        "schema_version": "p2_loho_checkpoint_freeze_v1",
        "status": "ALL_90_CHECKPOINTS_FROZEN",
        "frozen_at_utc": _now(),
        "analysis_plan_sha256": trainer.analysis_plan_identity()["analysis_plan_sha256"],
        "analysis_plan_document_sha256": trainer.analysis_plan_identity()["analysis_plan_document_sha256"],
        "c1_resolution_appendix_sha256": trainer.analysis_plan_identity()["c1_resolution_appendix_sha256"],
        "training_manifest_sha256": _sha256(TRAIN_MANIFEST_PATH),
        "queue_status_sha256": _sha256(QUEUE_STATUS_PATH),
        "sample_manifest_sha256": _sha256(SAMPLE_MANIFEST_PATH),
        "overlap_audit_sha256": _sha256(OVERLAP_AUDIT_PATH),
        "checkpoint_count": len(records),
        "checkpoint_set_sha256": checkpoints_digest,
        "checkpoints": records,
        "queue_attempt_audit": attempt_audit,
        "validation_split_manifest_gate": validation_split_audit,
        "test_inference_authorized": True,
        "test_inference_consumed": False,
        "test_inference_ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
        "original_train_manifest_test_authorization": manifest.get("test_inference_authorized"),
        "queue_status": queue.get("status"),
    }
    _atomic_json(FREEZE_PATH, freeze_doc)
    print(json.dumps({"status": freeze_doc["status"], "checkpoint_count": len(records),
                      "checkpoint_set_sha256": checkpoints_digest,
                      "test_values_or_labels_read": False}, indent=2))
    return 0


def _verify_freeze(trainer) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("P2 test inference requires a recorded 90-checkpoint freeze")
    freeze_doc = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (
        freeze_doc.get("status") != "ALL_90_CHECKPOINTS_FROZEN"
        or freeze_doc.get("test_inference_authorized") is not True
        or freeze_doc.get("test_inference_consumed") is not False
        or freeze_doc.get("checkpoint_count") != 90
        or freeze_doc.get("analysis_plan_sha256") != trainer.analysis_plan_identity()["analysis_plan_sha256"]
        or freeze_doc.get("analysis_plan_document_sha256") != trainer.analysis_plan_identity()["analysis_plan_document_sha256"]
        or freeze_doc.get("c1_resolution_appendix_sha256") != trainer.analysis_plan_identity()["c1_resolution_appendix_sha256"]
    ):
        raise RuntimeError("P2 freeze does not authorize this first test pass")
    manifest, _queue, current, attempt_audit, validation_split_audit = _completed_run_records(trainer)
    if (
        freeze_doc.get("training_manifest_sha256") != _sha256(TRAIN_MANIFEST_PATH)
        or freeze_doc.get("sample_manifest_sha256") != _sha256(SAMPLE_MANIFEST_PATH)
        or freeze_doc.get("overlap_audit_sha256") != _sha256(OVERLAP_AUDIT_PATH)
        or freeze_doc.get("checkpoint_set_sha256") != _stable_sha(current)
        or freeze_doc.get("checkpoints") != current
        or freeze_doc.get("queue_attempt_audit") != attempt_audit
        or freeze_doc.get("validation_split_manifest_gate") != validation_split_audit
    ):
        raise RuntimeError("P2 checkpoint/source set changed after freeze")
    return freeze_doc, manifest, current


def _prediction_path(record: dict[str, Any]) -> Path:
    return (PREDICTION_ROOT / f"fold{record['fold']}" / record["model"]
            / record["objective"] / f"seed{record['seed']}.npz")


def _write_npz_once(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite a P2 prediction file: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _carry_forward(example: dict[str, Any], standardizer: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    obs = example["obs"]
    variables = np.asarray(obs["var"], dtype=np.int64)
    times = np.asarray(obs["time"], dtype=np.float64)
    values = np.asarray(obs["value"], dtype=np.float32)
    values_raw = np.asarray(obs["value_raw"], dtype=np.float64)
    medians = np.asarray(standardizer["median"], dtype=np.float64)
    cf_standardized = np.zeros(N_VARIABLES, dtype=np.float32)
    cf_raw = medians.copy()
    for variable in range(N_VARIABLES):
        positions = np.flatnonzero(variables == variable)
        if not positions.size:
            continue
        latest_time = np.max(times[positions])
        # np.flatnonzero retains the cache's stable event order for tied readings.
        latest_position = int(positions[np.flatnonzero(times[positions] == latest_time)[0]])
        cf_standardized[variable] = values[latest_position]
        cf_raw[variable] = values_raw[latest_position]
    return cf_standardized, cf_raw


def _load_variable_names() -> list[str]:
    import yaml
    payload = yaml.safe_load(VARIABLE_MAP_PATH.read_text(encoding="utf-8"))
    names = [str(value) for value in payload["variable_names"]]
    if len(names) != N_VARIABLES:
        raise RuntimeError(f"expected {N_VARIABLES} frozen variable names, got {len(names)}")
    return names


def _stable_gpu_status(runner, gpu: int, minimum_free_mib: int) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for _ in range(5):
        current = runner._gpu_status(gpu, min_free_mib=minimum_free_mib,
                                     require_zero_utilization=True)
        samples.append({"memory_free_mib": current["memory_free_mib"],
                        "utilization_percent": current["utilization_percent"],
                        "free": current["free"]})
        if not current["free"]:
            raise RuntimeError(f"P2 one-pass test inference GPU preflight failed: {current}")
        time.sleep(2)
    current["stable_idle_samples"] = samples
    current["stable_idle_sample_count"] = len(samples)
    current["free"] = all(row["free"] for row in samples)
    return current


def _archived_failed_preinference_attempt() -> dict[str, Any] | None:
    """Verify and reference the preserved setup-only failure, if present."""
    if not FAILED_PREINFERENCE_LEDGER_PATH.is_file():
        return None
    failed = json.loads(FAILED_PREINFERENCE_LEDGER_PATH.read_text(encoding="utf-8"))
    if (
        failed.get("status") != "FAILED"
        or failed.get("failure") != "AttributeError: 'Data' object has no attribute 'examples'"
        or failed.get("failed_run_key") is not None
        or failed.get("run_results") != {}
        or failed.get("test_loader_constructed_after_freeze") is not True
        or failed.get("test_inference_started_after_freeze") is not False
        or failed.get("all_test_predictions_completed_after_freeze") is not False
    ):
        raise RuntimeError("archived P2 failure is not the expected no-forward setup attempt")
    return {
        "ledger_path": str(FAILED_PREINFERENCE_LEDGER_PATH.relative_to(ROOT)),
        "ledger_sha256": _sha256(FAILED_PREINFERENCE_LEDGER_PATH),
        "status": "FAILED_BEFORE_MODEL_FORWARD",
        "failure": str(failed["failure"]),
        "failed_run_key": None,
        "run_results_count": 0,
    }


def _resume_from_failed_attempt(
    failed_ledger_path: Path, frozen_records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = failed_ledger_path.resolve()
    archive_root = (ROOT / "results/jbi/_work/p2_loho_test_failed_attempts").resolve()
    if archive_root not in path.parents or not path.is_file():
        raise RuntimeError("resume requires a preserved P2 failed ledger inside the test-attempt archive")
    failed = json.loads(path.read_text(encoding="utf-8"))
    run_results = failed.get("run_results")
    if (
        failed.get("status") != "FAILED"
        or "OutOfMemoryError: CUDA out of memory" not in str(failed.get("failure", ""))
        or failed.get("test_inference_started_after_freeze") is not True
        or failed.get("all_test_predictions_completed_after_freeze") is not False
        or not isinstance(run_results, dict)
    ):
        raise RuntimeError("resume is limited to the archived partial P2 test OOM attempt")
    keys = list(run_results)
    expected_prefix = [row["run_key"] for row in frozen_records[:len(keys)]]
    if keys != expected_prefix or len(keys) >= len(frozen_records):
        raise RuntimeError("failed P2 test ledger is not a strict completed prefix of the frozen run order")
    failed_run_key = str(failed.get("failed_run_key"))
    if frozen_records[len(keys)]["run_key"] != failed_run_key:
        raise RuntimeError("failed run is not the next cell after the preserved completed prediction prefix")
    by_key = {row["run_key"]: row for row in frozen_records}
    expected_prediction_paths: set[Path] = set()
    for key in keys:
        result = run_results[key]
        record = by_key[key]
        expected_path = _prediction_path(record).resolve()
        recorded_path = (ROOT / str(result.get("path", ""))).resolve()
        if (
            result.get("status") != "completed"
            or recorded_path != expected_path
            or not expected_path.is_file()
            or _sha256(expected_path) != result.get("sha256")
        ):
            raise RuntimeError(f"{key}: preserved P2 prediction is missing, misplaced or hash-mismatched")
        with np.load(expected_path, allow_pickle=False) as prediction:
            if (
                str(prediction["run_key"].item()) != key
                or str(prediction["checkpoint_sha256"].item()) != record["checkpoint_sha256"]
            ):
                raise RuntimeError(f"{key}: preserved P2 prediction does not match its frozen checkpoint")
        expected_prediction_paths.add(expected_path)
    actual_prediction_paths = {
        path.resolve() for path in PREDICTION_ROOT.rglob("*.npz")
    } if PREDICTION_ROOT.exists() else set()
    if actual_prediction_paths != expected_prediction_paths:
        raise RuntimeError("P2 prediction directory does not exactly match the failed ledger's completed prefix")
    folds = failed.get("fold_batch_manifests")
    expected_folds = {str(by_key[key]["fold"]) for key in keys}
    if not isinstance(folds, dict) or set(folds) != expected_folds:
        raise RuntimeError("failed P2 ledger has incomplete fold batch-composition records")
    for fold, batch_doc in folds.items():
        payload = {key: value for key, value in batch_doc.items() if key != "sha256"}
        if _stable_sha(payload) != batch_doc.get("sha256"):
            raise RuntimeError(f"fold {fold}: preserved P2 batch manifest hash is invalid")
        for key in keys:
            if str(by_key[key]["fold"]) == str(fold) and run_results[key].get("fold_batch_manifest_sha256") != batch_doc.get("sha256"):
                raise RuntimeError(f"{key}: prediction and preserved batch-manifest hashes differ")
    failed_path = _prediction_path(by_key[failed_run_key])
    if failed_path.exists():
        raise RuntimeError("OOM-failed P2 checkpoint unexpectedly has a prediction file")
    reference = {
        "ledger_path": str(path.relative_to(ROOT)),
        "ledger_sha256": _sha256(path),
        "failed_script_sha256": failed.get("script_sha256"),
        "status": "FAILED_AFTER_PARTIAL_INFERENCE",
        "failure": str(failed["failure"]),
        "failed_run_key": failed_run_key,
        "completed_run_count": len(keys),
        "completed_prediction_keys": keys,
        "failed_run_prediction_written": False,
    }
    return failed, reference


def _test_model_forward(
    model: Any, raw_batch: dict[str, Any], device: Any, *, split_enabled: bool, budget: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forward the original loader batch, splitting only its oversized model call."""
    import torch

    tensors = {key: value for key, value in raw_batch.items() if isinstance(value, torch.Tensor)}
    options = {key: value for key, value in raw_batch.items() if key not in tensors}
    options["exp_stage"] = "test"
    if not split_enabled:
        batch = {key: value.to(device) for key, value in tensors.items()}
        output = model(**batch, **options)
        return output, {"mode": "full_original_batch", "was_split": False, "piece_count": 1}

    from p1_padded_batch_split import plan_padded_batch, slice_padded_batch

    plan = plan_padded_batch(tensors, budget, allow_oversized_singletons=True)
    sample_ids = [int(round(float(value))) for value in tensors["sample_ID"].detach().cpu().tolist()]
    audit = {
        "mode": "padded_size_split_when_needed",
        "padded_size_budget": int(budget),
        "original_padded_size": int(plan.original_padded_size),
        "was_split": bool(plan.was_split),
        "dense_sample_id": int(plan.dense_sample_id),
        "observation_padding_width": int(max(plan.observation_counts)),
        "piece_count": len(plan.piece_positions),
        "piece_sample_positions": [list(piece) for piece in plan.piece_positions],
        "piece_sample_ids": [[sample_ids[position] for position in piece] for piece in plan.piece_positions],
        "piece_padded_sizes": [int(size) for size in plan.piece_padded_sizes],
    }
    if not plan.was_split:
        batch = {key: value.to(device) for key, value in tensors.items()}
        return model(**batch, **options), audit

    target_rows = int(tensors["y_mask"].shape[1])
    prediction_by_position: dict[int, Any] = {}
    for positions in plan.piece_positions:
        piece = slice_padded_batch(tensors, positions)
        piece_batch = {key: value.to(device) for key, value in piece.items()}
        piece_output = model(
            **piece_batch,
            **options,
            p1_observation_padding_width=max(plan.observation_counts),
        )
        piece_pred = piece_output.get("pred")
        if not isinstance(piece_pred, torch.Tensor) or piece_pred.ndim != 3:
            raise RuntimeError("split P2 test forward returned no batched [B,T,V] prediction")
        if piece_pred.shape[0] != len(positions) or piece_pred.shape[1] > target_rows:
            raise RuntimeError("split P2 test prediction cannot be reassembled to the original batch grid")
        piece_pred = piece_pred.detach().cpu()
        piece_pred = torch.nn.functional.pad(piece_pred, (0, 0, 0, target_rows - piece_pred.shape[1]))
        for local_index, original_position in enumerate(positions):
            prediction_by_position[int(original_position)] = piece_pred[local_index]
        del piece_output, piece_batch
    if set(prediction_by_position) != set(range(len(sample_ids))):
        raise RuntimeError("split P2 test pieces do not cover each original sample exactly once")
    merged = torch.stack([prediction_by_position[position] for position in range(len(sample_ids))], dim=0)
    if tuple(merged.shape[:2]) != tuple(tensors["y_mask"].shape[:2]):
        raise RuntimeError("split P2 test prediction does not match the original outcome grid")
    return {"pred": merged}, audit


def run_test(gpu: int, resume_failed_ledger_path: Path | None = None) -> int:
    trainer = _load_trainer()
    freeze_doc, manifest, frozen_records = _verify_freeze(trainer)
    if LEDGER_PATH.exists() or LOCK_PATH.exists() or BATCH_MANIFEST_PATH.exists():
        raise FileExistsError("P2 canonical test artifacts must be archived before a registered resume")
    prior_failed_attempt = _archived_failed_preinference_attempt()
    resumed_ledger: dict[str, Any] | None = None
    prior_failed_forward_attempt: dict[str, Any] | None = None
    if resume_failed_ledger_path is None:
        existing = sorted(str(path) for path in PREDICTION_ROOT.rglob("*.npz")) if PREDICTION_ROOT.exists() else []
        if existing:
            raise FileExistsError(f"P2 prediction files exist before the single authorized pass: {existing[:3]}")
        initial_run_results: dict[str, Any] = {}
        initial_fold_manifests: dict[str, Any] = {}
        initial_forward_audit: dict[str, Any] = {}
    else:
        resumed_ledger, prior_failed_forward_attempt = _resume_from_failed_attempt(
            resume_failed_ledger_path, frozen_records
        )
        initial_run_results = dict(resumed_ledger["run_results"])
        initial_fold_manifests = dict(resumed_ledger["fold_batch_manifests"])
        initial_forward_audit = {
            key: {
                "status": "completed_before_registered_inference_memory_deviation",
                "mode": "full_original_dataloader_batch",
                "was_split": False,
                "batch_size": BATCH_SIZE,
                "split_batch_count": 0,
                "audit_limit": "the first OOM attempt recorded batch composition but not per-batch padded-size or split planning details",
            }
            for key in initial_run_results
        }
    maximum_peak = max(int(record["cuda_peak_reserved_mib"]) for record in frozen_records)
    minimum_free = maximum_peak + GPU_MARGIN_MIB
    runner = trainer._load_runner()
    gpu_info = _stable_gpu_status(runner, gpu, minimum_free)

    lock_fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(lock_fd, f"started_utc={_now()}\npid={os.getpid()}\n".encode("utf-8"))
    os.fsync(lock_fd)
    os.close(lock_fd)
    ledger: dict[str, Any] = {
        "schema_version": "p2_loho_test_inference_ledger_v2",
        "status": "RUNNING",
        "started_at_utc": _now(),
        "python": PYTHON,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_plan_sha256": freeze_doc["analysis_plan_sha256"],
        "analysis_plan_document_sha256": freeze_doc["analysis_plan_document_sha256"],
        "c1_resolution_appendix_sha256": freeze_doc["c1_resolution_appendix_sha256"],
        "checkpoint_set_sha256": freeze_doc["checkpoint_set_sha256"],
        "checkpoint_count": 90,
        "batch_size": BATCH_SIZE,
        "combined_cache_with_heldout_event_values_deserialized_before_freeze": True,
        "heldout_indices_selected_for_training_or_validation": False,
        "test_loader_constructed_after_freeze": False,
        "test_inference_started_after_freeze": False,
        "all_test_predictions_completed_after_freeze": False,
        "prior_failed_preinference_attempt": prior_failed_attempt,
        "prior_failed_model_forward_attempt": prior_failed_forward_attempt,
        "test_execution_memory_deviation": {
            "analysis_plan_appendix_sha256": trainer.analysis_plan_identity().get("execution_deviation_appendix_sha256"),
            "padded_size_budget": int(trainer.P2_PADDED_SIZE_BUDGET),
            "split_models": list(trainer.P2_SPLIT_MODELS),
            "rule": "preserve original sequential DataLoader batches of 128; split only over-budget GraFITi/HyperIMTS model forwards; reassemble predictions in original sample/time order",
            "carried_forward_predictions": sorted(initial_run_results),
        },
        "gpu": gpu_info,
        "minimum_free_memory_mib": minimum_free,
        "test_loader_order": "sequential, no shuffle, drop_last=False",
        "run_results": initial_run_results,
        "fold_batch_manifests": initial_fold_manifests,
        "test_forward_audit_by_run": initial_forward_audit,
        "resume_from_failed_attempt": prior_failed_forward_attempt,
    }
    if resumed_ledger is not None:
        ledger["test_loader_constructed_after_freeze"] = bool(
            resumed_ledger.get("test_loader_constructed_after_freeze")
        )
        ledger["test_inference_started_after_freeze"] = bool(
            resumed_ledger.get("test_inference_started_after_freeze")
        )
    _atomic_json(LEDGER_PATH, ledger)

    current_run_key: str | None = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        harness = trainer.HARNESS
        sys.path.insert(0, str(harness))
        os.chdir(harness)
        import torch
        from torch.utils.data import DataLoader
        from data.data_provider.datasets import ICUKAN as adapter
        from exp.exp_main import Exp_Main

        sample_manifest = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if sample_manifest.get("status") != "PASS":
            raise RuntimeError("frozen P2 sample manifest is not PASS")
        variable_names = _load_variable_names()
        import pickle
        with LOHO_CACHE_PATH.open("rb") as handle:
            # This inference process unpickles the shared cache after the freeze.
            # Earlier train/validation processes also deserialized this combined
            # cache before freeze; they exposed no held-out indices to their loaders.
            frozen_cache_payload = pickle.load(handle)
        if frozen_cache_payload.get("variable_names") != variable_names:
            raise RuntimeError("LOHO cache variable order differs from the frozen variable map")
        cache_holder: dict[str, Any] = {
            "examples": frozen_cache_payload["examples"],
            "standardizer": frozen_cache_payload["standardizer"],
            "variable_names": frozen_cache_payload["variable_names"],
        }
        active_fold: dict[str, int] = {"value": -1}

        def load_test_cache(self):
            # The cache is unpickled once, only after the frozen-set and GPU
            # gates above.  Each fold exposes only its frozen held-out indices.
            fold = int(active_fold["value"])
            fold_data = sample_manifest["folds"][str(fold)]
            self._examples = cache_holder["examples"]
            self._splits = {
                "train": np.asarray(fold_data["train_stay_indices"], dtype=np.int64),
                "val": np.asarray(fold_data["validation_stay_indices"], dtype=np.int64),
                "test": np.asarray(fold_data["test_stay_indices"], dtype=np.int64),
            }
            self._cohort = "eicu"
            self._data_variant = "p2_loho_frozen_test"
            self.data_provenance = {
                "data_source": "eICU LOHO frozen P2 fold",
                "fold": fold,
                "cache_path": str(LOHO_CACHE_PATH),
                "cache_sha256": _sha256(LOHO_CACHE_PATH),
                "sample_manifest_sha256": _sha256(SAMPLE_MANIFEST_PATH),
                "split_sizes": {key: int(len(value)) for key, value in self._splits.items()},
                "test_rows_exposed": int(len(self._splits["test"])),
                "freeze_checkpoint_set_sha256": freeze_doc["checkpoint_set_sha256"],
            }

        adapter.Data._load_cache = load_test_cache
        device = torch.device("cuda:0")
        all_run_keys: list[str] = list(ledger["run_results"])
        first_exp_by_fold: dict[int, Any] = {}
        test_data_by_fold: dict[int, Any] = {}
        base_loader_by_fold: dict[int, Any] = {}
        for fold in range(N_FOLDS):
            pending_fold_rows = [
                row for row in frozen_records
                if row["fold"] == fold and row["run_key"] not in ledger["run_results"]
            ]
            if not pending_fold_rows:
                continue
            active_fold["value"] = fold
            representative = pending_fold_rows[0]
            config, _config_path, _reference = trainer._configuration(
                representative["model"], representative["objective"], fold, representative["seed"]
            )
            config.p2_mode = "test"
            config.is_training = 0
            config.test_all = 0
            config.batch_size = BATCH_SIZE
            first_exp = Exp_Main(config)
            test_data, loader = first_exp._get_data(flag="test")
            test_indices = np.asarray(sample_manifest["folds"][str(fold)]["test_stay_indices"], dtype=np.int64)
            observed_ids = np.asarray(test_data.sample_index, dtype=np.int64)
            if len(test_data) != len(test_indices) or not np.array_equal(observed_ids, test_indices):
                raise RuntimeError(f"fold {fold}: test dataset does not equal the frozen held-out stay indices")
            if loader.batch_size != BATCH_SIZE or loader.drop_last:
                raise RuntimeError(f"fold {fold}: test loader batch policy differs from plan")
            if loader.sampler.__class__.__name__ != "SequentialSampler":
                raise RuntimeError(f"fold {fold}: test loader is not sequential")
            if len(loader) == 0:
                raise RuntimeError(f"fold {fold}: empty held-out test loader")
            first_exp_by_fold[fold] = first_exp
            test_data_by_fold[fold] = test_data
            base_loader_by_fold[fold] = loader

        ledger["test_loader_constructed_after_freeze"] = True
        _atomic_json(LEDGER_PATH, ledger)

        for fold in range(N_FOLDS):
            fold_rows = [
                row for row in frozen_records
                if row["fold"] == fold and row["run_key"] not in ledger["run_results"]
            ]
            if not fold_rows:
                continue
            active_fold["value"] = fold
            first_exp = first_exp_by_fold[fold]
            test_data = test_data_by_fold[fold]
            base_loader = base_loader_by_fold[fold]
            # ICUKAN's dataset adapter stores the frozen examples in `_examples`.
            examples = test_data._examples
            standardizer = cache_holder["standardizer"]
            medians = np.asarray(standardizer["median"], dtype=np.float64)
            iqrs = np.asarray(standardizer["iqr"], dtype=np.float64)
            if medians.shape != (N_VARIABLES,) or iqrs.shape != (N_VARIABLES,) or np.any(iqrs <= 0):
                raise RuntimeError("P2 frozen standardizer has an invalid shape or scale")
            saved_fold_manifest = ledger["fold_batch_manifests"].get(str(fold))
            fold_batches: list[dict[str, Any]] | None = (
                list(saved_fold_manifest["batches"]) if saved_fold_manifest else None
            )
            fold_batch_sha: str | None = (
                str(saved_fold_manifest["sha256"]) if saved_fold_manifest else None
            )
            for record in fold_rows:
                current_run_key = record["run_key"]
                if _sha256(ROOT / record["checkpoint_path"]) != record["checkpoint_sha256"]:
                    raise RuntimeError(f"frozen checkpoint changed immediately before inference: {current_run_key}")
                config, _config_path, _reference = trainer._configuration(
                    record["model"], record["objective"], fold, record["seed"]
                )
                config.p2_mode = "test"
                config.is_training = 0
                config.test_all = 0
                config.batch_size = BATCH_SIZE
                config.checkpoints_test = str((ROOT / record["checkpoint_path"]).parent)
                exp = Exp_Main(config)
                model = exp._build_model().to(device)
                state = exp._get_state_dict(ROOT / record["checkpoint_path"])
                model.load_state_dict(state, strict=True)
                model.eval()
                torch.cuda.reset_peak_memory_stats(device)
                parts: dict[str, list[np.ndarray]] = {key: [] for key in (
                    "event_sample_index", "event_stay_id", "event_patient_id", "event_hospital_id",
                    "event_variable_index", "event_time_hour", "truth_standardized", "prediction_standardized",
                    "carry_forward_standardized", "truth_raw", "prediction_raw", "carry_forward_raw",
                )}
                sample_indices: list[int] = []
                sample_stay_ids: list[int] = []
                sample_patient_ids: list[str] = []
                sample_hospital_ids: list[int] = []
                batch_rows: list[dict[str, Any]] = []
                forward_audit = {
                    "mode": "padded_size_split_when_needed" if record["model"] in trainer.P2_SPLIT_MODELS else "full_original_dataloader_batch",
                    "split_enabled_for_model": record["model"] in trainer.P2_SPLIT_MODELS,
                    "padded_size_budget": int(trainer.P2_PADDED_SIZE_BUDGET) if record["model"] in trainer.P2_SPLIT_MODELS else None,
                    "batch_size": BATCH_SIZE,
                    "loader_batch_count": int(len(base_loader)),
                    "split_batch_count": 0,
                    "split_piece_count": 0,
                    "maximum_piece_count": 1,
                    "split_batches": [],
                }
                if not ledger["test_inference_started_after_freeze"]:
                    ledger["test_inference_started_after_freeze"] = True
                    _atomic_json(LEDGER_PATH, ledger)
                with torch.inference_mode():
                    for batch_index, raw_batch in enumerate(base_loader):
                        ids = [int(round(float(value))) for value in raw_batch["sample_ID"].tolist()]
                        if len(ids) > BATCH_SIZE:
                            raise RuntimeError(f"{current_run_key}: batch exceeds fixed size 128")
                        if not torch.isfinite(raw_batch["y"]).all() or not torch.isfinite(raw_batch["x"]).all():
                            # Padding is represented as zeros by collate_fn; NaNs are not allowed into the model.
                            raise FloatingPointError(f"{current_run_key}: batch contains non-finite model inputs")
                        target_mask = raw_batch["y_mask"] > 0
                        batch_counts = target_mask.sum(dim=(1, 2)).cpu().numpy().astype(np.int32)
                        input_counts = (raw_batch["x_mask"] > 0).sum(dim=(1, 2)).cpu().numpy().astype(np.int32)
                        x_valid_rows = (raw_batch["x_mask"] > 0).any(-1).sum(1).cpu().numpy().astype(np.int32)
                        y_valid_rows = target_mask.any(-1).sum(1).cpu().numpy().astype(np.int32)
                        raw_targets = np.zeros(tuple(raw_batch["y"].shape), dtype=np.float64)
                        standardized_targets = np.zeros(tuple(raw_batch["y"].shape), dtype=np.float32)
                        cf_std_rows: list[np.ndarray] = []
                        cf_raw_rows: list[np.ndarray] = []
                        for batch_row, sample_id in enumerate(ids):
                            if sample_id < 0 or sample_id >= len(examples):
                                raise RuntimeError(f"{current_run_key}: invalid frozen sample index {sample_id}")
                            example = examples[sample_id]
                            target = example["target"]
                            time_std, dense_std = adapter._densify(
                                np.asarray(target["var"], dtype=np.int64),
                                np.asarray(target["time"], dtype=np.float32),
                                np.asarray(target["value"], dtype=np.float32),
                            )
                            _time_raw, dense_raw = adapter._densify(
                                np.asarray(target["var"], dtype=np.int64),
                                np.asarray(target["time"], dtype=np.float32),
                                np.asarray(target["value_raw"], dtype=np.float32),
                            )
                            n_rows = int(dense_std.shape[0])
                            if dense_std.shape != dense_raw.shape or n_rows > raw_batch["y"].shape[1]:
                                raise RuntimeError(f"{current_run_key}: target dense representation mismatch")
                            if not torch.equal(raw_batch["y_mask"][batch_row, :n_rows].cpu(), torch.isfinite(dense_std)):
                                raise RuntimeError(f"{current_run_key}: standardized target mask mismatch for sample {sample_id}")
                            if not torch.equal(raw_batch["y"][batch_row, :n_rows].cpu(), torch.nan_to_num(dense_std)):
                                raise RuntimeError(f"{current_run_key}: standardized target values mismatch for sample {sample_id}")
                            if n_rows and not np.allclose(
                                raw_batch["y_mark"][batch_row, :n_rows, 0].cpu().numpy(),
                                time_std.numpy(), atol=1e-7, rtol=0,
                            ):
                                raise RuntimeError(f"{current_run_key}: target time alignment mismatch for sample {sample_id}")
                            standardized_targets[batch_row, :n_rows] = torch.nan_to_num(dense_std).numpy()
                            raw_targets[batch_row, :n_rows] = torch.nan_to_num(dense_raw, nan=0.0).numpy()
                            cf_std, cf_raw = _carry_forward(example, standardizer)
                            cf_std_rows.append(cf_std)
                            cf_raw_rows.append(cf_raw)
                            sample_indices.append(sample_id)
                            sample_stay_ids.append(int(example["record_id"]))
                            sample_patient_ids.append(str(example["uniquepid"]))
                            sample_hospital_ids.append(int(example["hospital_id"]))
                        batch_rows.append({
                            "batch_index": int(batch_index),
                            "sample_indices": ids,
                            "sample_count": len(ids),
                            "input_padded_time_rows": int(raw_batch["x"].shape[1]),
                            "target_padded_time_rows": int(raw_batch["y"].shape[1]),
                            "input_measurements_by_sample": input_counts.tolist(),
                            "target_measurements_by_sample": batch_counts.tolist(),
                            "input_valid_time_rows_by_sample": x_valid_rows.tolist(),
                            "target_valid_time_rows_by_sample": y_valid_rows.tolist(),
                        })
                        outputs, split_audit = _test_model_forward(
                            model,
                            raw_batch,
                            device,
                            split_enabled=record["model"] in trainer.P2_SPLIT_MODELS,
                            budget=int(trainer.P2_PADDED_SIZE_BUDGET),
                        )
                        forward_audit["maximum_piece_count"] = max(
                            int(forward_audit["maximum_piece_count"]),
                            int(split_audit["piece_count"]),
                        )
                        if split_audit.get("was_split"):
                            forward_audit["split_batch_count"] += 1
                            forward_audit["split_piece_count"] += int(split_audit["piece_count"])
                            forward_audit["split_batches"].append({
                                "batch_index": int(batch_index),
                                "sample_indices": ids,
                                **split_audit,
                            })
                        pred = outputs["pred"].detach().cpu()
                        if pred.shape != raw_batch["y"].shape:
                            raise RuntimeError(f"{current_run_key}: prediction shape {tuple(pred.shape)} does not match target {tuple(raw_batch['y'].shape)}")
                        if not torch.isfinite(pred[target_mask]).all():
                            raise FloatingPointError(f"{current_run_key}: non-finite prediction at a true target measurement")
                        eb, er, ev = torch.nonzero(target_mask, as_tuple=True)
                        if eb.numel():
                            event_samples = np.asarray([ids[int(b)] for b in eb.tolist()], dtype=np.int64)
                            event_vars = ev.numpy().astype(np.int16)
                            truth_std = raw_batch["y"][eb, er, ev].numpy().astype(np.float32)
                            prediction_std = pred[eb, er, ev].numpy().astype(np.float32)
                            truth_raw = np.asarray([
                                raw_targets[int(b), int(r), int(v)]
                                for b, r, v in zip(eb.tolist(), er.tolist(), ev.tolist())
                            ], dtype=np.float64)
                            prediction_raw = prediction_std.astype(np.float64) * iqrs[event_vars] + medians[event_vars]
                            cf_std = np.asarray([cf_std_rows[int(b)][int(v)] for b, v in zip(eb.tolist(), ev.tolist())], dtype=np.float32)
                            cf_raw = np.asarray([cf_raw_rows[int(b)][int(v)] for b, v in zip(eb.tolist(), ev.tolist())], dtype=np.float64)
                            event_times = raw_batch["y_mark"][eb, er, 0].numpy().astype(np.float64) * L_TOTAL_HOURS
                            parts["event_sample_index"].append(event_samples)
                            parts["event_stay_id"].append(np.asarray([sample_stay_ids[-len(ids) + int(b)] for b in eb.tolist()], dtype=np.int64))
                            parts["event_patient_id"].append(np.asarray([sample_patient_ids[-len(ids) + int(b)] for b in eb.tolist()], dtype="U64"))
                            parts["event_hospital_id"].append(np.asarray([sample_hospital_ids[-len(ids) + int(b)] for b in eb.tolist()], dtype=np.int32))
                            parts["event_variable_index"].append(event_vars)
                            parts["event_time_hour"].append(event_times.astype(np.float32))
                            parts["truth_standardized"].append(truth_std)
                            parts["prediction_standardized"].append(prediction_std)
                            parts["carry_forward_standardized"].append(cf_std)
                            parts["truth_raw"].append(truth_raw)
                            parts["prediction_raw"].append(prediction_raw)
                            parts["carry_forward_raw"].append(cf_raw)
                if fold_batches is None:
                    fold_batches = batch_rows
                    batch_doc = {
                        "fold": int(fold), "batch_size": BATCH_SIZE,
                        "shuffle": False, "drop_last": False,
                        "test_sample_count": int(len(test_data)),
                        "batch_count": len(batch_rows), "batches": batch_rows,
                    }
                    fold_batch_sha = _stable_sha(batch_doc)
                    ledger["fold_batch_manifests"][str(fold)] = {
                        **batch_doc, "sha256": fold_batch_sha,
                    }
                elif _stable_sha(batch_rows) != _stable_sha(fold_batches):
                    raise RuntimeError(f"{current_run_key}: test loader batch order/composition changed")

                arrays: dict[str, Any] = {
                    "schema_version": np.asarray("p2_loho_test_predictions_v1"),
                    "run_key": np.asarray(current_run_key),
                    "fold": np.asarray(fold, dtype=np.int8),
                    "model": np.asarray(record["model"]),
                    "objective": np.asarray(record["objective"]),
                    "seed": np.asarray(record["seed"], dtype=np.int16),
                    "checkpoint_sha256": np.asarray(record["checkpoint_sha256"]),
                    "fold_batch_manifest_sha256": np.asarray(fold_batch_sha),
                    "variable_names": np.asarray(variable_names),
                    "test_sample_index": np.asarray(sample_indices, dtype=np.int64),
                    "test_stay_id": np.asarray(sample_stay_ids, dtype=np.int64),
                    "test_patient_id": np.asarray(sample_patient_ids, dtype="U64"),
                    "test_hospital_id": np.asarray(sample_hospital_ids, dtype=np.int32),
                }
                event_dtypes = {
                    "event_sample_index": np.int64, "event_stay_id": np.int64,
                    "event_patient_id": "U64", "event_hospital_id": np.int32,
                    "event_variable_index": np.int16, "event_time_hour": np.float32,
                    "truth_standardized": np.float32, "prediction_standardized": np.float32,
                    "carry_forward_standardized": np.float32, "truth_raw": np.float64,
                    "prediction_raw": np.float64, "carry_forward_raw": np.float64,
                }
                for key, dtype in event_dtypes.items():
                    arrays[key] = np.concatenate(parts[key]).astype(dtype, copy=False) if parts[key] else np.empty(0, dtype=dtype)
                path = _prediction_path(record)
                _write_npz_once(path, arrays)
                result = {
                    "status": "completed", "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path), "fold_batch_manifest_sha256": fold_batch_sha,
                    "test_stay_count": int(len(sample_indices)),
                    "test_event_count": int(arrays["event_sample_index"].size),
                    "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
                    "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) // (1024 * 1024)),
                    "test_forward_execution": forward_audit,
                    "completed_at_utc": _now(),
                }
                ledger["run_results"][current_run_key] = result
                ledger["test_forward_audit_by_run"][current_run_key] = forward_audit
                _atomic_json(LEDGER_PATH, ledger)
                all_run_keys.append(current_run_key)
                del model, exp, outputs
                torch.cuda.empty_cache()

        if len(all_run_keys) != 90 or set(all_run_keys) != {row["run_key"] for row in frozen_records}:
            raise RuntimeError("P2 test prediction ledger does not cover the frozen 90-cell set")
        fold_doc = {
            "schema_version": "p2_loho_test_batch_manifest_v1",
            "analysis_plan_sha256": freeze_doc["analysis_plan_sha256"],
            "analysis_plan_document_sha256": freeze_doc["analysis_plan_document_sha256"],
            "c1_resolution_appendix_sha256": freeze_doc["c1_resolution_appendix_sha256"],
            "checkpoint_set_sha256": freeze_doc["checkpoint_set_sha256"],
            "batch_size": BATCH_SIZE,
            "loader_order": "SequentialSampler; shuffle=false; drop_last=false",
            "folds": ledger["fold_batch_manifests"],
            "test_forward_audit_by_run": ledger["test_forward_audit_by_run"],
            "test_execution_memory_deviation": ledger["test_execution_memory_deviation"],
            "prior_failed_preinference_attempt": prior_failed_attempt,
            "prior_failed_model_forward_attempt": prior_failed_forward_attempt,
        }
        _atomic_json(BATCH_MANIFEST_PATH, fold_doc)
        ledger["test_batch_manifest_path"] = str(BATCH_MANIFEST_PATH.relative_to(ROOT))
        ledger["test_batch_manifest_sha256"] = _sha256(BATCH_MANIFEST_PATH)
        ledger["status"] = "COMPLETE"
        ledger["completed_at_utc"] = _now()
        ledger["completed_run_count"] = 90
        ledger["test_inference_consumed"] = True
        ledger["all_test_predictions_completed_after_freeze"] = True
        _atomic_json(LEDGER_PATH, ledger)
        freeze_doc["test_inference_consumed"] = True
        freeze_doc["test_inference_completed_at_utc"] = ledger["completed_at_utc"]
        freeze_doc["test_inference_ledger_sha256"] = _sha256(LEDGER_PATH)
        freeze_doc["prior_failed_preinference_attempt"] = prior_failed_attempt
        freeze_doc["prior_failed_model_forward_attempt"] = prior_failed_forward_attempt
        freeze_doc["test_execution_deviation_appendix_sha256"] = ledger[
            "test_execution_memory_deviation"
        ]["analysis_plan_appendix_sha256"]
        _atomic_json(FREEZE_PATH, freeze_doc)
        print(f"P2 one-pass test inference complete for {len(all_run_keys)} frozen checkpoints")
        return 0
    except BaseException as exc:
        ledger["status"] = "FAILED"
        ledger["failed_at_utc"] = _now()
        ledger["failed_run_key"] = current_run_key
        ledger["failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(LEDGER_PATH, ledger)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="freeze all 90 completed P2 checkpoints without loading test examples")
    test_parser = sub.add_parser("test", help="consume the one authorized P2 test inference pass")
    test_parser.add_argument("--gpu", required=True, type=int, choices=(0, 1))
    test_parser.add_argument(
        "--resume-failed-ledger", type=Path,
        help="resume only from a verified archived partial CUDA OOM attempt; prior NPZ predictions remain immutable",
    )
    args = parser.parse_args()
    if args.command == "freeze":
        return freeze()
    return run_test(args.gpu, args.resume_failed_ledger)


if __name__ == "__main__":
    raise SystemExit(main())
