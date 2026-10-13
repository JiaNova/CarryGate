#!/usr/bin/env python
"""Gated, training-only launcher for the frozen 72-hour P1 landmark runs.

The runner trains one cohort/model/seed at a time with ``Exp_Main.train``.
It never calls the harness test path.  Seed-2024 pilot validation is followed
by a validation-only t=24 comparison against the corresponding frozen Track-B
masked-L1 score; seeds 2025--2028 remain locked until the six-run trigger says
continue.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(os.environ.get("ICU_KAN_PYTHON", sys.executable))
HARNESS = ROOT / "external" / "PyOmniTS"
P1_PLAN = ROOT / "paper" / "预设用例分析计划_P1.md"
WORKORDER = ROOT / "paper" / "投稿前证据补强工作单.md"
ADAPTER = HARNESS / "data/data_provider/datasets/ICUKANLandmark.py"
PADDED_BATCH_SPLITTER = ROOT / "scripts/jbi/p1_padded_batch_split.py"
MICROBATCH_TRAINER = ROOT / "scripts/jbi/p1_microbatch_training.py"
MICROBATCH_POLICY_PATH = ROOT / "results/jbi/p1_microbatch_training_policy.json"
MICROBATCH_PROOF_PATH = ROOT / "results/jbi/p1_microbatch_gradient_equivalence.json"
MICROBATCH_PROOF_SCRIPT = ROOT / "scripts/jbi/47_verify_p1_microbatch_gradients.py"
MICROBATCH_PROFILE_SCRIPT = ROOT / "scripts/jbi/48_profile_p1_microbatch_memory.py"
GATE_TEST = ROOT / "tests/test_landmark_adapter.py"
GATE_PATH = ROOT / "results/jbi/p1_landmark_adapter_test_gate.json"
MANIFEST_PATH = ROOT / "results/jbi/p1_landmark_training_manifest.json"
TRIGGER_PATH = ROOT / "results/jbi/p1_landmark_pilot_trigger.json"
# Written by 50_p1_trigger_diagnostics.py only when the §17.3 rule holds.
TRIGGER_RESOLUTION_PATH = ROOT / "results/jbi/p1_landmark_pilot_trigger_resolution.json"
FAILURE_OVERRIDE_PATH = ROOT / "results/jbi/p1_landmark_continue_after_pilot_failure.json"
NOT_RUNNABLE_OVERRIDE_PATH = ROOT / "results/jbi/p1_landmark_continue_after_hyperimts_not_runnable.json"
CHECKPOINTS = ROOT / "results/jbi/p1_landmark_checkpoints"
EPOCH_DIR = ROOT / "results/jbi/p1_landmark_epoch_metrics"
TRAINING_SCRIPT = Path(__file__).resolve()
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
SEEDS = (2024, 2025, 2026, 2027, 2028)
LANDMARKS_PER_STAY = 7


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _run_lock():
    path = ROOT / "results/jbi/.p1_landmark_manifest.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _source_fingerprints() -> dict[str, str]:
    return {
        "p1_plan_sha256": sha256(P1_PLAN),
        "workorder_sha256": sha256(WORKORDER),
        "adapter_sha256": sha256(ADAPTER),
        "gate_test_sha256": sha256(GATE_TEST),
        "training_launcher_sha256": sha256(TRAINING_SCRIPT),
        "padded_batch_splitter_sha256": sha256(PADDED_BATCH_SPLITTER),
        "microbatch_trainer_sha256": sha256(MICROBATCH_TRAINER),
        "eicu_landmark_cache_sha256": sha256(ROOT / "data/processed/eicu_events_landmark_v1.pkl"),
        "mimiciv_landmark_cache_sha256": sha256(ROOT / "data/processed/mimiciv_events_landmark_v1.pkl"),
        "eicu_split_sha256": sha256(ROOT / "data/processed/eicu_split_indices_jbi_v1.json"),
        "mimiciv_split_sha256": sha256(ROOT / "data/processed/mimiciv_split_indices_jbi_v1.json"),
        "extraction_manifest_sha256": sha256(ROOT / "results/jbi/p1_landmark_extraction_manifest.json"),
    }


def _run_gate() -> int:
    started = time.perf_counter()
    command = [str(PYTHON), "-m", "pytest", "-q", str(GATE_TEST)]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout
    result = {
        "schema_version": "p1_landmark_adapter_gate_v1",
        "generated_at_utc": now_utc(),
        "python": str(PYTHON),
        "command": command,
        "returncode": int(completed.returncode),
        "duration_sec": time.perf_counter() - started,
        "status": "PASS" if completed.returncode == 0 and "6 passed" in output else "FAIL",
        "test_output": output,
        "source_fingerprints": _source_fingerprints(),
    }
    _atomic_json(GATE_PATH, result)
    print(output, end="")
    print(f"adapter gate: {result['status']} → {GATE_PATH}")
    return 0 if result["status"] == "PASS" else 2


def _assert_gate() -> dict[str, Any]:
    if not GATE_PATH.is_file():
        raise RuntimeError("six V1–V6 adapter tests must be run and recorded before training")
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("returncode") != 0:
        raise RuntimeError("P1 adapter gate did not pass")
    saved = gate.get("source_fingerprints", {})
    current = _source_fingerprints()
    for key in ("p1_plan_sha256", "adapter_sha256", "gate_test_sha256", "training_launcher_sha256",
                "padded_batch_splitter_sha256", "microbatch_trainer_sha256",
                "eicu_landmark_cache_sha256", "mimiciv_landmark_cache_sha256",
                "eicu_split_sha256", "mimiciv_split_sha256", "extraction_manifest_sha256"):
        if saved.get(key) != current.get(key):
            raise RuntimeError(f"P1 gate is stale: {key} changed after six tests passed")
    return gate


def _assert_microbatch_training_evidence(model_name: str, padded_size_budget: int) -> dict[str, Any]:
    """Require frozen calibration and three real-batch proofs before target runs."""
    if not MICROBATCH_POLICY_PATH.is_file() or not MICROBATCH_PROOF_PATH.is_file():
        raise RuntimeError("microbatch calibration policy and gradient proof are required before target training")
    policy = json.loads(MICROBATCH_POLICY_PATH.read_text(encoding="utf-8"))
    proof = json.loads(MICROBATCH_PROOF_PATH.read_text(encoding="utf-8"))
    if policy.get("status") != "GRADIENT_PROOF_PASS":
        raise RuntimeError(f"microbatch policy is not proof-frozen: {policy.get('status')}")
    if proof.get("status") != "PASS":
        raise RuntimeError("microbatch gradient-equivalence artifact did not pass")
    proof_ref = policy.get("gradient_equivalence_proof", {})
    if (
        proof_ref.get("status") != "PASS"
        or proof_ref.get("path") != str(MICROBATCH_PROOF_PATH.relative_to(ROOT))
        or proof_ref.get("sha256") != sha256(MICROBATCH_PROOF_PATH)
    ):
        raise RuntimeError("policy does not bind to the current passing gradient-proof artifact")
    if int(proof.get("padded_size_budget", -1)) != int(padded_size_budget):
        raise RuntimeError("target run budget differs from the budget used by the gradient proof")
    if int(policy.get("budget_values", {}).get(f"mimiciv/{model_name}", -2)) != int(padded_size_budget):
        raise RuntimeError(f"target run budget is not frozen for mimiciv/{model_name}")

    proof_models = proof.get("model_results", {})
    if set(("HyperIMTS", "GraFITi")) - set(proof_models):
        raise RuntimeError("gradient proof must cover both HyperIMTS and GraFITi")
    required_scenarios = {
        "frozen_budget_no_oversized_singleton_offload",
        "frozen_budget_oversized_singleton_offload",
        "rescue_budget_quarter_all_offload",
    }
    if (
        proof.get("production_execution_api") != "p1_microbatch_training.run_batch_plan"
        or set(proof.get("required_scenarios", [])) != required_scenarios
        or proof.get("gradient_relative_metric")
            != "max absolute elementwise gradient difference across all parameters divided by the maximum absolute elementwise full-batch gradient across all parameters"
    ):
        raise RuntimeError("gradient proof must exercise the production run_batch_plan API and all three scenarios")
    for proven_model in ("HyperIMTS", "GraFITi"):
        batches = proof_models[proven_model].get("batches", {})
        loader_order = proof_models[proven_model].get("loader_order_equivalence", {})
        if (
            set(batches) != required_scenarios
            or not all(bool(batch.get("pass")) for batch in batches.values())
            or any(float(batch.get("max_abs_gradient_relative_difference", float("inf"))) > 1e-5
                   for batch in batches.values())
            or any(float(batch.get("absolute_loss_difference", float("inf"))) > 1e-5
                   for batch in batches.values())
        ):
            raise RuntimeError(f"{proven_model} needs three passing real-batch production-path scenario comparisons")
        baseline = batches["frozen_budget_no_oversized_singleton_offload"]
        oversized = batches["frozen_budget_oversized_singleton_offload"]
        rescue = batches["rescue_budget_quarter_all_offload"]
        if (
            any(item.get("production_function_called") != "p1_microbatch_training.run_batch_plan"
                for item in batches.values())
            or any(item.get("optimizer_step_performed") is not False for item in batches.values())
            or int(baseline.get("padded_size_budget", -1)) != int(padded_size_budget)
            or baseline.get("allow_oversized_singletons") is not False
            or any(baseline.get("piece_offloaded", []))
            or int(oversized.get("padded_size_budget", -1)) != int(padded_size_budget)
            or oversized.get("allow_oversized_singletons") is not True
            or oversized.get("offload_all") is not False
            or oversized.get("rescue_attempt") is not False
            or not any(
                sample_ids == [53010]
                and piece_size > int(padded_size_budget)
                and offloaded
                for sample_ids, piece_size, offloaded in zip(
                    oversized.get("piece_sample_ids", []),
                    oversized.get("piece_padded_sizes", []),
                    oversized.get("piece_offloaded", []),
                )
            )
            or int(rescue.get("padded_size_budget", -1)) != max(1, int(padded_size_budget) // 4)
            or rescue.get("allow_oversized_singletons") is not True
            or rescue.get("offload_all") is not True
            or rescue.get("rescue_attempt") is not True
            or not rescue.get("piece_offloaded")
            or not all(rescue.get("piece_offloaded", []))
        ):
            raise RuntimeError(f"{proven_model} proof does not certify the exact three production scenarios")
        dense_batch = oversized
        if (
            (dense_batch.get("dense_window") or {}).get("input_distinct_timestamp_count_raw_float64") != 1394
            or not any(piece_ids == [53010] for piece_ids in dense_batch.get("piece_sample_ids", []))
        ):
            raise RuntimeError(f"{proven_model} proof must include and isolate sample 53010's 1,394-timestamp window")
        if (
            loader_order.get("status") != "PASS"
            or loader_order.get("same_sample_and_batch_order") is not True
            or int(loader_order.get("batch_size", -1)) != 128
        ):
            raise RuntimeError(f"{proven_model} proof did not confirm the original P1 loader order and batch size")
        if proof_models[proven_model].get("model_source_sha256") != sha256(
            HARNESS / "models" / f"{proven_model}.py"
        ):
            raise RuntimeError(f"{proven_model} source changed after gradient proof")
    if proof.get("splitter_sha256") != sha256(PADDED_BATCH_SPLITTER):
        raise RuntimeError("padded-batch splitter changed after gradient proof")
    if proof.get("trainer_sha256") != sha256(MICROBATCH_TRAINER):
        raise RuntimeError("microbatch trainer changed after gradient proof")
    if proof.get("proof_script_sha256") != sha256(MICROBATCH_PROOF_SCRIPT):
        raise RuntimeError("gradient-proof script changed after proof")

    calibration = policy.get("memory_calibration", {}).get(f"mimiciv/{model_name}", {})
    if calibration.get("status") != "PASS":
        raise RuntimeError(f"{model_name} memory calibration is not PASS")
    if int(calibration.get("padded_size_budget", -1)) != int(padded_size_budget):
        raise RuntimeError(f"{model_name} memory calibration used a different padded-size budget")
    calibration_path = (ROOT / str(calibration.get("path", ""))).resolve()
    if ROOT not in calibration_path.parents or not calibration_path.is_file():
        raise RuntimeError(f"{model_name} memory-calibration artifact is missing or outside the project")
    if sha256(calibration_path) != calibration.get("sha256"):
        raise RuntimeError(f"{model_name} memory-calibration artifact hash changed")
    calibration_report = json.loads(calibration_path.read_text(encoding="utf-8"))
    single_profiles = calibration_report.get("single_sample_profiles", [])
    offloaded_single_profiles = [
        item for item in single_profiles
        if item.get("offload_saved_tensors_to_pinned_cpu") is True
    ]
    no_offload_single_profiles = [
        item for item in single_profiles
        if item.get("offload_saved_tensors_to_pinned_cpu") is False
    ]
    required_profile_labels = {
        "median_proxy_typical_training_batch",
        "largest_full_batch_under_seed2024_loader",
        "batch_containing_1394_timestamp_window",
    }
    memory_preflight = calibration_report.get("memory_preflight", {})
    headroom_check = calibration_report.get("memory_headroom_check", {})
    shared_gpu_authorized = calibration_report.get("allow_shared_gpu") is True
    if (
        calibration_report.get("status") != "PASS"
        or calibration_report.get("model") != model_name
        or calibration_report.get("cohort") != "mimiciv"
        or int(calibration_report.get("padded_size_budget_tested", -1)) != int(padded_size_budget)
        or calibration_report.get("training_optimizer_step_performed") is not False
        or not all(item.get("fits_cuda") for item in offloaded_single_profiles)
        or not {53009, 53010}.issubset({
            int(item["sample_ids"][0]) for item in offloaded_single_profiles if item.get("sample_ids")
        })
        or not {53009, 53010}.issubset({
            int(item["sample_ids"][0]) for item in no_offload_single_profiles if item.get("sample_ids")
        })
        or len(offloaded_single_profiles) < 2
        or len(no_offload_single_profiles) < 2
        or not all(item.get("fits_cuda") for item in calibration_report.get("microbatch_profiles", []))
        or not required_profile_labels.issubset({
            str(item.get("label")) for item in calibration_report.get("microbatch_profiles", [])
        })
        or headroom_check.get("status") != "PASS"
        or float(headroom_check.get("minimum_fraction_of_preflight_free_memory", 0.0)) < 0.25
        or int(headroom_check.get("preflight_free_memory_mib", -1))
            != int(memory_preflight.get("memory_free_mib", -2))
        or int(headroom_check.get("required_headroom_mib", -1))
            < math.ceil(int(memory_preflight.get("memory_free_mib", 0)) * 0.25)
        or int(headroom_check.get("remaining_headroom_mib", -1))
            < int(headroom_check.get("required_headroom_mib", 0))
        or (
            not shared_gpu_authorized
            and (
                memory_preflight.get("compute_processes")
                or int(memory_preflight.get("memory_used_mib", 0)) > 512
                or int(memory_preflight.get("utilization_percent", 0)) != 0
            )
        )
        or (
            shared_gpu_authorized
            and int(memory_preflight.get("minimum_free_memory_mib", 0))
                > int(memory_preflight.get("memory_free_mib", -1))
        )
        or (
            shared_gpu_authorized
            and memory_preflight.get("require_zero_utilization") is not False
        )
        or calibration_report.get("splitter_sha256") != sha256(PADDED_BATCH_SPLITTER)
        or calibration_report.get("profile_script_sha256") != sha256(MICROBATCH_PROFILE_SCRIPT)
        or calibration_report.get("model_source_sha256") != sha256(HARNESS / "models" / f"{model_name}.py")
    ):
        raise RuntimeError(f"{model_name} memory-calibration report does not prove all pieces fit")
    singleton_ids = {
        int(sample_id)
        for item in calibration_report.get("single_sample_profiles", [])
        for sample_id in item.get("sample_ids", [])
    }
    if not {53009, 53010}.issubset(singleton_ids):
        raise RuntimeError(f"{model_name} calibration omitted one of the audited dense singleton samples")
    if not any(
        piece.get("sample_ids", []) == [53010]
        for batch in calibration_report.get("microbatch_profiles", [])
        for piece in batch.get("pieces", [])
    ):
        raise RuntimeError(f"{model_name} calibration did not isolate the 1,394-timestamp sample into its own piece")

    proof_ref = {
        "path": str(MICROBATCH_PROOF_PATH.relative_to(ROOT)),
        "sha256": sha256(MICROBATCH_PROOF_PATH),
        "status": proof["status"],
        "padded_size_budget": int(padded_size_budget),
    }
    calibration_ref = {
        "path": str(calibration_path.relative_to(ROOT)),
        "sha256": calibration["sha256"],
        "status": calibration["status"],
        "padded_size_budget": int(padded_size_budget),
    }
    return {"gradient_proof": proof_ref, "memory_calibration": calibration_ref}


def _manifest_default() -> dict[str, Any]:
    runs = []
    for seed in SEEDS:
        for cohort in COHORTS:
            for model in MODELS:
                runs.append({
                    "run_key": f"{cohort}/{model}/seed{seed}",
                    "cohort": cohort,
                    "model": model,
                    "seed": seed,
                    "phase": "pilot" if seed == 2024 else "remaining",
                    "status": "queued",
                })
    return {
        "schema_version": "p1_landmark_training_manifest_v1",
        "created_at_utc": now_utc(),
        "analysis_plan_path": str(P1_PLAN.relative_to(ROOT)),
        "analysis_plan_first_commit": "05e2590",
        "analysis_plan_sha256": sha256(P1_PLAN),
        "adapter_gate": {
            "path": str(GATE_PATH.relative_to(ROOT)),
            "status": "PASS",
            "generated_at_utc": json.loads(GATE_PATH.read_text(encoding="utf-8"))["generated_at_utc"],
            "source_fingerprints": _source_fingerprints(),
        },
        "run_count_expected": 30,
        "pilot_run_count_expected": 6,
        "remaining_run_count_expected": 24,
        "runs": runs,
        "all_checkpoints_frozen": False,
        "test_inference_authorized": False,
    }


def _load_manifest(create: bool = True) -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    elif create:
        manifest = _manifest_default()
        _atomic_json(MANIFEST_PATH, manifest)
    else:
        raise FileNotFoundError(MANIFEST_PATH)
    if manifest.get("analysis_plan_sha256") != sha256(P1_PLAN):
        raise RuntimeError("frozen P1 analysis plan hash differs from training manifest")
    if len(manifest.get("runs", [])) != 30:
        raise RuntimeError("P1 manifest must enumerate exactly 30 planned runs")
    return manifest


def _manifest_run(manifest: dict[str, Any], cohort: str, model: str, seed: int) -> dict[str, Any]:
    key = f"{cohort}/{model}/seed{seed}"
    matches = [run for run in manifest["runs"] if run.get("run_key") == key]
    if len(matches) != 1:
        raise RuntimeError(f"P1 run key missing or duplicated: {key}")
    return matches[0]


def _mark_failed(cohort: str, model: str, seed: int, exc: Exception, epoch_path: Path | None = None) -> None:
    lock = _run_lock()
    try:
        manifest = _load_manifest(create=False)
        current = _manifest_run(manifest, cohort, model, seed)
        current.update({
            "status": "failed",
            "finished_at_utc": now_utc(),
            "failure": f"{type(exc).__name__}: {exc}",
            "epoch_validation_log": str(epoch_path.relative_to(ROOT)) if epoch_path is not None else None,
            "epoch_validation_log_sha256": sha256(epoch_path) if epoch_path is not None and epoch_path.exists() else None,
            "test_inference": "not_run",
        })
        manifest["updated_at_utc"] = now_utc()
        _atomic_json(MANIFEST_PATH, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _mark_mimiciv_hyperimts_not_runnable(seed: int, exc: Exception) -> None:
    """Keep all five planned MIMIC-IV HyperIMTS cells in-denominator as failures."""
    lock = _run_lock()
    try:
        manifest = _load_manifest(create=False)
        source_key = f"mimiciv/HyperIMTS/seed{seed}"
        evidence = {
            "source_run_key": source_key,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "occurred_at_utc": now_utc(),
            "interpretation": "a complete singleton forward/backward raised CUDA out-of-memory under the frozen MIMIC-IV HyperIMTS budget",
            "cuda_evidence": getattr(exc, "evidence", {}),
            "windows_truncated": False,
            "model_replaced": False,
            "test_inference": "not_run",
        }
        for run in manifest["runs"]:
            if run.get("cohort") != "mimiciv" or run.get("model") != "HyperIMTS":
                continue
            run.update({
                "status": "not_runnable",
                "not_runnable_at_utc": now_utc(),
                "not_runnable_due_to": source_key,
                "not_runnable_evidence": evidence,
                "analysis_cell_status": "not_passing_in_fixed_denominator",
                "test_inference": "not_run",
            })
        manifest.setdefault("model_runnability", {})["mimiciv/HyperIMTS"] = {
            "status": "not_runnable",
            "planned_seed_cells": [2024, 2025, 2026, 2027, 2028],
            "fixed_denominator_retained": True,
            "evidence": evidence,
        }
        manifest["updated_at_utc"] = now_utc()
        _atomic_json(MANIFEST_PATH, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _gpu_status(
    gpu: int,
    allow_current_process: bool = False,
    min_free_mib: int | None = None,
    require_zero_utilization: bool = True,
) -> dict[str, Any]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, uuid = int(parts[0]), parts[1]
        rows[index] = {
            "index": index,
            "uuid": uuid,
            "memory_used_mib": int(parts[2]),
            "memory_free_mib": int(parts[3]),
            "utilization_percent": int(parts[4]),
        }
    if gpu not in rows:
        raise RuntimeError(f"GPU index {gpu} not found in nvidia-smi")
    apps_text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    apps = []
    for line in apps_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                app_mem = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            except ValueError:
                app_mem = 0
            apps.append({"gpu_uuid": parts[0], "pid": parts[1], "process_name": parts[2], "used_memory_mib": app_mem})
    selected = rows[gpu]
    selected["compute_processes"] = [app for app in apps if app["gpu_uuid"] == selected["uuid"]]
    blockers = [
        app for app in selected["compute_processes"]
        if not (allow_current_process and app["pid"] == str(os.getpid()))
    ]
    own_memory = sum(
        int(app.get("used_memory_mib", 0))
        for app in selected["compute_processes"]
        if allow_current_process and app["pid"] == str(os.getpid())
    )
    nonprocess_memory = max(0, selected["memory_used_mib"] - own_memory)
    selected["nonprocess_memory_used_mib"] = nonprocess_memory
    if min_free_mib is None:
        # Default to the original exclusive-use rule. A memory threshold is
        # only supplied when the operator explicitly authorizes shared use.
        selected["free"] = (
            not blockers
            and nonprocess_memory <= 512
            and selected["utilization_percent"] == 0
        )
        selected["memory_policy"] = "exclusive_idle"
    else:
        selected["free"] = (
            selected["memory_free_mib"] >= min_free_mib
            and (not require_zero_utilization or selected["utilization_percent"] == 0)
        )
        selected["memory_policy"] = (
            "shared_idle_with_minimum_free_memory"
            if require_zero_utilization
            else "shared_with_minimum_free_memory"
        )
        selected["minimum_free_memory_mib"] = min_free_mib
        selected["require_zero_utilization"] = require_zero_utilization
    return selected


def _reference_config(model: str, cohort: str) -> tuple[Path, dict[str, Any]]:
    tag = f"jbi_trackb_controls_factorial_{model}_mae_macro_v1_{cohort}"
    path = HARNESS / "configs" / model / tag / "ICUKAN.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("dataset_subset_name") != "jbi_patient":
        raise RuntimeError(f"reference config is not jbi_patient: {path}")
    if str(config.get("loss", "")).upper() != "MAE":
        raise RuntimeError(f"reference config is not masked L1: {path}")
    if str(config.get("checkpoint_selection_metric")) != "macro_persistence":
        raise RuntimeError(f"reference config has unexpected selection metric: {path}")
    # This is an explicit frozen value in P1 §3.2; fail if a cohort config
    # silently diverges from the plan.
    expected_dropout = 0.00 if model == "HyperIMTS" and cohort == "eicu" else 0.05
    if not math.isclose(float(config.get("dropout", float("nan"))), expected_dropout, abs_tol=1e-12):
        raise RuntimeError(f"{cohort}/{model}: Track-B masked-L1 dropout does not match P1 §3.2")
    if not math.isclose(float(config.get("learning_rate", float("nan"))), 0.001, abs_tol=1e-12):
        raise RuntimeError(f"{cohort}/{model}: Track-B masked-L1 learning rate does not match P1 §3.2")
    return path, config


def _get_configs(model: str, cohort: str, seed: int):
    ref_path, ref = _reference_config(model, cohort)
    ref_model_id = ref_path.parent.name
    # Reuse the existing ICUKAN parser/config reference path, then set values
    # from the frozen selected Track-B YAML before applying P1's time/data ID.
    sys.path.insert(0, str(HARNESS))
    os.chdir(HARNESS)
    from utils.configs import get_configs
    args = [
        "--model_name", model,
        "--model_id", ref_model_id,
        "--dataset_name", "ICUKAN",
        "--dataset_id", cohort,
        "--dataset_root_path", str(ROOT / "data/processed"),
    ]
    config = get_configs(args)
    for key, value in ref.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.model_name = model
    config.model_id = f"p1_landmark_{model}_{cohort}_seed{seed}"
    config.dataset_name = "ICUKANLandmark"
    config.dataset_root_path = str(ROOT / "data/processed")
    config.dataset_id = cohort
    config.dataset_subset_name = "jbi_patient"
    config.collate_fn = "collate_fn"
    config.seq_len = 48
    config.pred_len = 24
    config.enc_in = config.dec_in = config.c_out = 37
    config.checkpoints = str(CHECKPOINTS)
    config.checkpoints_test = None
    config.subfolder_train = "p1_landmark_v1"
    config.itr = 1
    config.itr_i = 0
    config.iteration_index = -1
    config.is_training = 1
    config.use_gpu = 1
    config.gpu_id = 0
    config.gpu_ids = None
    config.use_multi_gpu = 0
    config.batch_size = 128
    config.num_workers = 10
    config.learning_rate = 0.001
    config.dropout = 0.0 if model == "HyperIMTS" and cohort == "eicu" else 0.05
    config.loss = "MAE"
    config.checkpoint_selection_metric = "macro_persistence"
    config.train_epochs = 300
    config.patience = 10
    config.val_interval = 1
    config.train_val_loader_drop_last = 0
    config.train_val_loader_shuffle = 1
    config.wandb = 0
    config.sweep = 0
    config.save_arrays = 0
    config.save_cache_arrays = 0
    config.test_all = 0
    config.test_flop = 0
    config.test_gpu_memory = 0
    config.test_train_time = 0
    config.hyperimts_force_low_memory = 1 if model == "HyperIMTS" else 0
    config.device = "cuda"
    return config, ref_path, ref


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validation_t24_skill(exp, checkpoint: Path, torch) -> dict[str, Any]:
    """Score the selected checkpoint on t=24 validation events only."""
    from data.data_provider.datasets.ICUKANLandmark import LANDMARK_HOURS, carry_forward_target_values

    model = exp._build_model().to(exp.device)
    model.load_state_dict(exp._get_state_dict(checkpoint))
    model.eval()
    _val_data, loader = exp._get_data(flag="val")
    model_error = np.zeros(37, dtype=np.float64)
    carry_error = np.zeros(37, dtype=np.float64)
    count = np.zeros(37, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            sample_id = batch["sample_ID"].to(exp.device)
            batch = {key: value.to(exp.device) for key, value in batch.items()}
            output = model(**batch, exp_stage="val")
            pred = output["pred"]
            true = batch["y"]
            mask = batch["y_mask"] > 0
            first_landmark = torch.remainder(sample_id, len(LANDMARK_HOURS)) == 0
            keep = mask & first_landmark[:, None, None]
            carry = carry_forward_target_values(batch)
            abs_model = (pred - true).abs()
            abs_carry = (carry - true).abs()
            for variable in range(37):
                vm = keep[:, :, variable]
                count[variable] += int(vm.sum().item())
                model_error[variable] += float(abs_model[:, :, variable][vm].sum().item())
                carry_error[variable] += float(abs_carry[:, :, variable][vm].sum().item())
    scoreable = np.asarray([v for v in range(37) if v not in (6, 19, 32)], dtype=np.int64)
    valid = scoreable[(count[scoreable] >= 50) & (carry_error[scoreable] > 1e-9)]
    if not valid.size:
        raise RuntimeError("t=24 validation has no valid macro-persistence variables")
    score = float(np.mean(1.0 - model_error[valid] / carry_error[valid]))
    return {
        "validation_t24_macro_persistence_skill": score,
        "validation_t24_valid_variable_count": int(valid.size),
        "validation_t24_valid_variables": valid.tolist(),
        "validation_t24_event_count": int(count.sum()),
    }


def _trackb_val_skill(cohort: str, model: str, seed: int) -> dict[str, Any]:
    tag = f"jbi_trackb_controls_factorial_{model}_mae_macro_v1"
    control_dir = ROOT / "results/jbi" / (
        "trackb_controls" if cohort == "eicu" else "trackb_controls_mimiciv"
    )
    manifest_path = control_dir / "run_manifest.json"
    control_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"factorial/{model}/mae_macro/-"
    records = [record for record in control_manifest.get("runs", []) if record.get("key") == key]
    if len(records) != 1 or records[0].get("status") != "completed":
        raise RuntimeError(f"no unique completed Track-B manifest record for {cohort}/{model}/mae_macro")
    record = records[0]
    root = ROOT / str(record["output_root"])
    iteration = seed - 2024
    summary_path = root / f"iter{iteration}" / "training_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"manifest-selected Track-B summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stages = summary.get("stages", [])
    if len(stages) != 1:
        raise RuntimeError(f"unexpected Track-B training stages: {summary_path}")
    config_path = summary_path.parent / "configs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if (
        config.get("dataset_subset_name") != "jbi_patient"
        or int(config.get("seq_len", -1)) != 24
        or int(config.get("pred_len", -1)) != 24
        or str(config.get("loss", "")).upper() != "MAE"
        or config.get("checkpoint_selection_metric") != "macro_persistence"
    ):
        raise RuntimeError(f"Track-B validation reference config mismatch: {config_path}")
    score = float(stages[0].get("best_macro_persistence_skill", float("nan")))
    if not math.isfinite(score):
        raise RuntimeError(f"non-finite Track-B validation score: {summary_path}")
    return {
        "trackb_best_validation_macro_persistence_skill": score,
        "trackb_run_manifest": str(manifest_path.relative_to(ROOT)),
        "trackb_run_manifest_sha256": sha256(manifest_path),
        "trackb_training_summary": str(summary_path.relative_to(ROOT)),
        "trackb_training_summary_sha256": sha256(summary_path),
    }


def _run_one(
    cohort: str,
    model_name: str,
    seed: int,
    gpu: int,
    min_free_mib: int | None = None,
    offload_saved_tensors_to_cpu: bool = False,
    split_oversized_batches: bool = False,
    padded_size_budget: int | None = None,
    require_zero_utilization: bool = True,
    offload_oversized_pieces: bool = False,
    oom_rescue: bool = False,
) -> int:
    if cohort not in COHORTS or model_name not in MODELS or seed not in SEEDS:
        raise ValueError("invalid P1 cohort/model/seed")
    if min_free_mib is not None and min_free_mib < 1:
        raise ValueError("minimum free GPU memory must be positive")
    if not require_zero_utilization and min_free_mib is None:
        raise ValueError("sharing a busy GPU requires an explicit minimum-free-memory threshold")
    offload_allowed = model_name == "GraFITi" or (
        model_name == "HyperIMTS" and cohort == "mimiciv"
    )
    split_required = cohort == "mimiciv" and (
        model_name == "HyperIMTS"
        or (model_name == "GraFITi" and seed in {2025, 2026, 2027, 2028})
    )
    if split_required and not split_oversized_batches:
        raise ValueError(
            f"{cohort}/{model_name}/seed{seed} requires the author-approved padded-batch strategy"
        )
    if split_oversized_batches and not split_required:
        raise ValueError(f"padded-batch splitting is not authorized for {cohort}/{model_name}/seed{seed}")
    if split_oversized_batches and offload_saved_tensors_to_cpu:
        raise ValueError("padded-batch splitting replaces saved-tensor CPU offload; do not enable both")
    if split_oversized_batches and (padded_size_budget is None or int(padded_size_budget) < 1):
        raise ValueError("padded-size budget must be explicitly frozen for split-batch runs")
    if not split_oversized_batches and padded_size_budget is not None:
        raise ValueError("--padded-size-budget is only valid with padded-batch splitting")
    if offload_saved_tensors_to_cpu:
        # Retired 2026-10-09 (ISSUE-024): offloading every batch made eICU
        # GraFITi take 46-93 h per seed against 0.7 h without it. Completed
        # runs that used it keep their records; new runs offload only the
        # oversized pieces of a split batch.
        raise ValueError(
            "blanket saved-tensor CPU offload is retired; eICU runs fit without it and "
            "MIMIC-IV runs use --split-oversized-padded-batches --offload-oversized-pieces"
        )
    if (offload_oversized_pieces or oom_rescue) and not split_oversized_batches:
        raise ValueError("--offload-oversized-pieces and --oom-rescue apply only to padded-batch splitting runs")
    if offload_saved_tensors_to_cpu and not offload_allowed:
        raise ValueError(
            "saved-tensor CPU offload is currently restricted to GraFITi and "
            "the memory-audited MIMIC-IV HyperIMTS runs"
        )
    microbatch_evidence = (
        _assert_microbatch_training_evidence(model_name, int(padded_size_budget))
        if split_oversized_batches else None
    )
    gate = _assert_gate()
    if seed > 2024:
        trigger = json.loads(TRIGGER_PATH.read_text(encoding="utf-8")) if TRIGGER_PATH.exists() else {}
        trigger_allows = _effective_trigger_status() == "CONTINUE_REMAINING_SEEDS"
        override = (
            json.loads(FAILURE_OVERRIDE_PATH.read_text(encoding="utf-8"))
            if FAILURE_OVERRIDE_PATH.exists()
            else {}
        )
        override_allows = (
            not TRIGGER_PATH.exists()
            and override.get("status") == "CONTINUE_ELIGIBLE_REMAINING_RUNS"
            and f"{cohort}/{model_name}/seed{seed}" in override.get("eligible_run_keys", [])
        )
        not_runnable_override = (
            json.loads(NOT_RUNNABLE_OVERRIDE_PATH.read_text(encoding="utf-8"))
            if NOT_RUNNABLE_OVERRIDE_PATH.exists()
            else {}
        )
        not_runnable_override_allows = (
            not_runnable_override.get("status") == "CONTINUE_OTHER_MODELS_WITH_HYPERIMTS_NOT_RUNNABLE"
            and f"{cohort}/{model_name}/seed{seed}" in not_runnable_override.get("eligible_run_keys", [])
            and (not TRIGGER_PATH.exists() or trigger.get("pause_remaining_seeds") is False)
        )
        if not trigger_allows and not override_allows and not not_runnable_override_allows:
            raise RuntimeError("seeds 2025–2028 are locked until all six pilot runs pass the validation-only trigger")
    gpu_info = _gpu_status(
        gpu, min_free_mib=min_free_mib,
        require_zero_utilization=require_zero_utilization,
    )
    if not gpu_info["free"]:
        raise RuntimeError(f"GPU {gpu} does not meet its free-memory policy; refusing to start: {gpu_info}")

    lock = _run_lock()
    try:
        manifest = _load_manifest()
        run = _manifest_run(manifest, cohort, model_name, seed)
        if run.get("status") != "queued":
            raise RuntimeError(f"refusing to overwrite or rerun {run['run_key']} with status {run.get('status')}")
        config, ref_path, ref_yaml = _get_configs(model_name, cohort, seed)
        config.checkpoints = str(CHECKPOINTS)
        expected_checkpoint_dir = (
            CHECKPOINTS / config.dataset_name / cohort / model_name / config.model_id
            / f"{config.seq_len}_{config.pred_len}" / config.subfolder_train / "iter0"
        )
        if expected_checkpoint_dir.exists() and any(expected_checkpoint_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse existing checkpoint directory: {expected_checkpoint_dir}")
        expected_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        policy_config = None
        if split_oversized_batches:
            policy_config = {
                "enabled": True,
                "padded_size_budget": int(padded_size_budget),
                "padded_size_definition": "B * (max valid input rows + max valid outcome rows) * max observed measurements",
                "single_sample_exceeds_budget": "MIMIC-IV HyperIMTS not_runnable; retain all five cells as not passing in fixed denominator",
                "full_batch_normalizer": True,
                "one_optimizer_step_per_original_batch": True,
                "saved_tensors_cpu_offload": (
                    "oversized single-sample pieces only, pinned host memory"
                    if offload_oversized_pieces else False
                ),
                "offload_oversized_pieces": bool(offload_oversized_pieces),
                "oom_rescue": bool(oom_rescue),
                "oom_rescue_rule": (
                    "on CUDA OOM, zero this batch's gradients and retry the same samples once "
                    "at budget//4 with every piece offloaded; same normalizer, one optimizer step"
                    if oom_rescue else None
                ),
                "frozen_evidence": microbatch_evidence,
            }
        config_for_hash = asdict(config)
        if policy_config is not None:
            config_for_hash["p1_padded_microbatch"] = policy_config
        run.update({
            "status": "running",
            "started_at_utc": now_utc(),
            "gpu": gpu_info,
            "gpu_minimum_free_memory_mib": min_free_mib,
            "saved_tensors_cpu_offload": offload_saved_tensors_to_cpu,
            "saved_tensors_cpu_offload_pin_memory": False if offload_saved_tensors_to_cpu else None,
            "p1_padded_microbatch": policy_config,
            "p1_microbatch_evidence": microbatch_evidence,
            "configuration_reference": str(ref_path.relative_to(ROOT)),
            "configuration_reference_sha256": sha256(ref_path),
            "configuration_sha256": stable_json_sha(config_for_hash),
            "training_launcher_sha256": sha256(TRAINING_SCRIPT),
            "padded_batch_splitter_sha256": sha256(PADDED_BATCH_SPLITTER),
            "microbatch_trainer_sha256": sha256(MICROBATCH_TRAINER),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "model_source": f"external/PyOmniTS/models/{model_name}.py",
            "model_source_sha256": sha256(HARNESS / "models" / f"{model_name}.py"),
            "landmark_cache": str((ROOT / f"data/processed/{cohort}_events_landmark_v1.pkl").relative_to(ROOT)),
            "landmark_cache_sha256": sha256(ROOT / f"data/processed/{cohort}_events_landmark_v1.pkl"),
            "split_sha256": sha256(ROOT / f"data/processed/{cohort}_split_indices_jbi_v1.json"),
            "adapter_gate_generated_at_utc": gate["generated_at_utc"],
            "checkpoint_directory": str(expected_checkpoint_dir.relative_to(ROOT)),
            "test_inference": "not_run",
        })
        manifest["updated_at_utc"] = now_utc()
        _atomic_json(MANIFEST_PATH, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

    epoch_path = EPOCH_DIR / f"{cohort}_{model_name}_seed{seed}.jsonl"
    try:
        # Pin this process to the one verified physical A100 before importing
        # torch or the accelerator singleton.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        sys.path.insert(0, str(HARNESS))
        os.chdir(HARNESS)
        import torch
        from exp.exp_main import Exp_Main
        from utils.globals import accelerator, logger

        _seed_everything(seed, torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable after the free-GPU preflight")
        post_init_min_free_mib = 2048 if min_free_mib is not None else None
        gpu_after = _gpu_status(
            gpu,
            allow_current_process=True,
            min_free_mib=post_init_min_free_mib,
            require_zero_utilization=require_zero_utilization,
        )
        if not gpu_after["free"]:
            raise RuntimeError(f"GPU memory safety check failed before model construction: {gpu_after}")
        accelerator.project_configuration.set_directories(project_dir=expected_checkpoint_dir)

        # Save the actual final P1 config, not just the reference config.
        config_document = asdict(config)
        if policy_config is not None:
            config_document["p1_padded_microbatch"] = policy_config
        (expected_checkpoint_dir / "configs.yaml").write_text(
            yaml.safe_dump(config_document, sort_keys=True), encoding="utf-8"
        )
        if epoch_path.exists():
            raise FileExistsError(f"refusing to overwrite epoch log: {epoch_path}")
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_file = epoch_path.open("x", encoding="utf-8")
        exp = Exp_Main(config)
        if min_free_mib is not None:
            gpu_model = _gpu_status(
                gpu,
                allow_current_process=True,
                min_free_mib=post_init_min_free_mib,
                require_zero_utilization=False,
            )
            if not gpu_model["free"]:
                raise RuntimeError(f"GPU memory safety check failed after model construction: {gpu_model}")
        original_vali = exp.vali

        def guarded_vali(**kwargs):
            report = original_vali(**kwargs)
            selection_skill = float(report.get("macro_persistence_skill", float("nan")))
            val_loss = float(report.get("loss", float("nan")))
            pooled = float(report.get("pooled_mse", float("nan")))
            row = {
                "timestamp_utc": now_utc(),
                "epoch": int(kwargs.get("current_epoch", -1)),
                "train_stage": int(kwargs.get("train_stage", 1)),
                "validation_loss": val_loss,
                "validation_macro_persistence_skill": selection_skill,
                "validation_selection_value": -selection_skill,
                "validation_pooled_mse": pooled,
                "finite_selection_value": math.isfinite(-selection_skill),
            }
            epoch_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            epoch_file.flush()
            os.fsync(epoch_file.fileno())
            if not math.isfinite(selection_skill) or not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"non-finite validation metric at epoch {row['epoch']}: {row}"
                )
            return report

        exp.vali = guarded_vali
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        training_policy_result = None
        if offload_saved_tensors_to_cpu:
            logger.info(
                f"P1 memory policy: offload autograd saved tensors to CPU; "
                f"effective batch size remains {config.batch_size}"
            )
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                exp.train()  # training + validation only; autograd tensors return to their original device on backward
        elif split_oversized_batches:
            from p1_microbatch_training import train_with_padded_microbatches

            logger.info(
                "P1 memory policy: padded batch splitting; batch_size=%s, padded-size budget=%s; "
                "CPU saved-tensor offload disabled",
                config.batch_size,
                padded_size_budget,
            )
            training_policy_result = train_with_padded_microbatches(
                exp, int(padded_size_budget),
                offload_oversized_pieces=offload_oversized_pieces,
                oom_rescue=oom_rescue,
            )
        else:
            exp.train()  # deliberately training + validation only
        duration = time.perf_counter() - started
        # Recorded so dispatch thresholds can be set from measurements
        # instead of guesses (ISSUE-024).
        cuda_peak = {
            "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated(0) // (1024 * 1024)),
            "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(0) // (1024 * 1024)),
        }
        epoch_file.close()
        summary_path = expected_checkpoint_dir / "training_summary.json"
        checkpoint = expected_checkpoint_dir / "pytorch_model.bin"
        if not summary_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("training completed without its summary/checkpoint")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stages = summary.get("stages", [])
        if len(stages) != 1 or stages[0].get("best_epoch") is None:
            raise RuntimeError("training summary has no selected checkpoint epoch")
        val_rows = [json.loads(line) for line in epoch_path.read_text(encoding="utf-8").splitlines() if line]
        if not val_rows or any(not row["finite_selection_value"] for row in val_rows):
            raise RuntimeError("epoch validation selection scores are missing or non-finite")
        run_result: dict[str, Any] = {
            "status": "completed",
            "finished_at_utc": now_utc(),
            "duration_sec": duration,
            "epoch_validation_log": str(epoch_path.relative_to(ROOT)),
            "epoch_validation_log_sha256": sha256(epoch_path),
            "epochs_with_validation": len(val_rows),
            "all_validation_selection_values_finite": True,
            "best_epoch": int(stages[0]["best_epoch"]),
            "best_validation_macro_persistence_skill": float(stages[0]["best_macro_persistence_skill"]),
            "early_stopped": bool(stages[0]["early_stopped"]),
            "batch_size_effective": int(summary.get("batch_size_effective", -1)),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "training_summary_sha256": sha256(summary_path),
            "epoch_training_summary": str(summary_path.relative_to(ROOT)),
            "p1_padded_microbatch_result": training_policy_result,
            **cuda_peak,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "post_preflight_gpu": gpu_after,
            },
            "test_inference": "not_run",
        }
        if seed == 2024:
            t24 = _validation_t24_skill(exp, checkpoint, torch)
            trackb = _trackb_val_skill(cohort, model_name, seed)
            diff = abs(t24["validation_t24_macro_persistence_skill"] - trackb["trackb_best_validation_macro_persistence_skill"])
            run_result.update(t24)
            run_result.update(trackb)
            run_result["validation_t24_abs_difference_from_trackb"] = float(diff)
            run_result["validation_t24_adapter_trigger_gt_0_03"] = bool(diff > 0.03)
            run_result["validation_t24_trigger_interpretation"] = "validation-only troubleshooting; not a hypothesis test"
        lock = _run_lock()
        try:
            manifest = _load_manifest(create=False)
            current = _manifest_run(manifest, cohort, model_name, seed)
            if current.get("status") != "running":
                raise RuntimeError(f"manifest state changed during run: {current.get('status')}")
            current.update(run_result)
            manifest["updated_at_utc"] = now_utc()
            _atomic_json(MANIFEST_PATH, manifest)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        logger.info("P1 training-only run complete: %s/%s/seed%s", cohort, model_name, seed)
        print(json.dumps(run_result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        if "epoch_file" in locals() and not epoch_file.closed:
            epoch_file.close()
        if (
            cohort == "mimiciv"
            and model_name == "HyperIMTS"
            and split_oversized_batches
            and type(exc).__name__ == "SingleSampleCudaOutOfMemory"
        ):
            _mark_mimiciv_hyperimts_not_runnable(seed, exc)
            print(json.dumps({
                "status": "not_runnable",
                "run_key": f"{cohort}/{model_name}/seed{seed}",
                "fixed_denominator_retained": True,
                "evidence": str(exc),
                "cuda_evidence": getattr(exc, "evidence", {}),
            }, indent=2, ensure_ascii=False))
            return 0
        _mark_failed(cohort, model_name, seed, exc, epoch_path)
        raise


def _effective_trigger_status() -> str | None:
    """The pilot trigger's status, honouring a valid §17.3 resolution.

    The trigger file itself is never rewritten. A PAUSE is lifted only by a
    resolution that is bound to this exact trigger file and to unchanged
    evidence files, and that records a passing leakage probe and a completed
    control run.
    """
    if not TRIGGER_PATH.is_file():
        return None
    status = json.loads(TRIGGER_PATH.read_text(encoding="utf-8")).get("status")
    if status == "CONTINUE_REMAINING_SEEDS" or not TRIGGER_RESOLUTION_PATH.is_file():
        return status
    resolution = json.loads(TRIGGER_RESOLUTION_PATH.read_text(encoding="utf-8"))
    evidence = resolution.get("evidence_sha256") or {}
    valid = (
        resolution.get("status") == "RESOLVED_CONTINUE_REMAINING_SEEDS"
        and resolution.get("trigger_sha256") == sha256(TRIGGER_PATH)
        and resolution.get("leakage_probe_pass") is True
        and resolution.get("control_completed") is True
        and len(evidence) == 2
        and all((ROOT / path).is_file() and sha256(ROOT / path) == digest for path, digest in evidence.items())
    )
    return "CONTINUE_REMAINING_SEEDS" if valid else status


def _pilot_trigger() -> int:
    manifest = _load_manifest(create=False)
    pilot = [run for run in manifest["runs"] if int(run["seed"]) == 2024]
    missing = [run["run_key"] for run in pilot if run.get("status") != "completed"]
    incomplete_metrics = []
    for run in pilot:
        value = run.get("validation_t24_abs_difference_from_trackb")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            incomplete_metrics.append(run["run_key"])
    if missing or incomplete_metrics:
        raise RuntimeError(
            f"pilot trigger needs six complete seed-2024 runs with finite validation t=24 comparisons; "
            f"incomplete={missing}, missing metric={incomplete_metrics}"
        )
    details = [
        {
            "run_key": run["run_key"],
            "p1_validation_t24_macro_skill": run["validation_t24_macro_persistence_skill"],
            "trackb_validation_macro_skill": run["trackb_best_validation_macro_persistence_skill"],
            "absolute_difference": run["validation_t24_abs_difference_from_trackb"],
            "trigger_gt_0_03": bool(run["validation_t24_abs_difference_from_trackb"] > 0.03),
        }
        for run in sorted(pilot, key=lambda item: item["run_key"])
    ]
    pause = any(item["trigger_gt_0_03"] for item in details)
    result = {
        "schema_version": "p1_landmark_pilot_trigger_v1",
        "generated_at_utc": now_utc(),
        "analysis_plan_sha256": sha256(P1_PLAN),
        "pilot_runs_expected": 6,
        "pilot_runs_completed": 6,
        "all_comparisons_validation_only": True,
        "threshold_absolute_skill_difference": 0.03,
        "pause_remaining_seeds": pause,
        "status": "PAUSE_REMAINING_SEEDS" if pause else "CONTINUE_REMAINING_SEEDS",
        "interpretation": "adapter troubleshooting trigger only; not a hypothesis test",
        "comparisons": details,
    }
    _atomic_json(TRIGGER_PATH, result)
    manifest["pilot_trigger"] = {
        "path": str(TRIGGER_PATH.relative_to(ROOT)),
        "sha256": sha256(TRIGGER_PATH),
        "status": result["status"],
    }
    manifest["updated_at_utc"] = now_utc()
    _atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not pause else 3


def _freeze_checkpoints() -> int:
    manifest = _load_manifest(create=False)
    runs = manifest["runs"]
    if len(runs) != 30:
        raise RuntimeError(f"cannot freeze P1: expected 30 planned cells, found {len(runs)}")

    non_runnable = [run for run in runs if run.get("status") == "not_runnable"]
    not_completed = [run for run in runs if run.get("status") == "not_completed"]
    unresolved = [
        run["run_key"] for run in runs
        if run.get("status") not in {"completed", "not_runnable", "not_completed"}
    ]
    if unresolved:
        raise RuntimeError(
            f"cannot freeze P1 checkpoints; {len(unresolved)} planned runs are unresolved: {unresolved[:6]}"
        )
    if non_runnable and not_completed:
        raise RuntimeError("cannot combine not_runnable and S5 not_completed cells in one P1 freeze")

    not_runnable_keys = sorted(run["run_key"] for run in non_runnable)
    not_completed_keys = sorted(run["run_key"] for run in not_completed)
    not_completed_ledger_path = ROOT / "results/jbi/_work/p1_s5_time_estimates_20261009.jsonl"
    not_completed_ledger_sha256 = None
    if not_completed:
        eligible_s5_final_keys = {
            f"mimiciv/GraFITi/seed{seed}" for seed in (2026, 2027, 2028)
        } | {
            f"mimiciv/HyperIMTS/seed{seed}" for seed in (2027, 2028)
        }
        if (
            len(not_completed_keys) != len(set(not_completed_keys))
            or not set(not_completed_keys).issubset(eligible_s5_final_keys)
            or any(
                run.get("final_not_completed") is not True
                or run.get("requeue_authorized") is not False
                or run.get("analysis_cell_status") != "not_passing_in_fixed_denominator"
                or float(run.get("s5_projection_hours", 0.0)) <= 4.0
                for run in not_completed
            )
        ):
            raise RuntimeError(
                "cannot freeze: not_completed cells must be eligible §18.2 S5 cells, "
                "retain the fixed denominator, and have projections above 4 h"
            )
        if not not_completed_ledger_path.is_file():
            raise RuntimeError("cannot freeze not_completed cells without the S5 projection ledger")
        s5_records = []
        for line_number, line in enumerate(not_completed_ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    s5_records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid S5 projection ledger line {line_number}") from error
        for run in not_completed:
            key = run["run_key"]
            evidence = [record for record in s5_records if record.get("run_key") == key]
            if not evidence:
                raise RuntimeError(f"S5 projection ledger lacks final not_completed evidence for {key}")
            final_record = evidence[-1]
            if (
                final_record.get("final_cell_resolution") != "not_completed_no_requeue_under_work_order_18_2_item_4"
                or final_record.get("decision") not in {"stop_and_pause_model", "stop_final_not_completed_no_requeue"}
                or float(final_record.get("projected_total_hours", 0.0)) <= 4.0
            ):
                raise RuntimeError(f"S5 ledger does not authorize final fixed-denominator treatment for {key}")
        not_completed_ledger_sha256 = sha256(not_completed_ledger_path)

    if non_runnable:
        expected_not_runnable = [
            f"mimiciv/HyperIMTS/seed{seed}" for seed in SEEDS
        ]
        model_state = manifest.get("model_runnability", {}).get("mimiciv/HyperIMTS", {})
        if (
            not_runnable_keys != expected_not_runnable
            or model_state.get("status") != "not_runnable"
            or model_state.get("fixed_denominator_retained") is not True
            or any(
                run.get("analysis_cell_status") != "not_passing_in_fixed_denominator"
                for run in non_runnable
            )
        ):
            raise RuntimeError(
                "checkpoint freeze permits only the five predeclared MIMIC-IV HyperIMTS cells "
                "to be not_runnable in the fixed denominator"
            )
        if not NOT_RUNNABLE_OVERRIDE_PATH.is_file():
            raise RuntimeError("cannot freeze not-runnable cells without the continuation evidence record")
        not_runnable_override = json.loads(
            NOT_RUNNABLE_OVERRIDE_PATH.read_text(encoding="utf-8")
        )
        if (
            not_runnable_override.get("status") != "CONTINUE_OTHER_MODELS_WITH_HYPERIMTS_NOT_RUNNABLE"
            or sorted(not_runnable_override.get("excluded_run_keys", [])) != expected_not_runnable
            or not all(
                run.get("status") == "completed"
                for run in runs if run["run_key"] not in expected_not_runnable
            )
        ):
            raise RuntimeError(
                "cannot freeze: fixed HyperIMTS not-runnable branch lacks its matching override "
                "or another planned run is not complete"
            )
        if TRIGGER_PATH.exists():
            raise RuntimeError(
                "unexpected six-pilot trigger exists alongside the not-runnable pilot branch; inspect before freezing"
            )
        pilot_trigger_sha256 = None
        freeze_status = "FROZEN_WITH_NOT_RUNNABLE_CELLS"
    else:
        if not TRIGGER_PATH.is_file():
            raise RuntimeError("cannot freeze without the six-pilot validation trigger")
        if _effective_trigger_status() != "CONTINUE_REMAINING_SEEDS":
            raise RuntimeError("pilot trigger paused remaining seeds; inspect before freezing")
        pilot_trigger_sha256 = sha256(TRIGGER_PATH)
        if TRIGGER_RESOLUTION_PATH.is_file():
            manifest["pilot_trigger_resolution"] = {
                "path": str(TRIGGER_RESOLUTION_PATH.relative_to(ROOT)),
                "sha256": sha256(TRIGGER_RESOLUTION_PATH),
            }
        freeze_status = "FROZEN_WITH_NOT_COMPLETED_CELLS" if not_completed else "ALL_30_CHECKPOINTS_FROZEN"

    checkpoint_records = []
    for run in runs:
        if run.get("status") != "completed":
            continue
        checkpoint = ROOT / str(run.get("checkpoint_path", ""))
        if not checkpoint.is_file() or sha256(checkpoint) != run.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint absent/hash mismatch for {run['run_key']}")
        checkpoint_records.append({"run_key": run["run_key"], "path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint)})
    freeze = {
        "schema_version": "p1_landmark_checkpoint_freeze_v2",
        "status": freeze_status,
        "frozen_at_utc": now_utc(),
        "analysis_plan_sha256": sha256(P1_PLAN),
        "pilot_trigger_sha256": pilot_trigger_sha256,
        "planned_run_count": len(runs),
        "run_count": len(checkpoint_records),
        "not_runnable_run_keys": not_runnable_keys,
        "not_completed_run_keys": not_completed_keys,
        "not_completed_evidence_path": (
            str(not_completed_ledger_path.relative_to(ROOT)) if not_completed else None
        ),
        "not_completed_evidence_sha256": not_completed_ledger_sha256,
        "not_runnable_override_sha256": (
            sha256(NOT_RUNNABLE_OVERRIDE_PATH) if non_runnable else None
        ),
        "checkpoints": checkpoint_records,
        "checkpoint_set_sha256": stable_json_sha(checkpoint_records),
        "test_inference_run_keys": [record["run_key"] for record in checkpoint_records],
        "test_inference_authorized": True,
        "test_inference_runs_allowed": 1,
    }
    manifest["all_checkpoints_frozen"] = True
    manifest["checkpoints_frozen_at_utc"] = freeze["frozen_at_utc"]
    manifest["checkpoint_set_sha256"] = freeze["checkpoint_set_sha256"]
    manifest["checkpoint_set"] = checkpoint_records
    manifest["checkpoint_freeze_status"] = freeze_status
    manifest["planned_run_count_frozen"] = len(runs)
    manifest["not_runnable_run_keys_frozen"] = not_runnable_keys
    manifest["not_completed_run_keys_frozen"] = not_completed_keys
    manifest["not_completed_evidence_sha256"] = not_completed_ledger_sha256
    manifest["test_inference_authorized"] = True
    manifest["test_inference_runs_allowed"] = 1
    manifest["test_inference_run_keys"] = [record["run_key"] for record in checkpoint_records]
    manifest["updated_at_utc"] = now_utc()
    _atomic_json(MANIFEST_PATH, manifest)
    freeze_path = ROOT / "results/jbi/p1_landmark_checkpoint_freeze.json"
    _atomic_json(freeze_path, freeze)
    print(
        f"frozen {len(checkpoint_records)} runnable checkpoints across {len(runs)} planned cells "
        f"({len(not_runnable_keys)} not_runnable, {len(not_completed_keys)} not_completed); "
        f"one test inference pass authorized at {freeze_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("gate", help="run and record all six adapter tests")
    run_parser = subparsers.add_parser("run", help="run one training-only cohort/model/seed job")
    run_parser.add_argument("--cohort", choices=COHORTS, required=True)
    run_parser.add_argument("--model", choices=MODELS, required=True)
    run_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run_parser.add_argument("--gpu", type=int, required=True, help="physical A100 index")
    run_parser.add_argument(
        "--min-free-mib",
        type=int,
        default=None,
        help="allow shared use only when this much VRAM is free and GPU utilization is zero; omitted means exclusive idle use",
    )
    run_parser.add_argument(
        "--allow-nonzero-utilization",
        action="store_true",
        help="allow use alongside another GPU process when --min-free-mib is met; intended for author-authorized sharing",
    )
    run_parser.add_argument(
        "--offload-saved-tensors-to-cpu",
        action="store_true",
        help="for GraFITi and memory-audited MIMIC-IV HyperIMTS runs, trade host RAM and transfer time for lower activation VRAM while keeping the effective batch unchanged",
    )
    run_parser.add_argument(
        "--split-oversized-padded-batches",
        action="store_true",
        help="for the author-specified MIMIC-IV HyperIMTS/GraFITi runs, split only batches above the frozen padded-size budget and accumulate against the full-batch target count",
    )
    run_parser.add_argument(
        "--padded-size-budget",
        type=int,
        default=None,
        help="frozen threshold in B * (max valid input rows + max valid outcome rows) * max observed measurements; required with padded batch splitting",
    )
    run_parser.add_argument(
        "--offload-oversized-pieces",
        action="store_true",
        help="with padded-batch splitting, run any single sample above the budget with autograd tensors in pinned host memory instead of refusing it",
    )
    run_parser.add_argument(
        "--oom-rescue",
        action="store_true",
        help="with padded-batch splitting, retry a batch that hits CUDA OOM once at budget//4 with all pieces offloaded",
    )
    subparsers.add_parser("pilot-trigger", help="evaluate the six validation-only t=24 differences")
    subparsers.add_parser(
        "freeze",
        help="freeze all completed checkpoints after every planned cell is resolved; authorize one test pass",
    )
    args = parser.parse_args()
    if args.command == "gate":
        return _run_gate()
    if args.command == "run":
        return _run_one(
            args.cohort, args.model, args.seed, args.gpu, args.min_free_mib,
            args.offload_saved_tensors_to_cpu,
            args.split_oversized_padded_batches,
            args.padded_size_budget,
            not args.allow_nonzero_utilization,
            args.offload_oversized_pieces,
            args.oom_rescue,
        )
    if args.command == "pilot-trigger":
        return _pilot_trigger()
    return _freeze_checkpoints()


if __name__ == "__main__":
    sys.exit(main())
