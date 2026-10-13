#!/usr/bin/env python
"""Frozen P2 eICU leave-one-hospital-group-out training queue.

The ``self-check`` command verifies the registered plan, frozen fold samples,
source hashes and selected Track-B MAE configuration files. It does not load
the LOHO cache payload, inspect test outcomes, initialize CUDA or start a run.
Training remains locked until P1 test inference and I1/I2 analysis are complete.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from p2_plan_integrity import inspect_plan_identity


ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("ICU_KAN_PYTHON", sys.executable)
HARNESS = ROOT / "external/PyOmniTS"
PLAN = ROOT / "paper/跨医院验证分析计划_P2.md"
P1_PLAN = ROOT / "paper/预设用例分析计划_P1.md"
SAMPLE_MANIFEST = ROOT / "results/jbi/p2_training_sample_manifest.json"
OVERLAP_AUDIT = ROOT / "results/jbi/p2_hospital_overlap_audit.json"
LOHO_CACHE = ROOT / "data/processed/eicu_loho_events.pkl"
FOLD_MAP = ROOT / "results/phase3_eicu/e_loho/loho_folds.csv"
SELF_CHECK = ROOT / "results/jbi/p2_training_queue_self_check.json"
TRAIN_MANIFEST = ROOT / "results/jbi/p2_loho_training_manifest.json"
QUEUE_STATUS = ROOT / "results/jbi/p2_loho_queue_status.json"
LOCK_PATH = ROOT / "results/jbi/.p2_loho_training_manifest.lock"
CHECKPOINT_ROOT = ROOT / "results/jbi/p2_loho_checkpoints"
EPOCH_ROOT = ROOT / "results/jbi/p2_loho_epoch_metrics"
LOG_ROOT = ROOT / "results/jbi/p2_loho_queue_logs"
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
OBJECTIVES = ("MAE", "MSE")
FOLDS = tuple(range(5))
SEEDS = (2024, 2025, 2026)
EXPECTED_DROPOUT = {"ICUKANForecaster": 0.05, "GraFITi": 0.05, "HyperIMTS": 0.0}
P1_RUNNER_PATH = ROOT / "scripts/jbi/42_train_p1_landmark.py"
P2_RUN_WALLCLOCK_CAP_SEC = 3 * 60 * 60
P2_GPU_MIN_FREE_MIB = 39_000
P2_GPU_IDLE_SAMPLES = 5
P2_GPU_SAMPLE_INTERVAL_SEC = 2
# Work order §20.2: GraFITi and HyperIMTS train through P1's exact padded
# micro-batch path. Only original batches above this padded size (those
# holding the dense fold-0/2 stays) are split; every Track B and fold-1/3/4
# batch measured stays whole.
P2_SPLIT_MODELS = ("GraFITi", "HyperIMTS")
P2_PADDED_SIZE_BUDGET = 150_000_000
MICROBATCH_TRAINER = ROOT / "scripts/jbi/p1_microbatch_training.py"
PADDED_BATCH_SPLITTER = ROOT / "scripts/jbi/p1_padded_batch_split.py"
MICROBATCH_PROOF = ROOT / "results/jbi/p2_microbatch_gradient_equivalence.json"
FAILED_ATTEMPTS_ROOT = ROOT / "results/jbi/_work/p2_loho_failed_attempts"
# Work order §21.2: validation batches above the same budget are split too.
VALIDATION_SPLIT_PROOF = ROOT / "results/jbi/p2_validation_split_equivalence.json"
REQUEUE_REASONS = {
    "cuda_oom_before_p2_microbatch_path": "work order §20.2",
    "validation_oom_before_split_validation": "work order §21.2",
}


def _s5_wallclock_alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError(
        f"S5 eICU per-run wall-clock cap exceeded ({P2_RUN_WALLCLOCK_CAP_SEC} seconds)"
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analysis_plan_identity() -> dict[str, str | None]:
    """Return frozen design and resolution hashes after integrity validation."""
    return inspect_plan_identity(PLAN)


def analysis_plan_sha256() -> str:
    return str(analysis_plan_identity()["analysis_plan_sha256"])


def stable_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_runner_for_p2", P1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(P1_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_config(model: str) -> tuple[Path, dict[str, Any]]:
    tag = f"jbi_trackb_controls_factorial_{model}_mae_macro_v1_eicu"
    path = HARNESS / "configs" / model / tag / "ICUKAN.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen eICU Track-B masked-L1 config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "dataset_subset_name": "jbi_patient",
        "seq_len": 24,
        "pred_len": 24,
        "loss": "MAE",
        "checkpoint_selection_metric": "macro_persistence",
        "batch_size": 128,
    }
    for key, expected in required.items():
        actual = payload.get(key)
        if str(actual).upper() != str(expected).upper():
            raise RuntimeError(f"Track-B reference mismatch for {model}.{key}: {actual!r} != {expected!r}")
    if not math.isclose(float(payload.get("learning_rate", float("nan"))), 0.001, abs_tol=1e-12):
        raise RuntimeError(f"Track-B reference learning rate is not 0.001: {path}")
    if not math.isclose(float(payload.get("dropout", float("nan"))), EXPECTED_DROPOUT[model], abs_tol=1e-12):
        raise RuntimeError(f"Track-B reference dropout differs from P2 plan: {path}")
    return path, payload


def validate_sources() -> dict[str, Any]:
    for path in (PLAN, SAMPLE_MANIFEST, OVERLAP_AUDIT, LOHO_CACHE, FOLD_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan_sha = analysis_plan_sha256()
    samples = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(OVERLAP_AUDIT.read_text(encoding="utf-8"))
    if samples.get("status") != "PASS" or audit.get("training_gate") != "PASS":
        raise RuntimeError("P2 hospital-overlap audit or frozen sample manifest is not PASS")
    if samples.get("analysis_plan_sha256") != plan_sha or audit.get("analysis_plan_sha256") != plan_sha:
        raise RuntimeError("P2 plan hash differs from the frozen audit/sample manifest")
    cache_sha, folds_sha = sha256(LOHO_CACHE), sha256(FOLD_MAP)
    if samples.get("source_cache_sha256") != cache_sha or audit.get("cache_sha256") != cache_sha:
        raise RuntimeError("LOHO cache hash differs from the frozen P2 prep artifacts")
    if samples.get("source_fold_file_sha256") != folds_sha or audit.get("fold_file_sha256") != folds_sha:
        raise RuntimeError("LOHO fold-map hash differs from the frozen P2 prep artifacts")
    folds = samples.get("folds", {})
    if set(folds) != {str(fold) for fold in FOLDS}:
        raise RuntimeError("P2 sample manifest does not contain all five folds")
    train_val_hashes: dict[str, str] = {}
    for fold in FOLDS:
        item = folds[str(fold)]
        if item.get("train_count") != 8400 or item.get("validation_count") != 1800:
            raise RuntimeError(f"fold {fold} does not have the frozen 8,400/1,800 samples")
        train_val_hashes[str(fold)] = stable_sha({
            "train_stay_indices": item["train_stay_indices"],
            "validation_stay_indices": item["validation_stay_indices"],
            "train_stay_ids": item["train_stay_ids"],
            "validation_stay_ids": item["validation_stay_ids"],
        })
    references = {}
    for model in MODELS:
        path, config = _reference_config(model)
        references[model] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "loss_source": str(config["loss"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["learning_rate"]),
            "dropout": float(config["dropout"]),
            "selection_metric": config["checkpoint_selection_metric"],
        }
    return {
        "schema_version": "p2_loho_training_queue_self_check_v1",
        "generated_at_utc": now_utc(),
        "status": "PASS",
        "queue_script_sha256": sha256(Path(__file__).resolve()),
        "s5_wallclock_cap_seconds": P2_RUN_WALLCLOCK_CAP_SEC,
        "gpu_dispatch_policy": {
            "memory_policy": "shared_idle_with_minimum_free_memory",
            "minimum_free_memory_mib": P2_GPU_MIN_FREE_MIB,
            "require_zero_utilization": True,
            "stable_idle_samples": P2_GPU_IDLE_SAMPLES,
            "sample_interval_seconds": P2_GPU_SAMPLE_INTERVAL_SEC,
        },
        "training_started": False,
        "test_values_or_labels_read": False,
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": plan_sha},
        "p1_plan_sha256": sha256(P1_PLAN),
        "sample_manifest": {
            "path": str(SAMPLE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(SAMPLE_MANIFEST),
            "sample_ids_sha256": samples["sample_ids_sha256"],
        },
        "overlap_audit_sha256": sha256(OVERLAP_AUDIT),
        "loho_cache_sha256": cache_sha,
        "fold_map_sha256": folds_sha,
        "train_validation_sample_hashes": train_val_hashes,
        "reference_trackb_mae_configs": references,
        "planned_run_count": len(MODELS) * len(OBJECTIVES) * len(FOLDS) * len(SEEDS),
        "planned_folds": list(FOLDS),
        "planned_models": list(MODELS),
        "planned_objectives": list(OBJECTIVES),
        "planned_seeds": list(SEEDS),
        "frozen_training_settings": {
            "seq_len_hours": 24,
            "pred_len_hours": 24,
            "batch_size": 128,
            "train_epochs": 300,
            "patience": 10,
            "val_interval": 1,
            "learning_rate": 0.001,
            "selection_metric": "macro_persistence",
            "dropout_by_model": EXPECTED_DROPOUT,
        },
    }


def verify_self_check_current(
    current: dict[str, Any] | None = None,
    *,
    allow_registered_queue_code_update: bool = False,
) -> dict[str, Any]:
    if not SELF_CHECK.is_file():
        raise RuntimeError("run `self-check` before P2 training")
    current = current or validate_sources()
    saved = json.loads(SELF_CHECK.read_text(encoding="utf-8"))
    stable_fields = (
        "plan", "sample_manifest", "overlap_audit_sha256", "loho_cache_sha256",
        "fold_map_sha256", "train_validation_sample_hashes", "reference_trackb_mae_configs",
        "s5_wallclock_cap_seconds", "gpu_dispatch_policy",
    )
    for field in stable_fields:
        if saved.get(field) != current.get(field):
            raise RuntimeError(f"P2 self-check is stale for {field}; rerun `self-check`")
    if saved.get("queue_script_sha256") != current.get("queue_script_sha256"):
        if not allow_registered_queue_code_update:
            raise RuntimeError("P2 queue source changed after self-check; use the recorded resume path")
        manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")) if TRAIN_MANIFEST.is_file() else {}
        if manifest.get("training_started") is not True:
            raise RuntimeError("registered queue source update is allowed only after the frozen campaign started")
    return current


def _jobs() -> list[dict[str, Any]]:
    return [
        {"run_key": f"fold{fold}/{model}/{objective}/seed{seed}",
         "fold": fold, "model": model, "objective": objective, "seed": seed}
        for fold in FOLDS for model in MODELS for objective in OBJECTIVES for seed in SEEDS
    ]


def _p2_gpu_status(runner, gpu: int) -> dict[str, Any]:
    """Use the shared-idle gate without disturbing other GPU users."""
    return runner._gpu_status(
        gpu,
        min_free_mib=P2_GPU_MIN_FREE_MIB,
        require_zero_utilization=True,
    )


def _stable_p2_gpu_status(runner, gpu: int) -> dict[str, Any]:
    """Require five consecutive zero-utilization, 39-GB-free samples."""
    samples: list[dict[str, Any]] = []
    latest: dict[str, Any] | None = None
    for index in range(P2_GPU_IDLE_SAMPLES):
        latest = _p2_gpu_status(runner, gpu)
        samples.append({
            "memory_free_mib": latest["memory_free_mib"],
            "utilization_percent": latest["utilization_percent"],
            "free": latest["free"],
        })
        if not latest["free"]:
            break
        if index + 1 < P2_GPU_IDLE_SAMPLES:
            time.sleep(P2_GPU_SAMPLE_INTERVAL_SEC)
    assert latest is not None
    latest["stable_idle_samples"] = samples
    latest["stable_idle_sample_count"] = len(samples)
    latest["free"] = len(samples) == P2_GPU_IDLE_SAMPLES and all(item["free"] for item in samples)
    latest["memory_policy"] = "shared_idle_with_minimum_free_memory_stable_samples"
    return latest


def _manifest_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _load_training_manifest(create: bool = False, verify_source_hashes: bool = False) -> dict[str, Any]:
    expected = {job["run_key"] for job in _jobs()}
    if TRAIN_MANIFEST.is_file():
        manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
        if {item.get("run_key") for item in manifest.get("runs", [])} != expected:
            raise RuntimeError("existing P2 training manifest has an unexpected run set")
        if manifest.get("self_check_sha256") != sha256(SELF_CHECK):
            raise RuntimeError("P2 source/config self-check changed after manifest initialization")
        if analysis_plan_sha256() != manifest.get("analysis_plan_sha256"):
            raise RuntimeError("P2 analysis plan changed after manifest initialization")
        if sha256(SAMPLE_MANIFEST) != manifest.get("sample_manifest_sha256"):
            raise RuntimeError("P2 frozen sample manifest changed after initialization")
        if verify_source_hashes:
            current_check = validate_sources()
            if manifest.get("loho_cache_sha256") != current_check["loho_cache_sha256"]:
                raise RuntimeError("P2 LOHO cache changed after manifest initialization")
        return manifest
    if not create:
        raise FileNotFoundError(TRAIN_MANIFEST)
    if not SELF_CHECK.is_file():
        raise RuntimeError("run `self-check` before initializing the P2 queue")
    check = json.loads(SELF_CHECK.read_text(encoding="utf-8"))
    current = validate_sources()
    if check.get("plan", {}).get("sha256") != current["plan"]["sha256"]:
        raise RuntimeError("P2 self-check is stale")
    return {
        "schema_version": "p2_loho_training_manifest_v1",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "status": "QUEUED",
        "training_started": False,
        "test_inference_authorized": False,
        "self_check_sha256": sha256(SELF_CHECK),
        "analysis_plan_sha256": current["plan"]["sha256"],
        "sample_manifest_sha256": current["sample_manifest"]["sha256"],
        "loho_cache_sha256": current["loho_cache_sha256"],
        "fold_map_sha256": current["fold_map_sha256"],
        "reference_trackb_mae_configs": current["reference_trackb_mae_configs"],
        "runs": [{**job, "status": "queued"} for job in _jobs()],
    }


def _set_training_campaign_status(status_value: str) -> None:
    """Persist the campaign terminal state without changing any run record."""
    lock = _manifest_lock()
    try:
        manifest = _load_training_manifest(create=False)
        manifest["status"] = status_value
        manifest["updated_at_utc"] = now_utc()
        atomic_json(TRAIN_MANIFEST, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _update_run(run_key: str, fields: dict[str, Any], *, expected_status: str | None = None) -> None:
    lock = _manifest_lock()
    try:
        manifest = _load_training_manifest(create=False)
        record = next((item for item in manifest["runs"] if item["run_key"] == run_key), None)
        if record is None:
            raise KeyError(run_key)
        if expected_status and record.get("status") != expected_status:
            raise RuntimeError(f"{run_key}: expected {expected_status}, found {record.get('status')}")
        record.update(fields)
        manifest["updated_at_utc"] = now_utc()
        atomic_json(TRAIN_MANIFEST, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _p1_complete_gate() -> None:
    p1_manifest_path = ROOT / "results/jbi/p1_landmark_training_manifest.json"
    test_ledger_path = ROOT / "results/jbi/p1_landmark_test_inference_ledger.json"
    verdict_path = ROOT / "results/jbi/p1_landmark_verdict.json"
    if not all(path.is_file() for path in (p1_manifest_path, test_ledger_path, verdict_path)):
        raise RuntimeError("P2 is locked until P1 freeze, one-pass test inference and I1/I2 analysis exist")
    p1 = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(test_ledger_path.read_text(encoding="utf-8"))
    if p1.get("all_checkpoints_frozen") is not True or p1.get("test_inference_consumed") is not True:
        raise RuntimeError("P1 checkpoint freeze/test inference is not complete")
    if ledger.get("status") != "COMPLETE" or ledger.get("test_inference_consumed") is not True:
        raise RuntimeError("P1 one-pass test inference ledger is not COMPLETE")


def _parse_run_key(key: str) -> dict[str, Any]:
    parts = key.split("/")
    if len(parts) != 4 or not parts[0].startswith("fold") or not parts[3].startswith("seed"):
        raise ValueError(f"invalid P2 run key: {key}")
    job = {"fold": int(parts[0][4:]), "model": parts[1], "objective": parts[2], "seed": int(parts[3][4:]), "run_key": key}
    if job not in _jobs():
        raise ValueError(f"run key is outside the frozen P2 grid: {key}")
    return job


def _configuration(model: str, objective: str, fold: int, seed: int):
    runner = _load_runner()
    config, ref_path, ref = runner._get_configs(model, "eicu", seed)
    config.model_name = model
    config.model_id = f"p2_loho_{model}_{objective.lower()}_fold{fold}_seed{seed}"
    config.dataset_name = "ICUKAN"
    config.dataset_id = f"eicu_loho_fold{fold}"
    config.dataset_root_path = str(ROOT / "data/processed")
    config.dataset_subset_name = "p2_loho"
    config.p2_fold = int(fold)
    config.p2_mode = "train"
    config.seq_len = 24
    config.pred_len = 24
    config.enc_in = config.dec_in = config.c_out = 37
    config.checkpoints = str(CHECKPOINT_ROOT)
    config.checkpoints_test = None
    config.subfolder_train = "p2_loho_v1"
    config.itr = 1
    config.itr_i = 0
    config.iteration_index = -1
    config.is_training = 1
    config.use_gpu = 1
    config.gpu_id = 0
    config.gpu_ids = None
    config.use_multi_gpu = 0
    config.batch_size = 128
    config.learning_rate = 0.001
    config.dropout = EXPECTED_DROPOUT[model]
    config.loss = objective
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
    config.device = "cuda"
    # Track-B factorial HyperIMTS runs use this shipped low-memory path.
    config.hyperimts_force_low_memory = 1 if model == "HyperIMTS" else 0
    return config, ref_path, ref


def _install_train_only_loho_loader(fold: int):
    """Bind ICUKAN to frozen train/validation indices and hide test indices."""
    sys.path.insert(0, str(HARNESS))
    import data.data_provider.datasets.ICUKAN as adapter

    def load_cache(self):
        samples = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
        if samples.get("status") != "PASS":
            raise RuntimeError("P2 sample manifest is not PASS")
        fold_data = samples["folds"][str(fold)]
        with LOHO_CACHE.open("rb") as handle:
            payload = __import__("pickle").load(handle)
        self._examples = payload["examples"]
        self._splits = {
            "train": np.asarray(fold_data["train_stay_indices"], dtype=np.int64),
            "val": np.asarray(fold_data["validation_stay_indices"], dtype=np.int64),
            # No test rows are exposed to the train/validation data adapter.
            "test": np.empty(0, dtype=np.int64),
        }
        self._cohort = "eicu"
        self._data_variant = "p2_loho_train_validation_only"
        self.data_provenance = {
            "data_source": "eICU LOHO frozen P2 fold",
            "fold": int(fold),
            "cache_path": str(LOHO_CACHE),
            "cache_sha256": sha256(LOHO_CACHE),
            "sample_manifest_sha256": sha256(SAMPLE_MANIFEST),
            "split_sizes": {key: int(len(value)) for key, value in self._splits.items()},
            "test_rows_exposed": 0,
        }

    adapter.Data._load_cache = load_cache
    return adapter


def _microbatch_evidence() -> dict[str, Any]:
    """The P2 proof must pass for the exact trainer, splitter and budget in use."""
    if not MICROBATCH_PROOF.is_file():
        raise RuntimeError(f"P2 micro-batch proof is missing: {MICROBATCH_PROOF}")
    proof = json.loads(MICROBATCH_PROOF.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "padded_size_budget": P2_PADDED_SIZE_BUDGET,
        "trainer_sha256": sha256(MICROBATCH_TRAINER),
        "splitter_sha256": sha256(PADDED_BATCH_SPLITTER),
    }
    mismatched = {key: proof.get(key) for key, value in expected.items() if proof.get(key) != value}
    if mismatched:
        raise RuntimeError(f"P2 micro-batch proof does not match current code or budget: {mismatched}")
    return {"proof_path": str(MICROBATCH_PROOF.relative_to(ROOT)), "proof_sha256": sha256(MICROBATCH_PROOF), **expected}


def split_validation_model(model: Any, budget: int) -> Any:
    """Wrap ``model`` so an over-budget validation batch runs piece by piece.

    The pieces come from the same planner as training, receive the full
    batch's observation padding width, run under the caller's no_grad, and
    their outputs are concatenated back in batch order. Each piece is cropped
    to its own outcome rows, so ``pred``/``true``/``mask`` are zero-padded
    back to the full batch's outcome length; those rows carry mask 0 and
    drop out of the loss, pooled MSE and persistence counts. Any batch within
    the budget, and every non-validation call, goes to the model unchanged.
    """
    import torch
    from p1_padded_batch_split import plan_padded_batch, slice_padded_batch

    class _SplitValidation(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def forward(self, **batch: Any) -> dict[str, Any]:
            tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            options = {key: value for key, value in batch.items() if key not in tensors}
            if options.get("exp_stage") != "val":
                return self.inner(**batch)
            plan = plan_padded_batch(tensors, budget, allow_oversized_singletons=True)
            if not plan.was_split:
                return self.inner(**batch)
            size = int(tensors["y_mask"].shape[0])
            parts = [
                self.inner(**slice_padded_batch(tensors, positions), **options,
                           p1_observation_padding_width=max(plan.observation_counts))
                for positions in plan.piece_positions
            ]
            outcome_rows = int(tensors["y_mask"].shape[1])
            merged = {}
            for key, value in parts[0].items():
                if not isinstance(value, torch.Tensor) or value.dim() == 0 or value.shape[0] != len(plan.piece_positions[0]):
                    raise RuntimeError(f"cannot reassemble non-batch validation output {key!r}")
                pieces = [part[key] for part in parts]
                if key in ("pred", "true", "mask"):
                    pieces = [
                        torch.nn.functional.pad(piece, (0, 0) * (piece.dim() - 2) + (0, outcome_rows - piece.shape[1]))
                        for piece in pieces
                    ]
                merged[key] = torch.cat(pieces, dim=0)
                if merged[key].shape[0] != size:
                    raise RuntimeError(f"reassembled validation output {key!r} has the wrong batch size")
            if tuple(merged["pred"].shape[:2]) != tuple(tensors["y_mask"].shape[:2]):
                raise RuntimeError("reassembled validation predictions do not match the full outcome grid")
            return merged

    return _SplitValidation(model)


def _validation_split_evidence() -> dict[str, Any]:
    """The validation-split proof must match the wrapper source and budget in use."""
    import inspect

    if not VALIDATION_SPLIT_PROOF.is_file():
        raise RuntimeError(f"P2 validation-split proof is missing: {VALIDATION_SPLIT_PROOF}")
    proof = json.loads(VALIDATION_SPLIT_PROOF.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "padded_size_budget": P2_PADDED_SIZE_BUDGET,
        "wrapper_source_sha256": hashlib.sha256(inspect.getsource(split_validation_model).encode("utf-8")).hexdigest(),
        "splitter_sha256": sha256(PADDED_BATCH_SPLITTER),
    }
    mismatched = {key: proof.get(key) for key, value in expected.items() if proof.get(key) != value}
    if mismatched:
        raise RuntimeError(f"P2 validation-split proof does not match current code or budget: {mismatched}")
    return {"proof_path": str(VALIDATION_SPLIT_PROOF.relative_to(ROOT)), "proof_sha256": sha256(VALIDATION_SPLIT_PROOF), **expected}


def train_one(run_key: str, gpu: int, *, allow_registered_queue_code_update: bool = False) -> int:
    job = _parse_run_key(run_key)
    split_path = job["model"] in P2_SPLIT_MODELS
    microbatch = _microbatch_evidence() if split_path else None
    if split_path:
        microbatch = {**microbatch, "validation_split": _validation_split_evidence()}
    _p1_complete_gate()
    check = validate_sources()
    verify_self_check_current(check, allow_registered_queue_code_update=allow_registered_queue_code_update)
    if not SELF_CHECK.is_file() or sha256(SELF_CHECK) != json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")).get("self_check_sha256"):
        raise RuntimeError("P2 self-check changed or is absent")
    runner = _load_runner()
    gpu_info = _p2_gpu_status(runner, gpu)
    if not gpu_info.get("free"):
        raise RuntimeError(f"P2 training requires a shared-idle A100 with at least {P2_GPU_MIN_FREE_MIB} MiB free and zero utilization: {gpu_info}")

    lock = _manifest_lock()
    try:
        manifest = _load_training_manifest(create=False)
        record = next(item for item in manifest["runs"] if item["run_key"] == run_key)
        if record.get("status") != "queued":
            raise RuntimeError(f"refusing to overwrite non-queued P2 run: {run_key}={record.get('status')}")
        config, ref_path, _ref = _configuration(job["model"], job["objective"], job["fold"], job["seed"])
        output = (
            CHECKPOINT_ROOT / "ICUKAN" / config.dataset_id / job["model"] / config.model_id
            / "24_24" / config.subfolder_train / "iter0"
        )
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to reuse P2 checkpoint directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        config_sha = stable_sha({
            "reference_yaml_sha256": sha256(ref_path),
            "model": job["model"], "objective": job["objective"], "fold": job["fold"], "seed": job["seed"],
            "seq_len": 24, "pred_len": 24, "batch_size": 128, "epochs": 300, "patience": 10,
            "learning_rate": 0.001, "dropout": EXPECTED_DROPOUT[job["model"]],
            "selection": "macro_persistence", "train_validation_sample_sha256": check["train_validation_sample_hashes"][str(job["fold"])],
        })
        record.update({
            "status": "running", "started_at_utc": now_utc(), "gpu": gpu_info,
            "training_entrypoint_sha256": sha256(Path(__file__).resolve()),
            "registered_queue_source_update": bool(allow_registered_queue_code_update),
            "configuration_reference": str(ref_path.relative_to(ROOT)),
            "configuration_reference_sha256": sha256(ref_path),
            "configuration_sha256": config_sha,
            "train_validation_sample_sha256": check["train_validation_sample_hashes"][str(job["fold"])],
            "sample_manifest_sha256": sha256(SAMPLE_MANIFEST),
            "loho_cache_sha256": sha256(LOHO_CACHE),
            "fold_map_sha256": sha256(FOLD_MAP),
            "s5_wallclock_cap_seconds": P2_RUN_WALLCLOCK_CAP_SEC,
            "checkpoint_directory": str(output.relative_to(ROOT)),
            "p2_padded_microbatch": microbatch,
            "test_inference": "not_run",
        })
        manifest["training_started"] = True
        manifest["status"] = "RUNNING"
        manifest["updated_at_utc"] = now_utc()
        atomic_json(TRAIN_MANIFEST, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

    epoch_path = EPOCH_ROOT / (run_key.replace("/", "_") + ".jsonl")
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_file = None
    started = time.perf_counter()
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _s5_wallclock_alarm)
    signal.setitimer(signal.ITIMER_REAL, P2_RUN_WALLCLOCK_CAP_SEC)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        if split_path:
            # Allocation behaviour only, as for the P1 GraFITi/HyperIMTS runs.
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        sys.path.insert(0, str(HARNESS))
        os.chdir(HARNESS)
        import torch
        from exp.exp_main import Exp_Main
        from utils.globals import accelerator

        config, ref_path, _ref = _configuration(job["model"], job["objective"], job["fold"], job["seed"])
        config.gpu_id = 0
        adapter = _install_train_only_loho_loader(job["fold"])
        runner._seed_everything(job["seed"], torch)
        accelerator.project_configuration.set_directories(project_dir=output)
        (output / "configs.yaml").write_text(yaml.safe_dump(config.__dict__, sort_keys=True), encoding="utf-8")
        epoch_file = epoch_path.open("x", encoding="utf-8")
        exp = Exp_Main(config)
        original_vali = exp.vali

        def finite_logged_validation(**kwargs):
            if split_path:
                kwargs["model_train"] = split_validation_model(kwargs["model_train"], P2_PADDED_SIZE_BUDGET)
            report = original_vali(**kwargs)
            skill = float(report.get("macro_persistence_skill", float("nan")))
            loss = float(report.get("loss", float("nan")))
            if not math.isfinite(skill) or not math.isfinite(loss):
                raise FloatingPointError(f"non-finite P2 validation metric at epoch {kwargs.get('current_epoch')}")
            row = {"epoch": int(kwargs.get("current_epoch", -1)), "validation_macro_persistence_skill": skill, "validation_loss": loss}
            epoch_file.write(json.dumps(row) + "\n")
            epoch_file.flush()
            os.fsync(epoch_file.fileno())
            return report

        exp.vali = finite_logged_validation
        torch.cuda.reset_peak_memory_stats(0)
        # Train and validation only; the test split is an empty index set in this adapter.
        if split_path:
            sys.path.insert(0, str(MICROBATCH_TRAINER.parent))
            from p1_microbatch_training import train_with_padded_microbatches

            microbatch_summary = train_with_padded_microbatches(
                exp, P2_PADDED_SIZE_BUDGET, offload_oversized_pieces=True, oom_rescue=True,
            )
        else:
            exp.train()
            microbatch_summary = None
        elapsed = time.perf_counter() - started
        summary_path = output / "training_summary.json"
        checkpoint = output / "pytorch_model.bin"
        if not summary_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("P2 training ended without its selected checkpoint and summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stages = summary.get("stages", [])
        if len(stages) != 1 or stages[0].get("best_epoch") is None:
            raise RuntimeError("P2 training summary lacks a best validation epoch")
        rows = [json.loads(line) for line in epoch_path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise RuntimeError("P2 training wrote no validation rows")
        result = {
            "status": "completed", "finished_at_utc": now_utc(), "duration_sec": elapsed,
            "s5_wallclock_cap_seconds": P2_RUN_WALLCLOCK_CAP_SEC,
            "s5_wallclock_cap_triggered": False,
            "best_epoch": int(stages[0]["best_epoch"]),
            "best_validation_macro_persistence_skill": float(stages[0]["best_macro_persistence_skill"]),
            "validation_epoch_count": len(rows), "all_validation_selection_values_finite": True,
            "epoch_validation_log": str(epoch_path.relative_to(ROOT)), "epoch_validation_log_sha256": sha256(epoch_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
            "training_summary_path": str(summary_path.relative_to(ROOT)), "training_summary_sha256": sha256(summary_path),
            "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated(0) // (1024 * 1024)),
            "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(0) // (1024 * 1024)),
            "p2_padded_microbatch_summary": microbatch_summary,
            "test_inference": "not_run",
        }
        epoch_file.close()
        epoch_file = None
        _update_run(run_key, result, expected_status="running")
        print(json.dumps({"run_key": run_key, **result}, indent=2))
        return 0
    except BaseException as exc:
        if epoch_file is not None and not epoch_file.closed:
            epoch_file.close()
        failure_record = {
            "status": "failed", "finished_at_utc": now_utc(),
            "failure": f"{type(exc).__name__}: {exc}", "test_inference": "not_run",
        }
        if isinstance(exc, TimeoutError) and "S5 eICU per-run wall-clock cap" in str(exc):
            failure_record.update({
                "s5_wallclock_cap_seconds": P2_RUN_WALLCLOCK_CAP_SEC,
                "s5_wallclock_cap_triggered": True,
                "duration_sec_at_s5_stop": time.perf_counter() - started,
            })
        _update_run(run_key, failure_record, expected_status="running")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)


def queue(gpus: list[int], poll_seconds: int) -> int:
    _p1_complete_gate()
    if not SELF_CHECK.is_file():
        raise RuntimeError("run `self-check` before the P2 queue")
    current_check = verify_self_check_current()
    if QUEUE_STATUS.exists():
        raise FileExistsError(f"refusing to overwrite P2 queue status: {QUEUE_STATUS}")
    lock = _manifest_lock()
    try:
        manifest = _load_training_manifest(create=True, verify_source_hashes=True)
        atomic_json(TRAIN_MANIFEST, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    status: dict[str, Any] = {
        "schema_version": "p2_loho_queue_status_v1", "started_at_utc": now_utc(),
        "status": "RUNNING", "gpus": gpus, "poll_seconds": poll_seconds,
        "s5_wallclock_cap_seconds": P2_RUN_WALLCLOCK_CAP_SEC,
        "gpu_dispatch_policy": {
            "memory_policy": "shared_idle_with_minimum_free_memory_stable_samples",
            "minimum_free_memory_mib": P2_GPU_MIN_FREE_MIB,
            "require_zero_utilization": True,
            "stable_idle_samples": P2_GPU_IDLE_SAMPLES,
            "sample_interval_seconds": P2_GPU_SAMPLE_INTERVAL_SEC,
        },
        "queue_script_sha256": sha256(Path(__file__).resolve()),
        "training_manifest_sha256_at_start": sha256(TRAIN_MANIFEST),
        "runs_started": [], "runs_finished": [], "test_inference": "not_authorized_by_this_queue",
    }
    atomic_json(QUEUE_STATUS, status)
    runner = _load_runner()
    active: dict[int, tuple[subprocess.Popen, dict[str, Any], Any, Path]] = {}
    failed = False
    failed_run_key: str | None = None
    while True:
        for gpu, (proc, job, log_file, log_path) in list(active.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            log_file.close()
            status["runs_finished"].append({"run_key": job["run_key"], "gpu": gpu, "returncode": returncode, "log": str(log_path.relative_to(ROOT)), "finished_at_utc": now_utc()})
            del active[gpu]
            if returncode != 0:
                failed = True
                failed_run_key = job["run_key"]
                status["status"] = "STOPPING_AFTER_RUN_FAILURE"
                status["failed_run_key"] = failed_run_key
                atomic_json(QUEUE_STATUS, status)
                print(f"P2 run failed: {job['run_key']} (exit {returncode}); no new runs will dispatch", flush=True)
            atomic_json(QUEUE_STATUS, status)

        if failed and not active:
            status["status"] = "STOPPED_RUN_FAILED"
            status["finished_at_utc"] = now_utc()
            status["failed_run_key"] = failed_run_key
            atomic_json(QUEUE_STATUS, status)
            return 4

        manifest = _load_training_manifest(create=False)
        run_map = {row["run_key"]: row for row in manifest["runs"]}
        if any(row.get("status") == "failed" for row in manifest["runs"]):
            status["status"] = "STOPPED_RUN_FAILED"
            atomic_json(QUEUE_STATUS, status)
            return 4
        pending = [job for job in _jobs() if run_map[job["run_key"]].get("status") == "queued"]
        for gpu in ([] if failed else gpus):
            if gpu in active or not pending:
                continue
            gpu_info = _stable_p2_gpu_status(runner, gpu)
            if not gpu_info.get("free"):
                continue
            job = pending.pop(0)
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = LOG_ROOT / f"{job['run_key'].replace('/', '_')}_gpu{gpu}.log"
            if log_path.exists():
                raise FileExistsError(f"refusing to overwrite P2 run log: {log_path}")
            log_file = log_path.open("x", encoding="utf-8")
            log_file.write(json.dumps({"run_key": job["run_key"], "gpu_dispatch": gpu_info, "dispatched_at_utc": now_utc()}) + "\n")
            log_file.flush()
            command = [PYTHON, str(Path(__file__).resolve()), "run", "--run-key", job["run_key"], "--gpu", str(gpu)]
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=log_file, stderr=subprocess.STDOUT)
            active[gpu] = (child, job, log_file, log_path)
            status["runs_started"].append({"run_key": job["run_key"], "gpu": gpu, "pid": child.pid, "dispatch_gpu_state": gpu_info, "log": str(log_path.relative_to(ROOT)), "started_at_utc": now_utc(), "command": command})
            atomic_json(QUEUE_STATUS, status)
            print(f"started {job['run_key']} on GPU {gpu}; log={log_path}", flush=True)

        if not pending and not active:
            manifest = _load_training_manifest(create=False)
            if all(row.get("status") == "completed" for row in manifest["runs"]):
                _set_training_campaign_status("ALL_90_RUNS_COMPLETE_CHECKPOINT_FREEZE_REQUIRED")
                status["status"] = "ALL_90_RUNS_COMPLETE_CHECKPOINT_FREEZE_REQUIRED"
                status["finished_at_utc"] = now_utc()
                atomic_json(QUEUE_STATUS, status)
                return 0
            # A non-queued unresolved state is not implicitly requeued.
            status["status"] = "STOPPED_UNRESOLVED_MANIFEST_STATE"
            status["finished_at_utc"] = now_utc()
            atomic_json(QUEUE_STATUS, status)
            return 5
        time.sleep(poll_seconds)


def requeue_oom(run_key: str, reason: str = "cuda_oom_before_p2_microbatch_path") -> int:
    """Archive one CUDA-OOM attempt of a split-path model and requeue it once per reason."""
    job = _parse_run_key(run_key)
    if job["model"] not in P2_SPLIT_MODELS:
        raise RuntimeError(f"OOM requeue is limited to {P2_SPLIT_MODELS}: {run_key}")
    if reason not in REQUEUE_REASONS:
        raise ValueError(f"unknown requeue reason {reason!r}")
    evidence = _microbatch_evidence()
    if reason == "validation_oom_before_split_validation":
        if (job["fold"], job["model"]) != (4, "HyperIMTS"):
            raise RuntimeError("the validation-split requeue is limited to the six fold-4 HyperIMTS runs (§21.2)")
        evidence = {**evidence, "validation_split": _validation_split_evidence()}
    lock = _manifest_lock()
    try:
        manifest = _load_training_manifest(create=False)
        record = next(item for item in manifest["runs"] if item["run_key"] == run_key)
        failure = str(record.get("failure", ""))
        if record.get("status") != "failed" or "OutOfMemoryError" not in failure:
            raise RuntimeError(f"{run_key} is not a recorded CUDA OOM failure: {record.get('status')} {failure[:120]}")
        history = list(record.get("attempt_history", []))
        if any(item.get("reason") == reason for item in history):
            raise RuntimeError(f"{run_key} was already requeued once for {reason} ({REQUEUE_REASONS[reason]})")
        if reason == "validation_oom_before_split_validation" and not record.get("p2_padded_microbatch"):
            raise RuntimeError(f"{run_key} did not run the split training path, so its OOM is not a validation gap")
        attempt = len(history) + 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = FAILED_ATTEMPTS_ROOT / f"{run_key.replace('/', '_')}_attempt{attempt:02d}_{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        moved = {}
        sources = {
            "checkpoint_directory": ROOT / record["checkpoint_directory"] if record.get("checkpoint_directory") else None,
            "epoch_log": EPOCH_ROOT / (run_key.replace("/", "_") + ".jsonl"),
        }
        sources.update({
            f"run_log_{path.name}": path
            for path in sorted(LOG_ROOT.glob(run_key.replace("/", "_") + "_gpu*.log"))
        })
        for label, source in sources.items():
            if source is not None and source.exists():
                target = archive / label
                shutil.move(str(source), str(target))
                moved[label] = {"from": str(source.relative_to(ROOT)), "to": str(target.relative_to(ROOT))}
        event = {
            "attempt_number": attempt,
            "archived_at_utc": now_utc(),
            "reason": reason,
            "rule": REQUEUE_REASONS[reason],
            "failed_attempt_snapshot": dict(record),
            "archived": moved,
            "p2_padded_microbatch": evidence,
        }
        (archive / "retry_record.json").write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stable = {key: record[key] for key in ("run_key", "fold", "model", "objective", "seed")}
        record.clear()
        record.update({**stable, "status": "queued", "attempt_history": history + [event]})
        manifest["updated_at_utc"] = now_utc()
        atomic_json(TRAIN_MANIFEST, manifest)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(json.dumps({"run_key": run_key, "status": "queued", "archive": str(archive.relative_to(ROOT)), "archived": moved}, indent=2))
    return 0


def _matches_run_process(pid: int, run_key: str, gpu: int) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        return False
    try:
        argv = [part.decode("utf-8", errors="replace") for part in command_path.read_bytes().split(b"\0") if part]
    except OSError:
        return False
    return (
        any(part.endswith("/52_train_p2_loho.py") for part in argv)
        and "run" in argv and "--run-key" in argv and run_key in argv
        and "--gpu" in argv and str(gpu) in argv
    )


def _terminal_microbatch_oom_keys(manifest: dict[str, Any]) -> list[str]:
    """Return final OOM cells that the frozen P2-M rule forbids requeueing."""
    failed = [row for row in manifest["runs"] if row.get("status") == "failed"]
    invalid = []
    for row in failed:
        evidence = row.get("p2_padded_microbatch") or {}
        failure = str(row.get("failure", ""))
        if not (
            row.get("model") in P2_SPLIT_MODELS
            and "OutOfMemoryError" in failure
            and evidence.get("status") == "PASS"
            and evidence.get("padded_size_budget") == P2_PADDED_SIZE_BUDGET
        ):
            invalid.append(row.get("run_key", "<missing run key>"))
    if invalid:
        raise RuntimeError(
            "cannot continue past failed P2 cells unless every failure is a final "
            f"micro-batch CUDA OOM under the frozen P2-M rule: {invalid}"
        )
    return sorted(row["run_key"] for row in failed)


def resume_queue(
    gpus: list[int], poll_seconds: int, *, continue_after_terminal_oom: bool = False
) -> int:
    """Resume recorded work; optionally leave verified final P2-M OOM cells failed."""
    _p1_complete_gate()
    current_check = validate_sources()
    verify_self_check_current(current_check, allow_registered_queue_code_update=True)
    if not QUEUE_STATUS.is_file():
        raise FileNotFoundError("P2 queue resume requires the existing queue status ledger")
    status = json.loads(QUEUE_STATUS.read_text(encoding="utf-8"))
    resumable = status.get("status") == "RUNNING"
    current = _load_training_manifest(create=False)
    terminal_failed_keys: list[str] = []
    if continue_after_terminal_oom:
        terminal_failed_keys = _terminal_microbatch_oom_keys(current)
    if status.get("status") in {
        "STOPPED_RUN_FAILED", "RUNNING_WITH_TERMINAL_FAILURES",
        "ALL_90_RUNS_RESOLVED_WITH_FINAL_FAILURES_NO_FREEZE",
    }:
        # A failure stop is resumable only after every failed run has been
        # resolved through a registered path such as `requeue-oom`. The P2-M
        # terminal-OOM option instead preserves those cells as failed and only
        # dispatches rows that remain queued.
        has_failed = any(row.get("status") == "failed" for row in current["runs"])
        resumable = (not has_failed) or (continue_after_terminal_oom and bool(terminal_failed_keys))
        if resumable:
            status["status"] = "RUNNING_WITH_TERMINAL_FAILURES" if terminal_failed_keys else "RUNNING"
            if terminal_failed_keys:
                status["terminal_failed_run_keys"] = terminal_failed_keys
                status["continue_after_terminal_oom"] = True
            else:
                status.pop("failed_run_key", None)
            status.pop("finished_at_utc", None)
    if not resumable:
        raise RuntimeError(f"P2 queue is not resumable from status {status.get('status')!r}")
    if sorted(set(gpus)) != sorted(gpus) or not gpus or poll_seconds < 1:
        raise ValueError("resume requires distinct GPUs and a positive poll interval")
    manifest = _load_training_manifest(create=False)
    run_map = {row["run_key"]: row for row in manifest["runs"]}
    started = status.get("runs_started", [])
    # A requeued run starts more than once; pair starts with finishes per key
    # in order, so only the latest unfinished start can still be live.
    unmatched_finishes: dict[str, int] = {}
    for item in status.get("runs_finished", []):
        unmatched_finishes[item["run_key"]] = unmatched_finishes.get(item["run_key"], 0) + 1
    active: dict[int, dict[str, Any]] = {}
    recovered_finished: list[dict[str, Any]] = []
    for item in started:
        run_key = str(item["run_key"])
        if unmatched_finishes.get(run_key, 0) > 0:
            unmatched_finishes[run_key] -= 1
            continue
        gpu, pid = int(item["gpu"]), int(item["pid"])
        row = run_map[run_key]
        state = row.get("status")
        if state in {"completed", "failed"} and not _matches_run_process(pid, run_key, gpu):
            recovered_finished.append({
                "run_key": run_key, "gpu": gpu,
                "returncode": 0 if state == "completed" else 1,
                "log": item["log"], "finished_at_utc": row.get("finished_at_utc", now_utc()),
                "recovered_from_manifest": True, "process_exit_code_observed": False,
            })
            continue
        if state != "running" or not _matches_run_process(pid, run_key, gpu):
            detail = {"run_key": run_key, "gpu": gpu, "pid": pid,
                      "manifest_status": state, "process_matches": _matches_run_process(pid, run_key, gpu)}
            status["status"] = "STOPPED_UNRESOLVED_MANIFEST_STATE"
            status["resume_error"] = detail
            status["finished_at_utc"] = now_utc()
            atomic_json(QUEUE_STATUS, status)
            raise RuntimeError(f"cannot safely attach or requeue an in-flight P2 run: {detail}")
        if gpu in active:
            raise RuntimeError(f"queue ledger has more than one active P2 run on GPU {gpu}")
        if gpu not in gpus:
            raise RuntimeError(f"recorded active run uses GPU {gpu}, absent from resume GPU set {gpus}")
        active[gpu] = {"item": item, "proc": None, "log_file": None}
    status.setdefault("runs_finished", []).extend(recovered_finished)
    recovered_failures = [
        item["run_key"] for item in recovered_finished if int(item["returncode"]) != 0
    ]
    failed_run_key: str | None = next(
        (key for key in recovered_failures if key not in terminal_failed_keys), None
    )
    if recovered_failures and not continue_after_terminal_oom:
        failed_run_key = recovered_failures[0]
    status.setdefault("resume_events", []).append({
        "resumed_at_utc": now_utc(),
        "resumed_queue_script_sha256": sha256(Path(__file__).resolve()),
        "registered_self_check_queue_script_sha256": json.loads(SELF_CHECK.read_text(encoding="utf-8")).get("queue_script_sha256"),
        "attached_runs": [{"run_key": value["item"]["run_key"], "gpu": gpu, "pid": value["item"]["pid"]}
                          for gpu, value in sorted(active.items())],
        "recovered_finished_runs": recovered_finished,
        "continue_after_terminal_oom": continue_after_terminal_oom,
        "terminal_failed_run_keys": terminal_failed_keys,
        "training_code_change_scope": "plan identity accepts only the unchanged frozen P2 body plus the separately hashed C1 resolution appendix; run settings and model code are unchanged",
    })
    atomic_json(QUEUE_STATUS, status)
    runner = _load_runner()
    while True:
        manifest = _load_training_manifest(create=False)
        run_map = {row["run_key"]: row for row in manifest["runs"]}
        for gpu, active_state in list(active.items()):
            item = active_state["item"]
            run_key, pid = str(item["run_key"]), int(item["pid"])
            row = run_map[run_key]
            proc = active_state["proc"]
            process_alive = proc.poll() is None if proc is not None else _matches_run_process(pid, run_key, gpu)
            if row.get("status") in {"completed", "failed"}:
                if process_alive:
                    continue
                returncode = (proc.returncode if proc is not None else
                              (0 if row.get("status") == "completed" else 1))
                if active_state["log_file"] is not None:
                    active_state["log_file"].close()
                status["runs_finished"].append({
                    "run_key": run_key, "gpu": gpu, "returncode": int(returncode),
                    "log": item["log"], "finished_at_utc": now_utc(),
                    "recovered_after_queue_restart": proc is None,
                    "manifest_status": row.get("status"),
                })
                del active[gpu]
                if returncode != 0:
                    if continue_after_terminal_oom:
                        current = _load_training_manifest(create=False)
                        try:
                            verified_failures = _terminal_microbatch_oom_keys(current)
                        except RuntimeError:
                            failed_run_key = run_key
                        else:
                            terminal_failed_keys = verified_failures
                            if run_key not in terminal_failed_keys:
                                failed_run_key = run_key
                            else:
                                status["terminal_failed_run_keys"] = terminal_failed_keys
                                status["continue_after_terminal_oom"] = True
                                status["status"] = "RUNNING_WITH_TERMINAL_FAILURES"
                    else:
                        failed_run_key = run_key
                        status["status"] = "STOPPING_AFTER_RUN_FAILURE"
                atomic_json(QUEUE_STATUS, status)
            elif row.get("status") == "running":
                if not process_alive:
                    status["status"] = "STOPPED_UNRESOLVED_MANIFEST_STATE"
                    status["resume_error"] = {"run_key": run_key, "gpu": gpu, "pid": pid,
                                              "manifest_status": "running", "process_alive": False}
                    status["finished_at_utc"] = now_utc()
                    atomic_json(QUEUE_STATUS, status)
                    return 5
            else:
                status["status"] = "STOPPED_UNRESOLVED_MANIFEST_STATE"
                status["resume_error"] = {"run_key": run_key, "gpu": gpu, "manifest_status": row.get("status")}
                status["finished_at_utc"] = now_utc()
                atomic_json(QUEUE_STATUS, status)
                return 5

        if failed_run_key is not None and not active:
            status["status"] = "STOPPED_RUN_FAILED"
            status["finished_at_utc"] = now_utc()
            status["failed_run_key"] = failed_run_key
            atomic_json(QUEUE_STATUS, status)
            return 4

        manifest = _load_training_manifest(create=False)
        run_map = {row["run_key"]: row for row in manifest["runs"]}
        if any(row.get("status") == "failed" for row in manifest["runs"]):
            if continue_after_terminal_oom:
                try:
                    terminal_failed_keys = _terminal_microbatch_oom_keys(manifest)
                except RuntimeError:
                    status["status"] = "STOPPING_AFTER_RUN_FAILURE"
                    atomic_json(QUEUE_STATUS, status)
                    failed_run_key = next(
                        row["run_key"] for row in manifest["runs"] if row.get("status") == "failed"
                    )
                else:
                    status["status"] = "RUNNING_WITH_TERMINAL_FAILURES"
                    status["terminal_failed_run_keys"] = terminal_failed_keys
                    status["continue_after_terminal_oom"] = True
                    atomic_json(QUEUE_STATUS, status)
            else:
                status["status"] = "STOPPING_AFTER_RUN_FAILURE"
                atomic_json(QUEUE_STATUS, status)
                failed_run_key = next(
                    row["run_key"] for row in manifest["runs"] if row.get("status") == "failed"
                )
        pending = [job for job in _jobs() if run_map[job["run_key"]].get("status") == "queued"]
        for gpu in ([] if failed_run_key is not None else gpus):
            if gpu in active or not pending:
                continue
            gpu_info = _stable_p2_gpu_status(runner, gpu)
            if not gpu_info.get("free"):
                continue
            job = pending.pop(0)
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = LOG_ROOT / f"{job['run_key'].replace('/', '_')}_gpu{gpu}.log"
            if log_path.exists():
                raise FileExistsError(f"refusing to overwrite P2 run log: {log_path}")
            log_file = log_path.open("x", encoding="utf-8")
            log_file.write(json.dumps({"run_key": job["run_key"], "gpu_dispatch": gpu_info,
                                       "dispatched_at_utc": now_utc(),
                                       "resumed_queue_controller": True}) + "\n")
            log_file.flush()
            command = [PYTHON, str(Path(__file__).resolve()), "run", "--run-key", job["run_key"],
                       "--gpu", str(gpu), "--registered-source-update"]
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                     stdout=log_file, stderr=subprocess.STDOUT)
            item = {"run_key": job["run_key"], "gpu": gpu, "pid": child.pid,
                    "dispatch_gpu_state": gpu_info, "log": str(log_path.relative_to(ROOT)),
                    "started_at_utc": now_utc(), "command": command,
                    "resumed_queue_controller": True}
            active[gpu] = {"item": item, "proc": child, "log_file": log_file}
            status["runs_started"].append(item)
            atomic_json(QUEUE_STATUS, status)
            print(f"resumed queue: started {job['run_key']} on GPU {gpu}; log={log_path}", flush=True)

        if not pending and not active:
            manifest = _load_training_manifest(create=False)
            if all(row.get("status") == "completed" for row in manifest["runs"]):
                _set_training_campaign_status("ALL_90_RUNS_COMPLETE_CHECKPOINT_FREEZE_REQUIRED")
                status["status"] = "ALL_90_RUNS_COMPLETE_CHECKPOINT_FREEZE_REQUIRED"
                status["finished_at_utc"] = now_utc()
                atomic_json(QUEUE_STATUS, status)
                return 0
            if continue_after_terminal_oom and all(
                row.get("status") in {"completed", "failed"} for row in manifest["runs"]
            ):
                terminal_failed_keys = _terminal_microbatch_oom_keys(manifest)
                _set_training_campaign_status("ALL_90_RUNS_RESOLVED_WITH_FINAL_FAILURES_NO_FREEZE")
                status["terminal_failed_run_keys"] = terminal_failed_keys
                status["status"] = "ALL_90_RUNS_RESOLVED_WITH_FINAL_FAILURES_NO_FREEZE"
                status["finished_at_utc"] = now_utc()
                atomic_json(QUEUE_STATUS, status)
                print(
                    "all planned P2 cells are resolved; terminal OOM cells remain failed, "
                    "so checkpoint freeze and test inference are not authorized",
                    flush=True,
                )
                return 0
            status["status"] = "STOPPED_UNRESOLVED_MANIFEST_STATE"
            status["finished_at_utc"] = now_utc()
            atomic_json(QUEUE_STATUS, status)
            return 5
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check", help="verify frozen plan, config and data hashes without training")
    q = sub.add_parser("queue", help="queue all 90 frozen runs on stable, shared-idle A100 cards")
    q.add_argument("--gpus", type=int, nargs="+", required=True, choices=(0, 1))
    q.add_argument("--poll-seconds", type=int, default=20)
    resume = sub.add_parser("resume", help="resume the existing P2 queue without requeueing in-flight runs")
    resume.add_argument("--gpus", type=int, nargs="+", required=True, choices=(0, 1))
    resume.add_argument("--poll-seconds", type=int, default=20)
    resume.add_argument(
        "--continue-after-terminal-oom", action="store_true",
        help="continue queued P2 cells while retaining verified P2-M OOM cells as final failures",
    )
    rq = sub.add_parser("requeue-oom", help="archive one CUDA-OOM GraFITi/HyperIMTS attempt and requeue it once")
    rq.add_argument("--run-key", required=True)
    rq.add_argument("--reason", choices=sorted(REQUEUE_REASONS), default="cuda_oom_before_p2_microbatch_path")
    r = sub.add_parser("run", help=argparse.SUPPRESS)
    r.add_argument("--run-key", required=True)
    r.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    r.add_argument("--registered-source-update", action="store_true")
    args = parser.parse_args()
    if args.command == "self-check":
        report = validate_sources()
        atomic_json(SELF_CHECK, report)
        print(json.dumps({"status": report["status"], "run_count": report["planned_run_count"], "plan_sha256": report["plan"]["sha256"], "sample_manifest_sha256": report["sample_manifest"]["sha256"], "loho_cache_sha256": report["loho_cache_sha256"], "fold_map_sha256": report["fold_map_sha256"], "reference_configs": {key: value["sha256"] for key, value in report["reference_trackb_mae_configs"].items()}, "training_started": False, "test_values_or_labels_read": False}, indent=2))
        return 0
    if args.command == "run":
        return train_one(args.run_key, args.gpu,
                         allow_registered_queue_code_update=args.registered_source_update)
    if args.command == "requeue-oom":
        return requeue_oom(args.run_key, args.reason)
    if args.command == "resume":
        return resume_queue(
            args.gpus, args.poll_seconds,
            continue_after_terminal_oom=args.continue_after_terminal_oom,
        )
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.poll_seconds < 1:
        raise ValueError("queue requires distinct GPUs and a positive poll interval")
    return queue(args.gpus, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
