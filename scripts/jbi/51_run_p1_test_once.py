#!/usr/bin/env python
"""Run the single authorized, frozen-checkpoint P1 test inference pass.

The test split is opened only after the 30-run checkpoint freeze is verified.
Every checkpoint uses batch size 128 with the adapter's sequential test order.
The artifact records each window's batch index, sample IDs and padded batch
composition, together with SOFA-component predictions at both the fixed
hourly grid and actual outcome measurement times.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(os.environ.get("ICU_KAN_PYTHON", sys.executable))
HARNESS = ROOT / "external/PyOmniTS"
RUNNER_PATH = ROOT / "scripts/jbi/42_train_p1_landmark.py"
MANIFEST_PATH = ROOT / "results/jbi/p1_landmark_training_manifest.json"
FREEZE_PATH = ROOT / "results/jbi/p1_landmark_checkpoint_freeze.json"
LEDGER_PATH = ROOT / "results/jbi/p1_landmark_test_inference_ledger.json"
LOCK_PATH = ROOT / "results/jbi/.p1_test_inference_once.lock"
BATCH_MANIFEST_PATH = ROOT / "results/jbi/p1_landmark_test_batch_manifest.json"
OUTPUT_ROOT = ROOT / "results/jbi/p1_landmark_predictions"
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
SEEDS = (2024, 2025, 2026, 2027, 2028)
COMPONENT_VARIABLES = ("Platelets", "Bilirubin", "GCS", "Creatinine")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_landmark_runner_for_test", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_runs(runner) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not MANIFEST_PATH.is_file() or not FREEZE_PATH.is_file():
        raise RuntimeError("P1 test inference requires a recorded checkpoint freeze")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_status = str(freeze.get("status", ""))
    manifest_freeze_status = str(manifest.get("checkpoint_freeze_status", ""))
    allowed_freeze_statuses = {"ALL_30_CHECKPOINTS_FROZEN", "FROZEN_WITH_NOT_COMPLETED_CELLS"}
    if (
        freeze_status not in allowed_freeze_statuses
        or freeze.get("planned_run_count") != 30
        or manifest_freeze_status != freeze_status
        or manifest.get("all_checkpoints_frozen") is not True
        or manifest.get("test_inference_authorized") is not True
        or manifest.get("test_inference_runs_allowed") != 1
    ):
        raise RuntimeError("test split is locked until all P1 cells are resolved and completed checkpoints are frozen")
    if freeze.get("analysis_plan_sha256") != runner.sha256(runner.P1_PLAN):
        raise RuntimeError("frozen analysis plan hash differs from the current P1 plan")
    expected_keys = {
        f"{cohort}/{model}/seed{seed}"
        for cohort in COHORTS for model in MODELS for seed in SEEDS
    }
    runs = manifest.get("runs", [])
    if len(runs) != 30 or {str(run.get("run_key")) for run in runs} != expected_keys:
        raise RuntimeError("training manifest does not contain exactly the fixed 30 P1 cells")
    run_by_key = {str(run["run_key"]): run for run in runs}
    completed_keys = {key for key, run in run_by_key.items() if run.get("status") == "completed"}
    not_completed_keys = {key for key, run in run_by_key.items() if run.get("status") == "not_completed"}
    if completed_keys | not_completed_keys != expected_keys or completed_keys & not_completed_keys:
        raise RuntimeError("P1 test inference requires every planned cell to be completed or final not_completed")
    frozen_not_completed = set(str(key) for key in freeze.get("not_completed_run_keys", []))
    manifest_not_completed = set(str(key) for key in manifest.get("not_completed_run_keys_frozen", []))
    if frozen_not_completed != not_completed_keys or manifest_not_completed != not_completed_keys:
        raise RuntimeError("freeze and training manifest disagree on fixed-denominator not_completed cells")
    if freeze_status == "ALL_30_CHECKPOINTS_FROZEN":
        if not_completed_keys or len(completed_keys) != 30:
            raise RuntimeError("ALL_30_CHECKPOINTS_FROZEN requires all 30 cells completed")
    else:
        eligible_s5_final_keys = {
            f"mimiciv/GraFITi/seed{seed}" for seed in (2026, 2027, 2028)
        } | {
            f"mimiciv/HyperIMTS/seed{seed}" for seed in (2027, 2028)
        }
        if (
            not not_completed_keys
            or not not_completed_keys.issubset(eligible_s5_final_keys)
            or any(
                run_by_key[key].get("final_not_completed") is not True
                or run_by_key[key].get("requeue_authorized") is not False
                or run_by_key[key].get("analysis_cell_status") != "not_passing_in_fixed_denominator"
                or float(run_by_key[key].get("s5_projection_hours", 0.0)) <= 4.0
                for key in not_completed_keys
            )
        ):
            raise RuntimeError("FROZEN_WITH_NOT_COMPLETED_CELLS lacks valid final §18.2 S5 evidence")
        evidence_path = ROOT / "results/jbi/_work/p1_s5_time_estimates_20261009.jsonl"
        if (
            not evidence_path.is_file()
            or freeze.get("not_completed_evidence_path") != str(evidence_path.relative_to(ROOT))
            or freeze.get("not_completed_evidence_sha256") != _sha256(evidence_path)
            or manifest.get("not_completed_evidence_sha256") != _sha256(evidence_path)
        ):
            raise RuntimeError("frozen not_completed evidence ledger is absent or hash-mismatched")
        evidence_records = [
            json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for key in not_completed_keys:
            records_for_key = [record for record in evidence_records if record.get("run_key") == key]
            if not records_for_key:
                raise RuntimeError(f"S5 projection ledger lacks not_completed evidence for {key}")
            final_record = records_for_key[-1]
            if (
                final_record.get("final_cell_resolution") != "not_completed_no_requeue_under_work_order_18_2_item_4"
                or float(final_record.get("projected_total_hours", 0.0)) <= 4.0
            ):
                raise RuntimeError(f"S5 projection ledger does not validate final treatment for {key}")
    if freeze.get("run_count") != len(completed_keys):
        raise RuntimeError("freeze checkpoint count differs from completed training cells")
    records = []
    for run in runs:
        key = str(run["run_key"])
        if key not in completed_keys:
            continue
        checkpoint = (ROOT / str(run.get("checkpoint_path", ""))).resolve()
        if ROOT not in checkpoint.parents or not checkpoint.is_file():
            raise RuntimeError(f"missing or out-of-project checkpoint for {key}")
        actual_hash = _sha256(checkpoint)
        if actual_hash != run.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint hash changed for {key}")
        records.append({
            "run_key": key,
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": actual_hash,
        })
    frozen_records = freeze.get("checkpoints", [])
    if records != frozen_records or runner.stable_json_sha(records) != freeze.get("checkpoint_set_sha256"):
        raise RuntimeError("frozen checkpoint list or checkpoint-set digest does not match the manifest")
    if set(freeze.get("test_inference_run_keys", [])) != completed_keys:
        raise RuntimeError("freeze authorization does not cover exactly the completed checkpoints")
    records.sort(key=lambda row: row["run_key"])
    return manifest, freeze, [dict(record, run=run_by_key[record["run_key"]]) for record in records]


def _prediction_path(run: dict[str, Any]) -> Path:
    cohort, model, seed = run["run_key"].split("/")
    return OUTPUT_ROOT / cohort / model / f"seed{seed}.npz"


def _write_npz_once(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite a P1 prediction file: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _empty_parts() -> dict[str, list[np.ndarray]]:
    return {
        "sample_id": [], "batch_index": [], "stay_id": [], "patient_id": [],
        "landmark_hour": [], "outcome_exposure_hours": [],
        "reference_raw": [], "reference_standardized": [],
        "reference_seen": [],
        "grid_sample_id": [], "grid_batch_index": [], "grid_component_index": [],
        "grid_hour": [], "grid_pred_standardized": [], "grid_pred_raw": [],
        "event_sample_id": [], "event_batch_index": [], "event_component_index": [],
        "event_hour": [], "event_true_standardized": [], "event_true_raw": [],
        "event_pred_standardized": [], "event_pred_raw": [],
    }


def _join(parts: dict[str, list[np.ndarray]], key: str, dtype, tail: tuple[int, ...] = ()) -> np.ndarray:
    if not parts[key]:
        return np.empty((0, *tail), dtype=dtype)
    return np.concatenate(parts[key], axis=0).astype(dtype, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--allow-nonzero-utilization", action="store_true")
    parser.add_argument(
        "--min-free-mib", type=int, default=None,
        help="defaults to the largest completed-run reserved peak plus the frozen 2,048 MiB margin",
    )
    args = parser.parse_args()

    runner = _load_runner()
    manifest, freeze, run_records = _frozen_runs(runner)
    if LEDGER_PATH.exists() or LOCK_PATH.exists():
        raise FileExistsError("the authorized P1 test inference pass has already started")
    if BATCH_MANIFEST_PATH.exists():
        raise FileExistsError(f"refusing to replace existing test batch manifest: {BATCH_MANIFEST_PATH}")
    existing = [str(path) for path in OUTPUT_ROOT.rglob("*.npz")] if OUTPUT_ROOT.exists() else []
    if existing:
        raise FileExistsError(f"prediction outputs already exist; refusing a second pass: {existing[:3]}")

    observed_peaks = [int(row["run"].get("cuda_peak_reserved_mib", 0)) for row in run_records]
    min_free_mib = args.min_free_mib or max(observed_peaks) + 2048
    gpu_info = runner._gpu_status(
        args.gpu,
        min_free_mib=min_free_mib,
        require_zero_utilization=not args.allow_nonzero_utilization,
    )
    if not gpu_info.get("free"):
        raise RuntimeError(f"selected GPU does not meet the frozen inference VRAM preflight: {gpu_info}")

    lock_fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(lock_fd, f"started_utc={_now()}\npid={os.getpid()}\n".encode("utf-8"))
    os.fsync(lock_fd)
    os.close(lock_fd)
    ledger: dict[str, Any] = {
        "schema_version": "p1_landmark_test_inference_ledger_v1",
        "status": "RUNNING",
        "started_at_utc": _now(),
        "python": str(PYTHON),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_plan_sha256": freeze["analysis_plan_sha256"],
        "checkpoint_set_sha256": freeze["checkpoint_set_sha256"],
        "planned_run_count": int(freeze["planned_run_count"]),
        "completed_checkpoint_count": int(freeze["run_count"]),
        "not_completed_run_keys": list(freeze.get("not_completed_run_keys", [])),
        "fixed_denominator_run_count": int(freeze["planned_run_count"]),
        "freeze_file_sha256": _sha256(FREEZE_PATH),
        "training_manifest_sha256_before_inference": _sha256(MANIFEST_PATH),
        "test_inference_passes_authorized": 1,
        "batch_size": 128,
        "test_loader_order": "sequential, no shuffle, drop_last=False",
        "test_data_accessed_after_freeze_only": True,
        "gpu": gpu_info,
        "minimum_free_memory_mib": int(min_free_mib),
        "query_outputs": ["24 valid hourly midpoints", "actual outcome SOFA-component measurement times"],
        "run_results": {},
    }
    _atomic_json(LEDGER_PATH, ledger)

    cohort_batches: dict[str, dict[str, Any]] = {}
    current_run_key: str | None = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        sys.path.insert(0, str(HARNESS))
        os.chdir(HARNESS)
        import torch
        from torch.utils.data import DataLoader
        from exp.exp_main import Exp_Main
        from data.data_provider.datasets import ICUKANLandmark as adapter

        device = torch.device("cuda:0")
        all_saved_runs = []
        for cohort in COHORTS:
            runs = [row for row in run_records if row["run"]["cohort"] == cohort]
            # A dataset and its sequential loader are built once per cohort.
            first_config, _, _ = runner._get_configs(runs[0]["run"]["model"], cohort, int(runs[0]["run"]["seed"]))
            first_exp = Exp_Main(first_config)
            # Capture the already-loaded cache's small standardizer payload
            # without unpickling a second copy of its test examples.
            original_pickle_load = adapter.pickle.load
            captured_standardizer: dict[str, Any] = {}

            def capture_cache_payload(handle, *load_args, **load_kwargs):
                payload = original_pickle_load(handle, *load_args, **load_kwargs)
                if Path(handle.name).name == f"{cohort}_events_landmark_v1.pkl":
                    captured_standardizer.update(payload["standardizer"])
                return payload

            adapter.pickle.load = capture_cache_payload
            try:
                test_data, base_loader = first_exp._get_data(flag="test")
            finally:
                adapter.pickle.load = original_pickle_load
            standardizer = captured_standardizer
            if len(test_data) == 0:
                raise RuntimeError(f"frozen test split is empty for {cohort}")
            if not standardizer or not {"median", "iqr"}.issubset(standardizer):
                raise RuntimeError(f"could not capture frozen standardizer for {cohort}")
            if int(first_config.batch_size) != 128 or base_loader.batch_size != 128:
                raise RuntimeError("P1 test inference requires the frozen batch size of 128")
            if base_loader.sampler.__class__.__name__ != "SequentialSampler" or base_loader.drop_last:
                raise RuntimeError("P1 test loader is not sequential with drop_last=False")
            variable_names = tuple(test_data.variable_names)
            component_ids = np.asarray(
                [variable_names.index(name) for name in COMPONENT_VARIABLES], dtype=np.int64
            )
            component_index_by_variable = {
                int(variable_index): component_index
                for component_index, variable_index in enumerate(component_ids.tolist())
            }
            group_key = "uniquepid" if cohort == "eicu" else "subject_id"
            row_by_sample_id = {
                int(stay_index) * 7 + int(landmark_index): (int(stay_index), float(t), int(landmark_index))
                for stay_index, t, landmark_index in test_data.landmark_rows
            }
            meta_by_sample: dict[int, dict[str, Any]] = {}

            for run_record in runs:
                run = run_record["run"]
                current_run_key = run["run_key"]
                checkpoint = ROOT / run_record["path"]
                if _sha256(checkpoint) != run_record["sha256"]:
                    raise RuntimeError(f"checkpoint changed after freeze: {current_run_key}")
                config, _, _ = runner._get_configs(run["model"], cohort, int(run["seed"]))
                exp = Exp_Main(config)
                model = exp._build_model().to(device)
                model.load_state_dict(exp._get_state_dict(checkpoint), strict=True)
                model.eval()
                torch.cuda.reset_peak_memory_stats(device)
                parts = _empty_parts()
                batch_rows: list[dict[str, Any]] = []

                with torch.inference_mode():
                    for batch_index, raw_batch in enumerate(base_loader):
                        ids = [int(value) for value in raw_batch["sample_ID"].tolist()]
                        if len(ids) > 128:
                            raise RuntimeError(f"batch {batch_index} exceeds 128 windows")
                        x_counts = (raw_batch["x_mask"] > 0).sum((1, 2)).cpu().numpy().astype(int)
                        y_counts = (raw_batch["y_mask"] > 0).sum((1, 2)).cpu().numpy().astype(int)
                        x_rows = (raw_batch["x_mask"] > 0).any(-1).sum(1).cpu().numpy().astype(int)
                        y_rows = (raw_batch["y_mask"] > 0).any(-1).sum(1).cpu().numpy().astype(int)
                        grid_rows = []
                        windows = []
                        metadata = []
                        for sample_id in ids:
                            if sample_id not in row_by_sample_id:
                                raise RuntimeError(f"test loader produced an unknown sample_ID={sample_id}")
                            stay_index, t, landmark_index = row_by_sample_id[sample_id]
                            example = test_data.examples[stay_index]
                            window = adapter.build_landmark_window(
                                example["events"], example["outtime_hour"], t
                            )
                            grid_local = adapter.hourly_grid_local_hours(example["outtime_hour"], t)
                            last_std, last_raw, seen = adapter.carry_forward_input_reference(
                                example["events"], example["outtime_hour"], t
                            )
                            if sample_id not in meta_by_sample:
                                patient_id = example.get(group_key)
                                if patient_id is None:
                                    raise KeyError(f"cache example lacks frozen bootstrap group field {group_key}")
                                meta_by_sample[sample_id] = {
                                    "sample_id": sample_id,
                                    "stay_id": int(example.get("record_id", stay_index)),
                                    "patient_id": str(patient_id),
                                    "landmark_hour": t,
                                    "outcome_exposure_hours": min(24.0, max(0.0, float(example["outtime_hour"]) - t)),
                                    "reference_raw": np.asarray(last_raw[component_ids], dtype=np.float64),
                                    "reference_standardized": np.asarray(last_std[component_ids], dtype=np.float32),
                                    "reference_seen": np.asarray(seen[component_ids], dtype=np.bool_),
                                }
                            # Preserve original raw-unit outcomes; normalized cache values may be clipped.
                            raw_events = dict(example["events"])
                            raw_events["value"] = np.asarray(example["events"]["value_raw"], dtype=np.float32)
                            _, raw_y, raw_y_mask = adapter._events_to_dense(
                                raw_events, window.selection.outcome_indices, window.outcome_local_hour
                            )
                            raw_rows = int(raw_y.shape[0])
                            if not torch.equal(raw_y_mask, raw_batch["y_mask"][len(windows), :raw_rows].cpu()):
                                raise RuntimeError(f"raw and standardized target masks differ for sample_ID={sample_id}")
                            if not torch.equal(
                                adapter._events_to_dense(
                                    example["events"], window.selection.outcome_indices,
                                    window.outcome_local_hour,
                                )[0],
                                raw_batch["y_mark"][len(windows), :raw_rows].cpu(),
                            ):
                                raise RuntimeError(f"raw and standardized target rows differ for sample_ID={sample_id}")
                            windows.append((window, raw_y))
                            grid_rows.append(int(len(grid_local)))
                            metadata.append(meta_by_sample[sample_id])

                        batch_record = {
                            "batch_index": int(batch_index),
                            "sample_IDs": ids,
                            "sample_count": len(ids),
                            "input_padded_time_rows": int(raw_batch["x"].shape[1]),
                            "actual_target_padded_time_rows": int(raw_batch["y"].shape[1]),
                            "input_measurements_by_sample": x_counts.tolist(),
                            "actual_target_measurements_by_sample": y_counts.tolist(),
                            "input_valid_time_rows_by_sample": x_rows.tolist(),
                            "actual_target_valid_time_rows_by_sample": y_rows.tolist(),
                            "hourly_grid_points_by_sample": grid_rows,
                            "hourly_component_queries_by_sample": [4 * value for value in grid_rows],
                            "hyperimts_observation_padding_width_actual_targets": int(np.max(x_counts + y_counts)),
                            "hyperimts_observation_padding_width_hourly_grid": int(np.max(x_counts + 4 * np.asarray(grid_rows))),
                        }
                        batch_rows.append(batch_record)

                        batch_device = {
                            key: value.to(device) for key, value in raw_batch.items()
                        }
                        event_input = dict(batch_device)
                        event_input["y"] = torch.zeros_like(batch_device["y"])
                        event_outputs = model(**event_input, exp_stage="test")
                        event_pred = event_outputs["pred"].detach().cpu()
                        if not torch.isfinite(event_pred[raw_batch["y_mask"] > 0]).all():
                            raise FloatingPointError(f"non-finite event-query predictions for {current_run_key}, batch {batch_index}")

                        grid_y = torch.zeros((len(ids), 24, len(variable_names)), dtype=torch.float32)
                        grid_mark = torch.zeros((len(ids), 24, 1), dtype=torch.float32)
                        grid_mask = torch.zeros_like(grid_y)
                        grid_hour_abs = np.zeros((len(ids), 24), dtype=np.float64)
                        for b, ((window, _raw_y), grid_local) in enumerate(zip(windows, [
                            adapter.hourly_grid_local_hours(
                                test_data.examples[row_by_sample_id[sample_id][0]]["outtime_hour"],
                                row_by_sample_id[sample_id][1],
                            ) for sample_id in ids
                        ])):
                            n_grid = len(grid_local)
                            if n_grid:
                                grid_mark[b, :n_grid, 0] = torch.as_tensor(grid_local / 72.0)
                                grid_mask[b, :n_grid, component_ids] = 1.0
                                t = row_by_sample_id[ids[b]][1]
                                grid_hour_abs[b, :n_grid] = t + grid_local.astype(np.float64) - 48.0
                        grid_input = {
                            key: value for key, value in batch_device.items()
                            if key not in {"y", "y_mark", "y_mask"}
                        }
                        grid_input.update({
                            "y": grid_y.to(device),
                            "y_mark": grid_mark.to(device),
                            "y_mask": grid_mask.to(device),
                        })
                        grid_outputs = model(**grid_input, exp_stage="test")
                        grid_pred = grid_outputs["pred"].detach().cpu()
                        if not torch.isfinite(grid_pred[grid_mask > 0]).all():
                            raise FloatingPointError(f"non-finite hourly-query predictions for {current_run_key}, batch {batch_index}")

                        if current_run_key in ledger["run_results"]:
                            raise RuntimeError(f"duplicate checkpoint inference attempt: {current_run_key}")

                        # Store one row for every window, and one long-form row per requested component query.
                        parts["sample_id"].append(np.asarray(ids, dtype=np.int64))
                        parts["batch_index"].append(np.full(len(ids), batch_index, dtype=np.int32))
                        parts["stay_id"].append(np.asarray([m["stay_id"] for m in metadata], dtype=np.int64))
                        parts["patient_id"].append(np.asarray([m["patient_id"] for m in metadata], dtype="U64"))
                        parts["landmark_hour"].append(np.asarray([m["landmark_hour"] for m in metadata], dtype=np.float32))
                        parts["outcome_exposure_hours"].append(np.asarray([m["outcome_exposure_hours"] for m in metadata], dtype=np.float32))
                        parts["reference_raw"].append(np.stack([m["reference_raw"] for m in metadata]))
                        parts["reference_standardized"].append(np.stack([m["reference_standardized"] for m in metadata]))
                        parts["reference_seen"].append(np.stack([m["reference_seen"] for m in metadata]))

                        event_mask = raw_batch["y_mask"][:, :, component_ids] > 0
                        eb, er, ec = torch.nonzero(event_mask, as_tuple=True)
                        if eb.numel():
                            event_var = component_ids[ec.numpy()]
                            event_true_std = raw_batch["y"][eb, er, torch.as_tensor(event_var)].numpy()
                            event_true_raw = np.concatenate([
                                windows[int(b)][1][int(row), int(var):int(var)+1].numpy()
                                for b, row, var in zip(eb.tolist(), er.tolist(), event_var.tolist())
                            ]).astype(np.float64)
                            event_pred_std = event_pred[eb, er, torch.as_tensor(event_var)].numpy()
                            event_hours = np.asarray([
                                row_by_sample_id[ids[int(b)]][1]
                                + float(raw_batch["y_mark"][b, row, 0]) * 72.0 - 48.0
                                for b, row in zip(eb.tolist(), er.tolist())
                            ], dtype=np.float64)
                            med = np.asarray(standardizer["median"], dtype=np.float64)[event_var]
                            scale = np.asarray(standardizer["iqr"], dtype=np.float64)[event_var]
                            parts["event_sample_id"].append(np.asarray([ids[int(b)] for b in eb.tolist()], dtype=np.int64))
                            parts["event_batch_index"].append(np.full(eb.numel(), batch_index, dtype=np.int32))
                            parts["event_component_index"].append(ec.numpy().astype(np.int8))
                            parts["event_hour"].append(event_hours)
                            parts["event_true_standardized"].append(event_true_std.astype(np.float32))
                            parts["event_true_raw"].append(event_true_raw)
                            parts["event_pred_standardized"].append(event_pred_std.astype(np.float32))
                            parts["event_pred_raw"].append(event_pred_std.astype(np.float64) * scale + med)

                        gb, gr, gc = torch.nonzero(grid_mask > 0, as_tuple=True)
                        if gb.numel():
                            # torch.nonzero returns indices in the full 37-variable
                            # space because grid_mask has the original model width.
                            grid_var = gc.numpy()
                            grid_pred_std = grid_pred[gb, gr, torch.as_tensor(grid_var)].numpy()
                            med = np.asarray(standardizer["median"], dtype=np.float64)[grid_var]
                            scale = np.asarray(standardizer["iqr"], dtype=np.float64)[grid_var]
                            parts["grid_sample_id"].append(np.asarray([ids[int(b)] for b in gb.tolist()], dtype=np.int64))
                            parts["grid_batch_index"].append(np.full(gb.numel(), batch_index, dtype=np.int32))
                            parts["grid_component_index"].append(np.asarray(
                                [component_index_by_variable[int(variable)] for variable in gc.tolist()],
                                dtype=np.int8,
                            ))
                            parts["grid_hour"].append(np.asarray([grid_hour_abs[int(b), int(r)] for b, r in zip(gb.tolist(), gr.tolist())], dtype=np.float64))
                            parts["grid_pred_standardized"].append(grid_pred_std.astype(np.float32))
                            parts["grid_pred_raw"].append(grid_pred_std.astype(np.float64) * scale + med)

                if cohort not in cohort_batches:
                    cohort_batches[cohort] = {
                        "batch_size": 128,
                        "shuffle": False,
                        "drop_last": False,
                        "n_test_windows": int(len(test_data)),
                        "n_batches": int(len(batch_rows)),
                        "batches": batch_rows,
                    }
                    cohort_batches[cohort]["batch_manifest_sha256"] = _stable_sha(cohort_batches[cohort])
                else:
                    candidate = {
                        "batch_size": 128,
                        "shuffle": False,
                        "drop_last": False,
                        "n_test_windows": int(len(test_data)),
                        "n_batches": int(len(batch_rows)),
                        "batches": batch_rows,
                    }
                    if _stable_sha(candidate) != cohort_batches[cohort]["batch_manifest_sha256"]:
                        raise RuntimeError(f"test batch order/composition changed for {current_run_key}")

                batch_doc = {
                    "schema_version": "p1_landmark_test_batch_manifest_v1",
                    "analysis_plan_sha256": freeze["analysis_plan_sha256"],
                    "checkpoint_set_sha256": freeze["checkpoint_set_sha256"],
                    "batch_size": 128,
                    "loader_order": "SequentialSampler; shuffle=false; drop_last=false",
                    "cohorts": cohort_batches,
                }
                _atomic_json(BATCH_MANIFEST_PATH, batch_doc)
                batch_manifest_sha = _sha256(BATCH_MANIFEST_PATH)

                run_arrays = {
                    "schema_version": np.asarray("p1_landmark_test_predictions_v1"),
                    "run_key": np.asarray(current_run_key),
                    "cohort": np.asarray(cohort),
                    "model": np.asarray(run["model"]),
                    "seed": np.asarray(run["seed"], dtype=np.int32),
                    "checkpoint_sha256": np.asarray(run_record["sha256"]),
                    "batch_manifest_sha256": np.asarray(cohort_batches[cohort]["batch_manifest_sha256"]),
                    "sample_id": _join(parts, "sample_id", np.int64),
                    "batch_index": _join(parts, "batch_index", np.int32),
                    "stay_id": _join(parts, "stay_id", np.int64),
                    "patient_id": _join(parts, "patient_id", "U64"),
                    "landmark_hour": _join(parts, "landmark_hour", np.float32),
                    "outcome_exposure_hours": _join(parts, "outcome_exposure_hours", np.float32),
                    "component_variables": np.asarray(COMPONENT_VARIABLES),
                    "component_variable_indices": component_ids,
                    "reference_raw": _join(parts, "reference_raw", np.float64, (4,)),
                    "reference_standardized": _join(parts, "reference_standardized", np.float32, (4,)),
                    "reference_seen": _join(parts, "reference_seen", np.bool_, (4,)),
                    "grid_sample_id": _join(parts, "grid_sample_id", np.int64),
                    "grid_batch_index": _join(parts, "grid_batch_index", np.int32),
                    "grid_component_index": _join(parts, "grid_component_index", np.int8),
                    "grid_hour": _join(parts, "grid_hour", np.float64),
                    "grid_pred_standardized": _join(parts, "grid_pred_standardized", np.float32),
                    "grid_pred_raw": _join(parts, "grid_pred_raw", np.float64),
                    "event_sample_id": _join(parts, "event_sample_id", np.int64),
                    "event_batch_index": _join(parts, "event_batch_index", np.int32),
                    "event_component_index": _join(parts, "event_component_index", np.int8),
                    "event_hour": _join(parts, "event_hour", np.float64),
                    "event_true_standardized": _join(parts, "event_true_standardized", np.float32),
                    "event_true_raw": _join(parts, "event_true_raw", np.float64),
                    "event_pred_standardized": _join(parts, "event_pred_standardized", np.float32),
                    "event_pred_raw": _join(parts, "event_pred_raw", np.float64),
                }
                path = _prediction_path(run_record)
                _write_npz_once(path, run_arrays)
                result = {
                    "status": "completed",
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                    "n_test_windows": int(len(test_data)),
                    "n_test_batches": int(len(batch_rows)),
                    "n_hourly_component_queries": int(run_arrays["grid_sample_id"].size),
                    "n_actual_component_measurements": int(run_arrays["event_sample_id"].size),
                    "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
                    "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) // (1024 * 1024)),
                    "batch_manifest_sha256": cohort_batches[cohort]["batch_manifest_sha256"],
                    "completed_at_utc": _now(),
                }
                ledger["run_results"][current_run_key] = result
                _atomic_json(LEDGER_PATH, ledger)
                all_saved_runs.append(current_run_key)
                del model, exp, event_outputs, grid_outputs, event_input, grid_input, batch_device
                torch.cuda.empty_cache()

        ledger["status"] = "COMPLETE"
        ledger["completed_at_utc"] = _now()
        ledger["test_batch_manifest_path"] = str(BATCH_MANIFEST_PATH.relative_to(ROOT))
        ledger["test_batch_manifest_sha256"] = _sha256(BATCH_MANIFEST_PATH)
        ledger["completed_run_count"] = len(all_saved_runs)
        ledger["test_inference_consumed"] = True
        _atomic_json(LEDGER_PATH, ledger)
        manifest["test_inference_consumed"] = True
        manifest["test_inference_ledger"] = {
            "path": str(LEDGER_PATH.relative_to(ROOT)),
            "sha256": _sha256(LEDGER_PATH),
            "completed_run_count": len(all_saved_runs),
        }
        manifest["updated_at_utc"] = _now()
        runner._atomic_json(MANIFEST_PATH, manifest)
        print(f"P1 one-pass test inference complete for {len(all_saved_runs)} frozen checkpoints")
        return 0
    except BaseException as exc:
        ledger["status"] = "FAILED"
        ledger["failed_at_utc"] = _now()
        ledger["failed_run_key"] = current_run_key
        ledger["failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(LEDGER_PATH, ledger)
        raise


if __name__ == "__main__":
    sys.exit(main())
