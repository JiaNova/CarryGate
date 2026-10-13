#!/usr/bin/env python
"""Profile the largest MIMIC-IV P1 microbatch pieces before freezing a budget.

This is a forward/backward memory calibration only: it never calls an
optimizer step and never writes a checkpoint. The chosen budget must be
supplied explicitly so its measured peak can be reviewed before the numeric
value is frozen in each run config.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import fcntl
import os
import sys
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import hashlib


ROOT = Path(__file__).resolve().parents[2]
TIMESTAMP_DENSE_SAMPLE_ID = 53010
SINGLETON_AUDIT_PATH = ROOT / "results/jbi/_work/p1_singleton_proxy_audit.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("HyperIMTS", "GraFITi"), required=True)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument("--padded-size-budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-free-mib", type=int)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="allow other GPU processes when the author has explicitly authorized sharing; retain the 25%% headroom check",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_singleton_proxy_audit(runner: Any) -> tuple[dict[str, Any], str]:
    """Require current, complete evidence for the global maximum singleton."""
    if not SINGLETON_AUDIT_PATH.is_file():
        raise FileNotFoundError(
            "run scripts/jbi/49_audit_p1_singleton_proxy.py before memory calibration"
        )
    report = json.loads(SINGLETON_AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"MIMIC-IV singleton padded-size audit did not pass: {report.get('status')}")
    expected_hashes = {
        "cache_sha256": ROOT / "data/processed/mimiciv_events_landmark_v1.pkl",
        "train_split_sha256": ROOT / "data/processed/mimiciv_split_indices_jbi_v1.json",
        "adapter_sha256": ROOT / "external/PyOmniTS/data/data_provider/datasets/ICUKANLandmark.py",
    }
    stale = {}
    for field, path in expected_hashes.items():
        current_hash = runner.sha256(path)
        if report.get(field) != current_hash:
            stale[field] = {"audited": report.get(field), "current": current_hash}
    if stale:
        raise RuntimeError(f"singleton audit is stale against P1 sources: {stale}")
    if int(report.get("sample_count", 0)) <= 0:
        raise RuntimeError("singleton audit contains no MIMIC-IV training landmark samples")
    maximum = report.get("global_max_singleton")
    if not isinstance(maximum, dict) or int(maximum.get("sample_id", -1)) < 0:
        raise RuntimeError("singleton audit does not identify its global maximum sample")
    sample_ids = [int(value) for value in report.get("profile_sample_ids", [])]
    expected_ids = list(dict.fromkeys([int(maximum["sample_id"]), TIMESTAMP_DENSE_SAMPLE_ID]))
    if sample_ids != expected_ids:
        raise RuntimeError(
            "singleton audit profile sample IDs must be its global maximum plus sample 53010: "
            f"expected={expected_ids}, observed={sample_ids}"
        )
    dense = report.get("documented_dense_timestamp_sample", {})
    if (
        int(dense.get("sample_id", -1)) != TIMESTAMP_DENSE_SAMPLE_ID
        or not dense.get("timestamp_count_matches_documented_1394")
    ):
        raise RuntimeError("singleton audit did not confirm sample 53010's 1,394 input timestamps")
    return report, runner.sha256(SINGLETON_AUDIT_PATH)


def _record_calibration(args: argparse.Namespace, report: dict[str, Any]) -> None:
    policy_path = ROOT / "results/jbi/p1_microbatch_training_policy.json"
    lock_path = policy_path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            model_key = f"mimiciv/{args.model}"
            artifact = args.output.resolve()
            record = {
                "status": report["status"],
                "padded_size_budget": int(args.padded_size_budget),
                "path": str(artifact.relative_to(ROOT)),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "allow_shared_gpu": bool(args.allow_shared_gpu),
            }
            policy.setdefault("memory_calibration_history", {}).setdefault(model_key, []).append(record)
            policy.setdefault("memory_calibration", {})[model_key] = record
            policy.setdefault("budget_values", {})[model_key] = int(args.padded_size_budget)

            hyper = policy["memory_calibration"].get("mimiciv/HyperIMTS", {})
            grafiti = policy["memory_calibration"].get("mimiciv/GraFITi", {})
            same_budget = (
                int(policy["budget_values"].get("mimiciv/HyperIMTS", -1))
                == int(policy["budget_values"].get("mimiciv/GraFITi", -2))
            )
            if same_budget and hyper.get("status") == "PASS" and grafiti.get("status") == "PASS":
                policy["status"] = "MEMORY_CALIBRATION_PASS"
                policy["budget_value"] = int(args.padded_size_budget)
            elif (
                same_budget
                and hyper.get("status") == "SINGLE_SAMPLE_OOM"
                and grafiti.get("status") == "PASS"
            ):
                policy["status"] = "MEMORY_CALIBRATION_READY_HYPERIMTS_NOT_RUNNABLE"
                policy["budget_value"] = int(args.padded_size_budget)
            elif report["status"] in {
                "PIECE_OOM", "SINGLE_SAMPLE_OOM", "SINGLE_SAMPLE_OOM_SHARED_GPU", "HEADROOM_FAILED"
            }:
                policy["status"] = "MEMORY_CALIBRATION_FAILED"
            else:
                policy["status"] = "MEMORY_CALIBRATION_IN_PROGRESS"
            _atomic_json(policy_path, policy)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> int:
    started_at = time.perf_counter()
    args = _parse_args()
    if args.padded_size_budget < 1:
        raise ValueError("--padded-size-budget must be positive")
    if args.min_free_mib is None:
        args.min_free_mib = 37000 if args.model == "HyperIMTS" else 36000
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / (
            f"results/jbi/_work/p1_microbatch_memory_{args.model}_gpu{args.gpu}_"
            f"budget{args.padded_size_budget}_{stamp}.json"
        )
    args.output = args.output.expanduser().resolve()
    if args.output == ROOT or ROOT not in args.output.parents:
        raise ValueError("--output must be inside the ICU-KAN project root")

    sys.path.insert(0, str(ROOT / "scripts/jbi"))
    runner = importlib.import_module("42_train_p1_landmark")
    singleton_audit, singleton_audit_sha256 = _load_singleton_proxy_audit(runner)
    initial_gpu = runner._gpu_status(
        args.gpu,
        min_free_mib=args.min_free_mib,
        require_zero_utilization=not args.allow_shared_gpu,
    )
    if not initial_gpu["free"]:
        raise RuntimeError(f"GPU fails the frozen memory preflight: {initial_gpu}")
    if not args.allow_shared_gpu and (
        initial_gpu.get("compute_processes")
        or int(initial_gpu.get("memory_used_mib", 0)) > 512
        or int(initial_gpu.get("utilization_percent", 0)) != 0
    ):
        raise RuntimeError(
            "memory calibration requires an exclusive idle GPU with no other compute process "
            f"and at most 512 MiB non-process use: {initial_gpu}"
        )

    # Set the device visibility before importing torch or PyOmniTS globals.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(runner.HARNESS))
    import numpy as np
    import torch

    from p1_padded_batch_split import plan_padded_batch, slice_padded_batch

    torch.set_num_threads(max(1, int(args.threads)))
    config, ref_path, ref = runner._get_configs(args.model, "mimiciv", args.seed)
    from exp.exp_main import Exp_Main

    runner._seed_everything(args.seed, torch)
    config.use_gpu = 1
    config.gpu_id = 0
    config.device = "cuda"
    config.num_workers = 0
    config.batch_size = 128
    config.checkpoints = str(ROOT / "results/jbi/_work/p1_microbatch_profile_no_checkpoint")

    exp = Exp_Main(config)
    train_data, train_loader = exp._get_data(flag="train")
    _vali_data, _vali_loader = exp._get_data(flag="val")
    model = exp._build_model().to(exp.device)
    batch_norm_layers = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if batch_norm_layers:
        raise RuntimeError(f"microbatch memory profile assumes no BatchNorm; found {batch_norm_layers}")
    criterion = exp._select_criterion()
    optimizer = exp._select_optimizer(model)
    optimizer_state_bytes = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state[parameter]
        # Match Adam's persistent first/second-moment allocation after its
        # first real update, without changing any model parameter or step.
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)
        optimizer_state_bytes += 2 * parameter.numel() * parameter.element_size()
    model.train()  # match activation paths used by the actual training run

    # Use the data-audited global maximum singleton, plus the separately
    # documented 1,394-timestamp sample when it is not the same example.
    sample_lookup = {
        int(stay_index) * 7 + int(landmark_index): index
        for index, (stay_index, _t, landmark_index) in enumerate(train_data.landmark_rows)
    }
    from data.data_provider.datasets.ICUKANLandmark import collate_fn
    singleton_profile_ids = [int(value) for value in singleton_audit["profile_sample_ids"]]
    missing_profile_ids = [value for value in singleton_profile_ids if value not in sample_lookup]
    if missing_profile_ids:
        raise RuntimeError(
            "audited singleton profile samples are absent from this MIMIC-IV train dataset: "
            f"{missing_profile_ids}"
        )

    # The production splitter fixes HyperIMTS's observation padding width to
    # the original DataLoader batch. Find that width for each audited sample;
    # the sample-only calibration below must use the same forward shape.
    loader_rng_state_before_singleton_scan = torch.get_rng_state().clone()
    singleton_observation_padding_widths: dict[int, int] = {}
    for batch in train_loader:
        ids = [int(value) for value in batch["sample_ID"].reshape(-1).tolist()]
        matching = set(ids) & set(singleton_profile_ids)
        if matching:
            plan = plan_padded_batch(
                batch, args.padded_size_budget, allow_oversized_singletons=True
            )
            width = max(plan.observation_counts)
            singleton_observation_padding_widths.update({sample_id: width for sample_id in matching})
        if set(singleton_observation_padding_widths) == set(singleton_profile_ids):
            break
    torch.set_rng_state(loader_rng_state_before_singleton_scan)
    if set(singleton_observation_padding_widths) != set(singleton_profile_ids):
        raise RuntimeError(
            "could not identify the original DataLoader batches for all calibrated singleton samples"
        )

    def profile_piece_batch(
        batch: dict[str, torch.Tensor],
        name: str,
        *,
        offload_saved_tensors: bool,
        observation_padding_width: int,
    ) -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        before = torch.cuda.memory_allocated(0)
        targets = float(batch["y_mask"].sum().item())
        if targets <= 0:
            raise RuntimeError(f"{name} has no real measured targets")
        batch_device = outputs = loss_result = None
        try:
            context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if offload_saved_tensors else contextlib.nullcontext()
            )
            with context:
                batch_device = {key: value.to(exp.device) for key, value in batch.items()}
                outputs = model(
                    **batch_device,
                    exp_stage="train",
                    p1_observation_padding_width=int(observation_padding_width),
                )
                loss_result = criterion(**outputs, exp_stage="train", model=model)
                if not torch.equal(outputs["mask"], batch_device["y_mask"]):
                    raise AssertionError("model output mask differs from adapter target mask")
                loss_result["loss"].backward()
                torch.cuda.synchronize(0)
            peak_allocated = int(torch.cuda.max_memory_allocated(0))
            peak_reserved = int(torch.cuda.max_memory_reserved(0))
            return {
                "label": name,
                "offload_saved_tensors_to_pinned_cpu": bool(offload_saved_tensors),
                "sample_ids": [int(value) for value in batch["sample_ID"].reshape(-1).tolist()],
                "sample_count": int(batch["sample_ID"].shape[0]),
                "measured_target_count": int(targets),
                "p1_observation_padding_width": int(observation_padding_width),
                "loss": float(loss_result["loss"].detach().item()),
                "memory_before_allocated_mib": before // (1024 * 1024),
                "peak_allocated_mib": peak_allocated // (1024 * 1024),
                "peak_reserved_mib": peak_reserved // (1024 * 1024),
                "fits_cuda": True,
            }
        except torch.cuda.OutOfMemoryError as exc:
            try:
                torch.cuda.synchronize(0)
            except Exception:
                pass
            return {
                "label": name,
                "offload_saved_tensors_to_pinned_cpu": bool(offload_saved_tensors),
                "sample_ids": [int(value) for value in batch["sample_ID"].reshape(-1).tolist()],
                "sample_count": int(batch["sample_ID"].shape[0]),
                "measured_target_count": int(targets),
                "p1_observation_padding_width": int(observation_padding_width),
                "memory_before_allocated_mib": before // (1024 * 1024),
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated(0)) // (1024 * 1024),
                "peak_reserved_mib": int(torch.cuda.max_memory_reserved(0)) // (1024 * 1024),
                "fits_cuda": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            model.zero_grad(set_to_none=True)
            batch_device = outputs = loss_result = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    sample_profiles = []
    global_max_id = int(singleton_audit["global_max_singleton"]["sample_id"])
    for sample_id in singleton_profile_ids:
        if sample_id not in sample_lookup:
            raise RuntimeError(f"training split does not contain required sample ID {sample_id}")
        sample = train_data[sample_lookup[sample_id]]
        one = collate_fn([sample])
        label = "global_max_proxy_singleton" if sample_id == global_max_id else "1394_timestamp_singleton"
        stay_index, t, landmark_index = train_data.landmark_rows[sample_lookup[sample_id]]
        events = train_data.examples[stay_index]["events"]
        raw_hours = np.asarray(events["time_raw"], dtype=np.float64)
        input_event_mask = (raw_hours >= max(0.0, float(t) - 48.0)) & (raw_hours < float(t))
        sample_detail = {
            "proxy_audit_rank": 1 if sample_id == global_max_id else None,
            "stay_index": int(stay_index),
            "landmark_hour": float(t),
            "landmark_index": int(landmark_index),
            "input_distinct_timestamp_count_raw_float64": int(np.unique(raw_hours[input_event_mask]).size),
            "input_event_count": int(input_event_mask.sum()),
            "input_valid_row_count": int((one["x_mask"].sum(dim=-1) > 0).sum().item()),
            "outcome_event_count": int(one["y_mask"].sum().item()),
            "outcome_valid_row_count": int((one["y_mask"].sum(dim=-1) > 0).sum().item()),
        }
        for offload_saved_tensors in (False, True):
            mode = "offload" if offload_saved_tensors else "no_offload"
            profile = profile_piece_batch(
                one,
                f"{label}_{sample_id}_{mode}",
                offload_saved_tensors=offload_saved_tensors,
                observation_padding_width=singleton_observation_padding_widths[sample_id],
            )
            profile["sample_detail"] = sample_detail
            sample_profiles.append(profile)

    largest_batch = None
    dense_timestamp_batch = None
    batch_proxy_sizes = []
    loader_rng_state_before_scan = torch.get_rng_state().clone()
    for batch_index, batch in enumerate(train_loader):
        plan = plan_padded_batch(
            batch, args.padded_size_budget, allow_oversized_singletons=True
        )
        batch_proxy_sizes.append((
            int(plan.original_padded_size),
            int(batch_index),
            hashlib.sha256(",".join(map(str, plan.original_sample_ids)).encode()).hexdigest(),
        ))
        if largest_batch is None or plan.original_padded_size > largest_batch[1].original_padded_size:
            largest_batch = (batch, plan, batch_index)
        if TIMESTAMP_DENSE_SAMPLE_ID in plan.original_sample_ids:
            dense_timestamp_batch = (batch, plan, batch_index)
        if batch_index % 50 == 0:
            print(f"scanned train batch {batch_index}/{len(train_loader)}", flush=True)

    loader_rng_state_after_scan = torch.get_rng_state().clone()
    if largest_batch is None or dense_timestamp_batch is None or not batch_proxy_sizes:
        raise RuntimeError("could not identify the largest batch and 1,394-timestamp sample batch")

    # Select the median-proxy batch as a representative typical training batch.
    typical_proxy_size, typical_batch_index, typical_batch_sample_ids_sha256 = sorted(
        batch_proxy_sizes, key=lambda item: (item[0], item[1])
    )[
        len(batch_proxy_sizes) // 2
    ]
    typical_batch = None
    torch.set_rng_state(loader_rng_state_before_scan)
    try:
        for batch_index, batch in enumerate(train_loader):
            if batch_index == typical_batch_index:
                plan = plan_padded_batch(
                    batch, args.padded_size_budget, allow_oversized_singletons=True
                )
                actual_sample_ids_sha256 = hashlib.sha256(
                    ",".join(map(str, plan.original_sample_ids)).encode()
                ).hexdigest()
                if (
                    int(plan.original_padded_size) != typical_proxy_size
                    or actual_sample_ids_sha256 != typical_batch_sample_ids_sha256
                ):
                    raise RuntimeError(
                        "reloaded median-proxy DataLoader batch differs from the calibration scan"
                    )
                typical_batch = (batch, plan, batch_index)
                break
    finally:
        torch.set_rng_state(loader_rng_state_after_scan)
    if typical_batch is None:
        raise RuntimeError(f"could not reload median-proxy training batch {typical_batch_index}")

    def run_microbatch_profile(entry, label):
        batch, plan, batch_index = entry
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        full_targets = float(batch["y_mask"].sum().item())
        piece_results = []
        try:
            for piece_index, positions in enumerate(plan.piece_positions):
                piece = slice_padded_batch(batch, positions)
                piece_offloaded = int(plan.piece_padded_sizes[piece_index]) > int(args.padded_size_budget)
                context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if piece_offloaded else contextlib.nullcontext()
                )
                with context:
                    piece_device = {key: value.to(exp.device) for key, value in piece.items()}
                    outputs = model(
                        **piece_device,
                        exp_stage="train",
                        p1_observation_padding_width=max(plan.observation_counts),
                    )
                    loss = criterion(**outputs, exp_stage="train", model=model)["loss"]
                    piece_targets = float(piece_device["y_mask"].sum().item())
                    if not torch.equal(outputs["mask"], piece_device["y_mask"]):
                        raise AssertionError("piece output mask differs from adapter target mask")
                    (loss * (piece_targets / full_targets)).backward()
                piece_results.append({
                    "piece_index": piece_index,
                    "sample_ids": [int(value) for value in piece["sample_ID"].tolist()],
                    "padded_size": int(plan.piece_padded_sizes[piece_index]),
                    "measured_target_count": int(piece_targets),
                    "loss": float(loss.detach().item()),
                    "offload_saved_tensors_to_pinned_cpu": bool(piece_offloaded),
                })
                del piece, piece_device, outputs, loss
            torch.cuda.synchronize(0)
            return {
                "label": label,
                "loader_batch_index": int(batch_index),
                "sample_count": int(batch["sample_ID"].shape[0]),
                "sample_ids_sha256": __import__("hashlib").sha256(
                    ",".join(map(str, plan.original_sample_ids)).encode()
                ).hexdigest(),
                "original_padded_size": int(plan.original_padded_size),
                "padded_size_budget": int(plan.padded_size_budget),
                "piece_positions": [list(x) for x in plan.piece_positions],
                "piece_padded_sizes": list(plan.piece_padded_sizes),
                "pieces": piece_results,
                "full_target_count": int(full_targets),
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated(0)) // (1024 * 1024),
                "peak_reserved_mib": int(torch.cuda.max_memory_reserved(0)) // (1024 * 1024),
                "fits_cuda": True,
            }
        except torch.cuda.OutOfMemoryError as exc:
            try:
                torch.cuda.synchronize(0)
            except Exception:
                pass
            return {
                "label": label,
                "loader_batch_index": int(batch_index),
                "sample_count": int(batch["sample_ID"].shape[0]),
                "original_padded_size": int(plan.original_padded_size),
                "padded_size_budget": int(plan.padded_size_budget),
                "piece_padded_sizes": list(plan.piece_padded_sizes),
                "peak_allocated_mib": int(torch.cuda.max_memory_allocated(0)) // (1024 * 1024),
                "peak_reserved_mib": int(torch.cuda.max_memory_reserved(0)) // (1024 * 1024),
                "fits_cuda": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    profile_entries = [
        (typical_batch, "median_proxy_typical_training_batch"),
        (largest_batch, "largest_full_batch_under_seed2024_loader"),
        (dense_timestamp_batch, "batch_containing_1394_timestamp_window"),
    ]
    microbatch_profiles = []
    for entry, label in profile_entries:
        profile = run_microbatch_profile(entry, label)
        if label == "median_proxy_typical_training_batch":
            profile["typical_batch_selection"] = {
                "method": "median original_padded_size across the seed-2024 training DataLoader",
                "rank_ascending": int(len(batch_proxy_sizes) // 2 + 1),
                "batch_count": int(len(batch_proxy_sizes)),
            }
        microbatch_profiles.append(profile)
    final_gpu = runner._gpu_status(
        args.gpu,
        allow_current_process=True,
        min_free_mib=args.min_free_mib,
        require_zero_utilization=not args.allow_shared_gpu,
    )
    report = {
        "schema_version": "p1_microbatch_memory_calibration_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": "mimiciv",
        "model": args.model,
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "allow_shared_gpu": bool(args.allow_shared_gpu),
        "memory_preflight": initial_gpu,
        "memory_post_profile": final_gpu,
        "padded_size_budget_tested": int(args.padded_size_budget),
        "batch_size": 128,
        "training_optimizer_step_performed": False,
        "adam_moment_buffers_preallocated_without_step": True,
        "adam_moment_state_bytes": int(optimizer_state_bytes),
        "configuration_reference": str(ref_path.relative_to(ROOT)),
        "configuration_reference_sha256": runner.sha256(ref_path),
        "singleton_proxy_audit": {
            "path": str(SINGLETON_AUDIT_PATH.relative_to(ROOT)),
            "sha256": singleton_audit_sha256,
            "status": singleton_audit["status"],
            "sample_count": int(singleton_audit["sample_count"]),
            "global_max_singleton": singleton_audit["global_max_singleton"],
            "documented_dense_timestamp_sample": singleton_audit["documented_dense_timestamp_sample"],
            "profile_sample_ids": singleton_profile_ids,
        },
        "profile_script_sha256": runner.sha256(Path(__file__).resolve()),
        "model_source_sha256": runner.sha256(ROOT / f"external/PyOmniTS/models/{args.model}.py"),
        "splitter_sha256": runner.sha256(ROOT / "scripts/jbi/p1_padded_batch_split.py"),
        "single_sample_profiles": sample_profiles,
        "microbatch_profiles": microbatch_profiles,
        "status": "IN_PROGRESS",
        "memory_headroom_check": {},
        "duration_sec": time.perf_counter() - started_at,
    }
    offloaded_single_profiles = [
        item for item in sample_profiles
        if item["offload_saved_tensors_to_pinned_cpu"]
    ]
    planned_profiles = offloaded_single_profiles + microbatch_profiles
    successful_peaks = [
        int(item["peak_reserved_mib"])
        for item in planned_profiles if item.get("fits_cuda") and item.get("peak_reserved_mib") is not None
    ]
    initial_free_mib = int(initial_gpu["memory_free_mib"])
    required_headroom_mib = int(np.ceil(initial_free_mib * 0.25))
    measured_peak_reserved_mib = max(successful_peaks, default=0)
    measured_headroom_mib = initial_free_mib - measured_peak_reserved_mib
    headroom_pass = measured_headroom_mib >= required_headroom_mib
    report["memory_headroom_check"] = {
        "status": "PASS" if headroom_pass else "FAIL",
        "minimum_fraction_of_preflight_free_memory": 0.25,
        "preflight_free_memory_mib": initial_free_mib,
        "required_headroom_mib": required_headroom_mib,
        "maximum_successful_planned_path_peak_reserved_mib": measured_peak_reserved_mib,
        "remaining_headroom_mib": measured_headroom_mib,
        "measured_paths": "offloaded singletons and budgeted typical/largest/sample-53010 batches; unoffloaded singleton is diagnostic only",
    }
    offloaded_samples_fit = (
        {int(item["sample_ids"][0]) for item in offloaded_single_profiles if item["fits_cuda"]}
        >= set(singleton_profile_ids)
    )
    no_offload_samples_present = {
        int(item["sample_ids"][0]) for item in sample_profiles
        if not item["offload_saved_tensors_to_pinned_cpu"]
    } >= set(singleton_profile_ids)
    microbatch_pass = bool(microbatch_profiles) and all(
        item.get("fits_cuda") for item in microbatch_profiles
    )
    report["status"] = (
        "SINGLE_SAMPLE_OOM_SHARED_GPU" if not offloaded_samples_fit and args.allow_shared_gpu
        else "SINGLE_SAMPLE_OOM" if not offloaded_samples_fit
        else "HEADROOM_FAILED" if not headroom_pass
        else "PASS" if microbatch_pass and no_offload_samples_present
        else "PIECE_OOM"
    )
    report["single_sample_comparison_complete"] = bool(
        offloaded_samples_fit and no_offload_samples_present
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite memory-calibration evidence: {args.output}")
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_output.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temp_output.replace(args.output)
    _record_calibration(args, report)
    if report["status"] == "SINGLE_SAMPLE_OOM" and args.model == "HyperIMTS":
        from p1_microbatch_training import SingleSampleCudaOutOfMemory
        failed_sample = next(profile for profile in sample_profiles if not profile["fits_cuda"])
        evidence = {
            "model_id": config.model_id,
            "sample_id": failed_sample.get("sample_ids", [None])[0],
            "profile_artifact": str(args.output),
            "sample_profile": failed_sample,
            "memory_preflight": initial_gpu,
            "padded_size_budget_tested": int(args.padded_size_budget),
            "test_inference": "not_run",
        }
        exc = SingleSampleCudaOutOfMemory(
            "singleton forward/backward OOM during pre-training memory calibration",
            evidence,
        )
        runner._mark_mimiciv_hyperimts_not_runnable(args.seed, exc)
    print(json.dumps({
        "model": args.model,
        "status": report["status"],
        "single_sample_peak_mib": [x.get("peak_allocated_mib") for x in sample_profiles],
        "piece_profile_peaks_mib": [x.get("peak_allocated_mib") for x in microbatch_profiles],
        "output": str(args.output),
    }, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
