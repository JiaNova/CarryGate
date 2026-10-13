#!/usr/bin/env python
"""Prove masked-L1 gradient equivalence for real MIMIC-IV P1 batches.

The data loader and model setup mirror the beginning of the frozen P1
training loop. Full-batch and piecewise gradients are both evaluated on CPU,
which provides a reference when the unpartitioned dense batch cannot fit on
the available GPU. Dropout is disabled by putting the model in eval mode;
the model is still called with ``exp_stage='train'`` and no optimizer step is
performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results/jbi/p1_microbatch_gradient_equivalence.json"
POLICY_PATH = ROOT / "results/jbi/p1_microbatch_training_policy.json"
TARGET_DENSE_SAMPLE_ID = 53010
REL_TOLERANCE = 1e-5
LOSS_TOLERANCE = 1e-5


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def archive_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = ROOT / "results/jbi/_work" / f"{path.stem}_previous_{stamp}{path.suffix}"
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archived))


def _batch_metadata(batch: dict[str, torch.Tensor], dataset: Any, padded_size: int) -> dict[str, Any]:
    ids = [int(value) for value in batch["sample_ID"].reshape(-1).tolist()]
    n_vars = int(batch["x"].shape[-1])
    dense_detail = None
    if TARGET_DENSE_SAMPLE_ID in ids:
        position = ids.index(TARGET_DENSE_SAMPLE_ID)
        stay_index, t, landmark_index = dataset.landmark_rows[
            next(i for i, row in enumerate(dataset.landmark_rows)
                 if row[0] * 7 + row[2] == TARGET_DENSE_SAMPLE_ID)
        ]
        events = dataset.examples[stay_index]["events"]
        raw_hours = np.asarray(events["time_raw"], dtype=np.float64)
        input_mask = (raw_hours >= max(0.0, float(t) - 48.0)) & (raw_hours < float(t))
        distinct_times = int(np.unique(raw_hours[input_mask]).size)
        dense_detail = {
            "sample_id": TARGET_DENSE_SAMPLE_ID,
            "stay_index": int(stay_index),
            "landmark_hour": float(t),
            "landmark_index": int(landmark_index),
            "input_distinct_timestamp_count_raw_float64": distinct_times,
            "input_valid_row_count": int((batch["x_mask"][position].sum(dim=-1) > 0).sum().item()),
            "input_measurement_count": int(batch["x_mask"][position].sum().item()),
            "outcome_valid_row_count": int((batch["y_mask"][position].sum(dim=-1) > 0).sum().item()),
            "outcome_measurement_count": int(batch["y_mask"][position].sum().item()),
        }
    return {
        "sample_ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
        "sample_count": len(ids),
        "first_sample_ids": ids[:8],
        "last_sample_ids": ids[-8:],
        "measured_target_count": int(batch["y_mask"].sum().item()),
        "padded_size": int(padded_size),
        "n_variables": n_vars,
        "dense_window": dense_detail,
    }


def _loader_order_equivalence(
    train_loader: Any,
    dataset: Any,
    accelerator: Any,
    torch_module: Any,
    epochs: int = 3,
) -> dict[str, Any]:
    """Compare legacy device placement with the CPU-planning loader path.

    Iterating each prepared loader's BatchSampler avoids materializing tensors
    or transferring batches to the GPU while comparing the actual P1 sample
    IDs and batch boundaries for multiple epochs.
    """
    from data.data_provider.datasets.ICUKANLandmark import LANDMARK_HOURS

    sample_ids = [
        int(stay_index) * len(LANDMARK_HOURS) + int(landmark_index)
        for stay_index, _landmark_hour, landmark_index in dataset.landmark_rows
    ]
    shuffle = isinstance(train_loader.sampler, RandomSampler)
    saved_rng_state = torch_module.get_rng_state().clone()
    common_rng_state = saved_rng_state.clone()
    fingerprints: dict[str, list[dict[str, Any]]] = {}
    try:
        for label, device_placement in (("legacy_device_placement", True), ("cpu_before_split", False)):
            torch_module.set_rng_state(common_rng_state)
            clone = DataLoader(
                dataset,
                batch_size=train_loader.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=train_loader.drop_last,
                collate_fn=train_loader.collate_fn,
            )
            prepared = accelerator.prepare_data_loader(
                clone, device_placement=device_placement
            )
            epoch_records = []
            for epoch in range(epochs):
                batch_sampler = prepared.batch_sampler
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(epoch)
                digest = hashlib.sha256()
                sample_count = 0
                batch_count = 0
                for indices in batch_sampler:
                    ids = [sample_ids[int(index)] for index in indices]
                    digest.update((",".join(map(str, ids)) + "\n").encode("ascii"))
                    sample_count += len(ids)
                    batch_count += 1
                epoch_records.append({
                    "epoch_index": epoch,
                    "batch_count": batch_count,
                    "sample_count": sample_count,
                    "sample_and_batch_order_sha256": digest.hexdigest(),
                })
            fingerprints[label] = epoch_records
    finally:
        torch_module.set_rng_state(saved_rng_state)

    same_order = fingerprints["legacy_device_placement"] == fingerprints["cpu_before_split"]
    return {
        "status": "PASS" if same_order else "FAIL",
        "epochs_compared": int(epochs),
        "sample_count_dataset": len(dataset),
        "batch_size": train_loader.batch_size,
        "shuffle": bool(shuffle),
        "drop_last": bool(train_loader.drop_last),
        "gpu_batch_transfer_performed": False,
        "paths": fingerprints,
        "same_sample_and_batch_order": bool(same_order),
    }


def _collect_batches(model_name: str, budget: int, seed: int, torch_threads: int):
    sys.path.insert(0, str(ROOT / "scripts/jbi"))
    runner = importlib.import_module("42_train_p1_landmark")
    sys.path.insert(0, str(runner.HARNESS))
    config, reference_config_path, reference_config = runner._get_configs(
        model_name, "mimiciv", seed
    )
    if int(config.batch_size) != 128:
        raise AssertionError(f"{model_name}: gradient proof must use the frozen batch size 128")
    if (float(config.seq_len), float(config.pred_len)) != (48.0, 24.0):
        raise AssertionError(
            f"{model_name}: gradient proof must use the frozen 48/24-hour windows"
        )
    if float(config.dropout) != float(reference_config["dropout"]):
        raise AssertionError(
            f"{model_name}: P1 dropout differs from its frozen MIMIC-IV reference config"
        )
    if str(config.loss).upper() != str(reference_config["loss"]).upper():
        raise AssertionError(
            f"{model_name}: P1 loss differs from its frozen MIMIC-IV reference config"
        )
    if model_name == "HyperIMTS":
        reference_low_memory = int(reference_config.get("hyperimts_force_low_memory", 0))
        actual_low_memory = int(getattr(config, "hyperimts_force_low_memory", 0))
        if actual_low_memory != reference_low_memory:
            raise AssertionError(
                "HyperIMTS gradient proof must keep the frozen MIMIC-IV low-memory setting: "
                f"reference={reference_low_memory}, effective={actual_low_memory}"
            )
        if reference_low_memory != 1:
            raise AssertionError(
                "the frozen MIMIC-IV HyperIMTS reference config no longer selects the "
                "P1 low-memory computation path"
            )
    config.use_gpu = 0
    config.device = "cpu"
    config.num_workers = 0
    config.batch_size = 128
    torch.set_num_threads(max(1, int(torch_threads)))
    from exp.exp_main import Exp_Main

    runner._seed_everything(seed, torch)
    from p1_padded_batch_split import plan_padded_batch

    exp = Exp_Main(config)
    train_data, train_loader = exp._get_data(flag="train")
    # Match P1's construction sequence: both loaders exist before model init.
    _vali_data, _vali_loader = exp._get_data(flag="val")
    model = exp._build_model().cpu()
    from utils.globals import accelerator
    if int(accelerator.gradient_accumulation_steps) != 1:
        raise RuntimeError(
            "production run_batch_plan proof requires gradient_accumulation_steps=1; "
            f"observed {accelerator.gradient_accumulation_steps}"
        )

    loader_order_check = _loader_order_equivalence(
        train_loader, train_data, accelerator, torch
    )
    if loader_order_check["status"] != "PASS":
        raise AssertionError(f"CPU-before-split loader changed the P1 sample/batch order: {loader_order_check}")
    batch_norm_layers = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if batch_norm_layers:
        raise RuntimeError(f"gradient proof assumes no BatchNorm; found {batch_norm_layers}")
    model.eval()

    selected: dict[str, tuple[int, dict[str, torch.Tensor], dict[str, Any], Any, bool, bool]] = {}
    ordinary_candidates: list[tuple[int, dict[str, torch.Tensor], dict[str, Any], Any]] = []
    dense_candidate = None
    for batch_index, batch in enumerate(train_loader):
        plan = plan_padded_batch(batch, budget, allow_oversized_singletons=True)
        meta = _batch_metadata(batch, train_data, plan.original_padded_size)
        if meta["dense_window"] is not None:
            if meta["dense_window"]["input_distinct_timestamp_count_raw_float64"] != 1394:
                raise AssertionError(
                    "target dense sample no longer has the documented 1,394 distinct input timestamps: "
                    f"{meta['dense_window']}"
                )
            dense_positions = [
                (piece_index, positions)
                for piece_index, positions in enumerate(plan.piece_positions)
                if len(positions) == 1
                and plan.original_sample_ids[positions[0]] == TARGET_DENSE_SAMPLE_ID
            ]
            if len(dense_positions) != 1:
                raise AssertionError("the 1,394-timestamp window was not isolated into its own piece")
            dense_piece_index, _dense_piece = dense_positions[0]
            if plan.piece_padded_sizes[dense_piece_index] <= budget:
                raise AssertionError(
                    "the frozen budget must place sample 53010 above budget to prove singleton offload"
                )
            dense_candidate = (batch_index, batch, meta, plan)
        elif (
            plan.was_split
            and all(piece_size <= budget for piece_size in plan.piece_padded_sizes)
            and len(ordinary_candidates) < 2
        ):
            ordinary_plan = plan_padded_batch(batch, budget)
            ordinary_candidates.append((batch_index, batch, meta, ordinary_plan))
        if dense_candidate is not None and len(ordinary_candidates) == 2:
            break

    if dense_candidate is None:
        raise RuntimeError(f"train loader did not contain dense sample ID {TARGET_DENSE_SAMPLE_ID}")
    if len(ordinary_candidates) != 2:
        observed_sizes = [x[2]["padded_size"] for x in ordinary_candidates]
        raise RuntimeError(
            "could not collect two distinct natural training batches whose split pieces fit the frozen budget; "
            f"budget={budget}, observed={observed_sizes}"
        )

    normal_index, normal_batch, normal_meta, normal_plan = ordinary_candidates[0]
    selected["frozen_budget_no_oversized_singleton_offload"] = (
        normal_index, normal_batch, normal_meta, normal_plan, False, False
    )

    dense_index, dense_batch, dense_meta, dense_plan = dense_candidate
    selected["frozen_budget_oversized_singleton_offload"] = (
        dense_index, dense_batch, dense_meta, dense_plan, False, False
    )

    rescue_index, rescue_batch, rescue_meta, _rescue_base_plan = ordinary_candidates[1]
    rescue_plan = plan_padded_batch(
        rescue_batch, max(1, int(budget) // 4), allow_oversized_singletons=True
    )
    selected["rescue_budget_quarter_all_offload"] = (
        rescue_index, rescue_batch, rescue_meta, rescue_plan, True, True
    )
    return exp, model, selected, loader_order_check, reference_config_path, reference_config


def _loss_and_outputs(
    model,
    batch: dict[str, torch.Tensor],
    criterion,
    *,
    train_stage: int = 1,
    epoch: int = 0,
    observation_padding_width: int | None = None,
):
    model_kwargs = {}
    if observation_padding_width is not None:
        model_kwargs["p1_observation_padding_width"] = int(observation_padding_width)
    outputs = model(
        **batch,
        exp_stage="train",
        train_stage=train_stage,
        current_epoch=epoch,
        **model_kwargs,
    )
    loss_result = criterion(
        **outputs,
        exp_stage="train",
        train_stage=train_stage,
        current_epoch=epoch,
        model=model,
    )
    if not torch.equal(outputs["mask"], batch["y_mask"]):
        raise AssertionError("model output mask differs from the adapter's measured-target mask")
    return loss_result["loss"], outputs


def _grad_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    result = {}
    for name, parameter in model.named_parameters():
        result[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    return result


def _compare_one_batch(
    exp,
    model,
    optimizer,
    criterion,
    batch,
    plan,
    *,
    scenario: str,
    loader_batch_index: int,
    frozen_budget: int,
    allow_oversized_singletons: bool,
    offload_all: bool,
    rescue: bool,
):
    from p1_microbatch_training import run_batch_plan

    if not plan.was_split or len(plan.piece_positions) < 2:
        raise AssertionError(f"{scenario}: equivalence proof batch is not split into multiple pieces")
    full_targets = float(batch["y_mask"].sum().item())
    if full_targets <= 0:
        raise AssertionError("selected training batch has no measured outcome targets")

    model.zero_grad(set_to_none=True)
    full_loss, full_outputs = _loss_and_outputs(
        model,
        batch,
        criterion,
        observation_padding_width=max(plan.observation_counts),
    )
    full_loss_value = float(full_loss.detach().item())
    if not torch.isfinite(full_loss):
        raise FloatingPointError("full-batch MAE is non-finite")
    full_loss.backward()
    full_grads = _grad_snapshot(model)
    del full_outputs, full_loss
    model.zero_grad(set_to_none=True)

    production_result = run_batch_plan(
        exp,
        model,
        optimizer,
        criterion,
        batch,
        plan,
        train_stage=1,
        epoch=0,
        batch_index=loader_batch_index,
        offload_all=offload_all,
        rescue=rescue,
    )
    if production_result.get("nan_found"):
        raise FloatingPointError(f"production run_batch_plan returned NaN for {scenario}")
    if int(production_result.get("piece_target_total", -1)) != int(full_targets):
        raise AssertionError(f"{scenario}: production function changed the full-batch target count")
    split_loss_value = float(production_result["scaled_batch_loss"])
    audit = production_result["audit"]
    if scenario == "frozen_budget_no_oversized_singleton_offload":
        if (
            int(plan.padded_size_budget) != int(frozen_budget)
            or allow_oversized_singletons
            or offload_all
            or rescue
            or any(plan.piece_padded_sizes[i] > frozen_budget for i in range(len(plan.piece_positions)))
            or any(audit["piece_offloaded"])
        ):
            raise AssertionError("frozen-budget baseline did not use the plain no-offload plan")
    elif scenario == "frozen_budget_oversized_singleton_offload":
        dense_pieces = [
            i for i, sample_ids in enumerate(
                [[plan.original_sample_ids[position] for position in piece]
                 for piece in plan.piece_positions]
            )
            if sample_ids == [TARGET_DENSE_SAMPLE_ID]
        ]
        if (
            int(plan.padded_size_budget) != int(frozen_budget)
            or not allow_oversized_singletons
            or offload_all
            or rescue
            or len(dense_pieces) != 1
            or plan.piece_padded_sizes[dense_pieces[0]] <= frozen_budget
            or not audit["piece_offloaded"][dense_pieces[0]]
        ):
            raise AssertionError("frozen-budget scenario did not offload the over-budget sample 53010 singleton")
    elif scenario == "rescue_budget_quarter_all_offload":
        expected_rescue_budget = max(1, int(frozen_budget) // 4)
        if (
            int(plan.padded_size_budget) != expected_rescue_budget
            or not allow_oversized_singletons
            or not offload_all
            or not rescue
            or not all(audit["piece_offloaded"])
        ):
            raise AssertionError("rescue scenario did not use budget//4 with every piece offloaded")
    else:
        raise ValueError(f"unknown P1 microbatch proof scenario: {scenario}")

    split_grads = _grad_snapshot(model)
    per_parameter: dict[str, Any] = {}
    max_relative_gradient_difference = 0.0
    global_reference_gradient_max = 0.0
    global_gradient_difference_max = 0.0
    missing_gradient_pairs = []
    for name in full_grads:
        reference = full_grads[name]
        split = split_grads[name]
        if reference is None or split is None:
            if reference is not None or split is not None:
                difference = 1e300
                missing_gradient_pairs.append(name)
                ref_max = None if reference is None else float(reference.abs().max().item())
                diff_max = None if reference is None else float(reference.abs().max().item())
            else:
                difference = 0.0
                ref_max = 0.0
                diff_max = 0.0
            per_parameter[name] = {
                "reference_gradient_max_abs": ref_max,
                "gradient_difference_max_abs": diff_max,
                "relative_gradient_difference": difference,
                "both_gradients_none": reference is None and split is None,
            }
            max_relative_gradient_difference = max(max_relative_gradient_difference, difference)
            continue
        diff_max_tensor = (reference - split).abs().max()
        reference_max_tensor = reference.abs().max()
        ref_max = float(reference_max_tensor.item())
        diff_max = float(diff_max_tensor.item())
        global_reference_gradient_max = max(global_reference_gradient_max, ref_max)
        global_gradient_difference_max = max(global_gradient_difference_max, diff_max)
        if ref_max == 0.0:
            relative = 0.0 if diff_max == 0.0 else 1e300
        else:
            relative = diff_max / ref_max
        per_parameter[name] = {
            "reference_gradient_max_abs": ref_max,
            "gradient_difference_max_abs": diff_max,
            "relative_gradient_difference": relative,
        }
        max_relative_gradient_difference = max(max_relative_gradient_difference, relative)

    if global_reference_gradient_max == 0.0:
        global_relative_gradient_difference = (
            0.0 if global_gradient_difference_max == 0.0 else 1e300
        )
    else:
        global_relative_gradient_difference = (
            global_gradient_difference_max / global_reference_gradient_max
        )
    loss_difference = abs(full_loss_value - split_loss_value)
    model.zero_grad(set_to_none=True)
    del full_grads, split_grads
    gc.collect()
    return {
        "original_padded_size": plan.original_padded_size,
        "padded_size_budget": plan.padded_size_budget,
        "execution_scenario": scenario,
        "production_function_called": "p1_microbatch_training.run_batch_plan",
        "allow_oversized_singletons": bool(allow_oversized_singletons),
        "offload_all": bool(offload_all),
        "rescue_attempt": bool(rescue),
        "optimizer_step_performed": False,
        "run_batch_plan_audit": audit,
        "piece_positions": [list(piece) for piece in plan.piece_positions],
        "piece_sample_ids": [
            [plan.original_sample_ids[position] for position in piece]
            for piece in plan.piece_positions
        ],
        "piece_padded_sizes": list(plan.piece_padded_sizes),
        "piece_offloaded": list(audit["piece_offloaded"]),
        "dense_sample_id": plan.dense_sample_id,
        "full_target_count": int(full_targets),
        "full_batch_loss": full_loss_value,
        "split_accumulated_loss": split_loss_value,
        "absolute_loss_difference": loss_difference,
        "global_reference_gradient_max_abs": global_reference_gradient_max,
        "global_gradient_difference_max_abs": global_gradient_difference_max,
        "max_abs_gradient_relative_difference": global_relative_gradient_difference,
        "max_per_parameter_relative_gradient_difference": max_relative_gradient_difference,
        "gradient_tolerance": REL_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "parameters": per_parameter,
        "missing_gradient_pairs": missing_gradient_pairs,
        "pass": (
            global_relative_gradient_difference <= REL_TOLERANCE
            and loss_difference <= LOSS_TOLERANCE
            and not missing_gradient_pairs
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--padded-size-budget", type=int, required=True)
    parser.add_argument("--models", nargs="+", choices=("HyperIMTS", "GraFITi"), default=("HyperIMTS", "GraFITi"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    if args.padded_size_budget < 1:
        parser.error("--padded-size-budget must be positive")
    args.output = args.output.expanduser().resolve()
    if args.output == ROOT or ROOT not in args.output.parents:
        parser.error("--output must be inside the ICU-KAN project root")
    if set(args.models) != {"HyperIMTS", "GraFITi"}:
        parser.error("the frozen P1 gradient proof must cover both HyperIMTS and GraFITi")

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    allowed_memory_states = {
        "MEMORY_CALIBRATION_PASS",
        "MEMORY_CALIBRATION_READY_HYPERIMTS_NOT_RUNNABLE",
        # A failed proof must be reproducible after fixing the proof/model path.
        # Keep the existing frozen-budget and report-hash checks below as the
        # authorization boundary; the failed proof status alone must not lock
        # out a corrected rerun.
        "GRADIENT_PROOF_FAILED",
    }
    if policy.get("status") not in allowed_memory_states:
        raise RuntimeError(f"numeric budget and memory profiles are not frozen: {policy.get('status')}")
    if policy.get("budget_freeze", {}).get("status") != "FROZEN":
        raise RuntimeError("gradient proof requires an explicitly frozen memory budget")
    budget_values = policy.get("budget_values", {})
    if any(int(budget_values.get(f"mimiciv/{model}", -1)) != args.padded_size_budget
           for model in ("HyperIMTS", "GraFITi")):
        raise RuntimeError("proof budget must match both frozen target-model budgets")
    calibration = policy.get("memory_calibration", {})
    expected_calibration_status = {
        "mimiciv/HyperIMTS": {"PASS", "SINGLE_SAMPLE_OOM"},
        "mimiciv/GraFITi": {"PASS"},
    }
    for key, allowed_statuses in expected_calibration_status.items():
        entry = calibration.get(key, {})
        if entry.get("status") not in allowed_statuses:
            raise RuntimeError(f"missing acceptable memory calibration for {key}")
        if int(entry.get("padded_size_budget", -1)) != args.padded_size_budget:
            raise RuntimeError(f"memory calibration for {key} used a different budget")
        artifact = (ROOT / str(entry.get("path", ""))).resolve()
        if ROOT not in artifact.parents or not artifact.is_file() or file_sha256(artifact) != entry.get("sha256"):
            raise RuntimeError(f"memory calibration artifact for {key} is missing or has changed")

    from p1_padded_batch_split import plan_padded_batch

    report: dict[str, Any] = {
        "schema_version": "p1_microbatch_gradient_equivalence_v2",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": "mimiciv",
        "seed_for_batch_and_model_setup": int(args.seed),
        "batch_size": 128,
        "device": "cpu",
        "dropout": "disabled via model.eval(); forward called with exp_stage=train",
        "production_execution_api": "p1_microbatch_training.run_batch_plan",
        "required_scenarios": [
            "frozen_budget_no_oversized_singleton_offload",
            "frozen_budget_oversized_singleton_offload",
            "rescue_budget_quarter_all_offload",
        ],
        "padded_size_budget": int(args.padded_size_budget),
        "gradient_relative_tolerance": REL_TOLERANCE,
        "gradient_relative_metric": "max absolute elementwise gradient difference across all parameters divided by the maximum absolute elementwise full-batch gradient across all parameters",
        "loss_absolute_tolerance": LOSS_TOLERANCE,
        "splitter_sha256": file_sha256(ROOT / "scripts/jbi/p1_padded_batch_split.py"),
        "trainer_sha256": file_sha256(ROOT / "scripts/jbi/p1_microbatch_training.py"),
        "proof_script_sha256": file_sha256(Path(__file__).resolve()),
        "model_results": {},
    }
    overall_pass = True
    for model_name in args.models:
        started = time.perf_counter()
        (
            exp,
            model,
            selected,
            loader_order_check,
            reference_config_path,
            reference_config,
        ) = _collect_batches(
            model_name, args.padded_size_budget, args.seed, args.threads
        )
        criterion = exp._select_criterion()
        optimizer = exp._select_optimizer(model)
        model_id = tensor_state_sha256(model)
        model_result: dict[str, Any] = {
            "model": model_name,
            "model_source_sha256": file_sha256(
                ROOT / f"external/PyOmniTS/models/{model_name}.py"
            ),
            "reference_config_path": str(reference_config_path.resolve().relative_to(ROOT)),
            "reference_config_sha256": file_sha256(reference_config_path),
            "reference_config_dropout": float(reference_config.get("dropout")),
            "effective_config": {
                "batch_size": int(exp.configs.batch_size),
                "input_hours": float(exp.configs.seq_len),
                "outcome_hours": float(exp.configs.pred_len),
                "dropout": float(exp.configs.dropout),
                "loss": str(exp.configs.loss),
            },
            "hyperimts_force_low_memory": int(
                getattr(exp.configs, "hyperimts_force_low_memory", 0)
            ),
            "batch_norm_layers": [],
            "initial_model_state_sha256": model_id,
            "model_eval_dropout_off": not model.training,
            "gradient_accumulation_steps": int(
                __import__("utils.globals", fromlist=["accelerator"]).accelerator.gradient_accumulation_steps
            ),
            "loader_order_equivalence": loader_order_check,
            "batches": {},
        }
        for label, (batch_index, batch, metadata, plan, offload_all, rescue) in selected.items():
            print(f"[{model_name}] proving {label}, DataLoader batch={batch_index}", flush=True)
            compared = _compare_one_batch(
                exp,
                model,
                optimizer,
                criterion,
                batch,
                plan,
                scenario=label,
                loader_batch_index=batch_index,
                frozen_budget=args.padded_size_budget,
                allow_oversized_singletons=(label != "frozen_budget_no_oversized_singleton_offload"),
                offload_all=offload_all,
                rescue=rescue,
            )
            combined = {"loader_batch_index": int(batch_index), **metadata, **compared}
            model_result["batches"][label] = combined
            overall_pass = overall_pass and bool(combined["pass"])
            print(json.dumps({
                "model": model_name,
                "batch": label,
                "loss_difference": combined["absolute_loss_difference"],
                "max_abs_gradient_relative_difference": combined["max_abs_gradient_relative_difference"],
                "max_per_parameter_relative_gradient_difference_diagnostic": combined[
                    "max_per_parameter_relative_gradient_difference"
                ],
                "pass": combined["pass"],
            }), flush=True)
            del batch
            gc.collect()
        model_result["duration_sec"] = float(time.perf_counter() - started)
        report["model_results"][model_name] = model_result
        del model, exp, criterion, selected
        gc.collect()

    report["status"] = "PASS" if overall_pass else "FAIL"
    report["all_model_batch_comparisons_passed"] = bool(overall_pass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == RESULT_PATH.resolve():
        archive_existing(RESULT_PATH)
    elif args.output.exists():
        raise FileExistsError(f"refusing to overwrite gradient proof attempt: {args.output}")
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    published = args.output
    if overall_pass and args.output.resolve() != RESULT_PATH.resolve():
        archive_existing(RESULT_PATH)
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, RESULT_PATH)
        published = RESULT_PATH
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    proof_ref = {
        "status": report["status"],
        "path": str(published.resolve().relative_to(ROOT)),
        "sha256": file_sha256(published),
        "padded_size_budget": int(args.padded_size_budget),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    policy.setdefault("gradient_equivalence_history", []).append(proof_ref)
    policy["gradient_equivalence_proof"] = proof_ref
    policy["status"] = "GRADIENT_PROOF_PASS" if overall_pass else "GRADIENT_PROOF_FAILED"
    atomic_json(POLICY_PATH, policy)
    print(f"gradient equivalence: {report['status']} -> {published}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
