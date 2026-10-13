#!/usr/bin/env python
"""Gate for the P2 exact padded micro-batch path (work order §20.2).

``proof`` (CPU, dropout off): in folds 0 and 2, take the training batch with
the largest padded size and compare full-batch gradients with the production
``run_batch_plan`` at the P2 budget and at the rescue budget (budget//4, all
pieces offloaded), for GraFITi and HyperIMTS under MAE and MSE.

``calibrate`` (one idle GPU): peak VRAM of the split training step on that
batch and of an unsplit no-grad forward on the densest validation batch.

``proof`` writes the gate file; ``calibrate`` adds its section, and the file
is PASS only when both sections pass. Train/validation indices only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/jbi"))

_spec = importlib.util.spec_from_file_location("p2_loho_trainer", ROOT / "scripts/jbi/52_train_p2_loho.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)

FOLDS = (0, 2)
TOLERANCE = 1e-5
CALIBRATION_LIMIT_MIB = 30 * 1024
SEED = 2024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiment(model: str, objective: str, fold: int, device: str):
    import torch

    config, _ref_path, _ref = P2._configuration(model, objective, fold, SEED)
    if device == "cpu":
        config.use_gpu = 0
    config.device = device
    config.gpu_id = 0
    config.num_workers = 0
    P2._install_train_only_loho_loader(fold)
    sys.path.insert(0, str(P2.HARNESS))
    os.chdir(P2.HARNESS)
    from exp.exp_main import Exp_Main

    P2._load_runner()._seed_everything(SEED, torch)
    exp = Exp_Main(config)
    return exp


def _densest(loader) -> tuple[int, dict[str, Any], int]:
    from p1_padded_batch_split import plan_padded_batch

    best = None
    for index, batch in enumerate(loader):
        size = plan_padded_batch(batch, budget=10**18).original_padded_size
        if best is None or size > best[2]:
            best = (index, batch, size)
    return best


def proof(threads: int) -> dict[str, Any]:
    import torch
    from p1_padded_batch_split import plan_padded_batch
    from p1_microbatch_training import run_batch_plan

    p1_proof = importlib.import_module("47_verify_p1_microbatch_gradients")
    torch.set_num_threads(threads)
    results = []
    for fold in FOLDS:
        for model_name in P2.P2_SPLIT_MODELS:
            for objective in P2.OBJECTIVES:
                exp = _experiment(model_name, objective, fold, "cpu")
                _train_data, train_loader = exp._get_data(flag="train")
                batch_index, batch, padded_size = _densest(train_loader)
                model = exp._build_model().cpu()
                model.eval()  # dropout off; forward still uses exp_stage="train"
                optimizer = exp._select_optimizer(model)
                criterion = exp._select_criterion()
                base = plan_padded_batch(batch, P2.P2_PADDED_SIZE_BUDGET, allow_oversized_singletons=True)

                model.zero_grad(set_to_none=True)
                full_loss, _ = p1_proof._loss_and_outputs(
                    model, batch, criterion, observation_padding_width=max(base.observation_counts),
                )
                full_loss.backward()
                full = p1_proof._grad_snapshot(model)
                reference_max = max(float(g.abs().max()) for g in full.values() if g is not None)

                for scenario, budget, offload_all in (
                    ("p2_budget", P2.P2_PADDED_SIZE_BUDGET, False),
                    ("rescue_quarter_budget_all_offload", P2.P2_PADDED_SIZE_BUDGET // 4, True),
                ):
                    plan = plan_padded_batch(batch, budget, allow_oversized_singletons=True)
                    model.zero_grad(set_to_none=True)
                    result = run_batch_plan(
                        exp, model, optimizer, criterion, batch, plan,
                        train_stage=1, epoch=0, batch_index=batch_index + 1,
                        offload_all=offload_all, rescue=offload_all,
                    )
                    split = p1_proof._grad_snapshot(model)
                    missing = [name for name in full if (full[name] is None) != (split[name] is None)]
                    diff = max(
                        (float((full[name] - split[name]).abs().max()) for name in full
                         if full[name] is not None and split[name] is not None),
                        default=0.0,
                    )
                    relative = diff / reference_max if reference_max > 0 else float("inf")
                    loss_relative = abs(result["scaled_batch_loss"] - float(full_loss)) / max(abs(float(full_loss)), 1e-12)
                    passed = (
                        plan.was_split and len(plan.piece_positions) >= 2 and not missing
                        and not result["nan_found"] and relative <= TOLERANCE and loss_relative <= TOLERANCE
                        and int(result["piece_target_total"]) == int(batch["y_mask"].sum())
                    )
                    row = {
                        "fold": fold, "model": model_name, "objective": objective, "scenario": scenario,
                        "loader_batch_index": batch_index, "original_padded_size": padded_size,
                        "budget": budget, "piece_count": len(plan.piece_positions),
                        "max_abs_gradient_difference_over_max_full_gradient": relative,
                        "relative_loss_difference": loss_relative,
                        "missing_gradient_parameters": missing, "pass": bool(passed),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
    return {
        "device": "cpu",
        "dropout": "disabled via model.eval(); forward called with exp_stage=train",
        "tolerance": TOLERANCE,
        "comparisons": results,
        "status": "PASS" if results and all(row["pass"] for row in results) else "FAIL",
    }


def calibrate(gpu: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from p1_padded_batch_split import plan_padded_batch
    from p1_microbatch_training import run_batch_plan

    runner = P2._load_runner()
    gpu_state = runner._gpu_status(gpu, min_free_mib=P2.P2_GPU_MIN_FREE_MIB, require_zero_utilization=True)
    if not gpu_state.get("free"):
        raise RuntimeError(f"calibration needs an idle GPU with {P2.P2_GPU_MIN_FREE_MIB} MiB free: {gpu_state}")
    rows = []
    for fold in FOLDS:
        for model_name in P2.P2_SPLIT_MODELS:
            exp = _experiment(model_name, "MAE", fold, "cuda")
            _t, train_loader = exp._get_data(flag="train")
            _v, vali_loader = exp._get_data(flag="val")
            train_index, train_batch, train_size = _densest(train_loader)
            vali_index, vali_batch, vali_size = _densest(vali_loader)
            model = exp._build_model().to(exp.device)
            optimizer = exp._select_optimizer(model)
            criterion = exp._select_criterion()

            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            plan = plan_padded_batch(train_batch, P2.P2_PADDED_SIZE_BUDGET, allow_oversized_singletons=True)
            run_batch_plan(exp, model, optimizer, criterion, train_batch, plan,
                           train_stage=1, epoch=0, batch_index=train_index + 1,
                           offload_all=False, rescue=False)
            train_peak = int(torch.cuda.max_memory_reserved() // (1024 * 1024))

            model.eval()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model(**{key: value.to(exp.device) for key, value in vali_batch.items()},
                      exp_stage="val", train_stage=1, current_epoch=0)
            vali_peak = int(torch.cuda.max_memory_reserved() // (1024 * 1024))
            row = {
                "fold": fold, "model": model_name,
                "train_batch_index": train_index, "train_original_padded_size": train_size,
                "train_piece_count": len(plan.piece_positions), "train_split_step_peak_reserved_mib": train_peak,
                "validation_batch_index": vali_index, "validation_padded_size": vali_size,
                "validation_unsplit_no_grad_peak_reserved_mib": vali_peak,
                "pass": train_peak < CALIBRATION_LIMIT_MIB and vali_peak < CALIBRATION_LIMIT_MIB,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, optimizer, exp
            torch.cuda.empty_cache()
    return {
        "gpu_index": gpu, "gpu_state_before": gpu_state, "limit_mib": CALIBRATION_LIMIT_MIB,
        "allocator": "expandable_segments:True", "rows": rows,
        "status": "PASS" if rows and all(row["pass"] for row in rows) else "FAIL",
    }


VALIDATION_FOLDS = (4, 2)
VALIDATION_SEQUENCE_LIMIT_MIB = 34 * 1024


def _validation_identity() -> dict[str, Any]:
    import inspect

    return {
        "padded_size_budget": P2.P2_PADDED_SIZE_BUDGET,
        "wrapper_source_sha256": hashlib.sha256(
            inspect.getsource(P2.split_validation_model).encode("utf-8")
        ).hexdigest(),
        "splitter_sha256": _sha256(P2.PADDED_BATCH_SPLITTER),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }


def _validation_metrics(exp, criterion, outputs, batch) -> dict[str, Any]:
    error, baseline, counts = exp._validation_persistence_counts(outputs["pred"], batch)
    mask = batch["y_mask"].to(outputs["pred"].dtype)
    loss = criterion(**outputs, exp_stage="val", train_stage=1, current_epoch=0, model=None)["loss"]
    return {
        "loss": float(loss), "model_error": error, "baseline_error": baseline, "counts": counts,
        "pooled_squared_error": float((((outputs["pred"] - batch["y"]) ** 2) * mask).sum()),
        "masked_pred": outputs["pred"] * mask,
    }


def validation_proof(threads: int) -> dict[str, Any]:
    import torch
    from p1_padded_batch_split import plan_padded_batch

    torch.set_num_threads(threads)
    rows = []
    for fold in VALIDATION_FOLDS:
        for model_name in P2.P2_SPLIT_MODELS:
            exp = _experiment(model_name, "MAE", fold, "cpu")
            exp._get_data(flag="train")  # sets the padding lengths the validation loader uses
            _v, vali_loader = exp._get_data(flag="val")
            batch_index, batch, padded_size = _densest(vali_loader)
            plan = plan_padded_batch(batch, P2.P2_PADDED_SIZE_BUDGET, allow_oversized_singletons=True)
            model = exp._build_model().cpu()
            model.eval()
            criterion = exp._select_criterion()
            wrapped = P2.split_validation_model(model, P2.P2_PADDED_SIZE_BUDGET)
            with torch.no_grad():
                options = {"exp_stage": "val", "train_stage": 1, "current_epoch": 0}
                full = _validation_metrics(exp, criterion, model(**batch, **options), batch)
                split = _validation_metrics(exp, criterion, wrapped(**batch, **options), batch)
            scale = float(full["masked_pred"].abs().max())
            pred_relative = float((full["masked_pred"] - split["masked_pred"]).abs().max()) / scale
            error_relative = float(((full["model_error"] - split["model_error"]).abs()
                                    / full["model_error"].abs().clamp_min(1e-12)).max())
            loss_relative = abs(full["loss"] - split["loss"]) / max(abs(full["loss"]), 1e-12)
            pooled_relative = abs(full["pooled_squared_error"] - split["pooled_squared_error"]) / max(full["pooled_squared_error"], 1e-12)
            passed = (
                plan.was_split
                and torch.equal(full["counts"], split["counts"])
                and torch.equal(full["baseline_error"], split["baseline_error"])
                and max(pred_relative, error_relative, loss_relative, pooled_relative) <= TOLERANCE
            )
            row = {
                "fold": fold, "model": model_name, "validation_batch_index": batch_index,
                "original_padded_size": padded_size, "piece_count": len(plan.piece_positions),
                "masked_prediction_max_relative_difference": pred_relative,
                "per_variable_model_error_max_relative_difference": error_relative,
                "loss_relative_difference": loss_relative,
                "pooled_squared_error_relative_difference": pooled_relative,
                "event_counts_and_carry_forward_error_identical": bool(
                    torch.equal(full["counts"], split["counts"]) and torch.equal(full["baseline_error"], split["baseline_error"])
                ),
                "pass": bool(passed),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"device": "cpu", "tolerance": TOLERANCE, "comparisons": rows,
            "status": "PASS" if rows and all(row["pass"] for row in rows) else "FAIL"}


def validation_calibrate(gpu: int) -> dict[str, Any]:
    """Replay the failing order: a training step, then validation, no cache flush."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from p1_padded_batch_split import plan_padded_batch
    from p1_microbatch_training import run_batch_plan

    runner = P2._load_runner()
    gpu_state = runner._gpu_status(gpu, min_free_mib=P2.P2_GPU_MIN_FREE_MIB, require_zero_utilization=True)
    if not gpu_state.get("free"):
        raise RuntimeError(f"calibration needs an idle GPU with {P2.P2_GPU_MIN_FREE_MIB} MiB free: {gpu_state}")
    rows = []
    for model_name in P2.P2_SPLIT_MODELS:
        exp = _experiment(model_name, "MAE", 4, "cuda")
        _t, train_loader = exp._get_data(flag="train")
        _v, vali_loader = exp._get_data(flag="val")
        train_index, train_batch, train_size = _densest(train_loader)
        vali_index, vali_batch, vali_size = _densest(vali_loader)
        model = exp._build_model().to(exp.device)
        optimizer = exp._select_optimizer(model)
        criterion = exp._select_criterion()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        plan = plan_padded_batch(train_batch, P2.P2_PADDED_SIZE_BUDGET, allow_oversized_singletons=True)
        run_batch_plan(exp, model, optimizer, criterion, train_batch, plan, train_stage=1, epoch=0,
                       batch_index=train_index + 1, offload_all=False, rescue=False)
        optimizer.step()
        wrapped = P2.split_validation_model(model, P2.P2_PADDED_SIZE_BUDGET)
        wrapped.eval()
        with torch.no_grad():
            wrapped(**{key: value.to(exp.device) for key, value in vali_batch.items()},
                    exp_stage="val", train_stage=1, current_epoch=0)
        peak = int(torch.cuda.max_memory_reserved() // (1024 * 1024))
        row = {"fold": 4, "model": model_name, "train_original_padded_size": train_size,
               "validation_original_padded_size": vali_size,
               "train_then_validation_peak_reserved_mib": peak,
               "pass": peak < VALIDATION_SEQUENCE_LIMIT_MIB}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, exp, wrapped
        torch.cuda.empty_cache()
    return {"gpu_index": gpu, "gpu_state_before": gpu_state, "limit_mib": VALIDATION_SEQUENCE_LIMIT_MIB,
            "sequence": "training step then validation, no cache flush", "rows": rows,
            "status": "PASS" if rows and all(row["pass"] for row in rows) else "FAIL"}


def validation_main(command: str, threads: int, gpu: int | None) -> int:
    path = P2.VALIDATION_SPLIT_PROOF
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    identity = _validation_identity()
    if command == "validation-proof":
        record = {"schema_version": "p2_validation_split_gate_v1", "rule": "work order §21.2", **identity,
                  "proof": validation_proof(threads), "proof_at_utc": _now()}
        section = record["proof"]
    else:
        if any(record.get(key) != value for key, value in identity.items()):
            raise RuntimeError("run `validation-proof` first with the current code and budget")
        record["gpu_calibration"] = validation_calibrate(gpu)
        record["gpu_calibration_at_utc"] = _now()
        section = record["gpu_calibration"]
    record["status"] = (
        "PASS" if record.get("proof", {}).get("status") == "PASS"
        and record.get("gpu_calibration", {}).get("status") == "PASS" else "INCOMPLETE_OR_FAIL"
    )
    P2.atomic_json(path, record)
    print(json.dumps({"status": record["status"]}))
    return 0 if section["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("proof")
    p.add_argument("--threads", type=int, default=16)
    c = sub.add_parser("calibrate")
    c.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    vp = sub.add_parser("validation-proof")
    vp.add_argument("--threads", type=int, default=16)
    vc = sub.add_parser("validation-calibrate")
    vc.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    if args.command.startswith("validation-"):
        return validation_main(args.command, getattr(args, "threads", 16), getattr(args, "gpu", None))

    record = json.loads(P2.MICROBATCH_PROOF.read_text(encoding="utf-8")) if P2.MICROBATCH_PROOF.is_file() else {}
    identity = {
        "trainer_sha256": _sha256(P2.MICROBATCH_TRAINER),
        "splitter_sha256": _sha256(P2.PADDED_BATCH_SPLITTER),
        "padded_size_budget": P2.P2_PADDED_SIZE_BUDGET,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    if args.command == "proof":
        record = {"schema_version": "p2_microbatch_gate_v1", "rule": "work order §20.2", **identity,
                  "gradient_proof": proof(args.threads), "gradient_proof_at_utc": _now()}
    else:
        if any(record.get(key) != value for key, value in identity.items()):
            raise RuntimeError("run `proof` first with the current code and budget")
        record["gpu_calibration"] = calibrate(args.gpu)
        record["gpu_calibration_at_utc"] = _now()
    record["status"] = (
        "PASS" if record.get("gradient_proof", {}).get("status") == "PASS"
        and record.get("gpu_calibration", {}).get("status") == "PASS" else "INCOMPLETE_OR_FAIL"
    )
    P2.atomic_json(P2.MICROBATCH_PROOF, record)
    print(json.dumps({"status": record["status"]}))
    section = record["gradient_proof"] if args.command == "proof" else record["gpu_calibration"]
    return 0 if section["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
