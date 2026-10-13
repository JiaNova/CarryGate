#!/usr/bin/env python
"""Validation-only diagnostics for the P1 pilot trigger (work order §17).

The 0.03 trigger fired for MIMIC-IV HyperIMTS (P1 t=24 validation skill
0.1364 against Track B 0.1031). P1 legitimately differs from Track B -- it
trains on every landmark, selects checkpoints on every landmark, and uses a
72-hour time axis -- so the trigger cannot by itself separate an adapter error
from a training difference. The one adapter error that *raises* skill is
outcome information leaking into the model input. Three subcommands:

``probe``    D1, decisive. For each seed-2024 checkpoint, predictions must not
             change when every outcome value ``y`` is replaced by noise, the
             carry-forward reference must not change either, and every valid
             input event must lie before the landmark while every outcome
             event lies at or after it.
``control``  D2, descriptive. Retrain MIMIC-IV HyperIMTS seed 2024 exactly as
             the pilot, except that training and validation use only the
             t=24 landmark. Writes to its own directory, never the manifest.
``resolve``  Applies the rule registered in §17.3 and, only if it holds,
             writes the resolution file that lets the remaining seeds run.

Nothing here reads the test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/jbi/42_train_p1_landmark.py"
DIAG_DIR = ROOT / "results/jbi/p1_trigger_diagnostics"
PROBE_PATH = ROOT / "results/jbi/p1_trigger_probe.json"
CONTROL_PATH = ROOT / "results/jbi/p1_trigger_control_t24only.json"
POLICY_PATH = ROOT / "results/jbi/p1_microbatch_training_policy.json"
PROBE_BATCHES = 25
PROBE_NOISE_SEED = 20261009
HYPERIMTS_MIN_FREE_MIB = 14932
TRACKB_GAP_TOLERANCE = 0.015


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_landmark_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _enter_harness(gpu: int):
    # Pin the device before torch initialises CUDA, as the training entry does.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(RUNNER.HARNESS))
    sys.path.insert(0, str(ROOT / "scripts/jbi"))
    os.chdir(RUNNER.HARNESS)
    import torch
    return torch


def _pilot_runs() -> list[dict[str, Any]]:
    manifest = RUNNER._load_manifest(create=False)
    pilots = sorted(
        (run for run in manifest["runs"] if int(run["seed"]) == 2024),
        key=lambda run: run["run_key"],
    )
    if len(pilots) != 6 or any(run.get("status") != "completed" for run in pilots):
        raise RuntimeError("probe needs all six completed seed-2024 pilot checkpoints")
    return pilots


def _time_rows(mark, mask):
    """Per-row time and validity for [B,T] or [B,T,1] marks and [B,T,V] masks."""
    times = mark.squeeze(-1) if mark.dim() == 3 else mark
    valid = (mask > 0).reshape(mask.shape[0], mask.shape[1], -1).any(dim=-1)
    return times, valid


def probe(args) -> int:
    torch = _enter_harness(args.gpu)
    from exp.exp_main import Exp_Main
    from utils.globals import accelerator
    from data.data_provider.datasets.ICUKANLandmark import (
        INPUT_HOURS, TOTAL_HOURS, carry_forward_target_values,
    )

    results = []
    for run in _pilot_runs():
        cohort, model_name = run["cohort"], run["model"]
        config, _, _ = RUNNER._get_configs(model_name, cohort, 2024)
        scratch = DIAG_DIR / "probe_scratch" / f"{cohort}_{model_name}"
        scratch.mkdir(parents=True, exist_ok=True)
        config.checkpoints = str(scratch)          # nothing may write into frozen run directories
        accelerator.project_configuration.set_directories(project_dir=scratch)
        RUNNER._seed_everything(2024, torch)
        exp = Exp_Main(config)
        model = exp._build_model().to(exp.device)
        checkpoint = ROOT / run["checkpoint_path"]
        if RUNNER.sha256(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash changed for {run['run_key']}")
        model.load_state_dict(exp._get_state_dict(checkpoint))
        model.eval()
        _, loader = exp._get_data(flag="val")
        noise = torch.Generator().manual_seed(PROBE_NOISE_SEED)

        d_rep = d_pert = 0.0
        carry_forward_unchanged = True
        max_input_time = -float("inf")
        min_outcome_time = float("inf")
        batches = windows = 0
        with torch.no_grad():
            for batch in loader:
                if batches >= args.batches:
                    break
                batch = {key: value.to(exp.device) for key, value in batch.items()}
                first = model(**batch, exp_stage="val")["pred"]
                second = model(**batch, exp_stage="val")["pred"]
                perturbed = dict(batch)
                perturbed["y"] = 3.0 * torch.randn(batch["y"].shape, generator=noise).to(exp.device)
                shuffled = model(**perturbed, exp_stage="val")["pred"]
                d_rep = max(d_rep, float((first - second).abs().max().item()))
                d_pert = max(d_pert, float((first - shuffled).abs().max().item()))
                carry_forward_unchanged &= bool(torch.equal(
                    carry_forward_target_values(batch), carry_forward_target_values(perturbed)
                ))
                x_time, x_valid = _time_rows(batch["x_mark"], batch["x_mask"])
                y_time, y_valid = _time_rows(batch["y_mark"], batch["y_mask"])
                if x_valid.any():
                    max_input_time = max(max_input_time, float(x_time[x_valid].max().item()))
                if y_valid.any():
                    min_outcome_time = min(min_outcome_time, float(y_time[y_valid].min().item()))
                batches += 1
                windows += int(batch["sample_ID"].shape[0])
        # Marks are either local hours or local hours / 72; the boundary is 48 h either way.
        normalized = min_outcome_time <= 1.0 + 1e-6
        boundary = float(torch.tensor(INPUT_HOURS / TOTAL_HOURS if normalized else INPUT_HOURS,
                                      dtype=torch.float32).item())
        tolerance = max(10.0 * d_rep, 1e-6)
        record = {
            "run_key": run["run_key"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "validation_batches": batches,
            "validation_windows": windows,
            "d_rep_max_abs": d_rep,
            "d_pert_max_abs": d_pert,
            "tolerance": tolerance,
            "outcome_values_do_not_reach_predictions": d_pert <= tolerance,
            "carry_forward_reference_unchanged": carry_forward_unchanged,
            "time_axis": "normalized" if normalized else "hours",
            "input_boundary": boundary,
            "max_valid_input_time": max_input_time,
            "min_valid_outcome_time": min_outcome_time,
            "inputs_before_landmark": max_input_time < boundary,
            "outcomes_at_or_after_landmark": min_outcome_time >= boundary - 1e-6,
        }
        record["pass"] = all(record[key] for key in (
            "outcome_values_do_not_reach_predictions", "carry_forward_reference_unchanged",
            "inputs_before_landmark", "outcomes_at_or_after_landmark",
        ))
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
        del model, exp
        torch.cuda.empty_cache()

    report = {
        "schema_version": "p1_trigger_probe_v1",
        "generated_at_utc": RUNNER.now_utc(),
        "rule": "work order §17.3 D1: d_pert <= max(10*d_rep, 1e-6), carry-forward unchanged, input/outcome times on their side of the landmark; all six must pass",
        "split_used": "validation only",
        "script_sha256": RUNNER.sha256(Path(__file__).resolve()),
        "trigger_sha256": RUNNER.sha256(RUNNER.TRIGGER_PATH),
        "checkpoints": results,
        "all_pass": all(item["pass"] for item in results),
    }
    output = Path(args.output) if args.output else PROBE_PATH
    RUNNER._atomic_json(output, report)
    print(json.dumps({"probe_all_pass": report["all_pass"], "path": str(output)}, ensure_ascii=False))
    return 0 if report["all_pass"] else 3


def control(args) -> int:
    cohort, model_name, seed = "mimiciv", "HyperIMTS", 2024
    if CONTROL_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {CONTROL_PATH}")
    gpu_info = RUNNER._gpu_status(args.gpu, min_free_mib=HYPERIMTS_MIN_FREE_MIB, require_zero_utilization=False)
    if not gpu_info["free"]:
        raise RuntimeError(f"GPU {args.gpu} lacks {HYPERIMTS_MIN_FREE_MIB} MiB free: {gpu_info}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("budget_freeze", {}).get("status") != "FROZEN":
        raise RuntimeError("micro-batch budget is not frozen")
    budget = int(policy["budget_values"][f"{cohort}/{model_name}"])

    torch = _enter_harness(args.gpu)
    from exp.exp_main import Exp_Main
    from utils.globals import accelerator
    from data.data_provider.datasets import ICUKANLandmark as adapter
    from p1_microbatch_training import train_with_padded_microbatches

    # The only difference from the pilot: both loaders keep the t=24 landmark alone.
    original_init = adapter.Data.__init__

    def t24_only(self, configs, flag="train", **kwargs):
        original_init(self, configs, flag=flag, **kwargs)
        self.landmark_rows = [row for row in self.landmark_rows if row[1] == 24.0]

    adapter.Data.__init__ = t24_only

    config, _, _ = RUNNER._get_configs(model_name, cohort, seed)
    config.model_id = f"{config.model_id}_t24only_control"
    config.checkpoints = str(DIAG_DIR / "control_checkpoints")
    run_dir = (
        Path(config.checkpoints) / config.dataset_name / cohort / model_name / config.model_id
        / f"{config.seq_len}_{config.pred_len}" / config.subfolder_train / "iter0"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator.project_configuration.set_directories(project_dir=run_dir)
    RUNNER._seed_everything(seed, torch)
    exp = Exp_Main(config)

    epochs_path = run_dir / "epoch_validation.jsonl"
    original_vali = exp.vali

    def logged_vali(**kwargs):
        report = original_vali(**kwargs)
        with epochs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": int(kwargs.get("current_epoch", -1)),
                "validation_macro_persistence_skill": float(report.get("macro_persistence_skill", float("nan"))),
            }) + "\n")
        return report

    exp.vali = logged_vali
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    training = train_with_padded_microbatches(
        exp, budget, offload_oversized_pieces=True, oom_rescue=True,
    )
    duration = time.perf_counter() - started
    checkpoint = run_dir / "pytorch_model.bin"
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    t24 = RUNNER._validation_t24_skill(exp, checkpoint, torch)

    manifest = RUNNER._load_manifest(create=False)
    pilot = next(run for run in manifest["runs"] if run["run_key"] == f"{cohort}/{model_name}/seed{seed}")
    pilot_skill = float(pilot["validation_t24_macro_persistence_skill"])
    trackb_skill = float(pilot["trackb_best_validation_macro_persistence_skill"])
    control_skill = float(t24["validation_t24_macro_persistence_skill"])
    gap = pilot_skill - trackb_skill
    if abs(control_skill - trackb_skill) <= TRACKB_GAP_TOLERANCE:
        category = "mainly_more_training_landmarks"
    elif abs(control_skill - pilot_skill) <= TRACKB_GAP_TOLERANCE:
        category = "mainly_time_axis_or_checkpoint_selection"
    else:
        category = "both"
    report = {
        "schema_version": "p1_trigger_control_t24only_v1",
        "generated_at_utc": RUNNER.now_utc(),
        "rule": "work order §17.3 D2: descriptive only; no pass threshold",
        "split_used": "train and validation, t=24 landmark only",
        "script_sha256": RUNNER.sha256(Path(__file__).resolve()),
        "run_key": f"{cohort}/{model_name}/seed{seed}",
        "padded_size_budget": budget,
        "duration_sec": duration,
        "best_epoch": summary["stages"][0]["best_epoch"],
        "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(0) // (1024 * 1024)),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": RUNNER.sha256(checkpoint),
        "training": training,
        "control_validation_t24_skill": control_skill,
        "pilot_validation_t24_skill": pilot_skill,
        "trackb_validation_skill": trackb_skill,
        "gap_explained_by_extra_landmarks": (pilot_skill - control_skill) / gap if gap else None,
        "category": category,
        **{key: value for key, value in t24.items() if key != "validation_t24_macro_persistence_skill"},
    }
    RUNNER._atomic_json(CONTROL_PATH, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def resolve(_args) -> int:
    if not PROBE_PATH.is_file() or not CONTROL_PATH.is_file():
        raise RuntimeError("resolution needs both the D1 probe and the D2 control reports")
    trigger_sha = RUNNER.sha256(RUNNER.TRIGGER_PATH)
    probe_report = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    control_report = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    if probe_report.get("trigger_sha256") != trigger_sha:
        raise RuntimeError("probe was run against a different trigger file")
    probe_pass = bool(probe_report.get("all_pass")) and len(probe_report.get("checkpoints", [])) == 6
    control_done = control_report.get("control_validation_t24_skill") is not None
    resolution = {
        "schema_version": "p1_landmark_pilot_trigger_resolution_v1",
        "generated_at_utc": RUNNER.now_utc(),
        "rule": "work order §17.3: resume iff D1 passes for all six checkpoints and D2 has completed",
        "trigger_sha256": trigger_sha,
        "evidence_sha256": {
            str(PROBE_PATH.relative_to(ROOT)): RUNNER.sha256(PROBE_PATH),
            str(CONTROL_PATH.relative_to(ROOT)): RUNNER.sha256(CONTROL_PATH),
        },
        "leakage_probe_pass": probe_pass,
        "control_completed": control_done,
        "control_category": control_report.get("category"),
        "status": "RESOLVED_CONTINUE_REMAINING_SEEDS" if probe_pass and control_done else "UNRESOLVED_KEEP_PAUSED",
    }
    if resolution["status"] != "RESOLVED_CONTINUE_REMAINING_SEEDS":
        print(json.dumps(resolution, indent=2, ensure_ascii=False))
        return 3
    RUNNER._atomic_json(RUNNER.TRIGGER_RESOLUTION_PATH, resolution)
    print(json.dumps(resolution, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="D1 leakage probe on the six pilot checkpoints")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batches", type=int, default=PROBE_BATCHES)
    p.add_argument("--output", default=None, help="smoke tests only; the registered D1 report uses the default path")
    c = sub.add_parser("control", help="D2 t=24-only retraining of MIMIC-IV HyperIMTS seed 2024")
    c.add_argument("--gpu", type=int, required=True)
    sub.add_parser("resolve", help="apply the §17.3 rule and write the resolution file")
    args = parser.parse_args()
    return {"probe": probe, "control": control, "resolve": resolve}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
