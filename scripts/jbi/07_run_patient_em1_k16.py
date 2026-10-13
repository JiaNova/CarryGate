#!/usr/bin/env python
"""Run a patient-level E-M1-K16 arm on a JBI cache.

The registered E-M1-K16 architecture, optimization budget, and 15 seeds are
kept fixed.  The eICU R2 invocation changes only the patient-grouped cache/split
pair; the same entry point is also used for the MIMIC-IV patient-level rerun.
Multiple processes may run disjoint seed shards on available GPUs;
``--merge`` validates and combines the formal result files, then removes the
shard CSVs when ``--clean-shards`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.p12_events import TIME_SERIES_VARIABLES  # noqa: E402
from src.data.p12_forecast_dataset import load_forecast_splits  # noqa: E402
from src.models.gates.gate_bases import suspend_coverage  # noqa: E402
from src.training.forecast_train import train_forecast_run  # noqa: E402
from src.utils.provenance import provenance_columns, write_config_snapshot  # noqa: E402


SEEDS = tuple(range(15))
ARM_NAME = "E-M1-K16"
ARM = {
    "gate_type": "kan",
    "gate_scope": "per_variable",
    "recency_basis": "rbf",
    "recency_num_centers": 16,
    "recency_parameterization": "adaptive",
}
EXCLUDED = (6, 19, 32)
SCORING_VARIABLES = tuple(
    index for index in range(len(TIME_SERIES_VARIABLES)) if index not in EXCLUDED
)
CSV_FLOAT_FORMAT = "%.17g"
PROTOCOL = "JBI_R2_E-M1-K16_only_split_change"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"configuration must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def _assert_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("matrix_arms") != {ARM_NAME: ARM}:
        raise SystemExit("R2 must contain exactly the registered E-M1-K16 arm")
    if tuple(int(seed) for seed in cfg.get("matrix_seeds", ())) != SEEDS:
        raise SystemExit("R2 must use exactly seeds 0..14")
    if tuple(int(value) for value in cfg.get("excluded_variables", ())) != EXCLUDED:
        raise SystemExit("R2 scoring exclusions differ from the registered protocol")
    if tuple(int(value) for value in cfg.get("test_variables", ())) != SCORING_VARIABLES:
        raise SystemExit("R2 test variable set differs from the registered protocol")
    expected = {
        "epochs": 80,
        "patience": 5,
        "batch_size": 128,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "d_ff": 128,
        "calibration_batches": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "mortality_weight": 0.1,
    }
    for key, value in expected.items():
        actual = cfg.get(key)
        if isinstance(value, float):
            if not np.isclose(float(actual), value, rtol=0.0, atol=1e-12):
                raise SystemExit(f"R2 setting {key} differs: {actual} != {value}")
        elif int(actual) != value:
            raise SystemExit(f"R2 setting {key} differs: {actual} != {value}")
    if cfg.get("head_type") != "residual" or cfg.get("encoder_type") != "identity":
        raise SystemExit("R2 must use the registered residual/identity model")
    if cfg.get("grouping") != "shared" or cfg.get("deterministic") is not True:
        raise SystemExit("R2 grouping/determinism differs from the registered protocol")
    if not np.isclose(float(cfg.get("eps_repro")), 1.0e-12, rtol=0.0, atol=1.0e-15):
        raise SystemExit("R2 eps_repro differs from the registered E-M1 value")
    objective = str(cfg.get("training_objective", "event_weighted_l1"))
    if objective not in {
        "event_weighted_l1",
        "event_weighted_mse",
        "variable_balanced_l1",
    }:
        raise SystemExit(f"unsupported training objective: {objective}")
    selection = str(
        cfg.get(
            "checkpoint_selection_metric",
            "macro_persistence",
        )
    )
    if selection not in {"macro_persistence", "pooled_mse"}:
        raise SystemExit(f"unsupported checkpoint selection metric: {selection}")


def _split_audit(bundle: Any) -> dict[str, Any]:
    examples = bundle.cache["examples"]
    candidates = ("uniquepid", "subject_id", "patient_id")
    group_column = next(
        (
            candidate
            for candidate in candidates
            if examples and all(
                candidate in example and example[candidate] is not None
                for example in examples
            )
        ),
        None,
    )
    if group_column is None:
        raise SystemExit("patient-level cache has no complete patient identity field")
    data_source = "eICU" if group_column == "uniquepid" else "MIMIC-IV"
    identities: dict[str, set[str]] = {}
    stays: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        values = []
        for index in bundle.split_indices[split_name]:
            example = examples[int(index)]
            patient = example.get(group_column)
            if patient is None:
                raise SystemExit(f"patient cache example {index} lacks {group_column}")
            values.append(str(patient))
        identities[split_name] = set(values)
        stays[split_name] = len(values)
    overlaps = {
        f"{left}&{right}": len(identities[left] & identities[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise SystemExit(f"R2 patient split has overlap: {overlaps}")
    return {
        "split_unit": "patient_grouped",
        "group_column": group_column,
        "data_source": data_source,
        "stays": stays,
        "patients": {name: len(values) for name, values in identities.items()},
        "overlaps": overlaps,
        "n_stays": int(sum(stays.values())),
        "n_patients": int(len(set().union(*identities.values()))),
    }


def _save_run(
    result: Mapping[str, Any], output_dir: Path, name: str, seed: int, n_variables: int
) -> None:
    test = result["test"]
    prediction_dir = output_dir / "predictions"
    curve_dir = output_dir / "gate_curves"
    stratified_dir = output_dir / "stratified"
    for directory in (prediction_dir, curve_dir, stratified_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{int(seed)}"
    np.savez_compressed(
        prediction_dir / f"{stem}.npz",
        prediction=test["prediction"],
        target=test["target"],
        variable=test["variable"],
        x_last=test["x_last"],
        staleness=test["staleness"],
    )
    for key, suffix in (
        ("skill_frame", "per_variable"),
        ("by_dynamics", "by_dynamics"),
        ("by_staleness", "by_staleness"),
    ):
        result[key].to_csv(
            stratified_dir / f"{stem}_{suffix}.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )

    hours = np.linspace(0.0, 48.0, 241, dtype=np.float32)
    gate = result["model"].forecast_head.gate
    if gate is None:
        omega = np.ones((n_variables, len(hours)), dtype=np.float32)
    else:
        gate_device = next(gate.parameters()).device
        log_delta = torch.as_tensor(np.log1p(hours), dtype=torch.float32, device=gate_device)
        variable = torch.arange(n_variables, dtype=torch.long, device=gate_device)
        variable = variable.reshape(-1, 1).expand(n_variables, len(hours))
        inputs = log_delta.reshape(1, -1).expand(n_variables, -1)
        with torch.no_grad(), suspend_coverage(gate):
            omega = gate(variable, inputs).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(curve_dir / f"{stem}.npz", hours=hours, omega=omega)


def _result_row(
    result: Mapping[str, Any],
    cfg: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": str(cfg.get("protocol", PROTOCOL)),
        "control_against": "registered E-M1-K16",
        "change_under_test": str(cfg.get("only_changed", "split_unit_only")),
        "config": ARM_NAME,
        "gate_type": ARM["gate_type"],
        "gate_scope": ARM["gate_scope"],
        "recency_basis": ARM["recency_basis"],
        "recency_num_centers": ARM["recency_num_centers"],
        "recency_parameterization": ARM["recency_parameterization"],
        "seed": int(result["seed"]),
        "epochs": int(cfg["epochs"]),
        "patience": int(cfg["patience"]),
        "training_objective": str(cfg.get("training_objective", "event_weighted_l1")),
        "checkpoint_selection_metric": str(
            cfg.get("checkpoint_selection_metric", "macro_persistence")
        ),
        "best_epoch": int(result["best_epoch"]),
        "epochs_run": int(result["epochs_run"]),
        "early_stopped": bool(
            int(result["best_epoch"]) <= int(cfg["epochs"]) - int(cfg["patience"])
        ),
        "status": result["status"],
        "grid_status": result["grid_status"],
        "determinism": result["determinism"],
        "skill": float(result["skill"]),
        "nmae": float(result["nmae"]),
        "nrmse": float(result["nrmse"]),
        "n_params": int(result["n_params"]),
        "duration_sec": float(result["duration_sec"]),
        "n_events_test": int(result["n_events_test"]),
        "excluded_variables": json.dumps(result["skill_excluded"], sort_keys=True),
        "registered_excluded_variables": json.dumps(EXCLUDED),
        "forecast_cache": str(Path(cfg["forecast_cache"]).resolve()),
        "forecast_split": str(Path(cfg["forecast_split"]).resolve()),
        "data_source": split_audit["data_source"],
        "split_unit": split_audit["split_unit"],
        "group_column": split_audit["group_column"],
        "n_stays_total": split_audit["n_stays"],
        "n_patients_total": split_audit["n_patients"],
        **{
            f"n_stays_{name}": value
            for name, value in split_audit["stays"].items()
        },
        **{
            f"n_patients_{name}": value
            for name, value in split_audit["patients"].items()
        },
        **provenance_columns(result),
    }


def _shard_path(output_dir: Path, shard_index: int, num_shards: int) -> Path:
    return output_dir / (
        "raw_results.csv"
        if int(num_shards) == 1
        else f"raw_results_shard_{int(shard_index)}.csv"
    )


def _merge(output_dir: Path, cfg: Mapping[str, Any], clean_shards: bool) -> None:
    _assert_config(cfg)
    paths = sorted(output_dir.glob("raw_results_shard_*.csv"))
    final = output_dir / "raw_results.csv"
    frames = [pd.read_csv(path, float_precision="round_trip") for path in paths]
    if final.exists():
        frames.insert(0, pd.read_csv(final, float_precision="round_trip"))
    if not frames:
        raise SystemExit(f"no R2 shard results found in {output_dir}")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["config", "seed"], keep="last"
    )
    merged = merged.sort_values(["config", "seed"])
    expected = {(ARM_NAME, seed) for seed in SEEDS}
    observed = {(str(row.config), int(row.seed)) for row in merged.itertuples()}
    errors = []
    if observed != expected or len(merged) != len(expected):
        errors.append(f"expected 15 arm/seed rows, observed {len(merged)}: {sorted(observed)}")
    for column, value in (("status", "ok"), ("grid_status", "ok"), ("determinism", "strict")):
        if len(merged) and not bool((merged[column].astype(str) == value).all()):
            errors.append(f"not all rows have {column}={value}")
    if len(merged) and not bool(np.isfinite(merged["skill"].to_numpy(dtype=float)).all()):
        errors.append("non-finite skill")
    if len(merged) and not bool((merged["split_unit"] == "patient_grouped").all()):
        errors.append("split_unit provenance is not patient_grouped")
    if "change_under_test" in merged:
        merged["change_under_test"] = str(
            cfg.get("only_changed", "split_unit_only")
        )
    merged["training_objective"] = str(
        cfg.get("training_objective", "event_weighted_l1")
    )
    merged["checkpoint_selection_metric"] = str(
        cfg.get("checkpoint_selection_metric", "macro_persistence")
    )
    merged.to_csv(final, index=False, float_format=CSV_FLOAT_FORMAT)
    report = {
        "protocol": str(cfg.get("protocol", PROTOCOL)),
        "passed": not errors,
        "n_runs": int(len(merged)),
        "seeds": list(SEEDS),
        "arm": ARM,
        "errors": errors,
        "mean_skill": float(merged["skill"].mean()) if len(merged) else None,
        "skill_std_across_seeds": float(merged["skill"].std(ddof=1)) if len(merged) > 1 else None,
        "split_unit": "patient_grouped",
        "forecast_cache": str(Path(cfg["forecast_cache"]).resolve()),
        "forecast_split": str(Path(cfg["forecast_split"]).resolve()),
    }
    (output_dir / "r2_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "r2_split_audit.json").write_text(
        json.dumps(
            {
                "protocol": str(cfg.get("protocol", PROTOCOL)),
                "split_unit": "patient_grouped",
                "group_column": str(
                    cfg.get("group_column", "uniquepid")
                ),
                "n_stays": {
                    key: int(merged[f"n_stays_{key}"].iloc[0])
                    for key in ("train", "val", "test")
                },
                "n_patients": {
                    key: int(merged[f"n_patients_{key}"].iloc[0])
                    for key in ("train", "val", "test")
                },
                "overlaps": {"train&val": 0, "train&test": 0, "val&test": 0},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if clean_shards:
        for path in paths:
            path.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eicu_jbi_patient_m1k16.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--clean-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    cfg = _load_config(Path(args.config))
    _assert_config(cfg)
    output_dir = Path(args.output_dir or cfg["phase5_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(output_dir, cfg, bool(args.clean_shards))
        return
    if args.device:
        cfg["device"] = args.device
    if args.shard_index == 0:
        write_config_snapshot(output_dir, cfg)
        (output_dir / "protocol.json").write_text(
            json.dumps(
                {
                    "protocol": str(cfg.get("protocol", PROTOCOL)),
                    "arm": ARM,
                    "seeds": list(SEEDS),
                    "epochs": int(cfg["epochs"]),
                    "only_changed": str(
                        cfg.get("only_changed", "patient_grouped_split")
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    bundle = load_forecast_splits(cfg["forecast_cache"], cfg["forecast_split"])
    split_audit = _split_audit(bundle)
    n_variables = int(bundle.standardizer.n_variables)
    jobs = [(ARM_NAME, seed) for seed in SEEDS][args.shard_index :: args.num_shards]
    raw_path = _shard_path(output_dir, args.shard_index, args.num_shards)
    completed: set[tuple[str, int]] = set()
    if args.resume and raw_path.exists():
        old = pd.read_csv(raw_path, float_precision="round_trip")
        completed = {(str(row.config), int(row.seed)) for row in old.itertuples()}
    for name, seed in jobs:
        if (name, seed) in completed:
            print(f"{name} seed={seed} resumed", flush=True)
            continue
        run_cfg = dict(cfg)
        run_cfg.update(ARM)
        run_cfg.update(
            {
                "head_type": "residual",
                "encoder_type": "identity",
                "grouping": "shared",
                "gate_scope": "per_variable",
                "recency_x_range": tuple(cfg.get("recency_x_range", (0.0, 4.0))),
            }
        )
        result = train_forecast_run(
            bundle,
            encoder_type="identity",
            seed=seed,
            cfg=run_cfg,
            output_dir=None,
        )
        _save_run(result, output_dir, name, seed, n_variables)
        row = _result_row(result, cfg, split_audit)
        if raw_path.exists():
            frame = pd.read_csv(raw_path, float_precision="round_trip")
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame = frame.drop_duplicates(["config", "seed"], keep="last").sort_values(
            ["config", "seed"]
        )
        frame.to_csv(raw_path, index=False, float_format=CSV_FLOAT_FORMAT)
        print(f"{name} seed={seed} skill={float(result['skill']):.8f}", flush=True)
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(raw_path)


if __name__ == "__main__":
    main()
