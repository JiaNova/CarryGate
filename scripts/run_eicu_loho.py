#!/usr/bin/env python
"""Run the preregistered eICU leave-one-hospital-group-out experiment.

The runner consumes only the complete selected-hospital cache produced by
``prepare_eicu_loho.py``.  The 145-hospital frame is defined by the raw
E-T0 long-stay count; only its registered first-unit/age/status stays enter
the forecasting cache.  It constructs deterministic, disjoint hospital
groups, keeps every held-out hospital out of train/validation, and reuses the
MIMIC-IV standardizer stored in the cache.  LOHO is descriptive: this script
does not contain a success threshold or a result-dependent retry path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.p12_events import TIME_SERIES_VARIABLES, load_p12_forecast_cache  # noqa: E402
from src.data.p12_forecast_dataset import (  # noqa: E402
    ForecastSplitBundle,
    P12ForecastDataset,
    load_forecast_split,
)
from src.models.gates.gate_bases import suspend_coverage  # noqa: E402
from src.training.forecast_train import train_forecast_run  # noqa: E402
from src.utils.provenance import provenance_columns, write_config_snapshot  # noqa: E402


LOHO_ARMS: dict[str, dict[str, Any]] = {
    "K-0": {"gate_type": "none", "gate_scope": "shared"},
    "K-M8": {"gate_type": "mlp", "gate_scope": "per_variable", "gate_hidden": 8},
    "K-K8": {
        "gate_type": "kan",
        "gate_scope": "per_variable",
        "recency_basis": "rbf",
        "recency_num_centers": 8,
        "recency_parameterization": "adaptive",
    },
}
LOHO_SEEDS = tuple(range(5))
LOHO_FOLDS = 5
LOHO_SEED = 42
LOHO_MIN_HOSPITAL_STAYS = 100
LOHO_EXPECTED_HOSPITALS = 145
LOHO_EXPECTED_ROWS = LOHO_FOLDS * len(LOHO_ARMS) * len(LOHO_SEEDS)
REGISTERED_EXCLUDED = (6, 19, 32)
REGISTERED_SCORING_VARIABLES = tuple(
    index for index in range(len(TIME_SERIES_VARIABLES)) if index not in REGISTERED_EXCLUDED
)
CSV_FLOAT_FORMAT = "%.17g"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"configuration must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def _assert_config(cfg: Mapping[str, Any]) -> None:
    if tuple(int(value) for value in cfg.get("excluded_variables", ())) != REGISTERED_EXCLUDED:
        raise SystemExit("LOHO excluded variables must remain exactly (6, 19, 32)")
    if tuple(int(value) for value in cfg.get("test_variables", ())) != REGISTERED_SCORING_VARIABLES:
        raise SystemExit("LOHO test_variables must be the 34 non-structural P12 IDs")
    if tuple(int(value) for value in cfg.get("loho_seeds", ())) != LOHO_SEEDS:
        raise SystemExit("LOHO seed list must remain [0, 1, 2, 3, 4]")
    frozen = {
        "loho_seed": LOHO_SEED,
        "loho_folds": LOHO_FOLDS,
        "loho_min_hospital_stays": LOHO_MIN_HOSPITAL_STAYS,
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
    for key, expected in frozen.items():
        actual = cfg.get(key)
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
                raise SystemExit(f"frozen LOHO setting {key} was modified: {actual}")
        elif int(actual) != expected:
            raise SystemExit(f"frozen LOHO setting {key} was modified: {actual}")
    if cfg.get("head_type") != "residual" or cfg.get("encoder_type") != "identity":
        raise SystemExit("LOHO must use the registered residual/identity model")
    if cfg.get("grouping") != "shared" or cfg.get("deterministic") is not True:
        raise SystemExit("LOHO grouping/determinism settings were modified")


def _standardizer_equal(left: object, right: object) -> bool:
    fields = ("variable_names", "median", "iqr", "lower_clip", "upper_clip", "train_mean", "clipping_ratio")
    return all(
        getattr(left, field, None) == getattr(right, field, None)
        if field == "variable_names"
        else np.array_equal(getattr(left, field, None), getattr(right, field, None))
        for field in fields
    )


def _load_full_bundle(cfg: Mapping[str, Any]) -> tuple[ForecastSplitBundle, dict[int, int]]:
    cache_path = Path(str(cfg["forecast_cache"]))
    split_path = Path(str(cfg["forecast_split"]))
    manifest_path = Path(
        str(cfg.get("manifest", "results/phase3_eicu/e_loho/eicu_loho_manifest.json"))
    )
    if not cache_path.exists() or not split_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "complete LOHO cache/split/manifest is missing; run "
            "scripts/prepare_eicu_loho.py first "
            f"({cache_path}, {split_path}, {manifest_path})"
        )
    cache, _ = load_forecast_split(cache_path, split_path)
    examples = cache["examples"]
    if tuple(cache.get("variable_names", ())) != TIME_SERIES_VARIABLES:
        raise SystemExit("LOHO cache variable order is not the registered P12 order")
    standardizer_cache = Path(str(cfg.get("mimic_standardizer_cache", "data/processed/mimiciv_events.pkl")))
    mimic = load_p12_forecast_cache(standardizer_cache)
    if not _standardizer_equal(cache["standardizer"], mimic["standardizer"]):
        raise SystemExit("LOHO cache standardizer differs from the frozen MIMIC-IV standardizer")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filter_report = manifest.get("queue", {}).get("filter", {})
    if int(filter_report.get("hospital_min_stays", -1)) != LOHO_MIN_HOSPITAL_STAYS:
        raise SystemExit("LOHO manifest has the wrong raw long-stay hospital threshold")
    if not str(filter_report.get("hospital_count_basis", "")).startswith(
        "all patient rows with unitdischargeoffset >= 2880"
    ):
        raise SystemExit("LOHO manifest does not record the registered raw long-stay criterion")
    selected_hospitals = {
        int(hospital) for hospital in filter_report.get("selected_hospital_ids", ())
    }
    raw_counts = {
        int(hospital): int(count)
        for hospital, count in filter_report.get("long_stay_hospital_counts", {}).items()
    }
    if len(selected_hospitals) != LOHO_EXPECTED_HOSPITALS:
        raise SystemExit(
            f"LOHO manifest must contain exactly {LOHO_EXPECTED_HOSPITALS} selected hospitals"
        )
    if set(raw_counts) != selected_hospitals or any(
        count < LOHO_MIN_HOSPITAL_STAYS for count in raw_counts.values()
    ):
        raise SystemExit("LOHO manifest raw long-stay hospital counts violate the registered contract")
    if int(manifest.get("examples", -1)) != len(examples):
        raise SystemExit("LOHO manifest example count does not match the cache")
    counts = Counter(int(example.get("hospital_id", -1)) for example in examples)
    if -1 in counts:
        raise SystemExit("LOHO cache has an example without hospital_id")
    if set(counts) != selected_hospitals:
        raise SystemExit(
            "LOHO cache hospitals do not match the manifest's selected raw-long-stay hospitals; "
            f"cache contains {len(counts)}, manifest contains {len(selected_hospitals)}"
        )
    if any(count < 1 for count in counts.values()):
        raise SystemExit(
            "LOHO selected hospitals must each contribute at least one registered "
            "forecast-cohort stay"
        )
    eligible_counts = {
        int(hospital): int(count)
        for hospital, count in filter_report.get("eligible_stay_counts", {}).items()
    }
    if eligible_counts != {int(hospital): int(count) for hospital, count in counts.items()}:
        raise SystemExit("LOHO manifest eligible stay counts do not match the cache")
    return (
        ForecastSplitBundle(
            standardizer=cache["standardizer"],
            cache=cache,
            datasets={},
            split_indices={},
        ),
        {int(hospital): int(count) for hospital, count in counts.items()},
    )


def build_hospital_folds(
    hospital_counts: Mapping[int, int],
    *,
    n_folds: int = LOHO_FOLDS,
    seed: int = LOHO_SEED,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Assign every selected hospital once, balancing held-out stays by layer."""

    hospitals = sorted(int(hospital) for hospital in hospital_counts)
    if len(hospitals) != LOHO_EXPECTED_HOSPITALS:
        raise ValueError(f"expected {LOHO_EXPECTED_HOSPITALS} hospitals, got {len(hospitals)}")
    layers = [list(map(int, values)) for values in np.array_split(hospitals, int(n_folds))]
    rng = np.random.default_rng(int(seed))
    assignment: dict[int, int] = {}
    total_held = [0 for _ in range(int(n_folds))]
    rows: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        target_count = int(np.ceil(len(layer) / int(n_folds)))
        layer_counts = [0 for _ in range(int(n_folds))]
        random_priority = {hospital: float(rng.random()) for hospital in layer}
        ordered = sorted(
            layer,
            key=lambda hospital: (-int(hospital_counts[hospital]), random_priority[hospital], hospital),
        )
        for hospital in ordered:
            candidates = [
                fold for fold in range(int(n_folds)) if layer_counts[fold] < target_count
            ]
            minimum_stays = min(total_held[fold] for fold in candidates)
            candidates = [fold for fold in candidates if total_held[fold] == minimum_stays]
            fold = int(candidates[int(rng.integers(len(candidates)))])
            assignment[hospital] = fold
            layer_counts[fold] += 1
            total_held[fold] += int(hospital_counts[hospital])
            rows.append(
                {
                    "hospital_id": hospital,
                    "n_stays": int(hospital_counts[hospital]),
                    "size_layer": layer_id,
                    "fold": fold,
                }
            )
    if set(assignment) != set(hospitals) or len(set(assignment.values())) != int(n_folds):
        raise AssertionError("LOHO hospital assignment is not complete")
    rows.sort(key=lambda row: (int(row["fold"]), int(row["size_layer"]), int(row["hospital_id"])))
    return assignment, rows


def _train_val_split(
    indices: Sequence[int],
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    val_fraction: float = 0.15,
) -> tuple[list[int], list[int]]:
    """Make a deterministic mortality-stratified 85/15 split of non-held stays."""

    rng = np.random.default_rng(int(seed))
    by_label: dict[int, list[int]] = {}
    for index in indices:
        label = int(float(examples[int(index)]["label"]))
        by_label.setdefault(label, []).append(int(index))
    train: list[int] = []
    val: list[int] = []
    for label in sorted(by_label):
        values = np.asarray(sorted(by_label[label]), dtype=np.int64)
        rng.shuffle(values)
        if len(values) <= 1:
            n_val = 0
        else:
            n_val = min(len(values) - 1, max(1, int(round(len(values) * float(val_fraction)))))
        val.extend(int(value) for value in values[:n_val])
        train.extend(int(value) for value in values[n_val:])
    if not train or not val:
        raise ValueError("LOHO non-held train/validation split is empty")
    return sorted(train), sorted(val)


def _fold_bundle(
    base: ForecastSplitBundle,
    assignment: Mapping[int, int],
    fold: int,
) -> tuple[ForecastSplitBundle, dict[str, Any]]:
    examples = base.cache["examples"]
    test_indices = sorted(
        int(index)
        for index, example in enumerate(examples)
        if int(assignment[int(example["hospital_id"])]) == int(fold)
    )
    train_pool = sorted(set(range(len(examples))) - set(test_indices))
    train_indices, val_indices = _train_val_split(
        train_pool, examples, seed=LOHO_SEED + int(fold)
    )
    if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
        raise AssertionError("LOHO train/val/test overlap")
    held_hospitals = sorted(
        hospital for hospital, assigned_fold in assignment.items() if int(assigned_fold) == int(fold)
    )
    datasets = {
        name: P12ForecastDataset(
            examples,
            indices,
            train_median=np.zeros(base.standardizer.n_variables, dtype=np.float32),
        )
        for name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        )
    }
    bundle = ForecastSplitBundle(
        standardizer=base.standardizer,
        cache=base.cache,
        datasets=datasets,
        split_indices={"train": train_indices, "val": val_indices, "test": test_indices},
    )
    metadata = {
        "fold": int(fold),
        "held_hospitals": held_hospitals,
        "n_held_hospitals": len(held_hospitals),
        "n_train_stays": len(train_indices),
        "n_val_stays": len(val_indices),
        "n_test_stays": len(test_indices),
    }
    return bundle, metadata


def _gate_metadata(arm: Mapping[str, Any], n_variables: int) -> dict[str, int]:
    scope = str(arm["gate_scope"])
    gate_type = str(arm["gate_type"])
    groups = 1 if scope == "shared" else int(n_variables)
    if gate_type == "none":
        per_group = 0
    elif gate_type == "mlp":
        per_group = 3 * int(arm["gate_hidden"]) + 1
    elif gate_type == "kan":
        per_group = 3 * int(arm["recency_num_centers"]) + 1
    else:
        raise ValueError(f"unsupported LOHO gate type: {gate_type}")
    return {
        "gate_groups": groups,
        "gate_params_per_group": per_group,
        "gate_n_params": groups * per_group,
    }


def _save_run(
    result: Mapping[str, Any],
    output_dir: Path,
    fold: int,
    name: str,
    seed: int,
    n_variables: int,
) -> float | None:
    prediction_dir = output_dir / "predictions"
    curve_dir = output_dir / "gate_curves"
    stratified_dir = output_dir / "stratified"
    for directory in (prediction_dir, curve_dir, stratified_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_fold{int(fold)}_{int(seed)}"
    test = result["test"]
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
    staleness = np.asarray(test["staleness"], dtype=float)
    finite = staleness[np.isfinite(staleness)]
    return float(np.median(finite)) if len(finite) else None


def _result_row(
    result: Mapping[str, Any],
    name: str,
    arm: Mapping[str, Any],
    cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    held_staleness_median: float | None,
) -> dict[str, Any]:
    gate = _gate_metadata(arm, len(TIME_SERIES_VARIABLES))
    return {
        "fold": int(metadata["fold"]),
        "config": name,
        "gate_type": arm["gate_type"],
        "gate_scope": arm["gate_scope"],
        "gate_hidden": arm.get("gate_hidden"),
        "recency_basis": arm.get("recency_basis"),
        "recency_num_centers": arm.get("recency_num_centers"),
        "recency_parameterization": arm.get("recency_parameterization"),
        "seed": int(result["seed"]),
        "epochs": int(cfg["epochs"]),
        "patience": int(cfg["patience"]),
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
        **gate,
        "duration_sec": float(result["duration_sec"]),
        "n_events_test": int(result["n_events_test"]),
        "n_train_stays": int(metadata["n_train_stays"]),
        "n_val_stays": int(metadata["n_val_stays"]),
        "n_test_stays": int(metadata["n_test_stays"]),
        "n_held_hospitals": int(metadata["n_held_hospitals"]),
        "held_hospitals": json.dumps(metadata["held_hospitals"]),
        "held_staleness_median": held_staleness_median,
        "excluded_variables": json.dumps(result["skill_excluded"], sort_keys=True),
        "registered_excluded_variables": json.dumps(REGISTERED_EXCLUDED),
        **provenance_columns(result),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    current = pd.read_csv(path, float_precision="round_trip") if path.exists() else pd.DataFrame()
    addition = pd.DataFrame([dict(row) for row in rows])
    merged = pd.concat([current, addition], ignore_index=True)
    merged = merged.drop_duplicates(["fold", "config", "seed"], keep="last")
    merged = merged.sort_values(["fold", "config", "seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)


def _merge(output_dir: Path, clean_shards: bool) -> Path:
    final = output_dir / "raw_results.csv"
    paths = sorted(output_dir.glob("raw_results_shard_*.csv"))
    frames = [pd.read_csv(final, float_precision="round_trip")] if final.exists() else []
    frames.extend(pd.read_csv(path, float_precision="round_trip") for path in paths)
    if not frames:
        raise SystemExit(f"no LOHO result files found in {output_dir}")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["fold", "config", "seed"], keep="last")
    merged = merged.sort_values(["fold", "config", "seed"])
    expected = {
        (fold, config, seed)
        for fold in range(LOHO_FOLDS)
        for config in LOHO_ARMS
        for seed in LOHO_SEEDS
    }
    observed = {
        (int(row.fold), str(row.config), int(row.seed))
        for row in merged.itertuples(index=False)
    }
    matrix_errors: list[str] = []
    if len(merged) != LOHO_EXPECTED_ROWS or observed != expected:
        matrix_errors.append(
            f"expected exactly {LOHO_EXPECTED_ROWS} fold/config/seed rows, observed {len(merged)}"
        )
    if not set(merged["config"].astype(str)).issubset(LOHO_ARMS):
        matrix_errors.append("merged LOHO matrix contains an unregistered arm")
    merged.to_csv(final, index=False, float_format=CSV_FLOAT_FORMAT)
    summary = (
        merged.groupby(["fold", "config"], sort=True)
        .agg(
            n_seeds=("seed", "size"),
            mean_skill=("skill", "mean"),
            seed_std=("skill", "std"),
            n_train_stays=("n_train_stays", "first"),
            n_val_stays=("n_val_stays", "first"),
            n_test_stays=("n_test_stays", "first"),
            n_held_hospitals=("n_held_hospitals", "first"),
            held_staleness_median=("held_staleness_median", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "loho_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    summary_json_path = output_dir / "loho_summary.json"
    summary_json: dict[str, Any] = {}
    if summary_json_path.exists():
        payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            summary_json = payload
    summary_json.update(
        {
            "status": "complete" if not matrix_errors else "incomplete",
            "completed_rows": int(len(merged)),
            "expected_rows": LOHO_EXPECTED_ROWS,
            "matrix_errors": matrix_errors,
        }
    )
    summary_json_path.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if clean_shards and not matrix_errors:
        for path in paths:
            path.unlink()
    if matrix_errors:
        raise SystemExit("LOHO merge incomplete: " + "; ".join(matrix_errors))
    print(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eicu_loho.yaml")
    parser.add_argument("--output-dir", default="results/phase3_eicu/e_loho")
    parser.add_argument("--device")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--clean-shards", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        _merge(output_dir, clean_shards=args.clean_shards)
        return
    cfg = _load_yaml(Path(args.config))
    if args.device:
        cfg["device"] = args.device
    _assert_config(cfg)
    base, hospital_counts = _load_full_bundle(cfg)
    assignment, fold_rows = build_hospital_folds(
        hospital_counts,
        n_folds=int(cfg["loho_folds"]),
        seed=int(cfg["loho_seed"]),
    )
    if args.shard_index == 0:
        write_config_snapshot(output_dir, cfg)
        pd.DataFrame(fold_rows).to_csv(
            output_dir / "loho_folds.csv", index=False, float_format=CSV_FLOAT_FORMAT
        )
        fold_summary = []
        for fold in range(LOHO_FOLDS):
            held = [hospital for hospital, assigned in assignment.items() if assigned == fold]
            fold_summary.append(
                {
                    "fold": fold,
                    "n_held_hospitals": len(held),
                    "n_held_stays": int(sum(hospital_counts[hospital] for hospital in held)),
                    "held_hospitals": held,
                }
            )
        (output_dir / "loho_summary.json").write_text(
            json.dumps(
                {
                    "protocol": "E-LOHO",
                    "status": "running",
                    "n_hospitals": len(hospital_counts),
                    "n_stays": int(sum(hospital_counts.values())),
                    "n_folds": LOHO_FOLDS,
                    "fold_seed": LOHO_SEED,
                    "minimum_hospital_stays": LOHO_MIN_HOSPITAL_STAYS,
                    "folds": fold_summary,
                    "no_threshold": True,
                    "standardizer": "frozen MIMIC-IV train split; no held-out hospital data used",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    jobs = []
    for fold in range(LOHO_FOLDS):
        for name, arm in LOHO_ARMS.items():
            for seed in LOHO_SEEDS:
                jobs.append((fold, name, arm, seed))
    jobs = jobs[args.shard_index :: args.num_shards]
    raw_path = output_dir / (
        "raw_results.csv"
        if int(args.num_shards) == 1
        else f"raw_results_shard_{int(args.shard_index)}.csv"
    )
    existing: set[tuple[int, str, int]] = set()
    if args.resume and raw_path.exists():
        prior = pd.read_csv(raw_path, float_precision="round_trip")
        existing = {
            (int(row.fold), str(row.config), int(row.seed))
            for row in prior.itertuples(index=False)
        }
    for fold, name, arm, seed in jobs:
        if (fold, name, seed) in existing:
            continue
        bundle, metadata = _fold_bundle(base, assignment, fold)
        run_cfg = dict(cfg)
        run_cfg.update(
            {
                "head_type": "residual",
                "encoder_type": "identity",
                "grouping": "shared",
                "gate_use_staleness": True,
                "selection_variables": REGISTERED_SCORING_VARIABLES,
                "test_variables": REGISTERED_SCORING_VARIABLES,
                "held_out_variables": None,
                "recency_x_range": tuple(cfg.get("recency_x_range", (0.0, 4.0))),
            }
        )
        run_cfg.update(dict(arm))
        result = train_forecast_run(
            bundle,
            encoder_type="identity",
            seed=seed,
            cfg=run_cfg,
            output_dir=None,
        )
        held_median = _save_run(
            result,
            output_dir,
            fold,
            name,
            seed,
            base.standardizer.n_variables,
        )
        row = _result_row(result, name, arm, run_cfg, metadata, held_median)
        _write_rows(raw_path, [row])
        print(
            f"fold={fold} {name} seed={seed} skill={float(result['skill']):.8f}",
            flush=True,
        )
    print(raw_path)


if __name__ == "__main__":
    main()
