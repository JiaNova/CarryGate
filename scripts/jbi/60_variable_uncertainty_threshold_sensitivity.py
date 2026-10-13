#!/usr/bin/env python
"""Registered post hoc per-variable uncertainty and threshold analysis (R5).

This reads frozen Track B evaluation outputs and the 90 frozen P2 prediction
files. It does not train models, choose checkpoints, or change thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi" / "revision_sensitivity"
PAPER_OUT = ROOT / "results" / "paper"
PLAN = ROOT / "paper" / "JAMIA修订补充分析计划.md"
PLAN_MANIFEST = OUT / "analysis_plan_manifest.json"
PREFLIGHT = OUT / "r5_preflight_manifest.json"
R4_SCRIPT = ROOT / "scripts" / "jbi" / "59_objective_metric_matrix.py"
R4_PREFLIGHT = OUT / "r4_preflight_manifest.json"
R4_MANIFEST = OUT / "r4_manifest.json"
P2_LEDGER = ROOT / "results" / "jbi" / "p2_loho_test_inference_ledger.json"
P2_SAMPLE = ROOT / "results" / "jbi" / "p2_training_sample_manifest.json"
P2_CACHE = ROOT / "data" / "processed" / "eicu_loho_events.pkl"
P2_FOLD_MAP = ROOT / "results" / "phase3_eicu" / "e_loho" / "loho_folds.csv"
P2_OVERLAP = ROOT / "results" / "jbi" / "p2_hospital_overlap_audit.json"

EXPECTED_PLAN_SHA = "fade749e20c54e9340dbbb01ec9a6c8e12a8ea912eb34baec7ccced30e38e320"
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261012
EVENT_THRESHOLDS = (50, 100, 200)
VARIABLES = 37
UNSCORED = frozenset({6, 19, 32})
SCORED_VARIABLES = tuple(v for v in range(VARIABLES) if v not in UNSCORED)
TRACKB_COHORTS = ("eicu", "mimiciv")
TRACKB_MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")
P2_MODELS = TRACKB_MODELS[:3]
OBJECTIVES = ("MSE", "MAE")
OBJECTIVE_LABEL = {"MSE": "MSE", "MAE": "masked L1"}
DISPLAY_MODEL = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi",
                 "HyperIMTS": "HyperIMTS", "Warpformer": "Warpformer"}
TRACKB_RUN_MANIFESTS = {
    "eicu": ROOT / "results" / "jbi" / "trackb_controls" / "run_manifest.json",
    "mimiciv": ROOT / "results" / "jbi" / "trackb_controls_mimiciv" / "run_manifest.json",
}
TRACKB_SUMMARIES = {
    "eicu": ROOT / "results" / "jbi" / "trackb_controls" / "e2_trackb_controls_summary.csv",
    "mimiciv": ROOT / "results" / "jbi" / "trackb_controls_mimiciv" / "e2_trackb_controls_summary.csv",
}
OUTPUTS = {
    "variables": OUT / "r5_variable_metrics.csv",
    "thresholds": OUT / "r5_threshold_sensitivity.csv",
    "evidence": OUT / "r5_evidence_summary.csv",
    "report": OUT / "r5_report.json",
    "forest_pdf": PAPER_OUT / "F11_r5_trackb_variable_forest.pdf",
    "forest_png": PAPER_OUT / "F11_r5_trackb_variable_forest_page1.png",
}
OUTPUTS["manifest"] = OUT / "r5_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def r4_module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("registered_r4_for_r5", R4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R4 reader helpers: {R4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extra_input_paths() -> list[Path]:
    return [
        PLAN, PLAN_MANIFEST, R4_SCRIPT, R4_PREFLIGHT, R4_MANIFEST,
        ROOT / "scripts" / "jbi" / "23_adapter_vs_external_paired.py",
        ROOT / "scripts" / "d1_harness" / "rescore_macro.py",
        ROOT / "src" / "data" / "p12_events.py",
        P2_LEDGER, P2_SAMPLE, P2_CACHE, P2_FOLD_MAP, P2_OVERLAP,
        *TRACKB_RUN_MANIFESTS.values(), *TRACKB_SUMMARIES.values(),
    ]


def _input_hashes_from_r4() -> tuple[list[dict[str, Any]], list[str]]:
    """Revalidate every R4 frozen input, then add R5's training-frequency sources."""
    failures: list[str] = []
    r4_preflight = json.loads(R4_PREFLIGHT.read_text(encoding="utf-8"))
    if r4_preflight.get("status") != "PASS" or r4_preflight.get("input_hash_count") != 706:
        return [], ["R4 preflight is not PASS with the registered 706 source hashes"]
    if r4_preflight.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA:
        failures.append("R4 preflight refers to another analysis plan")
    if r4_preflight.get("script_sha256") != sha256_file(R4_SCRIPT):
        failures.append("R4 reader script changed after its source preflight")

    by_path: dict[str, dict[str, Any]] = {}
    for old in r4_preflight.get("input_hashes", []):
        relative = old["path"]
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing R4-frozen source: {relative}")
            continue
        stat = path.stat()
        digest = sha256_file(path)
        if (stat.st_size != old.get("size_bytes")
                or stat.st_mtime_ns != old.get("mtime_ns")
                or digest != old.get("sha256")):
            failures.append(f"R4-frozen source changed: {relative}")
            continue
        by_path[relative] = {
            "path": relative, "role": old.get("role", "R4-frozen input"),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }

    for path in _extra_input_paths():
        relative = str(path.relative_to(ROOT))
        if relative in by_path:
            continue
        if not path.is_file():
            failures.append(f"missing R5-specific input: {relative}")
            continue
        stat = path.stat()
        by_path[relative] = {
            "path": relative, "role": "R5-specific frozen input",
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return [by_path[key] for key in sorted(by_path)], failures


def preflight() -> dict[str, Any]:
    started = time.time()
    failures: list[str] = []
    if not PLAN.is_file() or sha256_file(PLAN) != EXPECTED_PLAN_SHA:
        failures.append("registered analysis plan SHA256 does not match")
    if not PLAN_MANIFEST.is_file():
        failures.append("registered analysis plan manifest is missing")
    else:
        plan_manifest = json.loads(PLAN_MANIFEST.read_text(encoding="utf-8"))
        if (plan_manifest.get("status") != "RULES_FROZEN_BEFORE_NEW_COMPUTATION"
                or plan_manifest.get("plan_sha256") != EXPECTED_PLAN_SHA
                or "R5" not in plan_manifest.get("frozen_tasks", [])):
            failures.append("R5 is not listed in the immutable analysis-plan registration")

    r4 = r4_module()
    plan_counts = {"trackb_cells": 0, "trackb_eval_directories": 0,
                   "p2_prediction_files": 0, "p2_cells": 0, "variable_families": 0}
    for cohort in TRACKB_COHORTS:
        manifest_path = TRACKB_RUN_MANIFESTS[cohort]
        if not manifest_path.is_file():
            failures.append(f"missing Track B run manifest: {manifest_path.relative_to(ROOT)}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                matches = [row for row in runs if row.get("phase") == "factorial"
                           and row.get("model") == model and row.get("loss") == objective
                           and row.get("selection") == "macro_persistence"]
                if len(matches) != 1:
                    failures.append(f"{cohort}/{model}/{objective}: expected one frozen macro run")
                    continue
                directories = r4.reader_module()._eval_dirs(model, matches[0]["tag"], cohort)
                plan_counts["trackb_cells"] += 1
                plan_counts["trackb_eval_directories"] += len(directories)
                if len(directories) != 5:
                    failures.append(f"{cohort}/{model}/{objective}: expected 5 eval directories")
                for directory in directories:
                    for filename in ("input_y_mask.npy", "output_pred.npy"):
                        if not (directory / filename).is_file():
                            failures.append(f"missing frozen prediction input: {directory / filename}")

    if not P2_LEDGER.is_file():
        failures.append("P2 test-inference ledger is missing")
    else:
        ledger = json.loads(P2_LEDGER.read_text(encoding="utf-8"))
        results = ledger.get("run_results", {})
        plan_counts["p2_prediction_files"] = sum(
            1 for row in results.values() if row.get("status") == "completed")
        plan_counts["p2_cells"] = 5 * len(P2_MODELS) * len(OBJECTIVES)
        if ledger.get("status") != "COMPLETE" or len(results) != 90:
            failures.append("P2 ledger is not the complete frozen 90-run ledger")
        if plan_counts["p2_prediction_files"] != 90:
            failures.append("P2 completed prediction count differs from 90")

    if P2_SAMPLE.is_file():
        sample = json.loads(P2_SAMPLE.read_text(encoding="utf-8"))
        if (sample.get("status") != "PASS"
                or sample.get("analysis_plan_sha256") is None
                or len(sample.get("folds", {})) != 5
                or any(sample["folds"][str(f)].get("train_count") != 8400 for f in range(5))):
            failures.append("P2 frozen 8,400-stay-per-fold training sample manifest is invalid")

    plan_counts["variable_families"] = (2 * len(TRACKB_MODELS) * len(OBJECTIVES)
                                         + 5 * len(P2_MODELS) * len(OBJECTIVES))
    inputs, input_failures = _input_hashes_from_r4()
    failures.extend(input_failures)
    if len(inputs) < 706:
        failures.append(f"only {len(inputs)} frozen input hashes available; expected at least 706")
    expected_counts = {"trackb_cells": 16, "trackb_eval_directories": 80,
                       "p2_prediction_files": 90, "p2_cells": 30,
                       "variable_families": 46}
    if plan_counts != expected_counts:
        failures.append(f"analysis input cardinality mismatch: {plan_counts}")
    return {
        "schema_version": "jamia_r5_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "preflighted_at_utc": utc_now(),
        "analysis_plan_sha256": EXPECTED_PLAN_SHA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "counts": plan_counts,
        "input_hash_count": len(inputs),
        "input_hashes": inputs,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _archive_existing() -> None:
    existing = [path for path in [*OUTPUTS.values(), PREFLIGHT] if path.exists()]
    if not existing:
        return
    archive = ROOT / "results" / "jbi" / "_archive" / (
        "r5_variable_uncertainty_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    for path in existing:
        dest = archive / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    write_json(archive / "archive_manifest.json", {
        "archived_at_utc": utc_now(), "reason": "--refresh-generated requested for R5",
        "files": [str(path.relative_to(ROOT)) for path in existing],
    })


def _training_frequency_trackb(cohort: str) -> tuple[np.ndarray, list[str]]:
    from scripts.d1_harness.rescore_macro import VARIANT_CACHES

    events_name, split_name = VARIANT_CACHES["jbi_patient"][cohort]
    with (ROOT / "data" / "processed" / events_name).open("rb") as handle:
        payload = pickle.load(handle)
    split = json.loads((ROOT / "data" / "processed" / split_name).read_text(encoding="utf-8"))
    examples = payload["examples"]
    counts = np.zeros(VARIABLES, dtype=np.int64)
    for index in split["train"]:
        vars_for_stay = np.asarray(examples[int(index)]["target"]["var"], dtype=np.int64)
        counts += np.bincount(vars_for_stay, minlength=VARIABLES)[:VARIABLES]
    names = list(payload.get("variable_names", []))
    if len(names) != VARIABLES:
        raise AssertionError(f"{cohort}: expected {VARIABLES} frozen variable names")
    return counts, names


def _training_frequency_p2(fold: int, payload: dict[str, Any], sample: dict[str, Any]
                           ) -> tuple[np.ndarray, list[str]]:
    examples = payload["examples"]
    indices = sample["folds"][str(fold)]["train_stay_indices"]
    counts = np.zeros(VARIABLES, dtype=np.int64)
    for index in indices:
        vars_for_stay = np.asarray(examples[int(index)]["target"]["var"], dtype=np.int64)
        counts += np.bincount(vars_for_stay, minlength=VARIABLES)[:VARIABLES]
    names = list(payload.get("variable_names", []))
    if len(names) != VARIABLES:
        raise AssertionError(f"P2 fold{fold}: expected {VARIABLES} frozen variable names")
    return counts, names


def _bootstrap_group(
    *, source: str, cohort: str, fold: int | None,
    cells: dict[tuple[str, str], dict[str, Any]],
    baseline: np.ndarray, counts: np.ndarray, variable_names: list[str],
    train_frequency: np.ndarray, seed_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    n_patients = int(baseline.shape[0])
    if baseline.shape != counts.shape or baseline.shape[1] != VARIABLES:
        raise AssertionError(f"{source}/{cohort}/{fold}: baseline/count shape mismatch")
    keys = [(model, objective) for model in (
        TRACKB_MODELS if source == "Track B" else P2_MODELS) for objective in OBJECTIVES]
    errors = []
    for key in keys:
        if key not in cells:
            raise AssertionError(f"{source}/{cohort}/{fold}: missing arm {key}")
        error = np.asarray(cells[key]["abs_error"], dtype=np.float64)
        if error.shape != baseline.shape or not np.isfinite(error).all():
            raise AssertionError(f"{source}/{cohort}/{fold}/{key}: error grid invalid")
        errors.append(error)
    error_cube = np.stack(errors, axis=-1)
    draws = np.empty((BOOTSTRAP_REPS, n_patients), dtype=np.int16)
    for b in range(BOOTSTRAP_REPS):
        draws[b] = np.bincount(rng.integers(0, n_patients, n_patients),
                               minlength=n_patients).astype(np.int16)
    boot_counts = draws @ np.asarray(counts, dtype=np.float64)
    boot_baseline = draws @ np.asarray(baseline, dtype=np.float64)
    boot_error = (draws @ error_cube.reshape(n_patients, -1)).reshape(
        BOOTSTRAP_REPS, VARIABLES, len(keys))
    point_counts = np.asarray(counts, dtype=np.float64).sum(axis=0)
    point_baseline = np.asarray(baseline, dtype=np.float64).sum(axis=0)
    scored = np.asarray([i not in UNSCORED for i in range(VARIABLES)], dtype=bool)
    point_baseline_ok = point_baseline > 1e-9
    rows: list[dict[str, Any]] = []
    for k, (model, objective) in enumerate(keys):
        point_error = error_cube[:, :, k].sum(axis=0)
        point_skill = np.divide(point_error, point_baseline,
                                out=np.full(VARIABLES, np.nan),
                                where=point_baseline_ok)
        point_skill = 1.0 - point_skill
        boot_skill = np.divide(
            boot_error[:, :, k], boot_baseline,
            out=np.full((BOOTSTRAP_REPS, VARIABLES), np.nan),
            where=boot_baseline > 1e-9,
        )
        boot_skill = 1.0 - boot_skill
        for variable in SCORED_VARIABLES:
            values = boot_skill[:, variable]
            finite = values[np.isfinite(values)]
            if finite.size:
                ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
                tail_p = (1.0 + float(np.count_nonzero(finite >= 0.0))) / (BOOTSTRAP_REPS + 1.0)
            else:
                ci_low = ci_high = float("nan")
                tail_p = 1.0
            skill = float(point_skill[variable])
            rows.append({
                "source": source, "cohort": cohort, "fold": fold,
                "model": model, "model_display": DISPLAY_MODEL[model],
                "objective": objective, "objective_label": OBJECTIVE_LABEL[objective],
                "variable_index": variable, "variable_name": variable_names[variable],
                "training_measurement_count": int(train_frequency[variable]),
                "test_event_count": int(point_counts[variable]),
                "point_model_abs_error_sum": float(point_error[variable]),
                "carry_forward_abs_error_sum": float(point_baseline[variable]),
                "point_skill": skill,
                "estimability_status": ("ESTIMABLE" if point_baseline_ok[variable]
                                        else "UNDEFINED_ZERO_CARRY_FORWARD_ERROR"),
                "ci_low": float(ci_low), "ci_high": float(ci_high),
                "ci_valid_replicates": int(finite.size),
                "point_below_carry_forward": bool(np.isfinite(skill) and skill < 0.0),
                "ci_entirely_below_zero": bool(np.isfinite(ci_high) and ci_high < 0.0),
                "negative_bootstrap_tail_p": float(tail_p),
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "patient",
                "seed_count": seed_count,
                "variable_in_original_scored_set": bool(scored[variable]),
            })
    return rows


def _bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH inputs must be finite p-values in [0,1]")
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.minimum(adjusted, 1.0)
    return q


def _attach_fdr(variable_frame: pd.DataFrame) -> pd.DataFrame:
    frame = variable_frame.copy()
    frame["bh_fdr_family_n"] = 34
    frame["bh_negative_fdr_q_lt_0_05"] = False
    family_keys = ["source", "cohort", "fold", "model", "objective"]
    for _, indices in frame.groupby(family_keys, dropna=False, sort=False).groups.items():
        ordered_indices = list(indices)
        if len(ordered_indices) != len(SCORED_VARIABLES):
            raise AssertionError(f"R5 FDR family has {len(ordered_indices)} rows, expected 34")
        q = _bh_qvalues(frame.loc[ordered_indices, "negative_bootstrap_tail_p"].to_numpy(float))
        frame.loc[ordered_indices, "bh_fdr_q"] = q
        frame.loc[ordered_indices, "bh_negative_fdr_q_lt_0_05"] = (
            (q < 0.05) & (frame.loc[ordered_indices, "point_skill"].to_numpy(float) < 0.0))
    return frame


def _ordered_names(group: pd.DataFrame, by: str) -> tuple[list[str], dict[str, int], dict[str, str]]:
    records = group[[by, "macro_skill", "model_display"]].drop_duplicates()
    if records.duplicated(by).any():
        raise AssertionError(f"duplicate ranking entries for {by}")
    records = records.sort_values(["macro_skill", "model_display"],
                                  ascending=[False, True], kind="mergesort")
    values = records.set_index(by)["macro_skill"].to_dict()
    names = records.set_index(by)["model_display"].to_dict()
    ordered = records[by].astype(str).tolist()
    ranks: dict[str, int] = {}
    tied_labels: dict[str, str] = {}
    for item in ordered:
        ranks[item] = 1 + sum(float(values[other]) > float(values[item]) for other in values)
        tied = sorted(((names[other] if by == "model" else OBJECTIVE_LABEL[str(other)])
                       for other in values
                       if float(values[other]) == float(values[item])))
        tied_labels[item] = " = ".join(tied)
    return ordered, ranks, tied_labels


def build_threshold_table(variable_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_groups = variable_frame[["source", "cohort", "fold"]].drop_duplicates()
    for group_row in source_groups.itertuples(index=False):
        group = variable_frame[(variable_frame.source == group_row.source)
                               & (variable_frame.cohort == group_row.cohort)
                               & ((variable_frame.fold.isna()) if pd.isna(group_row.fold)
                                  else (variable_frame.fold == group_row.fold))]
        for threshold in EVENT_THRESHOLDS:
            for model in (TRACKB_MODELS if group_row.source == "Track B" else P2_MODELS):
                for objective in OBJECTIVES:
                    cell = group[(group.model == model) & (group.objective == objective)]
                    eligible = ((cell.test_event_count.to_numpy(int) >= threshold)
                                & (cell.carry_forward_abs_error_sum.to_numpy(float) > 1e-9)
                                & np.isfinite(cell.point_skill.to_numpy(float)))
                    values = cell.point_skill.to_numpy(float)
                    macro = float(values[eligible].mean()) if eligible.any() else float("nan")
                    rows.append({
                        "source": group_row.source, "cohort": group_row.cohort,
                        "fold": group_row.fold, "event_threshold": threshold,
                        "model": model, "model_display": DISPLAY_MODEL[model],
                        "objective": objective, "objective_label": OBJECTIVE_LABEL[objective],
                        "eligible_variable_count": int(eligible.sum()),
                        "macro_skill": macro,
                        "point_below_carry_forward_variable_count": int(
                            np.count_nonzero(eligible & (values < 0.0))),
                    })
    frame = pd.DataFrame(rows)
    frame["architecture_rank_within_objective"] = pd.Series(dtype="Int64")
    frame["architecture_tie_names"] = ""
    frame["architecture_order"] = ""
    frame["objective_rank_within_model"] = pd.Series(dtype="Int64")
    frame["objective_tie_labels"] = ""
    frame["objective_order"] = ""
    arch_group_cols = ["source", "cohort", "fold", "event_threshold", "objective"]
    for _, indices in frame.groupby(arch_group_cols, dropna=False, sort=False).groups.items():
        idx = list(indices)
        order, rank, ties = _ordered_names(frame.loc[idx], "model")
        order_text = " > ".join(DISPLAY_MODEL[item] for item in order)
        for i in idx:
            model = str(frame.at[i, "model"])
            frame.at[i, "architecture_rank_within_objective"] = rank[model]
            frame.at[i, "architecture_tie_names"] = ties[model]
            frame.at[i, "architecture_order"] = order_text
    obj_group_cols = ["source", "cohort", "fold", "event_threshold", "model"]
    for _, indices in frame.groupby(obj_group_cols, dropna=False, sort=False).groups.items():
        idx = list(indices)
        order, rank, ties = _ordered_names(frame.loc[idx], "objective")
        order_text = " > ".join(OBJECTIVE_LABEL[item] for item in order)
        for i in idx:
            objective = str(frame.at[i, "objective"])
            frame.at[i, "objective_rank_within_model"] = rank[objective]
            frame.at[i, "objective_tie_labels"] = ties[objective]
            frame.at[i, "objective_order"] = order_text

    rank_groups: dict[str, list[str]] = {}
    changed: list[dict[str, Any]] = []
    for _, part in frame.groupby(["source", "cohort", "fold", "objective"],
                                 dropna=False, sort=False):
        baseline = part[part.event_threshold == 50].iloc[0]["architecture_order"]
        keyrow = part.iloc[0]
        group_key = f"{keyrow['source']}:{keyrow['cohort']}:fold{keyrow['fold']}:{keyrow['objective']}"
        for threshold in (100, 200):
            order = part.loc[part.event_threshold == threshold, "architecture_order"].iloc[0]
            is_changed = order != baseline
            rank_groups[f"{group_key}:threshold{threshold}"] = [order]
            if is_changed:
                changed.append({"source": keyrow["source"], "cohort": keyrow["cohort"],
                                "fold": None if pd.isna(keyrow["fold"]) else int(keyrow["fold"]),
                                "objective": keyrow["objective"], "event_threshold": threshold,
                                "order_at_50": baseline, "order_at_threshold": order})
    frame["architecture_order_changed_from_50"] = False
    for index, row in frame.iterrows():
        baseline = frame[(frame.source == row.source) & (frame.cohort == row.cohort)
                         & (frame.fold.isna() if pd.isna(row.fold) else frame.fold == row.fold)
                         & (frame.objective == row.objective)
                         & (frame.event_threshold == 50)].architecture_order.iloc[0]
        frame.at[index, "architecture_order_changed_from_50"] = (
            row.architecture_order != baseline)
    summary = {
        "rank_comparisons": len(rank_groups),
        "rank_changes_from_50": changed,
        "rank_change_count": len(changed),
    }
    return frame, summary


def build_evidence_summary(variable_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["source", "cohort", "fold", "model", "objective"]
    for keys, group in variable_frame.groupby(group_keys, dropna=False, sort=False):
        eligible50 = ((group.test_event_count >= 50)
                      & (group.carry_forward_abs_error_sum > 1e-9)
                      & np.isfinite(group.point_skill))
        rows.append({
            "source": keys[0], "cohort": keys[1], "fold": keys[2],
            "model": keys[3], "model_display": DISPLAY_MODEL[keys[3]],
            "objective": keys[4], "objective_label": OBJECTIVE_LABEL[keys[4]],
            "fixed_scored_variable_count": len(group),
            "variables_eligible_at_50_events": int(eligible50.sum()),
            "point_negative_at_50_events": int(
                (eligible50 & (group.point_skill < 0)).sum()),
            "ci_upper_below_zero_at_50_events": int(
                (eligible50 & group.ci_entirely_below_zero).sum()),
            "bh_fdr_negative_at_50_events": int(
                (eligible50 & group.bh_negative_fdr_q_lt_0_05).sum()),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "fdr_interpretation": "descriptive bootstrap tail BH; not confirmatory",
        })
    return pd.DataFrame(rows)


def make_forest(variable_frame: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    trackb = variable_frame[variable_frame.source == "Track B"]
    x_limits = (-1.5, 1.0)
    colors = {"MSE": "#2667A8", "MAE": "#C95B39"}
    markers = {"MSE": "s", "MAE": "o"}
    names_by_cohort = {
        cohort: (trackb[trackb.cohort == cohort]
                 .sort_values(["training_measurement_count", "variable_name"],
                              ascending=[False, True])
                 [["variable_index", "variable_name", "training_measurement_count"]]
                 .drop_duplicates("variable_index")
                 .sort_values(["training_measurement_count", "variable_name"],
                              ascending=[False, True]))
        for cohort in TRACKB_COHORTS
    }
    with PdfPages(path_pdf) as pdf:
        page_number = 0
        for cohort in TRACKB_COHORTS:
            labels = names_by_cohort[cohort]
            for model in TRACKB_MODELS:
                page_number += 1
                fig, ax = plt.subplots(figsize=(8.5, 11.0))
                ax.set_position([0.34, 0.17, 0.61, 0.68])
                y = np.arange(len(labels), dtype=float)
                for objective, offset in (("MSE", -0.12), ("MAE", 0.12)):
                    subset = trackb[(trackb.cohort == cohort)
                                    & (trackb.model == model)
                                    & (trackb.objective == objective)].set_index("variable_index")
                    for row_i, item in enumerate(labels.itertuples(index=False)):
                        row = subset.loc[int(item.variable_index)]
                        yy = y[row_i] + offset
                        ax.hlines(yy, row.ci_low, row.ci_high, color=colors[objective],
                                  linewidth=1.0, alpha=0.9)
                        ax.vlines([row.ci_low, row.ci_high], yy - 0.035, yy + 0.035,
                                  color=colors[objective], linewidth=0.8)
                        ax.scatter(row.point_skill, yy, s=15, marker=markers[objective],
                                   color=colors[objective], zorder=3, linewidths=0.25,
                                   edgecolors="white")
                        if row.ci_low < x_limits[0]:
                            ax.annotate("", xy=(x_limits[0], yy),
                                        xytext=(x_limits[0] + 0.10, yy),
                                        arrowprops={"arrowstyle": "-|>", "color": colors[objective],
                                                    "lw": 0.8}, clip_on=True)
                        if row.ci_high > x_limits[1]:
                            ax.annotate("", xy=(x_limits[1], yy),
                                        xytext=(x_limits[1] - 0.10, yy),
                                        arrowprops={"arrowstyle": "-|>", "color": colors[objective],
                                                    "lw": 0.8}, clip_on=True)
                ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
                ylabels = [f"{row.variable_name}  (train n={row.training_measurement_count:,})"
                           for row in labels.itertuples(index=False)]
                ax.set_yticks(y, ylabels, fontsize=6.2)
                ax.invert_yaxis()
                ax.set_xlim(*x_limits)
                ax.set_xlabel("Per-variable absolute-error persistence skill", fontsize=9)
                cohort_name = "eICU-CRD" if cohort == "eicu" else "MIMIC-IV"
                fig.text(0.34, 0.955, f"{cohort_name} — {DISPLAY_MODEL[model]}",
                         fontsize=11, weight="semibold")
                fig.text(0.34, 0.93,
                         "Track B; variables ordered by training-set target-event frequency",
                         fontsize=8)
                ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
                ax.spines[["top", "right", "left"]].set_visible(False)
                legend = [Line2D([0], [0], marker=markers[obj], color=colors[obj],
                                 label=OBJECTIVE_LABEL[obj], linestyle="-", markersize=5,
                                 linewidth=1.0) for obj in OBJECTIVES]
                fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.95, 0.08),
                           frameon=False, fontsize=8, ncol=2)
                fig.text(0.34, 0.02,
                         "Higher is better; 0 equals carry-forward. Bars show patient-bootstrap 95% CIs. "
                         "Arrows mark intervals extending beyond the display range; exact values are in Data S1.",
                         fontsize=6.4)
                fig.text(0.94, 0.97, f"S14 page {page_number}/8", ha="right", fontsize=7)
                pdf.savefig(fig, bbox_inches="tight")
                if page_number == 1:
                    fig.savefig(path_png, dpi=180, bbox_inches="tight")
                plt.close(fig)


def run_analysis() -> dict[str, Any]:
    started = time.time()
    r4 = r4_module()
    reader = r4.reader_module()
    run_manifests = {cohort: json.loads(path.read_text(encoding="utf-8"))
                     for cohort, path in TRACKB_RUN_MANIFESTS.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_rows: list[dict[str, Any]] = []

    for cohort in TRACKB_COHORTS:
        ref = r4.build_trackb_reference(cohort, reader)
        train_frequency, variable_names = _training_frequency_trackb(cohort)
        if len(variable_names) != VARIABLES:
            raise AssertionError(f"{cohort}: variable-name cardinality mismatch")
        cells: dict[tuple[str, str], dict[str, Any]] = {}
        for model in TRACKB_MODELS:
            for objective in OBJECTIVES:
                matches = [row for row in run_manifests[cohort].get("runs", [])
                           if row.get("phase") == "factorial" and row.get("model") == model
                           and row.get("loss") == objective
                           and row.get("selection") == "macro_persistence"]
                if len(matches) != 1:
                    raise AssertionError(f"{cohort}/{model}/{objective}: frozen run is ambiguous")
                cell = r4.load_trackb_cell(cohort, model, objective,
                                           matches[0]["tag"], ref, reader)
                if cell["n_seeds"] != 5:
                    raise AssertionError(f"{cohort}/{model}/{objective}: expected five seeds")
                cells[(model, objective)] = cell
        baseline = cells[(TRACKB_MODELS[0], OBJECTIVES[0])]["baseline_abs"]
        counts = cells[(TRACKB_MODELS[0], OBJECTIVES[0])]["counts"]
        for cell in cells.values():
            if not np.array_equal(cell["counts"], counts) or not np.array_equal(
                    cell["baseline_abs"], baseline):
                raise AssertionError(f"{cohort}: Track B fixed test denominator differs across arms")
        all_rows.extend(_bootstrap_group(
            source="Track B", cohort=cohort, fold=None, cells=cells,
            baseline=baseline, counts=counts, variable_names=variable_names,
            train_frequency=train_frequency, seed_count=5, rng=rng,
        ))

    ledger = json.loads(P2_LEDGER.read_text(encoding="utf-8"))
    sample = json.loads(P2_SAMPLE.read_text(encoding="utf-8"))
    with P2_CACHE.open("rb") as handle:
        p2_payload = pickle.load(handle)
    p2_names_reference: list[str] | None = None
    for fold in range(5):
        cells, baseline_abs, _, counts, _, meta = r4.load_p2_fold(fold, ledger, {})
        train_frequency, variable_names = _training_frequency_p2(fold, p2_payload, sample)
        if p2_names_reference is None:
            p2_names_reference = variable_names
        elif variable_names != p2_names_reference:
            raise AssertionError("P2 variable-name order changes across folds")
        if len(meta["patient_labels"]) != baseline_abs.shape[0]:
            raise AssertionError(f"P2 fold{fold}: patient rows differ from frozen baseline")
        all_rows.extend(_bootstrap_group(
            source="P2 eICU", cohort="eICU held-hospital", fold=fold,
            cells=cells, baseline=baseline_abs, counts=counts,
            variable_names=variable_names, train_frequency=train_frequency,
            seed_count=3, rng=rng,
        ))

    variable_frame = pd.DataFrame(all_rows)
    variable_frame = _attach_fdr(variable_frame)
    threshold_frame, ranking_summary = build_threshold_table(variable_frame)
    evidence_frame = build_evidence_summary(variable_frame)
    if len(variable_frame) != 1564 or len(threshold_frame) != 138 or len(evidence_frame) != 46:
        raise AssertionError(
            f"unexpected R5 shapes: variable={len(variable_frame)}, threshold={len(threshold_frame)}, "
            f"evidence={len(evidence_frame)}")
    zero_baseline = variable_frame.carry_forward_abs_error_sum.to_numpy(float) <= 1e-9
    empty_ci = variable_frame.ci_valid_replicates.to_numpy(int) == 0
    if (np.any(empty_ci & ~zero_baseline)
            or np.any((~empty_ci) & zero_baseline)
            or np.any(variable_frame.ci_valid_replicates.to_numpy(int) > BOOTSTRAP_REPS)
            or not (variable_frame.bh_fdr_family_n == 34).all()):
        raise AssertionError("R5 has an empty CI for a positive denominator or malformed FDR family")
    if not threshold_frame.eligible_variable_count.between(0, 34).all():
        raise AssertionError("R5 threshold variable count falls outside [0,34]")

    rank_changed = ranking_summary["rank_changes_from_50"]
    report = {
        "schema_version": "jamia_r5_variable_uncertainty_v1",
        "analysis_type": "post hoc sensitivity analysis; reporting rules fixed before computation",
        "estimand": "per-variable absolute-error persistence skill, 1 - mean-seed patient-summed model error / carry-forward error",
        "bootstrap": {"unit": "patient within cohort or held-hospital fold",
                      "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                      "interval": "percentile 95% CI",
                      "negative_tail_p": "(1 + number of bootstrap skills >= 0)/(B + 1)",
                      "fdr": "Benjamini-Hochberg within cohort/fold x architecture x objective over fixed 34 variables; descriptive only"},
        "thresholds": list(EVENT_THRESHOLDS),
        "threshold_ranking_comparisons": ranking_summary,
        "counts": {
            "trackb_variable_rows": int((variable_frame.source == "Track B").sum()),
            "p2_variable_rows": int((variable_frame.source == "P2 eICU").sum()),
            "variable_rows": len(variable_frame), "threshold_rows": len(threshold_frame),
            "evidence_families": len(evidence_frame),
            "trackb_families": 2 * len(TRACKB_MODELS) * len(OBJECTIVES),
            "p2_families": 5 * len(P2_MODELS) * len(OBJECTIVES),
            "bootstrap_reps": BOOTSTRAP_REPS,
        },
        "evidence_counts_at_original_50_event_threshold": {
            "point_negative": int(evidence_frame.point_negative_at_50_events.sum()),
            "ci_entirely_negative": int(evidence_frame.ci_upper_below_zero_at_50_events.sum()),
            "bh_fdr_negative_descriptive": int(evidence_frame.bh_fdr_negative_at_50_events.sum()),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return {"variables": variable_frame, "thresholds": threshold_frame,
            "evidence": evidence_frame, "report": report}


def _publish(result: dict[str, Any], preflight_result: dict[str, Any]) -> None:
    stage = Path(tempfile.mkdtemp(prefix="r5_stage_", dir=OUT))
    try:
        staged = {name: stage / path.name for name, path in OUTPUTS.items() if name != "manifest"}
        staged_manifest = stage / OUTPUTS["manifest"].name
        result["variables"].to_csv(staged["variables"], index=False, float_format="%.17g")
        result["thresholds"].to_csv(staged["thresholds"], index=False, float_format="%.17g")
        result["evidence"].to_csv(staged["evidence"], index=False, float_format="%.17g")
        write_json(staged["report"], result["report"])
        make_forest(result["variables"], staged["forest_pdf"], staged["forest_png"])
        manifest = {
            "schema_version": "jamia_r5_manifest_v1", "status": "COMPLETE",
            "generated_at_utc": utc_now(), "analysis_plan_sha256": EXPECTED_PLAN_SHA,
            "preflight_manifest_sha256": sha256_file(PREFLIGHT),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python_version": sys.version, "numpy_version": np.__version__,
            "pandas_version": pd.__version__, "preflight_input_count": preflight_result["input_hash_count"],
            "post_hoc_status": "post hoc sensitivity analysis; reporting rules fixed before computation",
            "counts": result["report"]["counts"], "outputs": {},
        }
        for name, path in staged.items():
            manifest["outputs"][name] = {"path": str(OUTPUTS[name].relative_to(ROOT)),
                                           "size_bytes": path.stat().st_size,
                                           "sha256": sha256_file(path)}
        write_json(staged_manifest, manifest)
        for name, path in staged.items():
            OUTPUTS[name].parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, OUTPUTS[name])
        os.replace(staged_manifest, OUTPUTS["manifest"])
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--refresh-generated", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("choose exactly one of --preflight-only or --run")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        if PREFLIGHT.exists() and not args.refresh_generated:
            parser.error("R5 preflight exists; use --refresh-generated to archive prior R5 outputs")
        if args.refresh_generated:
            _archive_existing()
        result = preflight()
        write_json(PREFLIGHT, result)
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "input_hash_count": result["input_hash_count"],
                          "failures": result["failures"],
                          "preflight_manifest": str(PREFLIGHT.relative_to(ROOT))},
                         ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    if not PREFLIGHT.is_file():
        raise SystemExit("R5 requires a successful --preflight-only run before computation")
    saved = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    stale = []
    for row in saved.get("input_hashes", []):
        path = ROOT / row["path"]
        if (not path.is_file() or path.stat().st_size != row.get("size_bytes")
                or path.stat().st_mtime_ns != row.get("mtime_ns")):
            stale.append(row["path"])
    if (saved.get("status") != "PASS"
            or saved.get("script_sha256") != sha256_file(Path(__file__).resolve())
            or saved.get("analysis_plan_sha256") != sha256_file(PLAN)
            or stale):
        raise SystemExit("R5 preflight is stale or failed; no metric computation was started")
    if any(path.exists() for key, path in OUTPUTS.items() if key != "manifest"):
        _archive_existing()
    result = run_analysis()
    _publish(result, saved)
    print(json.dumps({"status": "COMPLETE", "report": result["report"],
                      "outputs": {key: str(path.relative_to(ROOT)) for key, path in OUTPUTS.items()},
                      "manifest": json.loads(OUTPUTS["manifest"].read_text(encoding="utf-8"))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
