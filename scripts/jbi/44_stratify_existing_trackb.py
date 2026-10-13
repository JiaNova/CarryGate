#!/usr/bin/env python
"""CPU-only, exploratory strata from the frozen patient-grouped Track-B runs.

This implements P2 plan §6 without retraining.  It validates the saved sample
IDs, outcome masks, target values, model configuration, and five-seed roster
before calculating persistence skill, measurement-time SOFA grade errors, and
measurement-time grade-deterioration detection.  Missing demographic metadata
remain explicit NA strata.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi" / "p2_stratified_breakdown.csv"
AUDIT = ROOT / "results" / "jbi" / "p2_stratified_breakdown_audit.json"
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")
ARMS = ("macro", "mae_macro")
SEEDS = tuple(range(2024, 2029))
SCOREABLE = np.asarray([v for v in range(37) if v not in (6, 19, 32)], dtype=np.int64)
SOFA_COMPONENTS = (
    ("Coagulation", "Platelets", (150.0, 100.0, 50.0, 20.0), "low", None),
    ("Liver", "Bilirubin", (1.2, 2.0, 6.0, 12.0), "high", None),
    ("CNS", "GCS", (15.0, 13.0, 10.0, 6.0), "low", (3.0, 15.0)),
    ("Renal", "Creatinine", (1.2, 2.0, 3.5, 5.0), "high", None),
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACKB = _load_module("p2_strata_trackb", ROOT / "scripts/jbi/23_adapter_vs_external_paired.py")
SOFA = _load_module("p2_strata_sofa", ROOT / "scripts/analyze_sofa_grades.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense_order(time: np.ndarray, variable: np.ndarray) -> np.ndarray:
    """Return original event indices in the external adapter's dense order."""
    order = np.lexsort((variable, time))
    var_s, time_s = variable[order], time[order]
    _, inverse, counts = np.unique(
        np.stack([time_s, var_s]), axis=1, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    group_start = np.searchsorted(inverse, np.arange(counts.size))
    rank = np.arange(inverse.size) - group_start[inverse]
    uniq_time, time_index = np.unique(time_s, return_inverse=True)
    rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
    np.maximum.at(rows_per_time, time_index, rank + 1)
    row_start = np.concatenate(([0], np.cumsum(rows_per_time[:-1])))
    row_of = row_start[time_index] + rank
    dense = np.lexsort((var_s, row_of))
    return order[dense]


def _last_raw_by_variable(example: dict[str, Any], t_obs: float, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    obs = example["obs"]
    var = np.asarray(obs["var"], dtype=np.int64)
    time = np.asarray(obs["time"], dtype=np.float64)
    raw = np.asarray(obs["value_raw"], dtype=np.float64)
    value = np.asarray(obs["value"], dtype=np.float32)
    last_raw = np.full(n_vars, np.nan, dtype=np.float64)
    last_value = np.zeros(n_vars, dtype=np.float32)
    for variable in np.unique(var):
        candidates = np.flatnonzero((var == variable) & (time < float(t_obs)))
        if not candidates.size:
            continue
        chosen = int(candidates[np.argmax(time[candidates])])
        last_raw[int(variable)] = raw[chosen]
        last_value[int(variable)] = value[chosen]
    return last_raw, last_value


def _metadata_group(value: Any) -> str:
    if value is None:
        return "NA (missing)"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NA (missing)"
    if isinstance(value, (int, np.integer)) and int(value) == -1:
        return "NA (missing)"
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null", "-1"} else "NA (missing)"


def _age_group(value: Any) -> str:
    value = _metadata_group(value)
    if value == "NA (missing)":
        return "NA (missing)"
    try:
        age = float(value)
    except ValueError:
        return "NA"
    return "<65" if age < 65 else "≥65"


def _reference(cohort: str) -> dict[str, Any]:
    events_name, split_name = TRACKB.VARIANT_CACHES["jbi_patient"][cohort]
    cache_path = ROOT / "data/processed" / events_name
    split_path = ROOT / "data/processed" / split_name
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("split_unit") != "patient_grouped":
        raise AssertionError(f"{cohort}: expected patient_grouped split")
    test_indices = [int(index) for index in split["test"]]
    train_indices = [int(index) for index in split["train"]]
    variable_names = list(cache["variable_names"])
    name_to_index = {name: i for i, name in enumerate(variable_names)}
    reference = TRACKB._reference(cohort)
    stats = cache["standardizer"]
    n_vars = len(variable_names)

    rows: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "variable", "truth", "truth_raw", "x_last", "input_raw", "stay",
            "patient", "gender", "age_group", "icu_type",
        )
    }
    stay_meta: list[dict[str, Any]] = []
    for position, cache_index in enumerate(test_indices):
        example = cache["examples"][cache_index]
        target = example["target"]
        variable = np.asarray(target["var"], dtype=np.int64)
        time = np.asarray(target["time"], dtype=np.float64)
        truth = np.asarray(target["value"], dtype=np.float32)
        truth_raw = np.asarray(target["value_raw"], dtype=np.float64)
        dense = _dense_order(time, variable)
        input_raw_by_var, _ = _last_raw_by_variable(example, float(example["t_obs"]), n_vars)
        patient = str(example.get(split.get("group_column") or "") or example.get("uniquepid") or example.get("subject_id") or example["record_id"])
        metadata = {
            "stay_position": int(position),
            "record_id": int(example["record_id"]),
            "patient_id": patient,
            "gender": _metadata_group(example.get("gender")),
            "age_group": _age_group(example.get("anchor_age")),
            "icu_type": _metadata_group(example.get("icu_type")),
        }
        stay_meta.append(metadata)
        rows["variable"].append(variable[dense])
        rows["truth"].append(truth[dense])
        rows["truth_raw"].append(truth_raw[dense])
        # x_last is filled from the independently shared Track-B reference below.
        rows["x_last"].append(np.zeros(len(dense), dtype=np.float32))
        rows["input_raw"].append(input_raw_by_var[variable[dense]])
        rows["stay"].append(np.full(len(dense), position, dtype=np.int64))
        rows["patient"].append(np.full(len(dense), patient, dtype=object))
        rows["gender"].append(np.full(len(dense), metadata["gender"], dtype=object))
        rows["age_group"].append(np.full(len(dense), metadata["age_group"], dtype=object))
        rows["icu_type"].append(np.full(len(dense), metadata["icu_type"], dtype=object))

    arrays = {key: np.concatenate(value) for key, value in rows.items()}
    arrays["x_last"] = np.asarray(reference["dense_x_last"], dtype=np.float32)
    if not np.array_equal(arrays["variable"], np.asarray(reference["dense_var"], dtype=np.int64)):
        raise AssertionError(f"{cohort}: dense variable order differs from Track-B reference")
    if not np.allclose(arrays["truth"], reference["dense_truth"], atol=1e-5, rtol=0):
        raise AssertionError(f"{cohort}: target values differ from Track-B reference")
    if not np.allclose(
        arrays["x_last"], reference["dense_x_last"], atol=1e-6, rtol=0
    ):
        raise AssertionError(f"{cohort}: carry-forward reference differs")

    measured_by_variable = np.zeros(n_vars, dtype=np.int64)
    for index in train_indices:
        obs_var = np.asarray(cache["examples"][index]["obs"]["var"], dtype=np.int64)
        measured_by_variable[np.unique(obs_var)] += 1
    train_stays = len(train_indices)
    frequency = measured_by_variable / train_stays
    ordered = sorted(SCOREABLE.tolist(), key=lambda v: (frequency[v], v))
    quartile_by_var: dict[int, str] = {}
    for quartile, group in enumerate(np.array_split(np.asarray(ordered, dtype=np.int64), 4), start=1):
        for variable in group.tolist():
            quartile_by_var[int(variable)] = f"Q{quartile}"

    arrays.update({
        "cache": cache,
        "split": split,
        "stats": stats,
        "variable_names": variable_names,
        "name_to_index": name_to_index,
        "test_indices": test_indices,
        "train_stays": train_stays,
        "frequency_by_variable": frequency,
        "quartile_by_variable": quartile_by_var,
        "stay_meta": stay_meta,
        "cache_path": cache_path,
        "split_path": split_path,
    })
    return arrays


def _load_cell(cohort: str, model: str, arm: str, ref: dict[str, Any]) -> tuple[np.ndarray, list[int], list[str]]:
    tags = TRACKB._tags(cohort, arm)
    tag = tags[model]
    directories = TRACKB._eval_dirs(model, tag, cohort)
    failures: list[str] = []
    if len(directories) != len(SEEDS):
        failures.append(f"expected five archived seeds, found {len(directories)}")
        return np.empty((0, len(ref["truth"])), dtype=np.float32), [], failures
    by_seed: dict[int, np.ndarray] = {}
    for directory in directories:
        try:
            config_path = directory.parent / "configs.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            iteration = int(directory.parent.name.removeprefix("iter"))
            seed = 2024 + iteration
            if seed not in SEEDS or seed in by_seed:
                raise AssertionError(f"unexpected/duplicate seed inferred from {directory}")
            if config.get("dataset_subset_name") != "jbi_patient":
                raise AssertionError("non-patient-grouped cache")
            if int(config.get("seq_len", -1)) != 24 or int(config.get("pred_len", -1)) != 24:
                raise AssertionError("run is not the frozen 24→24 task")
            if config.get("checkpoint_selection_metric") != "macro_persistence":
                raise AssertionError("checkpoint did not use macro-persistence selection")
            expected_loss = "MAE" if arm == "mae_macro" else "MSE"
            if str(config.get("loss", "")).upper() != expected_loss:
                raise AssertionError(f"expected {expected_loss} loss")
            sample_id = np.load(directory / "input_sample_ID.npy").astype(np.int64)
            if sample_id.tolist() != ref["test_indices"]:
                raise AssertionError("test sample IDs differ")
            mask = np.load(directory / "input_y_mask.npy").astype(bool)
            truth = np.load(directory / "input_y.npy")[mask]
            if truth.size != len(ref["truth"]) or not np.allclose(
                truth, ref["truth"], atol=1e-5, rtol=0
            ):
                raise AssertionError("test target values/mask differ")
            prediction = np.asarray(np.load(directory / "output_pred.npy")[mask], dtype=np.float32)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise AssertionError("prediction shape or values are invalid")
            by_seed[seed] = prediction
        except Exception as exc:
            failures.append(f"{directory.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    if set(by_seed) != set(SEEDS):
        failures.append(f"seed roster mismatch: got {sorted(by_seed)}")
    if failures:
        return np.empty((0, len(ref["truth"])), dtype=np.float32), sorted(by_seed), failures
    return np.stack([by_seed[seed] for seed in SEEDS]), list(SEEDS), []


def _grade(values: np.ndarray, bounds: tuple[float, ...], direction: str, support) -> np.ndarray:
    return SOFA.grade(SOFA.project(values, support), bounds, direction)


def _metrics(
    ref: dict[str, Any], predictions: np.ndarray, stay_selector: np.ndarray,
    event_mask: np.ndarray, grades: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = event_mask & stay_selector[ref["stay"]]
    score_selected = selected & np.isin(ref["variable"], SCOREABLE)
    variables = ref["variable"][score_selected]
    truth = ref["truth"][score_selected].astype(np.float64)
    x_last = ref["x_last"][score_selected].astype(np.float64)
    pred = predictions[:, score_selected].astype(np.float64)
    event_count = int(selected.sum())

    # Current Track-B macro score: equal mean over valid variables, with each
    # variable's seed-averaged event MAE divided by that variable's carry error.
    var_skill: list[float] = []
    var_mae: list[float] = []
    var_cf_mae: list[float] = []
    for variable in SCOREABLE:
        mask = variables == int(variable)
        count = int(mask.sum())
        cf_error = float(np.abs(x_last[mask] - truth[mask]).sum())
        if count < 50 or cf_error <= 1e-9:
            continue
        model_error = float(np.abs(pred[:, mask] - truth[mask][None, :]).mean(axis=0).sum())
        var_skill.append(1.0 - model_error / cf_error)
        var_mae.append(model_error / count)
        var_cf_mae.append(cf_error / count)
    cf_total = float(np.abs(x_last - truth).sum())
    model_total = float(np.abs(pred - truth[None, :]).mean(axis=0).sum())

    component_events = selected & np.isin(ref["variable"], list(grades["component_vars"]))
    sofa_wrong_rates: list[float] = []
    detections: list[dict[str, float]] = []
    n_sofa = int(component_events.sum())
    if n_sofa:
        c_true = grades["truth"][component_events]
        c_cf = grades["carry"][component_events]
        actual_deterioration = c_true > c_cf
        for seed_index in range(predictions.shape[0]):
            c_pred = grades["prediction"][seed_index, component_events]
            sofa_wrong_rates.append(float(np.mean(c_pred != c_true)))
            alarm = c_pred > c_cf
            tp = int(np.sum(actual_deterioration & alarm))
            fp = int(np.sum(~actual_deterioration & alarm))
            fn = int(np.sum(actual_deterioration & ~alarm))
            sensitivity = tp / (tp + fn) if tp + fn else float("nan")
            ppv = tp / (tp + fp) if tp + fp else float("nan")
            detections.append({"sensitivity": sensitivity, "ppv": ppv, "tp": tp, "fp": fp, "fn": fn})
    sensitivity_values = [item["sensitivity"] for item in detections if np.isfinite(item["sensitivity"])]
    ppv_values = [item["ppv"] for item in detections if np.isfinite(item["ppv"])]
    return {
        "n_target_measurements": event_count,
        "n_sofa_component_measurements": n_sofa,
        "n_measurement_ascertained_deterioration_events": int(
            np.sum(grades["truth"][component_events] > grades["carry"][component_events])
        ) if n_sofa else 0,
        "measurement_ascertained_deterioration_rate": float(np.mean(
            grades["truth"][component_events] > grades["carry"][component_events]
        )) if n_sofa else float("nan"),
        "persistence_macro_skill": float(np.mean(var_skill)) if var_skill else float("nan"),
        "persistence_pooled_skill": 1.0 - model_total / cf_total if cf_total > 1e-9 else float("nan"),
        "persistence_valid_variable_count": len(var_skill),
        "persistence_valid_variables": ";".join(str(int(v)) for v in SCOREABLE if np.sum(variables == v) >= 50 and np.abs(x_last[variables == v] - truth[variables == v]).sum() > 1e-9),
        "persistence_model_mae_macro_variables": float(np.mean(var_mae)) if var_mae else float("nan"),
        "persistence_carry_mae_macro_variables": float(np.mean(var_cf_mae)) if var_cf_mae else float("nan"),
        "sofa_event_misclassification_seed_mean": float(np.mean(sofa_wrong_rates)) if sofa_wrong_rates else float("nan"),
        "carryforward_event_misclassification": float(np.mean(
            grades["carry"][component_events] != grades["truth"][component_events]
        )) if n_sofa else float("nan"),
        "measured_time_deterioration_sensitivity_seed_mean": float(np.mean(sensitivity_values)) if sensitivity_values else float("nan"),
        "measured_time_deterioration_ppv_seed_mean": float(np.mean(ppv_values)) if ppv_values else float("nan"),
        "measured_time_deterioration_alarm_rate_seed_mean": float(np.mean([
            (item["tp"] + item["fp"]) / n_sofa for item in detections
        ])) if detections and n_sofa else float("nan"),
        "carryforward_no_alarm_deterioration_sensitivity": 0.0 if n_sofa else float("nan"),
        "carryforward_no_alarm_deterioration_ppv": float("nan"),
        "measured_time_detection_seed_count_sensitivity_defined": len(sensitivity_values),
        "measured_time_detection_seed_count_ppv_defined": len(ppv_values),
        "measured_time_detection_tp_seed_mean": float(np.mean([item["tp"] for item in detections])) if detections else 0.0,
        "measured_time_detection_fp_seed_mean": float(np.mean([item["fp"] for item in detections])) if detections else 0.0,
        "measured_time_detection_fn_seed_mean": float(np.mean([item["fn"] for item in detections])) if detections else 0.0,
    }


def _event_component_grades(ref: dict[str, Any], predictions: np.ndarray) -> dict[str, Any]:
    n = len(ref["truth"])
    truth_grade = np.full(n, -1, dtype=np.int8)
    carry_grade = np.full(n, -1, dtype=np.int8)
    pred_grade = np.full((predictions.shape[0], n), -1, dtype=np.int8)
    component_vars: set[int] = set()
    for _organ, name, bounds, direction, support in SOFA_COMPONENTS:
        variable = ref["name_to_index"][name]
        component_vars.add(variable)
        selected = ref["variable"] == variable
        scale = float(ref["stats"]["iqr"][variable])
        center = float(ref["stats"]["median"][variable])
        truth_grade[selected] = _grade(ref["truth_raw"][selected], bounds, direction, support)
        # This is the input-window value (median prior when no input reading).
        carry_grade[selected] = _grade(
            ref["x_last"][selected] * scale + center, bounds, direction, support
        )
        pred_raw = predictions[:, selected].astype(np.float64) * scale + center
        for seed_index in range(predictions.shape[0]):
            pred_grade[seed_index, selected] = _grade(
                pred_raw[seed_index], bounds, direction, support
            )
    return {
        "truth": truth_grade,
        "carry": carry_grade,
        "prediction": pred_grade,
        "component_vars": component_vars,
    }


def _frequency_cutpoint(ref: dict[str, Any], name: str, variable: int, last_raw: np.ndarray) -> np.ndarray:
    _organ, _name, bounds, _direction, support = next(row for row in SOFA_COMPONENTS if row[1] == name)
    selected = ref["variable"] == variable
    values = last_raw[selected]
    valid = np.isfinite(values)
    near = np.zeros(values.shape, dtype=bool)
    if support is not None:
        clipped = np.clip(np.round(values[valid]), support[0], support[1])
        near[valid] = np.min(np.abs(clipped[:, None] - np.asarray(bounds)[None, :]), axis=1) <= 1.0
    else:
        near[valid] = np.any(
            np.abs(values[valid, None] - np.asarray(bounds)[None, :])
            <= 0.10 * np.abs(np.asarray(bounds)[None, :]), axis=1
        )
    labels = np.full(values.size, "no_input_reference", dtype=object)
    labels[valid & near] = "near_cutpoint"
    labels[valid & ~near] = "other"
    return labels


def _strata(ref: dict[str, Any]) -> list[tuple[str, str, np.ndarray, str]]:
    n_stays = len(ref["stay_meta"])
    output: list[tuple[str, str, np.ndarray, str]] = []
    var = np.asarray(ref["variable"], dtype=np.int64)
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        variables = [v for v, q in ref["quartile_by_variable"].items() if q == quartile]
        output.append(("measurement_frequency", quartile, np.ones(n_stays, dtype=bool), ";".join(map(str, variables))))
    for field, label in (("age_group", "age"), ("gender", "gender"), ("icu_type", "icu_type")):
        groups = sorted(
            {str(item[field]) for item in ref["stay_meta"]},
            key=lambda x: (x == "NA (missing)", x),
        )
        for group in groups:
            output.append((f"patient_{label}", group, np.asarray([item[field] == group for item in ref["stay_meta"]]), "all"))
    return output


def run() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "analysis_plan": "paper/跨医院验证分析计划_P2.md",
        "analysis_plan_commit": "28947c8",
        "analysis_plan_sha256": sha256(ROOT / "paper/跨医院验证分析计划_P2.md"),
        "status": "PASS",
        "cohorts": {},
        "cells": [],
        "interpretation": "Exploratory retrospective estimates conditional on real measurement times; not real-time alarm sensitivity.",
        "aggregation": {
            "persistence_skill": "seed-averaged event absolute errors followed by variable-level macro or pooled skill",
            "SOFA_misclassification_and_detection": "mean of seed-specific event metrics",
        },
        "seed_values": list(SEEDS),
        "models": list(MODELS),
        "arms": {"macro": "MSE", "mae_macro": "masked L1 / MAE"},
    }
    for cohort in COHORTS:
        ref = _reference(cohort)
        audit["cohorts"][cohort] = {
            "cache_path": str(ref["cache_path"].relative_to(ROOT)),
            "cache_sha256": sha256(ref["cache_path"]),
            "split_path": str(ref["split_path"].relative_to(ROOT)),
            "split_sha256": sha256(ref["split_path"]),
            "n_train_stays_for_frequency": int(ref["train_stays"]),
            "n_test_stays": len(ref["stay_meta"]),
            "n_test_patients": len({item["patient_id"] for item in ref["stay_meta"]}),
            "training_measurement_frequency_by_variable": {
                str(v): float(ref["frequency_by_variable"][v]) for v in range(37)
            },
            "frequency_quartile_variables": {
                q: sorted(v for v, label in ref["quartile_by_variable"].items() if label == q)
                for q in ("Q1", "Q2", "Q3", "Q4")
            },
            "metadata_group_counts": {
                key: {str(value): sum(item[key] == value for item in ref["stay_meta"]) for value in sorted({str(item[key]) for item in ref["stay_meta"]})}
                for key in ("age_group", "gender", "icu_type")
            },
            "alignment": "PASS",
        }
        stratum_rows = _strata(ref)
        for model in MODELS:
            for arm in ARMS:
                predictions, found_seeds, failures = _load_cell(cohort, model, arm, ref)
                cell_audit = {
                    "cohort": cohort, "model": model, "arm": arm,
                    "tag": TRACKB._tags(cohort, arm)[model],
                    "expected_seeds": list(SEEDS), "found_seeds": found_seeds,
                    "status": "PASS" if not failures else "MISSING",
                    "failures": failures,
                }
                audit["cells"].append(cell_audit)
                grade_arrays = _event_component_grades(ref, predictions) if not failures else None
                prediction_sha_material: list[str] = []
                if not failures:
                    for seed in SEEDS:
                        directories = TRACKB._eval_dirs(model, TRACKB._tags(cohort, arm)[model], cohort)
                        directory = next(
                            d for d in directories
                            if int(d.parent.name.removeprefix("iter")) == seed - 2024
                        )
                        prediction_sha_material.append(sha256(directory / "output_pred.npy"))
                for stratum_type, stratum, stay_selector, variables_text in stratum_rows:
                    if stratum_type == "measurement_frequency":
                        group_vars = {int(v) for v in variables_text.split(";") if v}
                        event_mask = np.isin(ref["variable"], list(group_vars))
                    else:
                        event_mask = np.ones(len(ref["variable"]), dtype=bool)
                    if failures:
                        metric_values = {}
                    else:
                        metric_values = _metrics(ref, predictions, stay_selector, event_mask, grade_arrays)
                    stay_ids = np.flatnonzero(stay_selector)
                    if stratum_type == "measurement_frequency":
                        contributing_stays = np.unique(ref["stay"][event_mask])
                        selected_patients = {
                            ref["stay_meta"][int(i)]["patient_id"] for i in contributing_stays
                        }
                        count_stays = len(contributing_stays)
                    else:
                        selected_patients = {ref["stay_meta"][i]["patient_id"] for i in stay_ids}
                        count_stays = len(stay_ids)
                    row = {
                        "analysis_plan_commit": "28947c8",
                        "cohort": cohort,
                        "data_variant": "jbi_patient",
                        "split_unit": "patient_grouped",
                        "model": "CarryGate" if model == "ICUKANForecaster" else model,
                        "model_id": model,
                        "arm": arm,
                        "training_objective": "MAE" if arm == "mae_macro" else "MSE",
                        "seed_count": 0 if failures else len(found_seeds),
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "variables_in_frequency_stratum": variables_text if stratum_type == "measurement_frequency" else "",
                        "n_patients": len(selected_patients),
                        "n_stays": int(count_stays),
                        "status": "MISSING" if failures else "PASS",
                        "missing_reason": " | ".join(failures),
                        "metric_condition": "conditional on real measured outcome events; retrospective",
                        "model_error_aggregation": "mean seed event absolute error; variable macro / pooled skill",
                        "clinical_metric_aggregation": "mean seed-specific event metric",
                        "prediction_sha256_by_seed": ";".join(prediction_sha_material),
                        **metric_values,
                    }
                    rows.append(row)

                if not failures:
                    # Near-cutpoint analysis is component-specific. Unmeasured
                    # input references are retained in their own stratum.
                    for _organ, name, _bounds, _direction, _support in SOFA_COMPONENTS:
                        variable = ref["name_to_index"][name]
                        component = ref["variable"] == variable
                        labels = _frequency_cutpoint(ref, name, variable, ref["input_raw"])
                        event_labels = np.full(len(ref["variable"]), "", dtype=object)
                        event_labels[component] = labels
                        for label in ("near_cutpoint", "other", "no_input_reference"):
                            selected_mask = component & (event_labels == label)
                            values = _metrics(
                                ref, predictions, np.ones(len(ref["stay_meta"]), dtype=bool),
                                selected_mask, grade_arrays,
                            )
                            selected_stays = np.unique(ref["stay"][selected_mask])
                            rows.append({
                                "analysis_plan_commit": "28947c8",
                                "cohort": cohort,
                                "data_variant": "jbi_patient",
                                "split_unit": "patient_grouped",
                                "model": "CarryGate" if model == "ICUKANForecaster" else model,
                                "model_id": model,
                                "arm": arm,
                                "training_objective": "MAE" if arm == "mae_macro" else "MSE",
                                "seed_count": len(found_seeds),
                                "stratum_type": "near_cutpoint",
                                "stratum": label,
                                "component": name,
                                "component_organ": _organ,
                                "n_patients": len({ref["stay_meta"][int(s)]["patient_id"] for s in selected_stays}),
                                "n_stays": int(len(selected_stays)),
                                "status": "PASS",
                                "missing_reason": "",
                                "metric_condition": "conditional on real measured outcome events; retrospective",
                                "model_error_aggregation": "mean seed event absolute error; variable macro / pooled skill",
                                "clinical_metric_aggregation": "mean seed-specific event metric",
                                "prediction_sha256_by_seed": ";".join(prediction_sha_material),
                                **values,
                            })
    return pd.DataFrame(rows), audit


def main() -> int:
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite prior P2 stratification artifact: {OUT} or {AUDIT}")
    frame, audit = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, float_format="%.17g")
    audit["output_path"] = str(OUT.relative_to(ROOT))
    audit["output_sha256"] = sha256(OUT)
    audit["n_rows"] = int(len(frame))
    audit["missing_cells"] = [cell for cell in audit["cells"] if cell["status"] != "PASS"]
    if audit["missing_cells"]:
        audit["status"] = "PARTIAL_MISSING_ARCHIVED_CELLS"
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(frame):,} rows: {OUT}")
    print(f"audit: {audit['status']} → {AUDIT}")
    for cohort in COHORTS:
        meta = audit["cohorts"][cohort]["metadata_group_counts"]
        print(f"{cohort}: test stays={audit['cohorts'][cohort]['n_test_stays']}, "
              f"patients={audit['cohorts'][cohort]['n_test_patients']}, metadata={meta}")
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
