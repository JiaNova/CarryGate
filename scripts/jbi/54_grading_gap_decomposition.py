#!/usr/bin/env python
"""Generate the post-hoc P1 grading-gap comparison, robustness bounds and F8.

Inputs are the archived patient-grouped 24-to-24-hour objective tables and the
frozen P1 agreement/net-benefit outputs. This script is descriptive: it does
not alter the registered I1/I2 estimands or decision rules.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/jbi"
PAPER = ROOT / "paper"
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
MODEL_LABEL = {
    "ICUKANForecaster": "CarryGate",
    "GraFITi": "GraFITi",
    "HyperIMTS": "HyperIMTS",
}
COHORTS = ("eicu", "mimiciv")
COHORT_LABEL = {"eicu": "eICU", "mimiciv": "MIMIC-IV"}
COMPONENTS = ("Platelets", "Bilirubin", "GCS", "Creatinine")
COMPONENT_LABEL = {
    "Platelets": "PLT",
    "Bilirubin": "Bili",
    "GCS": "GCS",
    "Creatinine": "Cr",
}
SETTING_ORDER = ("first_day_mse", "first_day_masked_l1", "p1_landmark_masked_l1")
SETTING_LABEL = {
    "first_day_mse": "First day, MSE",
    "first_day_masked_l1": "First day, masked L1",
    "p1_landmark_masked_l1": "Days 1–7 landmarks, masked L1",
}
# Objective/design colours follow the paper-wide ramp: MSE light grey, masked
# L1 in deepening blue as the design moves towards the bedside landmark use.
SETTING_COLOR = {
    "first_day_mse": "#B4B4B4",
    "first_day_masked_l1": "#6BA3CE",
    "p1_landmark_masked_l1": "#0F4C81",
}
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.30
THRESHOLD_STEP = 0.01
I2_THRESHOLDS_REQUIRED = 14

INPUTS = {
    "first_day_mse_eicu": RESULTS / "e8_sofa_by_objective_eicu_macro.csv",
    "first_day_mse_mimiciv": RESULTS / "e8_sofa_by_objective_mimiciv_macro.csv",
    "first_day_l1_eicu": RESULTS / "e8_sofa_by_objective_eicu_mae_macro.csv",
    "first_day_l1_mimiciv": RESULTS / "e8_sofa_by_objective_mimiciv_mae_macro.csv",
    "p1_agreement": RESULTS / "p1_landmark_agreement.csv",
    "p1_decision_curve": RESULTS / "p1_landmark_decision_curve.csv",
    "p1_verdict": RESULTS / "p1_landmark_verdict.json",
    "p1_ledger": RESULTS / "p1_landmark_test_inference_ledger.json",
    "p1_metrics": RESULTS / "p1_landmark_metrics.csv",
    "p1_ascertainment": RESULTS / "p1_landmark_ascertainment.csv",
}
OUT_SUMMARY_CSV = RESULTS / "p1_grading_gap_decomposition.csv"
OUT_DETAIL_CSV = RESULTS / "p1_grading_gap_decomposition_cells.csv"
OUT_ROBUSTNESS_JSON = RESULTS / "p1_grading_gap_decomposition_robustness.json"
OUT_SUMMARY_TEX = PAPER / "tables/T_p1_grading_gap_decomposition.tex"
OUT_DETAIL_TEX = PAPER / "tables/T_p1_grading_gap_decomposition_cells.tex"
OUT_FIG_PDF = ROOT / "results/paper/F8_p1_landmark_use_case.pdf"
OUT_FIG_PNG = ROOT / "results/paper/F8_p1_landmark_use_case.png"
OUT_SENSITIVITY_PDF = ROOT / "results/paper/F8_p1_sensitivity_alarm_burden_supplement.pdf"
OUT_SENSITIVITY_PNG = ROOT / "results/paper/F8_p1_sensitivity_alarm_burden_supplement.png"
TABLE_NOTE = (
    "Descriptive and post hoc after inspection of the landmark results. The three designs "
    "differ in several respects, including input length, landmark period and "
    "number of training windows; the contrast is not a single-factor causal estimate."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in row {row}")
    return value


def _objective_rows(path: Path, *, cohort: str, objective: str) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    chosen = [row for row in rows if row["model"] in MODELS]
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in chosen:
        if row["cohort"] != cohort or row["training_objective"].upper() != objective:
            raise ValueError(f"unexpected cohort/objective provenance in {path}: {row}")
        if int(row["n_seeds"]) != 5:
            raise ValueError(f"expected five seeds in existing comparison {path}: {row}")
        key = (row["model"], row["variable"])
        if key in out:
            raise ValueError(f"duplicate objective cell {key} in {path}")
        out[key] = row
    expected = {(model, component) for model in MODELS for component in COMPONENTS}
    if set(out) != expected:
        raise ValueError(f"incomplete 24-to-24 cells in {path}: missing={expected-set(out)}, extra={set(out)-expected}")
    return out


def build_gap_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_setting: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        key: {} for key in SETTING_ORDER
    }
    source_rows = {
        "first_day_mse": {
            "eicu": _objective_rows(INPUTS["first_day_mse_eicu"], cohort="eicu", objective="MSE"),
            "mimiciv": _objective_rows(INPUTS["first_day_mse_mimiciv"], cohort="mimiciv", objective="MSE"),
        },
        "first_day_masked_l1": {
            "eicu": _objective_rows(INPUTS["first_day_l1_eicu"], cohort="eicu", objective="MAE"),
            "mimiciv": _objective_rows(INPUTS["first_day_l1_mimiciv"], cohort="mimiciv", objective="MAE"),
        },
    }
    for setting, cohorts in source_rows.items():
        for cohort, rows in cohorts.items():
            for (model, component), row in rows.items():
                gap = fnum(row, "event_excess_over_carryforward")
                independently_computed = fnum(row, "event_misclassification") - fnum(row, "carryforward_event_misclassification")
                if not math.isclose(gap, independently_computed, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"stored grading gap disagrees with source rates in {setting}: {row}")
                by_setting[setting][(cohort, component, model)] = {
                    "setting": setting,
                    "setting_label": SETTING_LABEL[setting],
                    "cohort": cohort,
                    "component": component,
                    "model": model,
                    "model_event_misclassification_rate": fnum(row, "event_misclassification"),
                    "carry_forward_event_misclassification_rate": fnum(row, "carryforward_event_misclassification"),
                    "gap_percentage_points": 100.0 * gap,
                    "carry_forward_strictly_better": gap > 0.0,
                    "n_events": int(row["n_events"]),
                    "n_patients": int(row["n_patients"]),
                    "observed_seed_count": int(row["n_seeds"]),
                    "required_seed_count": int(row["n_seeds"]),
                    "coverage_complete": True,
                    "gap_ci_low_percentage_points": 100.0 * fnum(row, "event_ci_lo"),
                    "gap_ci_high_percentage_points": 100.0 * fnum(row, "event_ci_hi"),
                    "bootstrap_n": int(row["bootstrap_n"]),
                    "bootstrap_unit": row["bootstrap_unit"],
                }

    p1_rows = read_csv(INPUTS["p1_agreement"])
    expected_cells = {
        (cohort, component, model)
        for cohort in COHORTS for component in COMPONENTS for model in MODELS
    }
    for row in p1_rows:
        key = (row["cohort"], row["component"], row["model"])
        if key not in expected_cells or key in by_setting["p1_landmark_masked_l1"]:
            raise ValueError(f"unexpected or duplicate P1 agreement cell: {key}")
        by_setting["p1_landmark_masked_l1"][key] = {
            "setting": "p1_landmark_masked_l1",
            "setting_label": SETTING_LABEL["p1_landmark_masked_l1"],
            "cohort": row["cohort"],
            "component": row["component"],
            "model": row["model"],
            "model_event_misclassification_rate": fnum(row, "model_event_misclassification_rate"),
            "carry_forward_event_misclassification_rate": fnum(row, "carry_forward_event_misclassification_rate"),
            "gap_percentage_points": 100.0 * fnum(row, "model_minus_carry_forward_difference"),
            "carry_forward_strictly_better": row["carry_forward_strictly_better"].lower() == "true",
            "n_events": int(row["n_measured_events_mean_across_seeds"]),
            "n_patients": int(row["n_patients"]),
            "observed_seed_count": int(row["observed_seed_count"]),
            "required_seed_count": int(row["required_seed_count"]),
            "coverage_complete": row["seed_coverage_complete"].lower() == "true",
            "gap_ci_low_percentage_points": 100.0 * fnum(row, "difference_ci_low"),
            "gap_ci_high_percentage_points": 100.0 * fnum(row, "difference_ci_high"),
            "bootstrap_n": int(row["bootstrap_n"]),
            "bootstrap_unit": row["bootstrap_unit"],
        }
    if set(by_setting["p1_landmark_masked_l1"]) != expected_cells:
        raise ValueError("P1 agreement table does not contain all 24 fixed cells")

    detail_rows: list[dict[str, Any]] = []
    for setting in SETTING_ORDER:
        if len(by_setting[setting]) != 24:
            raise ValueError(f"{setting} contains {len(by_setting[setting])} cells, expected 24")
        detail_rows.extend(by_setting[setting][key] for key in sorted(by_setting[setting]))

    summary_rows = []
    for setting in SETTING_ORDER:
        rows = list(by_setting[setting].values())
        gap_values = [row["gap_percentage_points"] for row in rows]
        if setting == "p1_landmark_masked_l1":
            # The registered verdict deliberately treats incomplete seed coverage as non-favourable.
            n_cf_better = sum(bool(row["carry_forward_strictly_better"]) for row in rows)
            expected_i1 = json.loads(INPUTS["p1_verdict"].read_text(encoding="utf-8"))["I1"]
            if n_cf_better != expected_i1["favorable_cells"]:
                raise RuntimeError("P1 agreement tally differs from the frozen I1 verdict")
        else:
            n_cf_better = sum(float(row["gap_percentage_points"]) > 0.0 for row in rows)
        summary_rows.append({
            "setting": setting,
            "setting_label": SETTING_LABEL[setting],
            "cell_count": len(rows),
            "carry_forward_better_cells": int(n_cf_better),
            "gap_min_percentage_points": float(min(gap_values)),
            "gap_max_percentage_points": float(max(gap_values)),
            "gap_definition": "model event misclassification minus carry-forward event misclassification",
            "table_note": TABLE_NOTE,
        })
    return summary_rows, detail_rows


def _thresholds() -> list[float]:
    values = np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP), 2)
    if len(values) != 26 or not math.isclose(float(values[-1]), THRESHOLD_MAX):
        raise RuntimeError("the frozen P1 threshold grid is not 26 points from 0.05 to 0.30")
    return [float(value) for value in values]


def calculate_i2(decision_rows: list[dict[str, str]], verdict: dict[str, Any], scope: str) -> dict[str, Any]:
    expected_thresholds = _thresholds()
    complete = verdict["seed_coverage_by_cohort"]
    by_key: dict[tuple[str, str, str, float], dict[str, str]] = {}
    for row in decision_rows:
        if row["scope"] != scope or row["model_or_strategy"] not in MODELS:
            continue
        key = (row["cohort"], row["component"], row["model_or_strategy"], round(float(row["threshold_probability"]), 2))
        if key in by_key:
            raise ValueError(f"duplicate decision-curve key {key}")
        by_key[key] = row

    cell_rows = []
    for cohort in COHORTS:
        for component in COMPONENTS:
            model_passes: dict[str, bool] = {}
            model_wins: dict[str, int] = {}
            for model in MODELS:
                curve = [by_key.get((cohort, component, model, pt)) for pt in expected_thresholds]
                if any(item is None for item in curve):
                    raise ValueError(f"incomplete {scope} curve for {cohort}/{component}/{model}")
                points = []
                for item in curve:
                    assert item is not None
                    model_nb = fnum(item, "net_benefit")
                    no_alarm_nb = fnum(item, "treat_none_net_benefit")
                    alert_all_nb = fnum(item, "treat_all_net_benefit")
                    points.append(model_nb > max(no_alarm_nb, alert_all_nb))
                wins = int(sum(points))
                model_wins[model] = wins
                coverage = complete[cohort][model]["complete"]
                model_passes[model] = bool(coverage and wins >= I2_THRESHOLDS_REQUIRED)
            n_model_passes = int(sum(model_passes.values()))
            cell_rows.append({
                "cohort": cohort,
                "component": component,
                "model_threshold_wins": model_wins,
                "model_passes_14_of_26": model_passes,
                "models_passing": n_model_passes,
                "cell_pass": n_model_passes >= 2,
            })
    return {
        "scope": scope,
        "threshold_grid": expected_thresholds,
        "model_threshold_pass_rule": f">={I2_THRESHOLDS_REQUIRED} of 26 thresholds strictly above both no-alarm and alert-all",
        "cell_pass_rule": "at least two of three models pass",
        "cell_count": len(cell_rows),
        "passing_cells": sum(bool(row["cell_pass"]) for row in cell_rows),
        "descriptive_only": scope == "measured_windows_only",
        "cells": cell_rows,
    }


def build_robustness(detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict = json.loads(INPUTS["p1_verdict"].read_text(encoding="utf-8"))
    ledger = json.loads(INPUTS["p1_ledger"].read_text(encoding="utf-8"))
    if verdict.get("selected_branch") != "C" or ledger.get("status") != "COMPLETE":
        raise RuntimeError("robustness outputs require the frozen Branch C test analysis")
    not_completed = ledger.get("not_completed_run_keys", [])
    if set(not_completed) != {"mimiciv/GraFITi/seed2026", "mimiciv/GraFITi/seed2027"}:
        raise RuntimeError(f"unexpected unresolved P1 fixed-denominator cells: {not_completed}")

    p1_i1_rows = [row for row in detail_rows if row["setting"] == "p1_landmark_masked_l1"]
    observed_favorable = sum(bool(row["carry_forward_strictly_better"]) for row in p1_i1_rows)
    incomplete_i1_rows = [row for row in p1_i1_rows if not row["coverage_complete"]]
    verdict_favorable = int(verdict["I1"]["favorable_cells"])
    if observed_favorable != verdict_favorable:
        raise RuntimeError("generated P1 I1 coverage differs from the official verdict")
    i1_upper = observed_favorable + len(incomplete_i1_rows)
    i1_required = int(verdict["I1"]["required"])

    missing_model_by_cohort: dict[str, set[str]] = {cohort: set() for cohort in COHORTS}
    for run_key in not_completed:
        cohort, model, _seed = run_key.split("/")
        missing_model_by_cohort[cohort].add(model)
    possible_i2_cells = []
    for cell in verdict["I2_cells"]:
        cohort = cell["cohort"]
        missing_models = missing_model_by_cohort[cohort]
        known_passes = sum(
            bool(passed)
            for model, passed in cell["model_passes"].items()
            if model not in missing_models
        )
        unknown_count = len(missing_models)
        pass_if_unknown_models_fail = known_passes >= 2
        pass_if_unknown_models_pass = known_passes + unknown_count >= 2
        possible_i2_cells.append({
            "cohort": cohort,
            "component": cell["component"],
            "known_model_passes": known_passes,
            "unknown_model_count": unknown_count,
            "cell_pass_if_unknown_models_fail": pass_if_unknown_models_fail,
            "cell_pass_if_unknown_models_pass": pass_if_unknown_models_pass,
            "observed_cell_pass": bool(cell["cell_pass"]),
            "invariant": pass_if_unknown_models_fail == pass_if_unknown_models_pass == bool(cell["cell_pass"]),
        })
    i2_invariant = all(row["invariant"] for row in possible_i2_cells)
    i2_lower = sum(row["cell_pass_if_unknown_models_fail"] for row in possible_i2_cells)
    i2_upper = sum(row["cell_pass_if_unknown_models_pass"] for row in possible_i2_cells)
    i2_required = int(verdict["I2"]["required"])
    decision_rows = read_csv(INPUTS["p1_decision_curve"])
    measured_i2 = calculate_i2(decision_rows, verdict, "measured_windows_only")
    metrics_rows = [
        row for row in read_csv(INPUTS["p1_metrics"])
        if row["scope"] == "all_reference_windows" and row["model_or_strategy"] in MODELS
    ]
    ascertainment_rows = [
        row for row in read_csv(INPUTS["p1_ascertainment"])
        if row["model_or_strategy"] in MODELS
    ]
    if len(metrics_rows) != 24 or len(ascertainment_rows) != 24:
        raise RuntimeError("expected 24 model-by-cohort-by-component P1 primary-scope summary rows")
    sensitivity_values = [fnum(row, "window_sensitivity") for row in metrics_rows]
    alarm_burdens = [fnum(row, "alarms_per_reference_eligible_patient_day") for row in metrics_rows]
    measurement_rate_differences = [fnum(row, "measurement_rate_difference_alarm_minus_noalarm") for row in ascertainment_rows]

    input_hashes = {key: sha256(path) for key, path in INPUTS.items()}
    return {
        "schema_version": "p1_grading_gap_robustness_v1",
        "generated_from": {
            "generator_sha256": sha256(Path(__file__).resolve()),
            "input_sha256": input_hashes,
            "p1_selected_branch": verdict["selected_branch"],
        },
        "not_completed_run_keys": not_completed,
        "i1": {
            "observed_favorable_cells": observed_favorable,
            "incomplete_cohort_model_component_cells": len(incomplete_i1_rows),
            "upper_bound_if_every_incomplete_cell_favors_carry_forward": i1_upper,
            "fixed_denominator": int(verdict["I1"]["cell_count"]),
            "required_for_pass": i1_required,
            "upper_bound_fraction": i1_upper / int(verdict["I1"]["cell_count"]),
            "verdict_invariant_for_any_missing_cell_outcome": i1_upper < i1_required,
        },
        "i2": {
            "observed_passing_cells": int(verdict["I2"]["passing_cells"]),
            "fixed_denominator": int(verdict["I2"]["cell_count"]),
            "required_for_pass": int(verdict["I2"]["required"]),
            "minimum_passing_cells_over_any_missing_run_outcomes": i2_lower,
            "maximum_passing_cells_over_any_missing_run_outcomes": i2_upper,
            "verdict_invariant_for_any_missing_run_outcome": i2_invariant and ((i2_lower >= i2_required) == (i2_upper >= i2_required)),
            "affected_cell_details": possible_i2_cells,
        },
        "measured_windows_only_i2_descriptive": measured_i2,
        "p1_limitation_summaries": {
            "scope": "observed model-by-cohort-by-component cells; all-reference-windows metric rows",
            "cell_count": len(metrics_rows),
            "complete_seed_coverage_cells": sum(row["seed_coverage_complete"].lower() == "true" for row in metrics_rows),
            "maximum_measurement_ascertained_deterioration_sensitivity": max(sensitivity_values),
            "maximum_alarms_per_reference_eligible_patient_day": max(alarm_burdens),
            "alarmed_window_measurement_rate_higher_cells": sum(value > 0.0 for value in measurement_rate_differences),
            "measurement_rate_difference_range_percentage_points": [100.0 * min(measurement_rate_differences), 100.0 * max(measurement_rate_differences)],
            "measurement_rate_denominator": len(measurement_rate_differences),
        },
        "scope_note": "The measured-windows-only I2 count is descriptive and does not replace the registered all-reference-windows I2 verdict.",
    }


def _fmt_pp(value: float) -> str:
    return f"{value:+.1f}"


def write_latex(summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]], summary_path: Path, detail_path: Path) -> None:
    summary = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Carry-forward's grading advantage across training objective and evaluation design (descriptive, patient-grouped eICU and MIMIC-IV test splits).}",
        r"\label{tab:p1-grading-gap-decomposition}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Setting & Carry-forward better (of 24) & Gap range (pp) \\",
        r"\midrule",
    ]
    for row in summary_rows:
        summary.append(
            f"{row['setting_label']} & {row['carry_forward_better_cells']}/{row['cell_count']} & "
            f"{_fmt_pp(row['gap_min_percentage_points'])} to "
            f"{_fmt_pp(row['gap_max_percentage_points'])} \\\\")
    summary.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.96\linewidth}\vspace{2pt}\footnotesize\emph{Note.} " + TABLE_NOTE + r"\end{minipage}",
        r"\end{table}",
    ])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    lookup = {(row["setting"], row["cohort"], row["component"], row["model"]): row for row in detail_rows}
    details = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-cell event-level SOFA grading gap (model minus carry-forward, percentage points).}",
        r"\label{tab:p1-grading-gap-cells}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Cohort & Component & Model & First-day MSE & First-day masked L1 & Days 1--7 landmarks, masked L1 \\",
        r"\midrule",
    ]
    for cohort in COHORTS:
        for component in COMPONENTS:
            for model in MODELS:
                mse = lookup[("first_day_mse", cohort, component, model)]["gap_percentage_points"]
                l1 = lookup[("first_day_masked_l1", cohort, component, model)]["gap_percentage_points"]
                p1 = lookup[("p1_landmark_masked_l1", cohort, component, model)]
                p1_text = (
                    f"{_fmt_pp(p1['gap_percentage_points'])} "
                    f"[{_fmt_pp(p1['gap_ci_low_percentage_points'])}, {_fmt_pp(p1['gap_ci_high_percentage_points'])}]"
                )
                if not p1["coverage_complete"]:
                    p1_text += r"$^{\dagger}$"
                details.append(
                    f"{COHORT_LABEL[cohort]} & {component} & {MODEL_LABEL[model]} & "
                    f"{_fmt_pp(mse)} & {_fmt_pp(l1)} & {p1_text} \\\\")
    details.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{minipage}{0.98\linewidth}\vspace{2pt}\footnotesize\emph{Note.} " + TABLE_NOTE +
        r" Positive values mean greater event-level grade misclassification for the model than carry-forward. " +
        r"Landmark intervals are patient-cluster bootstrap 95\% intervals. $^{\dagger}$ Incomplete landmark seed coverage (3/5); these cells were not counted as favourable to carry-forward in the registered grading criterion.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    detail_path.write_text("\n".join(details) + "\n", encoding="utf-8")


def plot_figures(detail_rows: list[dict[str, Any]], robustness: dict[str, Any], stage: Path) -> None:
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import style

    style.use_style()
    fig = plt.figure(figsize=(7.4, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.55), hspace=0.36,
                             top=0.95, bottom=0.08, left=0.09, right=0.99)
    top = outer[0].subgridspec(1, 2, width_ratios=(1.0, 1.25), wspace=0.30)
    lookup = {(row["setting"], row["cohort"], row["component"], row["model"]): row for row in detail_rows}
    cells = [(cohort, component, model) for cohort in COHORTS for component in COMPONENTS for model in MODELS]
    better = {setting: sum(bool(lookup[(setting, *cell)]["carry_forward_strictly_better"]) for cell in cells)
              for setting in SETTING_ORDER}

    # A: all three designs on one linear axis; a thin line follows each cell.
    ax_a = fig.add_subplot(top[0])
    rng = np.random.default_rng(0)
    jitter = {cell: float(rng.uniform(-0.12, 0.12)) for cell in cells}
    for cell in cells:
        ys = [float(lookup[(setting, *cell)]["gap_percentage_points"]) for setting in SETTING_ORDER]
        xs = [k + jitter[cell] for k in range(3)]
        ax_a.plot(xs, ys, color=style.GRID, lw=0.6, zorder=1)
        for k, setting in enumerate(SETTING_ORDER):
            ax_a.scatter(xs[k], ys[k], s=11, marker=style.TRACKB_MARKER[cell[2]],
                         color=SETTING_COLOR[setting], edgecolor="white", linewidths=0.3, zorder=3)
    style.zero_reference(ax_a, axis="y")
    ymax = max(float(row["gap_percentage_points"]) for row in detail_rows)
    for k, setting in enumerate(SETTING_ORDER):
        ax_a.text(k, ymax * 1.12, f"{better[setting]}/{len(cells)}", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=SETTING_COLOR[setting])
    ax_a.text(1.0, ymax * 1.27, "comparisons in which carry-forward grades better",
              ha="center", va="bottom", fontsize=6.2, color=style.MUTED)
    ax_a.set_ylim(min(-2.0, ax_a.get_ylim()[0]), ymax * 1.42)
    ax_a.set_xlim(-0.45, 2.45)
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels(["first day\nMSE", "first day\nmasked L1", "day 1\u20137\nlandmarks"])
    ax_a.set_ylabel("model \u2212 carry-forward grading\nerror (percentage points)")
    style.soft_grid(ax_a, axis="y")
    style.panel_label(ax_a, "A", dx=-0.30, dy=1.08)

    # B: the two masked-L1 designs, magnified, with landmark intervals.
    ax_b = fig.add_subplot(top[1])
    for index, cell in enumerate(cells):
        first = float(lookup[("first_day_masked_l1", *cell)]["gap_percentage_points"])
        row = lookup[("p1_landmark_masked_l1", *cell)]
        y = float(row["gap_percentage_points"])
        low, high = float(row["gap_ci_low_percentage_points"]), float(row["gap_ci_high_percentage_points"])
        marker = style.TRACKB_MARKER[cell[2]]
        ax_b.scatter(index - 0.17, first, s=10, marker=marker, color=SETTING_COLOR["first_day_masked_l1"],
                     edgecolor="white", linewidths=0.3, zorder=3)
        face = SETTING_COLOR["p1_landmark_masked_l1"] if row["coverage_complete"] else "white"
        ax_b.errorbar(index + 0.17, y, yerr=[[max(0.0, y - low)], [max(0.0, high - y)]], fmt=marker,
                      color=SETTING_COLOR["p1_landmark_masked_l1"], markerfacecolor=face,
                      markersize=3.2, markeredgewidth=0.7, linewidth=0.7, capsize=0, zorder=4)
    style.zero_reference(ax_b, axis="y")
    for boundary in range(3, len(cells), 3):
        ax_b.axvline(boundary - 0.5, color=style.GRID, lw=0.5 if boundary % 12 else 1.0, zorder=0)
    ax_b.set_xticks([1 + 3 * k for k in range(8)])
    ax_b.set_xticklabels([COMPONENT_LABEL[c] for c in COMPONENTS] * 2, fontsize=6.4)
    for k, cohort in enumerate(COHORTS):
        ax_b.text(5.5 + 12 * k, 0.98, COHORT_LABEL[cohort], transform=ax_b.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.8, fontweight="bold", color=style.MUTED)
    ax_b.set_xlim(-0.6, len(cells) - 0.4)
    ax_b.set_ylabel("grading gap, masked L1 designs (pp)")
    low_b, high_b = ax_b.get_ylim()
    ax_b.set_ylim(low_b, high_b + 0.35)
    style.soft_grid(ax_b, axis="y")
    style.panel_label(ax_b, "B", dx=-0.16, dy=1.08)
    handles = [Line2D([], [], marker="o", ls="", color=SETTING_COLOR[s], label=SETTING_LABEL[s])
               for s in SETTING_ORDER]
    handles += [Line2D([], [], marker=style.TRACKB_MARKER[m], ls="", color=style.INK, markersize=3.5,
                       label=MODEL_LABEL[m]) for m in MODELS]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="white",
                       color=SETTING_COLOR["p1_landmark_masked_l1"], label="3/5 seeds")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.005), ncol=7,
               fontsize=6.0, handletextpad=0.2, columnspacing=0.8)

    # C: net benefit above the better default, per cohort and component.
    panel_c = outer[1].subgridspec(2, 4, wspace=0.28, hspace=0.42)
    axes = []
    decision_rows = read_csv(INPUTS["p1_decision_curve"])
    verdict = json.loads(INPUTS["p1_verdict"].read_text(encoding="utf-8"))
    all_i2 = {(row["cohort"], row["component"]): row["cell_pass"] for row in verdict["I2_cells"]}
    measured_i2 = robustness["measured_windows_only_i2_descriptive"]["passing_cells"]
    threshold_order = _thresholds()
    y_values = []
    decision: dict[tuple[str, str, str, str, float], dict[str, str]] = {}
    for row in decision_rows:
        if row["model_or_strategy"] not in MODELS:
            continue
        key = (row["cohort"], row["component"], row["model_or_strategy"], row["scope"], round(float(row["threshold_probability"]), 2))
        if key in decision:
            raise ValueError(f"duplicate decision-curve entry: {key}")
        decision[key] = row
    for cell_index, component in enumerate(COMPONENTS):
        for cohort_index, cohort in enumerate(COHORTS):
            ax = fig.add_subplot(panel_c[cohort_index, cell_index])
            axes.append(ax)
            for model in MODELS:
                for scope, linestyle in (("all_reference_windows", "-"), ("measured_windows_only", "--")):
                    values = []
                    for pt in threshold_order:
                        item = decision[(cohort, component, model, scope, pt)]
                        delta = fnum(item, "net_benefit") - max(fnum(item, "treat_none_net_benefit"), fnum(item, "treat_all_net_benefit"))
                        values.append(delta)
                        y_values.append(delta)
                    ax.plot(threshold_order, values, color=style.TRACKB_COLOUR[model], linestyle=linestyle,
                            linewidth=0.9, alpha=0.95)
            ax.axhline(0.0, color=style.CARRY, linestyle=(0, (3, 2)), linewidth=0.7)
            ax.set_title(f"{COHORT_LABEL[cohort]} \u00b7 {component}", fontsize=7.0, pad=2)
            ax.tick_params(axis="both", labelsize=5.8)
            ax.set_xticks([0.05, 0.15, 0.25])
            if cohort_index == 1:
                ax.set_xlabel(r"threshold $p_t$", fontsize=6.6)
            if cell_index == 0:
                ax.set_ylabel("NB \u2212 better default", fontsize=6.6)
            met = all_i2[(cohort, component)]
            ax.text(0.97, 0.06, "criterion met" if met else "criterion not met",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                    color=style.INK if met else style.WORSE, fontweight="bold" if not met else "normal")
    if y_values:
        # A shared linear axis let the large negative excursions at low
        # thresholds flatten the small positive margins that decide the
        # criterion; a shared symmetric-log axis keeps both readable and the
        # panels comparable.
        ymin, ymax_nb = min(y_values), max(y_values)
        from matplotlib.ticker import FuncFormatter, NullLocator
        for k, ax in enumerate(axes):
            ax.set_yscale("symlog", linthresh=0.005, linscale=1.0)
            ax.set_yticks([-0.1, -0.01, 0.0, 0.01])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(
                (lambda v, _: f"{v:g}".replace("-", "\u2212")) if k < 2 else (lambda v, _: "")))
            ax.set_ylim(ymin * 1.15, max(ymax_nb * 1.6, 0.02))
            ax.axhspan(0.0, max(ymax_nb * 1.6, 0.02), color=style.tint(style.BETTER, 0.90), zorder=0, linewidth=0)
    style.panel_label(axes[0], "C", dx=-0.42, dy=1.22)
    model_handles = [Line2D([0], [0], color=style.TRACKB_COLOUR[model], linewidth=1.4, label=MODEL_LABEL[model]) for model in MODELS]
    scope_handles = [
        Line2D([0], [0], color=style.MUTED, linewidth=1.2, linestyle="-", label="all reference windows"),
        Line2D([0], [0], color=style.MUTED, linewidth=1.2, linestyle="--", label="measured windows only"),
    ]
    fig.legend(handles=model_handles+scope_handles, loc="lower center", bbox_to_anchor=(0.54, 0.0),
               ncol=5, fontsize=6.2)
    fig.text(
        0.99, 0.605,
        f"criterion met in {verdict['I2']['passing_cells']}/{verdict['I2']['cell_count']} settings "
        f"(all windows); {measured_i2}/8 with measured windows only",
        ha="right", va="bottom", fontsize=6.0, color=style.MUTED,
    )
    fig.savefig(stage / "F8_p1_landmark_use_case.pdf", bbox_inches="tight")
    fig.savefig(stage / "F8_p1_landmark_use_case.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-generated", action="store_true", help="archive and replace a previous complete generation")
    args = parser.parse_args()
    required_inputs = list(INPUTS.values())
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not OUT_FIG_PDF.is_file() or not OUT_FIG_PNG.is_file():
        raise FileNotFoundError("the existing P1 sensitivity/alarm-burden F8 is required for supplemental preservation")
    replaceable = (
        OUT_SUMMARY_CSV, OUT_DETAIL_CSV, OUT_ROBUSTNESS_JSON, OUT_SUMMARY_TEX,
        OUT_DETAIL_TEX, OUT_SENSITIVITY_PDF, OUT_SENSITIVITY_PNG,
    )
    existing_replaceable = [path for path in replaceable if path.exists()]
    if existing_replaceable and not args.refresh_generated:
        raise FileExistsError("refusing to overwrite an existing P1 decomposition artifact; use --refresh-generated to archive it first")
    if args.refresh_generated and (len(existing_replaceable) != len(replaceable) or not all(path.is_file() for path in replaceable) or not all(path.is_file() for path in (OUT_FIG_PDF, OUT_FIG_PNG))):
        raise RuntimeError("--refresh-generated requires a complete prior generator output set")

    summary_rows, detail_rows = build_gap_rows()
    robustness = build_robustness(detail_rows)
    measured = calculate_i2(read_csv(INPUTS["p1_decision_curve"]), json.loads(INPUTS["p1_verdict"].read_text(encoding="utf-8")), "measured_windows_only")
    main_i2 = json.loads(INPUTS["p1_verdict"].read_text(encoding="utf-8"))["I2"]
    if robustness["i2"]["minimum_passing_cells_over_any_missing_run_outcomes"] != robustness["i2"]["maximum_passing_cells_over_any_missing_run_outcomes"]:
        raise RuntimeError("not-completed P1 run outcomes could change the I2 passing-cell count")
    if robustness["i1"]["upper_bound_if_every_incomplete_cell_favors_carry_forward"] >= robustness["i1"]["required_for_pass"]:
        raise RuntimeError("not-completed P1 cells could change the I1 verdict")

    stage_root = RESULTS / "_work"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix="p1_grading_gap_stage_", dir=stage_root))
    try:
        write_csv(stage_dir / OUT_SUMMARY_CSV.name, summary_rows)
        write_csv(stage_dir / OUT_DETAIL_CSV.name, detail_rows)
        write_latex(summary_rows, detail_rows, stage_dir / OUT_SUMMARY_TEX.name, stage_dir / OUT_DETAIL_TEX.name)
        (stage_dir / OUT_ROBUSTNESS_JSON.name).write_text(json.dumps(robustness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        plot_figures(detail_rows, robustness, stage_dir)

        # Preserve the pre-redraw sensitivity/alarm-burden panel as a supplement figure.
        sensitivity_source_pdf = OUT_SENSITIVITY_PDF if OUT_SENSITIVITY_PDF.is_file() else OUT_FIG_PDF
        sensitivity_source_png = OUT_SENSITIVITY_PNG if OUT_SENSITIVITY_PNG.is_file() else OUT_FIG_PNG
        shutil.copy2(sensitivity_source_pdf, stage_dir / OUT_SENSITIVITY_PDF.name)
        shutil.copy2(sensitivity_source_png, stage_dir / OUT_SENSITIVITY_PNG.name)
        # Add generator and final verdict fingerprints after plotting inputs were validated.
        payload = json.loads((stage_dir / OUT_ROBUSTNESS_JSON.name).read_text(encoding="utf-8"))
        payload["generated_from"]["generator_sha256"] = sha256(Path(__file__).resolve())
        payload["generated_from"]["input_sha256"] = {key: sha256(path) for key, path in INPUTS.items()}
        payload["registered_i2_verdict"] = main_i2
        payload["measured_windows_only_i2_descriptive_count"] = {
            "passing_cells": measured["passing_cells"], "cell_count": measured["cell_count"],
        }
        (stage_dir / OUT_ROBUSTNESS_JSON.name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        publish = [
            (stage_dir / OUT_SUMMARY_CSV.name, OUT_SUMMARY_CSV),
            (stage_dir / OUT_DETAIL_CSV.name, OUT_DETAIL_CSV),
            (stage_dir / OUT_ROBUSTNESS_JSON.name, OUT_ROBUSTNESS_JSON),
            (stage_dir / OUT_SUMMARY_TEX.name, OUT_SUMMARY_TEX),
            (stage_dir / OUT_DETAIL_TEX.name, OUT_DETAIL_TEX),
            (stage_dir / OUT_SENSITIVITY_PDF.name, OUT_SENSITIVITY_PDF),
            (stage_dir / OUT_SENSITIVITY_PNG.name, OUT_SENSITIVITY_PNG),
            (stage_dir / "F8_p1_landmark_use_case.pdf", OUT_FIG_PDF),
            (stage_dir / "F8_p1_landmark_use_case.png", OUT_FIG_PNG),
        ]
        collisions = [str(target) for _, target in publish if target.exists() and target not in (OUT_FIG_PDF, OUT_FIG_PNG) and not args.refresh_generated]
        if collisions:
            raise FileExistsError(f"refusing to overwrite generated P1 decomposition outputs: {collisions}")
        if args.refresh_generated:
            archive_root = RESULTS / "_archive" / f"p1_grading_gap_previous_generation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            archive_root.mkdir(parents=True, exist_ok=False)
            archived = []
            for _, target in publish:
                shutil.copy2(target, archive_root / target.name)
                archived.append({"path": str(target.relative_to(ROOT)), "sha256": sha256(target)})
            (archive_root / "archive_manifest.json").write_text(json.dumps({"reason": "preserve prior complete generator output before requested visual refinement", "files": archived}, indent=2) + "\n", encoding="utf-8")
        for staged, target in publish:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
        print(json.dumps({
            "status": "PASS",
            "summary": summary_rows,
            "detail_cell_rows": len(detail_rows),
            "i1_robustness": payload["i1"],
            "i2_robustness": {
                "invariant": payload["i2"]["verdict_invariant_for_any_missing_run_outcome"],
                "min_passing_cells": payload["i2"]["minimum_passing_cells_over_any_missing_run_outcomes"],
                "max_passing_cells": payload["i2"]["maximum_passing_cells_over_any_missing_run_outcomes"],
            },
            "measured_windows_only_i2": payload["measured_windows_only_i2_descriptive_count"],
            "outputs": [str(target.relative_to(ROOT)) for _, target in publish],
        }, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
