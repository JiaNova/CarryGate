#!/usr/bin/env python
"""Run the registered R2 ascertainment tipping-point sensitivity analysis.

Only the frozen P1 one-pass test predictions are read.  Expected counts are
computed analytically for the registered missing-outcome scenarios; no random
imputation or model inference is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/jbi"
PAPER_FIGURES = ROOT / "results/paper"
PAPER_PLAN = ROOT / "paper/JAMIA修订补充分析计划.md"
PLAN_MANIFEST = RESULTS / "revision_sensitivity/analysis_plan_manifest.json"
P1_SCRIPT = ROOT / "scripts/jbi/41_landmark_use_case.py"
P1_PLAN_PATH = "paper/预设用例分析计划_P1.md"
P1_PLAN_SOURCE_COMMIT = "05e2590ca0580e7ca29f41af8a36f5c7f9437444"
P1_LEDGER = RESULTS / "p1_landmark_test_inference_ledger.json"
P1_BATCH = RESULTS / "p1_landmark_test_batch_manifest.json"
P1_VERDICT = RESULTS / "p1_landmark_verdict.json"
P1_METRICS = RESULTS / "p1_landmark_metrics.csv"
P1_CURVES = RESULTS / "p1_landmark_decision_curve.csv"

OUT_CURVE = RESULTS / "p1_ascertainment_tipping_point.csv"
OUT_MODELS = RESULTS / "p1_ascertainment_tipping_point_models.csv"
OUT_THRESHOLDS = RESULTS / "p1_ascertainment_tipping_point_thresholds.csv"
OUT_REPORT = RESULTS / "p1_ascertainment_tipping_point_report.json"
OUT_MANIFEST = RESULTS / "p1_ascertainment_tipping_point_manifest.json"
OUT_PDF = PAPER_FIGURES / "F10_p1_ascertainment_tipping_point.pdf"
OUT_PNG = PAPER_FIGURES / "F10_p1_ascertainment_tipping_point.png"
OUTPUTS = (OUT_CURVE, OUT_MODELS, OUT_THRESHOLDS, OUT_REPORT, OUT_PDF, OUT_PNG, OUT_MANIFEST)

SCENARIOS = (
    ("non_differential", "Non-differential", "q_a=q_n=q"),
    ("model_adverse", "Adverse to model", "q_a=0; q_n=q"),
    ("model_favourable", "Favourable to model", "q_a=q; q_n=0"),
)
Q_GRID = tuple(float(value) for value in np.round(np.concatenate((np.arange(0.0, 0.201, 0.01), [0.30, 0.50, 1.00])), 2))
I2_REQUIRED_MODELS_PER_CELL = 2
I2_REQUIRED_THRESHOLD_WINS = 14
I2_REQUIRED_CELLS = 4
THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.05, 0.301, 0.01), 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_p1_module():
    spec = importlib.util.spec_from_file_location("p1_landmark_frozen", P1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the frozen P1 analysis module: {P1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def counts_for_policy(labels: dict[str, np.ndarray], alarm: np.ndarray) -> dict[str, float]:
    eligible = labels["reference_seen"].astype(bool)
    alarm = np.asarray(alarm, dtype=bool) & eligible
    outcome = labels["outcome"].astype(bool) & eligible
    measured = labels["measured"].astype(bool) & eligible
    unmeasured = eligible & ~measured
    no_alarm = eligible & ~alarm
    return {
        "N": float(eligible.sum()),
        "O": float(outcome.sum()),
        "TP": float((alarm & outcome).sum()),
        "FP": float((alarm & ~outcome).sum()),
        "FN": float((no_alarm & outcome).sum()),
        "TN": float((no_alarm & ~outcome).sum()),
        "A": float(alarm.sum()),
        "N_no_alarm": float(no_alarm.sum()),
        "measured": float(measured.sum()),
        "U_a": float((unmeasured & alarm).sum()),
        "U_n": float((unmeasured & no_alarm).sum()),
        "U": float(unmeasured.sum()),
    }


def scenario_probabilities(name: str, q: float) -> tuple[float, float]:
    if name == "non_differential":
        return q, q
    if name == "model_adverse":
        return 0.0, q
    if name == "model_favourable":
        return q, 0.0
    raise ValueError(f"unknown R2 scenario: {name}")


def adjusted_curve(counts: dict[str, float], q_a: float, q_n: float) -> tuple[float, np.ndarray, np.ndarray]:
    n = counts["N"]
    if n <= 0:
        raise RuntimeError("a P1 component cell has no eligible reference windows")
    tp = counts["TP"] + q_a * counts["U_a"]
    fp = counts["FP"] - q_a * counts["U_a"]
    fn = counts["FN"] + q_n * counts["U_n"]
    tn = counts["TN"] - q_n * counts["U_n"]
    if min(tp, fp, fn, tn) < -1e-10:
        raise RuntimeError("fractional missing-outcome update produced a negative confusion count")
    prevalence = (counts["O"] + q_a * counts["U_a"] + q_n * counts["U_n"]) / n
    odds = np.asarray(THRESHOLDS, dtype=np.float64) / (1.0 - np.asarray(THRESHOLDS, dtype=np.float64))
    model_nb = tp / n - fp / n * odds
    alert_all_nb = prevalence - (1.0 - prevalence) * odds
    return float(prevalence), model_nb, alert_all_nb


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verified_historical_p1_plan(ledger: dict[str, Any], scratch: Path) -> tuple[Path, str]:
    """Materialize the exact committed P1 plan whose hash the test ledger names."""
    plan_source = os.environ.get("ICU_KAN_P1_PLAN_SOURCE")
    if plan_source:
        plan_bytes = Path(plan_source).read_bytes()
    else:
        result = subprocess.run(
            ["git", "show", f"{P1_PLAN_SOURCE_COMMIT}:{P1_PLAN_PATH}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        plan_bytes = result.stdout
    digest = hashlib.sha256(plan_bytes).hexdigest()
    expected = str(ledger.get("analysis_plan_sha256", ""))
    if digest != expected:
        raise RuntimeError(
            "the committed historical P1 plan does not match the test ledger; "
            "refusing to continue with a reconstructed or edited plan"
        )
    path = scratch / "p1_analysis_plan_at_test_inference.md"
    path.write_bytes(plan_bytes)
    return path, digest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty R2 output: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(path: Path, curve_rows: list[dict[str, Any]], qstars: dict[str, float | None]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "scripts/figures"))
    import style

    style.use_style()
    colors = (style.TRACKB_COLOUR["ICUKANForecaster"], style.TRACKB_COLOUR["GraFITi"], style.TRACKB_COLOUR["HyperIMTS"])
    line_styles = ("-", "--", ":")
    labels = {name: label for name, label, _ in SCENARIOS}
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.78)
    for index, (name, _, _) in enumerate(SCENARIOS):
        rows = sorted((row for row in curve_rows if row["scenario"] == name), key=lambda row: float(row["q"]))
        x = np.asarray([float(row["q"]) for row in rows]) * 100.0
        y = np.asarray([int(row["passing_cells"]) for row in rows])
        ax.plot(x, y, color=colors[index], linestyle=line_styles[index], marker="o", markersize=2.8,
                linewidth=1.5, label=labels[name])
        qstar = qstars[name]
        if qstar is not None:
            ax.annotate(f"q*={100*qstar:g}%", (100*qstar, y[list(x).index(100*qstar)]),
                        xytext=(4, 6 + 8 * index), textcoords="offset points", fontsize=6.5,
                        color=colors[index])
    ax.axhline(I2_REQUIRED_CELLS, color=style.MUTED, linewidth=0.9, linestyle="-.",
               label="original I2 requirement (4/8)")
    ax.axvline(5.0, color=style.MUTED, linewidth=0.8, linestyle=":")
    ax.text(5.8, 0.25, "5% interpretation reference", fontsize=6.2, color=style.MUTED, rotation=90,
            va="bottom", ha="left")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 8.35)
    ax.set_xticks([0, 5, 10, 20, 30, 50, 100])
    ax.set_yticks(range(0, 9))
    ax.set_xlabel("Assumed deterioration among unmeasured windows (%)")
    ax.set_ylabel("I2 passing cohort–component cells (of 8)")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center",
               bbox_to_anchor=(0.53, 0.99), ncol=2, fontsize=6.5)
    ax.grid(axis="y", color=style.GRID, linewidth=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def archive_existing_outputs(refresh: bool) -> None:
    existing = [path for path in OUTPUTS if path.exists()]
    if not existing:
        return
    if not refresh:
        raise FileExistsError("R2 outputs already exist; use --refresh-generated to archive the complete prior generation")
    if len(existing) != len(OUTPUTS) or any(not path.is_file() for path in OUTPUTS):
        raise RuntimeError("--refresh-generated requires a complete prior R2 output generation")
    archive = RESULTS / "_archive" / f"r2_ascertainment_tipping_point_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    archive.mkdir(parents=True, exist_ok=False)
    records = []
    for path in OUTPUTS:
        shutil.copy2(path, archive / path.name)
        records.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    (archive / "archive_manifest.json").write_text(
        json.dumps({"reason": "requested refresh of a complete R2 generation", "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive and replace a complete prior R2 output generation")
    args = parser.parse_args()

    required = (PAPER_PLAN, PLAN_MANIFEST, P1_SCRIPT, P1_LEDGER, P1_BATCH, P1_VERDICT, P1_METRICS, P1_CURVES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    registered = json.loads(PLAN_MANIFEST.read_text(encoding="utf-8"))
    if (registered.get("status") != "RULES_FROZEN_BEFORE_NEW_COMPUTATION"
            or registered.get("plan_sha256") != sha256(PAPER_PLAN)
            or "R2" not in registered.get("frozen_tasks", [])):
        raise RuntimeError("R2 analysis plan is not frozen or its recorded SHA256 does not match")
    archive_existing_outputs(args.refresh_generated)

    p1 = load_p1_module()
    ledger_stub = json.loads(P1_LEDGER.read_text(encoding="utf-8"))
    scratch_root = RESULTS / "_work"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r2_p1_plan_validation_", dir=scratch_root) as scratch_name:
        historical_plan, historical_plan_sha = verified_historical_p1_plan(ledger_stub, Path(scratch_name))
        # The loader still performs its original SHA256 guard; point it at the
        # exact Git object named by that guard rather than today's amended file.
        p1.PLAN = historical_plan
        ledger, batch_manifest, files = p1._load_and_validate_inputs()
    verdict = json.loads(P1_VERDICT.read_text(encoding="utf-8"))
    if verdict.get("analysis_scope") != "frozen_test_analysis" or verdict.get("I2", {}).get("passing_cells") != 6:
        raise RuntimeError("R2 requires the frozen P1 test analysis and its archived 6/8 I2 result")

    original_decisions: dict[tuple[str, str, str, float], dict[str, str]] = {}
    for row in read_csv(P1_CURVES):
        if row["scope"] == "all_reference_windows" and row["model_or_strategy"] in p1.MODELS:
            key = (row["cohort"], row["component"], row["model_or_strategy"], round(float(row["threshold_probability"]), 2))
            if key in original_decisions:
                raise RuntimeError(f"duplicate frozen P1 decision curve for {key}")
            original_decisions[key] = row

    coverage = verdict["seed_coverage_by_cohort"]
    cell_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    q0_model_passes: dict[tuple[str, str, str], bool] = {}
    missing_summary: dict[tuple[str, str], dict[str, float]] = {}

    for cohort in p1.COHORTS:
        expected_sample: np.ndarray | None = None
        for component_index, component in enumerate(p1.COMPONENTS):
            labels_by_model_seed: dict[tuple[str, int], dict[str, np.ndarray]] = {}
            counts_by_model: dict[str, list[tuple[int, dict[str, float]]]] = {}
            for model in p1.MODELS:
                observed = coverage[cohort][model]["observed_seeds"]
                counts_by_model[model] = []
                for seed in observed:
                    artifact = files[cohort].get(f"{model}/seed{seed}")
                    if artifact is None:
                        raise RuntimeError(f"ledger/model seed mismatch for {cohort}/{model}/seed{seed}")
                    labels = p1._window_labels(artifact, component_index, component)
                    if expected_sample is None:
                        expected_sample = labels["sample_id"].astype(np.int64)
                    elif not np.array_equal(expected_sample, labels["sample_id"].astype(np.int64)):
                        raise RuntimeError(f"P1 sample order differs across seeds: {cohort}/{model}/seed{seed}")
                    labels_by_model_seed[(model, int(seed))] = labels
                    counts_by_model[model].append((int(seed), counts_for_policy(labels, labels["alarm"])))

            # Measurement ascertainment and reference windows are common to all
            # models/seeds. Assert this before using a shared prevalence model.
            reference_labels: dict[str, np.ndarray] | None = None
            for labels in labels_by_model_seed.values():
                identity = {key: labels[key] for key in ("reference_seen", "measured", "outcome", "reference_grade")}
                if reference_labels is None:
                    reference_labels = identity
                elif any(not np.array_equal(reference_labels[key], identity[key]) for key in reference_labels):
                    raise RuntimeError(f"P1 outcome/reference labels differ across models or seeds: {cohort}/{component['variable']}")
            assert reference_labels is not None
            base_eligible = reference_labels["reference_seen"].astype(bool)
            base_measured = reference_labels["measured"].astype(bool) & base_eligible
            missing_summary[(cohort, component["variable"])] = {
                "n_reference_windows": float(base_eligible.sum()),
                "n_measured_windows": float(base_measured.sum()),
                "n_unmeasured_windows": float((base_eligible & ~base_measured).sum()),
            }

            for scenario, scenario_label, assumption in SCENARIOS:
                for q in Q_GRID:
                    q_a, q_n = scenario_probabilities(scenario, q)
                    model_passes: dict[str, bool] = {}
                    for model in p1.MODELS:
                        seed_results = []
                        for seed, counts in counts_by_model[model]:
                            prevalence, model_nb, alert_all_nb = adjusted_curve(counts, q_a, q_n)
                            seed_results.append((counts, prevalence, model_nb, alert_all_nb))
                        if not seed_results:
                            raise RuntimeError(f"no completed seed predictions for {cohort}/{model}")
                        mean_nb = np.mean(np.stack([item[2] for item in seed_results]), axis=0)
                        mean_alert_all = np.mean(np.stack([item[3] for item in seed_results]), axis=0)
                        wins = (mean_nb > 0.0) & (mean_nb > mean_alert_all)
                        complete = bool(coverage[cohort][model]["complete"])
                        passed = bool(complete and int(wins.sum()) >= I2_REQUIRED_THRESHOLD_WINS)
                        model_passes[model] = passed
                        total_seed_count = len(seed_results)
                        mean_counts = {
                            key: float(np.mean([item[0][key] for item in seed_results]))
                            for key in ("N", "O", "TP", "FP", "FN", "TN", "A", "N_no_alarm", "measured", "U_a", "U_n", "U")
                        }
                        detail = {
                            "analysis_plan_sha256": registered["plan_sha256"],
                            "scenario": scenario,
                            "scenario_label": scenario_label,
                            "missingness_assumption": assumption,
                            "q": q,
                            "q_alarm": q_a,
                            "q_no_alarm": q_n,
                            "cohort": cohort,
                            "organ": component["organ"],
                            "component": component["variable"],
                            "model": model,
                            "observed_seeds": ",".join(str(seed) for seed, _ in counts_by_model[model]),
                            "observed_seed_count": total_seed_count,
                            "required_seed_count": len(p1.SEEDS),
                            "seed_coverage_complete": complete,
                            "n_reference_windows": mean_counts["N"],
                            "n_measured_windows": mean_counts["measured"],
                            "n_unmeasured_windows": mean_counts["U"],
                            "unmeasured_fraction": mean_counts["U"] / mean_counts["N"],
                            "mean_unmeasured_alarm_windows": mean_counts["U_a"],
                            "mean_unmeasured_non_alarm_windows": mean_counts["U_n"],
                            "fraction_alarm_windows_unmeasured": mean_counts["U_a"] / mean_counts["A"] if mean_counts["A"] else float("nan"),
                            "fraction_non_alarm_windows_unmeasured": mean_counts["U_n"] / mean_counts["N_no_alarm"] if mean_counts["N_no_alarm"] else float("nan"),
                            "mean_observed_positive_windows": mean_counts["O"],
                            "mean_adjusted_prevalence": float(np.mean([item[1] for item in seed_results])),
                            "mean_adjusted_tp": mean_counts["TP"] + q_a * mean_counts["U_a"],
                            "mean_adjusted_fp": mean_counts["FP"] - q_a * mean_counts["U_a"],
                            "mean_adjusted_fn": mean_counts["FN"] + q_n * mean_counts["U_n"],
                            "mean_adjusted_tn": mean_counts["TN"] - q_n * mean_counts["U_n"],
                            "thresholds_won": int(wins.sum()),
                            "required_threshold_wins": I2_REQUIRED_THRESHOLD_WINS,
                            "model_pass": passed,
                        }
                        model_rows.append(detail)
                        for index, threshold in enumerate(THRESHOLDS):
                            threshold_rows.append({
                                "analysis_plan_sha256": registered["plan_sha256"],
                                "scenario": scenario,
                                "q": q,
                                "cohort": cohort,
                                "component": component["variable"],
                                "model": model,
                                "threshold_probability": threshold,
                                "mean_model_net_benefit": float(mean_nb[index]),
                                "mean_no_alarm_net_benefit": 0.0,
                                "mean_alert_all_net_benefit": float(mean_alert_all[index]),
                                "strictly_exceeds_both": bool(wins[index]),
                                "seed_coverage_complete": complete,
                            })

                    n_pass = int(sum(model_passes.values()))
                    cell_rows.append({
                        "analysis_plan_sha256": registered["plan_sha256"],
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "missingness_assumption": assumption,
                        "q": q,
                        "cohort": cohort,
                        "organ": component["organ"],
                        "component": component["variable"],
                        "models_passing": n_pass,
                        "cell_pass": n_pass >= I2_REQUIRED_MODELS_PER_CELL,
                        "model_passes": json.dumps(model_passes, sort_keys=True),
                        "required_models_passing": I2_REQUIRED_MODELS_PER_CELL,
                    })

    if len(cell_rows) != len(SCENARIOS) * len(Q_GRID) * 8:
        raise RuntimeError(f"R2 fixed grid incomplete: expected {len(SCENARIOS) * len(Q_GRID) * 8} cell rows, got {len(cell_rows)}")

    original_cells = {
        (row["cohort"], row["component"]): row for row in verdict["I2_cells"]
    }
    q0_cells = {}
    for row in cell_rows:
        if row["scenario"] == "non_differential" and row["q"] == 0.0:
            q0_cells[(row["cohort"], row["component"])] = row
    if set(q0_cells) != set(original_cells):
        raise RuntimeError("R2 q=0 cell set differs from the frozen P1 I2 verdict")
    for key, row in q0_cells.items():
        expected = original_cells[key]
        observed_passes = json.loads(row["model_passes"])
        if observed_passes != expected["model_passes"] or bool(row["cell_pass"]) != bool(expected["cell_pass"]):
            raise RuntimeError(f"R2 q=0 model/cell decision differs from frozen P1 verdict for {key}")
    q0_count = sum(bool(row["cell_pass"]) for row in q0_cells.values())
    if q0_count != 6:
        raise RuntimeError(f"R2 q=0 must reproduce 6/8 I2; obtained {q0_count}/8")

    q0_thresholds = [row for row in threshold_rows if row["scenario"] == "non_differential" and row["q"] == 0.0]
    for row in q0_thresholds:
        key = (row["cohort"], row["component"], row["model"], round(float(row["threshold_probability"]), 2))
        original = original_decisions.get(key)
        if original is None:
            raise RuntimeError(f"missing frozen P1 net benefit for R2 q=0 check: {key}")
        if not np.isclose(row["mean_model_net_benefit"], float(original["net_benefit"]), atol=1e-12, rtol=0.0):
            raise RuntimeError(f"R2 q=0 model net benefit differs from frozen P1 curve for {key}")
        if not np.isclose(row["mean_alert_all_net_benefit"], float(original["treat_all_net_benefit"]), atol=1e-12, rtol=0.0):
            raise RuntimeError(f"R2 q=0 treat-all net benefit differs from frozen P1 curve for {key}")

    curve_rows = []
    qstars: dict[str, float | None] = {}
    for scenario, scenario_label, assumption in SCENARIOS:
        this_scenario = [row for row in cell_rows if row["scenario"] == scenario]
        for q in Q_GRID:
            at_q = [row for row in this_scenario if float(row["q"]) == q]
            passing = int(sum(bool(row["cell_pass"]) for row in at_q))
            curve_rows.append({
                "analysis_plan_sha256": registered["plan_sha256"],
                "scenario": scenario,
                "scenario_label": scenario_label,
                "missingness_assumption": assumption,
                "q": q,
                "passing_cells": passing,
                "total_cells": 8,
                "original_i2_required_cells": I2_REQUIRED_CELLS,
                "i2_remains_met": passing >= I2_REQUIRED_CELLS,
            })
        tipping = next((float(row["q"]) for row in curve_rows if row["scenario"] == scenario and int(row["passing_cells"]) < I2_REQUIRED_CELLS), None)
        qstars[scenario] = tipping

    # The adverse scenario is the pre-specified manuscript interpretation.
    adverse_qstar = qstars["model_adverse"]
    if adverse_qstar is None:
        wording = "I2 remained met throughout the evaluated adverse-missingness grid (q<=1.00)."
    elif adverse_qstar >= 0.05:
        wording = f"I2 remained met unless at least {100 * adverse_qstar:g}% of unmeasured non-alarmed windows concealed deterioration."
    else:
        wording = "The decision-value result was sensitive to plausible ascertainment bias."

    report = {
        "schema_version": "p1_ascertainment_tipping_point_v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_path": str(PAPER_PLAN.relative_to(ROOT)),
        "analysis_plan_sha256": registered["plan_sha256"],
        "analysis_plan_commit": registered["plan_commit"],
        "p1_plan_used_for_test_inference": {
            "git_commit": P1_PLAN_SOURCE_COMMIT,
            "git_path": P1_PLAN_PATH,
            "sha256": historical_plan_sha,
            "current_plan_sha256": sha256(ROOT / P1_PLAN_PATH),
            "current_document_differs_from_test_version": historical_plan_sha != sha256(ROOT / P1_PLAN_PATH),
            "validation": "the original P1 loader SHA256 check passed using this exact committed source",
        },
        "analysis_type": "post hoc sensitivity analysis; reporting rules fixed before computation",
        "test_predictions_previously_viewed": True,
        "n_prediction_files_loaded": len(ledger["run_results"]),
        "p1_completed_run_count": int(ledger["completed_run_count"]),
        "p1_planned_run_count": int(ledger["planned_run_count"]),
        "q_grid": list(Q_GRID),
        "scenarios": {name: {"label": label, "assumption": assumption, "q_star": qstars[name],
                              "q_star_reached": qstars[name] is not None}
                      for name, label, assumption in SCENARIOS},
        "threshold_grid": list(THRESHOLDS),
        "i2_rule": {
            "model_pass": "complete seed coverage and mean net benefit strictly > 0 and > alert-all for at least 14 of 26 thresholds",
            "cell_pass": "at least 2 of 3 models pass",
            "overall_pass": "at least 4 of 8 cohort-component cells pass",
        },
        "q0_self_check": {
            "status": "PASS",
            "passing_cells": q0_count,
            "matches_frozen_model_and_cell_decisions": True,
            "matches_frozen_model_and_alert_all_net_benefit_to_1e-12": True,
        },
        "adverse_scenario_manuscript_wording": wording,
        "unmeasured_window_summary": [
            {"cohort": cohort, "component": component,
             "n_reference_windows": int(values["n_reference_windows"]),
             "n_measured_windows": int(values["n_measured_windows"]),
             "n_unmeasured_windows": int(values["n_unmeasured_windows"]),
             "unmeasured_fraction": values["n_unmeasured_windows"] / values["n_reference_windows"]}
            for (cohort, component), values in sorted(missing_summary.items())
        ],
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PAPER_PLAN, PLAN_MANIFEST, ROOT / P1_PLAN_PATH, P1_SCRIPT,
                         P1_LEDGER, P1_BATCH, P1_VERDICT, P1_METRICS, P1_CURVES)
        },
        "p1_historical_plan_git_sha256": historical_plan_sha,
        "prediction_sha256": {
            key: value["sha256"] for key, value in sorted(ledger["run_results"].items())
        },
        "environment": {"python": sys.version, "numpy": np.__version__},
    }

    stage_root = RESULTS / "_work"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="r2_ascertainment_tipping_point_", dir=stage_root))
    try:
        staged_curve = stage / OUT_CURVE.name
        staged_models = stage / OUT_MODELS.name
        staged_thresholds = stage / OUT_THRESHOLDS.name
        staged_report = stage / OUT_REPORT.name
        staged_manifest = stage / OUT_MANIFEST.name
        staged_pdf = stage / OUT_PDF.name
        staged_png = stage / OUT_PNG.name
        write_csv(staged_curve, curve_rows)
        write_csv(staged_models, model_rows)
        write_csv(staged_thresholds, threshold_rows)
        draw_figure(staged_pdf, curve_rows, qstars)
        draw_figure(staged_png, curve_rows, qstars)
        staged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged_to_target = (
            (staged_curve, OUT_CURVE), (staged_models, OUT_MODELS), (staged_thresholds, OUT_THRESHOLDS),
            (staged_report, OUT_REPORT), (staged_pdf, OUT_PDF), (staged_png, OUT_PNG),
        )
        output_hashes = {str(target.relative_to(ROOT)): sha256(staged) for staged, target in staged_to_target}
        manifest = {
            "schema_version": "p1_ascertainment_tipping_point_manifest_v1",
            "status": "PASS",
            "analysis_plan_sha256": registered["plan_sha256"],
            "generator_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "generator_sha256": sha256(Path(__file__).resolve()),
            "generator_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "p1_analysis_module_sha256": sha256(P1_SCRIPT),
            "q0_self_check": report["q0_self_check"],
            "outputs": output_hashes,
            "generated_at_utc": report["generated_at_utc"],
        }
        staged_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for staged, target in staged_to_target + ((staged_manifest, OUT_MANIFEST),):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps({
        "status": "PASS",
        "q0_i2": f"{q0_count}/8",
        "q_star": qstars,
        "adverse_wording": wording,
        "curve_rows": len(curve_rows),
        "model_rows": len(model_rows),
        "threshold_rows": len(threshold_rows),
        "outputs": [str(path.relative_to(ROOT)) for path in OUTPUTS],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
