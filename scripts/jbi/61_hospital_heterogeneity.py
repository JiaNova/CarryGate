#!/usr/bin/env python
"""Registered post hoc hospital-level heterogeneity analysis (R6).

Only the 90 frozen P2 eICU test prediction files are used.  The preflight
binds the complete frozen inference ledger, fold batch manifests, the R4
preflight hashes, the analysis plan and every P2 NPZ before any new metric is
calculated.  No model is trained and no site-level output is included in the
figure or summary table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi" / "revision_sensitivity"
PAPER_OUT = ROOT / "results" / "paper"
SCRIPT_PATH = Path(__file__).resolve()
PLAN_PATH = ROOT / "paper" / "JAMIA修订补充分析计划.md"
PLAN_MANIFEST_PATH = OUT / "analysis_plan_manifest.json"
R4_PREFLIGHT_PATH = OUT / "r4_preflight_manifest.json"
LEDGER_PATH = ROOT / "results" / "jbi" / "p2_loho_test_inference_ledger.json"
BATCH_PATH = ROOT / "results" / "jbi" / "p2_loho_test_batch_manifest.json"
FREEZE_PATH = ROOT / "results" / "jbi" / "p2_loho_checkpoint_freeze.json"
PREFLIGHT_PATH = OUT / "r6_preflight_manifest.json"
EXPECTED_PLAN_SHA256 = "fade749e20c54e9340dbbb01ec9a6c8e12a8ea912eb34baec7ccced30e38e320"
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
DISPLAY_MODEL = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi", "HyperIMTS": "HyperIMTS"}
OBJECTIVES = ("MSE", "MAE")
OBJECTIVE_LABEL = {"MSE": "MSE", "MAE": "masked L1"}
SEEDS = (2024, 2025, 2026)
FOLDS = tuple(range(5))
N_VARIABLES = 37
UNSCORED = frozenset({6, 19, 32})
MIN_EVENTS = 50
MIN_HOSPITAL_STAYS = 100
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261012

OUTPUTS = {
    "hospital_scores": OUT / "r6_hospital_scores.csv",
    "hospital_values": OUT / "r6_hospital_values.csv",
    "summary": OUT / "r6_hospital_summary.csv",
    "small_summary": OUT / "r6_small_hospital_summary.csv",
    "correlations": OUT / "r6_hospital_correlations.csv",
    "report": OUT / "r6_report.json",
    "table_tex": OUT / "r6_S12_hospital_summary.tex",
    "figure_pdf": PAPER_OUT / "S12_r6_hospital_heterogeneity.pdf",
    "figure_png": PAPER_OUT / "S12_r6_hospital_heterogeneity.png",
    "manifest": OUT / "r6_manifest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_record(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        **extra,
    }


def expected_run_keys() -> set[str]:
    return {
        f"fold{fold}/{model}/{objective}/seed{seed}"
        for fold in FOLDS for model in MODELS
        for objective in OBJECTIVES for seed in SEEDS
    }


def npz_member_shape(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read an NPY header from an NPZ without inflating a large event array."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, _fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(x) for x in shape), np.dtype(dtype)


def _same_vector(arrays: Any, left: str, right: str) -> bool:
    return np.array_equal(arrays[left], arrays[right])


def preflight() -> dict[str, Any]:
    started = time.time()
    failures: list[str] = []
    input_hashes: list[dict[str, Any]] = []
    counts = {"p2_prediction_files": 0, "p2_completed_runs": 0,
              "folds": 0, "large_hospitals": 0, "small_hospitals": 0,
              "scoreable_variables": N_VARIABLES - len(UNSCORED)}

    plan_sha = sha256_file(PLAN_PATH)
    plan_manifest = load_json(PLAN_MANIFEST_PATH)
    if (plan_sha != EXPECTED_PLAN_SHA256
            or plan_manifest.get("status") != "RULES_FROZEN_BEFORE_NEW_COMPUTATION"
            or plan_manifest.get("new_computation_started") is not False
            or plan_manifest.get("plan_sha256") != EXPECTED_PLAN_SHA256
            or "R6" not in plan_manifest.get("frozen_tasks", [])):
        failures.append("registered R6 plan hash/status/task mismatch")
    for path, role in ((PLAN_PATH, "frozen_analysis_plan"),
                        (PLAN_MANIFEST_PATH, "analysis_plan_registration"),
                        (R4_PREFLIGHT_PATH, "R4_frozen_input_preflight"),
                        (LEDGER_PATH, "P2_test_inference_ledger"),
                        (BATCH_PATH, "P2_test_batch_manifest"),
                        (FREEZE_PATH, "P2_checkpoint_freeze")):
        input_hashes.append(hash_record(path, role))

    r4 = load_json(R4_PREFLIGHT_PATH)
    if r4.get("status") != "PASS" or r4.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA256:
        failures.append("R4 preflight is not a passing preflight under the frozen plan")
    r4_predictions = {
        row.get("run_key"): row for row in r4.get("input_hashes", [])
        if row.get("role") == "p2_frozen_prediction_npz"
    }
    ledger = load_json(LEDGER_PATH)
    batch_doc = load_json(BATCH_PATH)
    freeze_doc = load_json(FREEZE_PATH)
    if (ledger.get("status") != "COMPLETE" or ledger.get("checkpoint_count") != 90
            or ledger.get("completed_run_count") != 90
            or len(ledger.get("run_results", {})) != 90
            or set(ledger.get("run_results", {})) != expected_run_keys()):
        failures.append("P2 ledger does not contain exactly the 90 registered completed runs")
    if (freeze_doc.get("checkpoint_count") != 90
            or freeze_doc.get("test_inference_ledger_sha256") != sha256_file(LEDGER_PATH)
            or sha256_file(BATCH_PATH) != ledger.get("test_batch_manifest_sha256")):
        failures.append("P2 freeze/ledger/batch manifest hash binding mismatch")
    fold_docs = batch_doc.get("folds", {})
    ledger_fold_docs = ledger.get("fold_batch_manifests", {})
    if set(fold_docs) != {str(i) for i in FOLDS}:
        failures.append("P2 batch manifest does not contain exactly five folds")

    fold_identity: dict[int, dict[str, np.ndarray]] = {}
    fold_test_identity: dict[int, tuple[np.ndarray, ...]] = {}
    all_hospital_stays: dict[tuple[int, str], int] = {}
    seen_site_fold: dict[str, int] = {}
    for key in sorted(expected_run_keys()):
        record = ledger.get("run_results", {}).get(key, {})
        counts["p2_completed_runs"] += int(record.get("status") == "completed")
        relpath = record.get("path")
        path = ROOT / str(relpath)
        if record.get("status") != "completed" or not path.is_file():
            failures.append(f"missing or incomplete frozen P2 output: {key}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            failures.append(f"P2 ledger SHA256 mismatch: {key}")
        prior = r4_predictions.get(key)
        if not prior or prior.get("sha256") != actual_hash or prior.get("path") != str(path.relative_to(ROOT)):
            failures.append(f"P2 prediction does not match R4 preflight hash: {key}")
        input_hashes.append(hash_record(path, "P2_frozen_prediction_npz", run_key=key))
        counts["p2_prediction_files"] += 1
        fold = int(key.split("/", 1)[0].removeprefix("fold"))
        fold_info = fold_docs.get(str(fold), {})
        expected_batch_sha = ledger_fold_docs.get(str(fold), {}).get("sha256")
        if (record.get("fold_batch_manifest_sha256") != expected_batch_sha
                or fold_info.get("sha256") != expected_batch_sha):
            failures.append(f"P2 batch manifest identity mismatch: {key}")
        try:
            with np.load(path, allow_pickle=False) as z:
                if (str(z["run_key"].item()) != key
                        or int(z["fold"].item()) != fold
                        or str(z["model"].item()) != key.split("/")[1]
                        or str(z["objective"].item()) != key.split("/")[2]
                        or int(z["seed"].item()) != int(key.rsplit("seed", 1)[1])
                        or str(z["fold_batch_manifest_sha256"].item()) != expected_batch_sha):
                    failures.append(f"NPZ run identity metadata mismatch: {key}")
                sample_index = z["test_sample_index"].astype(np.int64)
                test_stay = z["test_stay_id"].astype(str)
                test_patient = z["test_patient_id"].astype(str)
                test_hospital = z["test_hospital_id"]
                event_count = int(record.get("test_event_count", -1))
                if (len(z["variable_names"]) != N_VARIABLES
                        or sample_index.size != int(record.get("test_stay_count", -2))
                        or test_stay.size != sample_index.size
                        or test_patient.size != sample_index.size
                        or test_hospital.size != sample_index.size
                        or len(set(sample_index.tolist())) != sample_index.size
                        or len(set(test_stay.tolist())) != test_stay.size):
                    failures.append(f"P2 NPZ event/stay dimensions or fixed roster mismatch: {key}")
                event_members = ("event_sample_index", "event_stay_id", "event_patient_id",
                                 "event_hospital_id", "event_variable_index", "truth_standardized",
                                 "prediction_standardized", "carry_forward_standardized")
                for member in event_members:
                    shape, _dtype = npz_member_shape(path, member)
                    if shape != (event_count,):
                        failures.append(f"P2 NPZ {member} event shape mismatch: {key}")
                batches = fold_info.get("batches", [])
                expected_samples = [int(index) for batch in batches
                                    for index in batch.get("sample_indices", [])]
                if expected_samples != sample_index.tolist():
                    failures.append(f"P2 sample sequence differs from frozen batch order: {key}")
                test_identity = (sample_index, test_stay, test_patient, test_hospital)
                if fold not in fold_test_identity:
                    fold_test_identity[fold] = tuple(np.asarray(x).copy() for x in test_identity)
                elif not all(np.array_equal(a, b) for a, b in zip(fold_test_identity[fold], test_identity)):
                    failures.append(f"P2 test stay/patient/hospital roster differs within fold: {key}")

                # One file per fold is sufficient for large event identity arrays;
                # every other file is already bound by its full hash and checked
                # against R4's passing 90-file preflight.
                if fold not in fold_identity:
                    event_sample = z["event_sample_index"].astype(np.int64)
                    event_hosp = z["event_hospital_id"]
                    var = z["event_variable_index"].astype(np.int64)
                    truth = z["truth_standardized"]
                    carry = z["carry_forward_standardized"]
                    if (var.min(initial=0) < 0 or var.max(initial=-1) >= N_VARIABLES
                            or not np.isfinite(truth).all() or not np.isfinite(carry).all()):
                        failures.append(f"P2 reference event identity/value range mismatch: {key}")
                    sample_sort = np.argsort(sample_index)
                    sorted_samples = sample_index[sample_sort]
                    event_pos = np.searchsorted(sorted_samples, event_sample)
                    if (np.any(event_pos >= sorted_samples.size)
                            or not np.array_equal(sorted_samples[event_pos], event_sample)):
                        failures.append(f"P2 event references an unknown test sample: {key}")
                    else:
                        event_rows = sample_sort[event_pos]
                        if not np.array_equal(event_hosp, test_hospital[event_rows]):
                            failures.append(f"P2 event-to-test-hospital mapping mismatch: {key}")
                    fold_identity[fold] = {
                        "event_sample_index": event_sample.copy(),
                        "event_hospital_id": event_hosp.copy(),
                        "event_variable_index": var.copy(),
                        "truth_standardized": truth.copy(),
                        "carry_forward_standardized": carry.copy(),
                    }
                site_sizes: dict[str, set[str]] = {}
                for h, stay in zip(test_hospital.tolist(), test_stay.tolist()):
                    hospital = str(h)
                    site_sizes.setdefault(hospital, set()).add(stay)
                for hospital, stays in site_sizes.items():
                    all_hospital_stays[(fold, hospital)] = len(stays)
                    if hospital in seen_site_fold and seen_site_fold[hospital] != fold:
                        failures.append(f"hospital identity occurs in multiple held-out folds: {hospital}")
                    seen_site_fold[hospital] = fold
        except Exception as exc:
            failures.append(f"unreadable or invalid P2 prediction {key}: {type(exc).__name__}: {exc}")

    counts["folds"] = len(fold_identity)
    counts["large_hospitals"] = sum(n >= MIN_HOSPITAL_STAYS for n in all_hospital_stays.values())
    counts["small_hospitals"] = sum(n < MIN_HOSPITAL_STAYS for n in all_hospital_stays.values())
    if (counts["p2_prediction_files"] != 90 or counts["p2_completed_runs"] != 90
            or counts["folds"] != 5 or not all_hospital_stays):
        failures.append("R6 fixed analysis cardinality failed: expected 90 predictions and five folds")
    if counts["large_hospitals"] == 0:
        failures.append("no held-out hospital meets the fixed 100-stay reporting threshold")

    return {
        "schema_version": "jamia_r6_hospital_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "preflighted_at_utc": utc_now(),
        "analysis_plan_sha256": plan_sha,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "python_version": sys.version,
        "counts": counts,
        "input_hashes": input_hashes,
        "input_hash_count": len(input_hashes),
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def verify_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("R6 preflight is missing; run --preflight-only first")
    record = load_json(PREFLIGHT_PATH)
    if (record.get("status") != "PASS"
            or record.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA256
            or record.get("script_sha256") != sha256_file(SCRIPT_PATH)):
        raise RuntimeError("R6 preflight does not match the current script/plan")
    for item in record.get("input_hashes", []):
        path = ROOT / item["path"]
        if (not path.is_file() or path.stat().st_size != item.get("size_bytes")
                or path.stat().st_mtime_ns != item.get("mtime_ns")
                or sha256_file(path) != item.get("sha256")):
            raise RuntimeError(f"R6 input changed after preflight: {item['path']}")
    return record


def _grouped_sum(hospital_code: np.ndarray, variable: np.ndarray,
                 value: np.ndarray, n_hospitals: int) -> np.ndarray:
    keep = ~np.isin(variable, tuple(UNSCORED))
    group = hospital_code[keep] * N_VARIABLES + variable[keep]
    return np.bincount(group, weights=value[keep], minlength=n_hospitals * N_VARIABLES).reshape(
        n_hospitals, N_VARIABLES
    )


def _macro_skill(model_error: np.ndarray, baseline_error: np.ndarray,
                 event_count: np.ndarray) -> tuple[float, int]:
    eligible = (event_count >= MIN_EVENTS) & (baseline_error > 1e-9)
    eligible[list(UNSCORED)] = False
    if not eligible.any():
        return float("nan"), 0
    values = 1.0 - model_error[eligible] / baseline_error[eligible]
    if not np.isfinite(values).all():
        raise AssertionError("non-finite hospital macro skill")
    return float(np.mean(values)), int(eligible.sum())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Hospital-level variation in eICU internal held-hospital test predictions. Individual hospital IDs and rankings are omitted; each summary includes only hospitals with at least 100 held-out test stays. Intervals are 2,000-replicate hospital-cluster bootstrap percentile intervals.}",
        r"\label{S-tab:r6-hospital-heterogeneity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Contrast & Hospitals & Positive, n/N (\%; 95\% CI) & Hospital median [95\% CI] & Patient-weighted mean [95\% CI] " + r"\\",
        r"\midrule",
    ]
    for row in summary_rows:
        if row["metric_type"] == "objective_effect":
            contrast = f"{row['group_label']}: masked L1 $-$ MSE"
        elif row["metric_type"] == "architecture_difference":
            contrast = f"{row['contrast_label']} ({row['group_label']})"
        else:
            contrast = f"C3 median pairwise gap ({row['group_label']})"
        contrast = contrast.replace(" − ", r" $-$ ")
        contrast = contrast.replace("&", r"\&")
        if row["metric_type"] == "c3_pairwise_median_absolute_gap":
            positive = "---"
        else:
            positive = (f"{int(row['positive_hospitals'])}/{int(row['n_hospitals'])} "
                        f"({100 * float(row['positive_fraction']):.1f}) "
                        f"[{100 * float(row['positive_fraction_ci_low']):.1f}, "
                        f"{100 * float(row['positive_fraction_ci_high']):.1f}]")
        median = (f"{float(row['hospital_median']):.3f} "
                  f"[{float(row['median_ci_low']):.3f}, {float(row['median_ci_high']):.3f}]")
        weighted = (f"{float(row['patient_weighted_mean']):.3f} "
                    f"[{float(row['weighted_mean_ci_low']):.3f}, "
                    f"{float(row['weighted_mean_ci_high']):.3f}]")
        lines.append(f"{contrast} & {int(row['n_hospitals'])} & {positive} & {median} & {weighted} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    OUTPUTS["table_tex"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_stats(values: np.ndarray, weights: np.ndarray,
                     rng: np.random.Generator) -> dict[str, float]:
    n = len(values)
    if n < 1:
        raise ValueError("cannot summarize a contrast with no eligible hospitals")
    draws = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    draw_values = values[draws]
    draw_weights = weights[draws]
    frac = np.mean(draw_values > 0, axis=1)
    med = np.median(draw_values, axis=1)
    weighted = np.sum(draw_values * draw_weights, axis=1) / np.sum(draw_weights, axis=1)
    return {
        "positive_fraction": float(np.mean(values > 0)),
        "positive_fraction_ci_low": float(np.percentile(frac, 2.5)),
        "positive_fraction_ci_high": float(np.percentile(frac, 97.5)),
        "hospital_median": float(np.median(values)),
        "median_ci_low": float(np.percentile(med, 2.5)),
        "median_ci_high": float(np.percentile(med, 97.5)),
        "patient_weighted_mean": float(np.average(values, weights=weights)),
        "weighted_mean_ci_low": float(np.percentile(weighted, 2.5)),
        "weighted_mean_ci_high": float(np.percentile(weighted, 97.5)),
        "bootstrap_reps": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _make_figure(detail: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.7), constrained_layout=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    colors = {"MSE": "#3977a8", "MAE": "#d08036"}

    targets = [row for row in detail if row["metric_type"] == "objective_effect"]
    groups = ["CarryGate", "GraFITi", "HyperIMTS"]
    distributions = [[float(r["estimate"]) for r in targets if r["group_label"] == g]
                     for g in groups]
    axes[0].boxplot(distributions, positions=np.arange(1, len(groups) + 1), widths=0.45,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.4})
    for x, (group, vals) in enumerate(zip(groups, distributions), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        axes[0].scatter(x + jitter, vals, s=18, alpha=0.55, color="#6b6b6b", linewidths=0)
        axes[0].text(x, 0.98, f"n={len(vals)}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
    axes[0].axhline(0, color="#a32f2f", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(1, 4), groups)
    axes[0].set_title("A. Objective effect by architecture")
    axes[0].set_ylabel("masked L1 − MSE persistence skill")

    arch_rows = [row for row in detail if row["metric_type"] == "architecture_difference"]
    arch_groups = [(obj, comp) for obj in OBJECTIVES for comp in
                   ("CarryGate − GraFITi", "CarryGate − HyperIMTS")]
    arch_data = [[float(r["estimate"]) for r in arch_rows
                  if r["contrast_label"] == comp and r["objective"] == obj]
                 for obj, comp in arch_groups]
    positions = np.arange(1, len(arch_groups) + 1)
    axes[1].boxplot(arch_data, positions=positions, widths=0.45, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.4})
    for x, ((obj, _), vals) in enumerate(zip(arch_groups, arch_data), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        axes[1].scatter(x + jitter, vals, s=17, alpha=0.5, color=colors[obj], linewidths=0)
        axes[1].text(x, 0.98, f"n={len(vals)}", transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
    axes[1].axhline(0, color="#a32f2f", linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, [f"{OBJECTIVE_LABEL[o]}\n{c.replace('CarryGate − ', 'vs ')}"
                                  for o, c in arch_groups], fontsize=8)
    axes[1].set_title("B. Paired architecture differences")
    axes[1].set_ylabel("CarryGate − comparator skill")

    c3_rows = [row for row in detail if row["metric_type"] == "c3_pairwise_median_absolute_gap"]
    c3_data = [[float(r["estimate"]) for r in c3_rows if r["objective"] == obj]
               for obj in OBJECTIVES]
    axes[2].boxplot(c3_data, positions=[1, 2], widths=0.45, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.4})
    for x, (obj, vals) in enumerate(zip(OBJECTIVES, c3_data), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        axes[2].scatter(x + jitter, vals, s=18, alpha=0.55, color=colors[obj], linewidths=0)
        axes[2].text(x, 0.98, f"n={len(vals)}", transform=axes[2].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
    axes[2].set_xticks([1, 2], [OBJECTIVE_LABEL[o] for o in OBJECTIVES])
    axes[2].set_title("C. P2 C3-style architecture gap")
    axes[2].set_ylabel("median pairwise absolute skill gap")
    fig.suptitle("Hospital-level heterogeneity in eICU internal held-hospital predictions\n"
                 "Post hoc sensitivity analysis; sites shown without identifiers or ranking",
                 fontsize=11)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS["figure_pdf"], bbox_inches="tight")
    fig.savefig(OUTPUTS["figure_png"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(preflight_doc: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    ledger = load_json(LEDGER_PATH)
    # Each site is keyed internally by fold and frozen hospital ID.  The
    # published detail file replaces these identifiers with an opaque row key.
    stay_sets: dict[tuple[int, int], set[str]] = {}
    patient_sets: dict[tuple[int, int], set[str]] = {}
    event_count_by_site_var: dict[tuple[int, int], np.ndarray] = {}
    baseline_abs_by_site_var: dict[tuple[int, int], np.ndarray] = {}
    model_abs: dict[tuple[int, int, str, str, int], np.ndarray] = {}
    scored_event_count_by_site: dict[tuple[int, int], int] = {}

    for key in sorted(expected_run_keys()):
        rec = ledger["run_results"][key]
        path = ROOT / rec["path"]
        fold = int(key.split("/", 1)[0].removeprefix("fold"))
        model, objective = key.split("/")[1:3]
        seed = int(key.rsplit("seed", 1)[1])
        with np.load(path, allow_pickle=False) as z:
            test_hosp = z["test_hospital_id"]
            test_stay = z["test_stay_id"].astype(str)
            test_patient = z["test_patient_id"].astype(str)
            event_hosp = z["event_hospital_id"]
            variable = z["event_variable_index"].astype(np.int64)
            truth = z["truth_standardized"].astype(np.float64)
            prediction = z["prediction_standardized"].astype(np.float64)
            carry = z["carry_forward_standardized"].astype(np.float64)
            if (not np.isfinite(truth).all() or not np.isfinite(prediction).all()
                    or not np.isfinite(carry).all()):
                raise AssertionError(f"non-finite frozen prediction values: {key}")
            unique_hosp, hosp_code = np.unique(event_hosp, return_inverse=True)
            for hospital in np.unique(test_hosp):
                site_key = (fold, int(hospital))
                sel_stay = test_hosp == hospital
                site_stays = set(test_stay[sel_stay].tolist())
                site_patients = set(test_patient[sel_stay].tolist())
                stay_sets.setdefault(site_key, set()).update(site_stays)
                patient_sets.setdefault(site_key, set()).update(site_patients)
            # The integer event_hospital_id column directly indexes hospital groups.
            event_hospital_code = hosp_code
            event_n = len(unique_hosp)
            base_count = _grouped_sum(event_hospital_code, variable,
                                      np.ones(variable.size), event_n)
            base_abs = _grouped_sum(event_hospital_code, variable,
                                    np.abs(carry - truth), event_n)
            pred_abs = _grouped_sum(event_hospital_code, variable,
                                    np.abs(prediction - truth), event_n)
            scoreable = ~np.isin(variable, tuple(UNSCORED))
            site_scored_count = np.bincount(event_hospital_code[scoreable],
                                            minlength=event_n)
            for i, hospital in enumerate(unique_hosp.tolist()):
                site_key = (fold, int(hospital))
                counts_i = base_count[i].astype(np.int64)
                if site_key in event_count_by_site_var:
                    if (not np.array_equal(event_count_by_site_var[site_key], counts_i)
                            or not np.array_equal(baseline_abs_by_site_var[site_key], base_abs[i])):
                        raise AssertionError(f"baseline/event grid changed across seeds: {site_key}")
                else:
                    event_count_by_site_var[site_key] = counts_i
                    baseline_abs_by_site_var[site_key] = base_abs[i].copy()
                scored_event_count_by_site[site_key] = int(site_scored_count[i])
                model_abs[(fold, int(hospital), model, objective, seed)] = pred_abs[i].copy()

    site_meta: dict[tuple[int, int], dict[str, Any]] = {}
    for site_key in sorted(stay_sets):
        stays = len(stay_sets[site_key])
        patients = len(patient_sets[site_key])
        site_meta[site_key] = {
            "fold": site_key[0], "n_test_stays": stays, "n_test_patients": patients,
            "n_scoreable_measurements": scored_event_count_by_site[site_key],
            "measurement_density": scored_event_count_by_site[site_key] / stays,
            "n_eligible_variables": int(((event_count_by_site_var[site_key] >= MIN_EVENTS)
                & (baseline_abs_by_site_var[site_key] > 1e-9)
                & ~np.isin(np.arange(N_VARIABLES), tuple(UNSCORED))).sum()),
        }
    large_sites = [site for site, meta in site_meta.items()
                   if meta["n_test_stays"] >= MIN_HOSPITAL_STAYS]
    small_sites = [site for site, meta in site_meta.items()
                   if meta["n_test_stays"] < MIN_HOSPITAL_STAYS]
    if not large_sites or not small_sites:
        raise AssertionError("expected both individually eligible and combined small hospitals")
    if any(site_meta[s]["n_eligible_variables"] == 0 for s in large_sites):
        bad = sum(site_meta[s]["n_eligible_variables"] == 0 for s in large_sites)
        raise AssertionError(f"{bad} >=100-stay hospitals have no estimable variables")

    seed_scores: dict[tuple[int, int, str, str, int], tuple[float, int]] = {}
    for site in site_meta:
        fold, hospital = site
        for model in MODELS:
            for objective in OBJECTIVES:
                for seed in SEEDS:
                    key = (fold, hospital, model, objective, seed)
                    skill, nvars = _macro_skill(model_abs[key], baseline_abs_by_site_var[site],
                                                event_count_by_site_var[site])
                    seed_scores[key] = (skill, nvars)

    average_scores: dict[tuple[int, int, str, str], float] = {}
    for site in site_meta:
        for model in MODELS:
            for objective in OBJECTIVES:
                vals = [seed_scores[(site[0], site[1], model, objective, seed)][0]
                        for seed in SEEDS]
                if not all(math.isfinite(v) for v in vals):
                    continue
                average_scores[(site[0], site[1], model, objective)] = float(np.mean(vals))

    # Opaque row indices allow pairing across contrasts in the machine-readable
    # detail file while excluding frozen hospital IDs from all outputs.
    row_rng = np.random.default_rng(BOOTSTRAP_SEED)
    ordered_sites = list(large_sites)
    row_rng.shuffle(ordered_sites)
    opaque_site_row = {site: f"site_{i + 1:03d}" for i, site in enumerate(ordered_sites)}
    detail_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for site in ordered_sites:
        meta = site_meta[site]
        for model in MODELS:
            for objective in OBJECTIVES:
                per_seed = [seed_scores[(site[0], site[1], model, objective, seed)][0]
                            for seed in SEEDS]
                score_rows.append({
                    "opaque_site_row": opaque_site_row[site], "fold": site[0],
                    "n_test_stays": meta["n_test_stays"],
                    "n_test_patients": meta["n_test_patients"],
                    "measurement_density_scoreable_events_per_stay": meta["measurement_density"],
                    "eligible_variable_count": seed_scores[(site[0], site[1], model, objective, SEEDS[0])][1],
                    "architecture": DISPLAY_MODEL[model], "objective": OBJECTIVE_LABEL[objective],
                    "seed_2024_macro_skill": per_seed[0], "seed_2025_macro_skill": per_seed[1],
                    "seed_2026_macro_skill": per_seed[2],
                    "mean_three_seed_macro_skill": float(np.mean(per_seed)),
                })
        for model in MODELS:
            l1 = average_scores.get((site[0], site[1], model, "MAE"), float("nan"))
            mse = average_scores.get((site[0], site[1], model, "MSE"), float("nan"))
            if math.isfinite(l1) and math.isfinite(mse):
                detail_rows.append({
                    "opaque_site_row": opaque_site_row[site], "fold": site[0],
                    "n_test_stays": meta["n_test_stays"],
                    "n_test_patients": meta["n_test_patients"],
                    "measurement_density_scoreable_events_per_stay": meta["measurement_density"],
                    "n_eligible_variables": meta["n_eligible_variables"],
                    "metric_type": "objective_effect", "objective": "MAE_minus_MSE",
                    "contrast_label": "masked L1 − MSE", "group_label": DISPLAY_MODEL[model],
                    "estimate": l1 - mse,
                })
        for objective in OBJECTIVES:
            cg = average_scores.get((site[0], site[1], MODELS[0], objective), float("nan"))
            for comparator in MODELS[1:]:
                other = average_scores.get((site[0], site[1], comparator, objective), float("nan"))
                if math.isfinite(cg) and math.isfinite(other):
                    detail_rows.append({
                        "opaque_site_row": opaque_site_row[site], "fold": site[0],
                        "n_test_stays": meta["n_test_stays"],
                        "n_test_patients": meta["n_test_patients"],
                        "measurement_density_scoreable_events_per_stay": meta["measurement_density"],
                        "n_eligible_variables": meta["n_eligible_variables"],
                        "metric_type": "architecture_difference", "objective": objective,
                        "contrast_label": f"CarryGate − {DISPLAY_MODEL[comparator]}",
                        "group_label": OBJECTIVE_LABEL[objective], "estimate": cg - other,
                    })
            scores = [average_scores.get((site[0], site[1], model, objective), float("nan"))
                      for model in MODELS]
            if all(math.isfinite(x) for x in scores):
                gaps = [abs(scores[i] - scores[j]) for i in range(3) for j in range(i + 1, 3)]
                detail_rows.append({
                    "opaque_site_row": opaque_site_row[site], "fold": site[0],
                    "n_test_stays": meta["n_test_stays"],
                    "n_test_patients": meta["n_test_patients"],
                    "measurement_density_scoreable_events_per_stay": meta["measurement_density"],
                    "n_eligible_variables": meta["n_eligible_variables"],
                    "metric_type": "c3_pairwise_median_absolute_gap", "objective": objective,
                    "contrast_label": "median of three pairwise absolute differences",
                    "group_label": OBJECTIVE_LABEL[objective], "estimate": float(np.median(gaps)),
                })

    summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    specs: list[tuple[str, str, str, str]] = []
    for model in MODELS:
        specs.append(("objective_effect", "MAE_minus_MSE", DISPLAY_MODEL[model], "masked L1 − MSE"))
    for objective in OBJECTIVES:
        for comparator in MODELS[1:]:
            specs.append(("architecture_difference", objective, OBJECTIVE_LABEL[objective],
                          f"CarryGate − {DISPLAY_MODEL[comparator]}"))
        specs.append(("c3_pairwise_median_absolute_gap", objective, OBJECTIVE_LABEL[objective],
                      "median of three pairwise absolute differences"))
    for metric_type, objective_key, group_label, contrast_label in specs:
        rows = [r for r in detail_rows if r["metric_type"] == metric_type
                and r["contrast_label"] == contrast_label
                and r["group_label"] == group_label
                and ((metric_type == "objective_effect") or r["objective"] == objective_key)]
        vals = np.asarray([r["estimate"] for r in rows], dtype=float)
        weights = np.asarray([r["n_test_patients"] for r in rows], dtype=float)
        stats = _bootstrap_stats(vals, weights, rng)
        summary_rows.append({
            "scope": "large_hospitals", "metric_type": metric_type,
            "objective": objective_key, "group_label": group_label,
            "contrast_label": contrast_label, "n_hospitals": len(rows),
            "positive_hospitals": int(np.sum(vals > 0)) if metric_type != "c3_pairwise_median_absolute_gap" else "",
            **stats,
        })

    # Exploratory Spearman correlations use large hospitals only and report no
    # p-values or significance labels.
    from scipy.stats import spearmanr
    correlation_rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = [r for r in detail_rows if r["metric_type"] == "objective_effect"
                    and r["group_label"] == DISPLAY_MODEL[model]]
        effects = np.asarray([r["estimate"] for r in selected], dtype=float)
        for covariate, field, definition in (
            ("test_stay_count", "n_test_stays", "test stays in held-out hospital fold"),
            ("measurement_density", "measurement_density_scoreable_events_per_stay",
             "scoreable target measurements per held-out test stay"),
        ):
            x = np.asarray([r[field] for r in selected], dtype=float)
            if len(x) < 2 or np.ptp(x) == 0 or np.ptp(effects) == 0:
                rho = float("nan")
                status = "undefined_constant_or_insufficient_values"
            else:
                rho = float(spearmanr(x, effects).statistic)
                status = "descriptive_only_no_p_value"
            correlation_rows.append({
                "architecture": DISPLAY_MODEL[model], "effect": "masked L1 − MSE",
                "covariate": covariate, "covariate_definition": definition,
                "spearman_rho": rho, "n_hospitals": len(x), "status": status,
            })

    # Combine all <100-stay hospitals before calculating macro skill.  The
    # pooled small-hospital group includes all five held-out folds.
    small_n_hosp = len(small_sites)
    small_n_stays = sum(site_meta[s]["n_test_stays"] for s in small_sites)
    small_n_patients = sum(site_meta[s]["n_test_patients"] for s in small_sites)
    small_event_count = np.sum([event_count_by_site_var[s] for s in small_sites], axis=0)
    small_baseline_abs = np.sum([baseline_abs_by_site_var[s] for s in small_sites], axis=0)
    small_score_by_cell: dict[tuple[str, str], float] = {}
    small_rows: list[dict[str, Any]] = []
    small_seed_scores: dict[tuple[str, str, int], tuple[float, int]] = {}
    for model in MODELS:
        for objective in OBJECTIVES:
            per_seed = []
            nvars_seed = []
            for seed in SEEDS:
                error_sum = np.zeros(N_VARIABLES, dtype=float)
                for site in small_sites:
                    error_sum += model_abs[(site[0], site[1], model, objective, seed)]
                skill, nvars = _macro_skill(error_sum, small_baseline_abs, small_event_count)
                small_seed_scores[(model, objective, seed)] = (skill, nvars)
                per_seed.append(skill)
                nvars_seed.append(nvars)
            score = float(np.mean(per_seed))
            small_score_by_cell[(model, objective)] = score
            small_rows.append({
                "scope": "combined_small_hospitals", "architecture": DISPLAY_MODEL[model],
                "objective": OBJECTIVE_LABEL[objective], "mean_three_seed_macro_skill": score,
                "seed_2024_skill": per_seed[0], "seed_2025_skill": per_seed[1],
                "seed_2026_skill": per_seed[2], "eligible_variables": nvars_seed[0],
                "scoreable_measurements": int(small_event_count.sum()),
                "hospitals_combined": small_n_hosp, "test_stays_combined": small_n_stays,
                "test_patients_combined": small_n_patients,
                "aggregation": "pool event errors within fixed variables, then variable macro mean",
            })
    small_objective_diffs = []
    for model in MODELS:
        small_objective_diffs.append({
            "scope": "combined_small_hospitals", "architecture": DISPLAY_MODEL[model],
            "objective": "masked L1 − MSE",
            "mean_three_seed_macro_skill": small_score_by_cell[(model, "MAE")] - small_score_by_cell[(model, "MSE")],
            "seed_2024_skill": "", "seed_2025_skill": "", "seed_2026_skill": "",
            "eligible_variables": int(np.sum((small_event_count >= MIN_EVENTS)
                & (small_baseline_abs > 1e-9) & ~np.isin(np.arange(N_VARIABLES), tuple(UNSCORED)))),
            "scoreable_measurements": int(small_event_count.sum()),
            "hospitals_combined": small_n_hosp, "test_stays_combined": small_n_stays,
            "test_patients_combined": small_n_patients,
            "aggregation": "pooled small-hospital objective effect",
        })
    small_rows.extend(small_objective_diffs)

    _write_csv(OUTPUTS["hospital_scores"], score_rows)
    _write_csv(OUTPUTS["hospital_values"], detail_rows)
    _write_csv(OUTPUTS["summary"], summary_rows)
    _write_csv(OUTPUTS["small_summary"], small_rows)
    _write_csv(OUTPUTS["correlations"], correlation_rows)
    _write_tex(summary_rows)
    _make_figure(detail_rows)

    report = {
        "schema_version": "jamia_r6_hospital_heterogeneity_report_v1",
        "status": "COMPLETE",
        "analysis_type": "post hoc sensitivity analysis; reporting rules fixed before computation",
        "scope": "eICU internal held-hospital test predictions; not independent external validation",
        "estimator": "per hospital, model, objective and seed: equal-variable mean of S_v=1-sum(abs(pred-truth))/sum(abs(carry-truth)); average the three seeds equally",
        "eligibility": {"hospital_min_test_stays": MIN_HOSPITAL_STAYS,
                        "variable_min_events": MIN_EVENTS,
                        "carry_forward_abs_error_minimum": 1e-9,
                        "scoreable_variable_count": N_VARIABLES - len(UNSCORED),
                        "unscored_variable_indices_zero_based": sorted(UNSCORED)},
        "bootstrap": {"cluster": "held-out hospital", "replicates": N_BOOTSTRAP,
                      "seed": BOOTSTRAP_SEED, "interval": "percentile 95%",
                      "patient_weight": "unique held-out test patients per hospital"},
        "counts": {
            "frozen_p2_predictions": 90, "held_out_folds": len(FOLDS),
            "all_hospitals": len(site_meta), "large_hospitals_individually_summarized": len(large_sites),
            "small_hospitals_combined": small_n_hosp, "large_hospital_stays": sum(site_meta[s]["n_test_stays"] for s in large_sites),
            "large_hospital_patients": sum(site_meta[s]["n_test_patients"] for s in large_sites),
            "small_hospital_stays": small_n_stays, "small_hospital_patients": small_n_patients,
            "large_hospitals_with_no_eligible_variables": sum(site_meta[s]["n_eligible_variables"] == 0 for s in large_sites),
        },
        "small_hospital_aggregation": "all hospitals with fewer than 100 held-out test stays pooled over five folds; no individual small-hospital estimates",
        "measurement_density_definition": "number of target events across the fixed 34 scoreable variables divided by held-out test stays",
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in OUTPUTS.items() if key != "manifest"},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    OUTPUTS["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    manifest = {
        "schema_version": "jamia_r6_hospital_heterogeneity_manifest_v1",
        "status": "COMPLETE", "created_at_utc": utc_now(),
        "analysis_plan_sha256": EXPECTED_PLAN_SHA256,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "preflight_manifest_sha256": sha256_file(PREFLIGHT_PATH),
        "python_version": sys.version,
        "input_hashes": preflight_doc["input_hashes"],
        "counts": report["counts"],
        "outputs": {},
    }
    for key, path in OUTPUTS.items():
        if key == "manifest":
            continue
        manifest["outputs"][key] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    OUTPUTS["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    report["manifest_sha256"] = sha256_file(OUTPUTS["manifest"])
    return report


def archive_existing(include_preflight: bool) -> None:
    paths = list(OUTPUTS.values())
    if include_preflight:
        paths.append(PREFLIGHT_PATH)
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = OUT / "_archive" / f"r6_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        rel = path.relative_to(ROOT)
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--refresh-generated", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.run:
        parser.error("choose exactly one of --preflight-only or --run")
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        if PREFLIGHT_PATH.exists() or any(path.exists() for path in OUTPUTS.values()):
            if not args.refresh_generated:
                raise RuntimeError("R6 outputs already exist; use --refresh-generated to archive them first")
            archive_existing(include_preflight=True)
        record = preflight()
        PREFLIGHT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(json.dumps({"status": record["status"], "counts": record["counts"],
                          "input_hash_count": record["input_hash_count"],
                          "failures": record["failures"],
                          "elapsed_seconds": record["elapsed_seconds"]},
                         indent=2, ensure_ascii=False))
        return 0 if record["status"] == "PASS" else 1
    preflight_doc = verify_preflight()
    if any(path.exists() for key, path in OUTPUTS.items() if key != "manifest") or OUTPUTS["manifest"].exists():
        if not args.refresh_generated:
            raise RuntimeError("R6 generated outputs already exist; use --refresh-generated to archive previous outputs")
        # Keep the verified preflight in place; archive only generated analysis outputs.
        archive_existing(include_preflight=False)
    report = run_analysis(preflight_doc)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "outputs": report["outputs"], "elapsed_seconds": report["elapsed_seconds"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
