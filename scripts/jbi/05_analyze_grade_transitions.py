#!/usr/bin/env python
"""Evaluate event/window SOFA-derived grades and grade transitions."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))
from analyze_sofa_grades import COMPONENTS, destandardise, grade, project  # noqa: E402
from rescore_macro import UNSCORED  # noqa: E402

OUT = ROOT / "results" / "jbi"
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = ROOT / "results" / "d3_clinical"
CACHES = {
    "p12": ("p12_events.pkl", "split_indices_12k.json"),
    "mimiciv": ("mimiciv_events.pkl", "mimiciv_split_indices_12k.json"),
    "eicu": ("eicu_events.pkl", "eicu_split_indices_12k.json"),
}
MATRIX = ROOT / "results" / "d1_external" / "d1c_matrix.csv"
N_BOOT = 2000
BOOTSTRAP_SEED = 20260913


def _dense_target(example: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = {key: np.asarray(value) for key, value in example["obs"].items()}
    tgt = {key: np.asarray(value) for key, value in example["target"].items()}
    prior = np.zeros(len(tgt["var"]), dtype=np.float32)
    for variable in np.unique(tgt["var"]):
        selected = tgt["var"] == variable
        before = np.flatnonzero((obs["var"] == variable) & (obs["time"] < float(example["t_obs"])))
        if len(before):
            prior[selected] = obs["value"][before[np.argmax(obs["time"][before])]]
    # Standardised training medians are zero; the baseline implementation uses
    # this exact fallback when a variable has no historical observation.
    order = np.lexsort((tgt["var"], tgt["time"]))
    var_s = tgt["var"][order]
    time_s = tgt["time"][order]
    truth_s = tgt["value"][order]
    prior_s = prior[order]
    _, inverse, counts = np.unique(
        np.stack([time_s, var_s]), axis=1, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    group_start = np.searchsorted(inverse, np.arange(counts.size))
    rank = np.arange(inverse.size) - group_start[inverse]
    uniq_time, time_index = np.unique(time_s, return_inverse=True)
    rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
    np.maximum.at(rows_per_time, time_index, rank + 1)
    row_start = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])
    row_of = row_start[time_index] + rank
    dense = np.lexsort((var_s, row_of))
    return var_s[dense], truth_s[dense], prior_s[dense]


def reference(cohort: str) -> dict[str, np.ndarray]:
    cache_name, split_name = CACHES[cohort]
    cache = pickle.load((PROCESSED / cache_name).open("rb"))
    split = json.loads((PROCESSED / split_name).read_text())
    var, truth, prior, stays, clusters = [], [], [], [], []
    cluster_units: set[str] = set()
    for stay_index, cache_index in enumerate(split["test"]):
        example = cache["examples"][int(cache_index)]
        v, y, x = _dense_target(example)
        cluster_key, cluster_unit = _cluster_key(example, cohort)
        cluster_units.add(cluster_unit)
        var.append(v)
        truth.append(y)
        prior.append(x)
        stays.append(np.full(len(v), stay_index, dtype=np.int64))
        clusters.append(np.full(len(v), cluster_key, dtype=object))
    return {
        "var": np.concatenate(var),
        "truth": np.concatenate(truth),
        "x_last": np.concatenate(prior),
        "stay": np.concatenate(stays),
        "cluster": np.concatenate(clusters),
        "bootstrap_unit": "patient" if cluster_units == {"patient"} else "stay",
    }


def raw_reference(cohort: str) -> dict[str, np.ndarray]:
    """Reference in the project model's native (per-stay target) order.

    The external harness flattens padded targets in dense row-major order,
    whereas ``evaluate_forecast`` concatenates each example's target events in
    their cache order.  Both are valid, but predictions must be compared with
    the matching order before attaching stay-level labels.
    """
    cache_name, split_name = CACHES[cohort]
    cache = pickle.load((PROCESSED / cache_name).open("rb"))
    split = json.loads((PROCESSED / split_name).read_text())
    var, truth, prior, stays, clusters = [], [], [], [], []
    cluster_units: set[str] = set()
    for stay_index, cache_index in enumerate(split["test"]):
        example = cache["examples"][int(cache_index)]
        obs = {key: np.asarray(value) for key, value in example["obs"].items()}
        tgt = {key: np.asarray(value) for key, value in example["target"].items()}
        prior_values = np.zeros(len(tgt["var"]), dtype=np.float32)
        for variable in np.unique(tgt["var"]):
            selected = tgt["var"] == variable
            before = np.flatnonzero((obs["var"] == variable) & (obs["time"] < float(example["t_obs"])))
            if len(before):
                prior_values[selected] = obs["value"][before[np.argmax(obs["time"][before])]]
        var.append(tgt["var"])
        truth.append(tgt["value"])
        prior.append(prior_values)
        stays.append(np.full(len(tgt["var"]), stay_index, dtype=np.int64))
        cluster_key, cluster_unit = _cluster_key(example, cohort)
        cluster_units.add(cluster_unit)
        clusters.append(np.full(len(tgt["var"]), cluster_key, dtype=object))
    return {
        "var": np.concatenate(var),
        "truth": np.concatenate(truth),
        "x_last": np.concatenate(prior),
        "stay": np.concatenate(stays),
        "cluster": np.concatenate(clusters),
        "bootstrap_unit": "patient" if cluster_units == {"patient"} else "stay",
    }


def _external_predictions(cohort: str, ref: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    matrix = pd.read_csv(MATRIX)
    result: dict[str, list[np.ndarray]] = {}
    for row in matrix[matrix.cohort == cohort].itertuples():
        z = np.load(ROOT / row.eval_dir / "output_pred.npy")
        mask = np.load(ROOT / row.eval_dir / "input_y_mask.npy").astype(bool)
        pred = z[mask]
        truth = np.load(ROOT / row.eval_dir / "input_y.npy")[mask]
        if len(pred) != len(ref["truth"]) or not np.allclose(truth, ref["truth"], atol=1e-5, rtol=0):
            raise AssertionError(f"{row.model}/{cohort}: target order does not match reference")
        result.setdefault(str(row.model), []).append(pred)
    return result


def _own_predictions(cohort: str, ref: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    result = []
    selected_ref = ref
    for path in sorted(PRED_DIR.glob(f"pred_{cohort}_seed*.npz")):
        z = np.load(path)
        if len(z["prediction"]) == len(ref["truth"]) and np.array_equal(z["variable"], ref["var"]):
            candidate = ref
        else:
            scored = ~np.isin(ref["var"], sorted(UNSCORED))
            if len(z["prediction"]) != int(scored.sum()) or not np.array_equal(z["variable"], ref["var"][scored]):
                raise AssertionError(f"This work/{cohort}: native prediction order does not match reference")
            candidate = {
                key: value[scored] if isinstance(value, np.ndarray) else value
                for key, value in ref.items()
            }
        if not np.allclose(z["target"], candidate["truth"], atol=1e-5, rtol=0):
            raise AssertionError(f"This work/{cohort}: native targets do not match reference")
        result.append(z["prediction"])
        selected_ref = candidate
    return result, selected_ref


def _classification_metrics(true_positive: np.ndarray, predicted_positive: np.ndarray) -> dict[str, float]:
    tp = float(np.sum(true_positive & predicted_positive))
    tn = float(np.sum(~true_positive & ~predicted_positive))
    fp = float(np.sum(~true_positive & predicted_positive))
    fn = float(np.sum(true_positive & ~predicted_positive))
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
    }


def _vectorised_binary_metrics(
    true_positive: np.ndarray,
    predicted_positive: np.ndarray,
    inverse: np.ndarray,
    n_clusters: int,
) -> dict[str, np.ndarray]:
    """Return confusion-matrix metrics for each resampled cluster total."""
    tp = np.bincount(
        inverse, weights=(true_positive & predicted_positive), minlength=n_clusters
    ).astype(float)
    tn = np.bincount(
        inverse, weights=(~true_positive & ~predicted_positive), minlength=n_clusters
    ).astype(float)
    fp = np.bincount(
        inverse, weights=(~true_positive & predicted_positive), minlength=n_clusters
    ).astype(float)
    fn = np.bincount(
        inverse, weights=(true_positive & ~predicted_positive), minlength=n_clusters
    ).astype(float)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_counts(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> dict[str, np.ndarray]:
    sensitivity = np.divide(
        tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0
    )
    specificity = np.divide(
        tn, tn + fp, out=np.full_like(tn, np.nan), where=(tn + fp) > 0
    )
    ppv = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    npv = np.divide(tn, tn + fn, out=np.full_like(tn, np.nan), where=(tn + fn) > 0)
    balanced = np.nanmean(np.stack([sensitivity, specificity]), axis=0)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": balanced,
    }


def _cluster_bootstrap_transition_ci(
    seed_arrays: list[dict[str, np.ndarray]],
    cluster: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    """Bootstrap seed-averaged transition metrics by patient/stay cluster."""
    cluster = np.asarray(cluster)
    _, inverse = np.unique(cluster, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    output: dict[str, float] = {}
    for transition in ("deterioration", "improvement"):
        draws_by_metric: dict[str, list[np.ndarray]] = {
            key: []
            for key in ("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy")
        }
        for item in seed_arrays:
            counts = _vectorised_binary_metrics(
                item[f"{transition}_true"],
                item[f"{transition}_pred"],
                inverse,
                n_clusters,
            )
            sampled = {
                key: values[picks].sum(axis=1)
                for key, values in counts.items()
            }
            metrics = _metrics_from_counts(
                sampled["tp"], sampled["tn"], sampled["fp"], sampled["fn"]
            )
            for key, values in metrics.items():
                draws_by_metric[key].append(values)
        for key, values in draws_by_metric.items():
            stack = np.stack(values, axis=0)
            if not np.isfinite(stack).any():
                output[f"{transition}_{key}_ci_lo"] = np.nan
                output[f"{transition}_{key}_ci_hi"] = np.nan
                continue
            valid_count = np.isfinite(stack).sum(axis=0)
            draws = np.divide(
                np.nansum(stack, axis=0),
                valid_count,
                out=np.full(valid_count.shape, np.nan, dtype=float),
                where=valid_count > 0,
            )
            finite = draws[np.isfinite(draws)]
            output[f"{transition}_{key}_ci_lo"] = float(np.percentile(finite, 2.5))
            output[f"{transition}_{key}_ci_hi"] = float(np.percentile(finite, 97.5))
    return output


def _cluster_key(example: dict, cohort: str) -> tuple[str, str]:
    """Return an auditable inference cluster and its identity level."""
    if cohort == "eicu" and example.get("uniquepid") is not None:
        return str(example["uniquepid"]), "patient"
    if example.get("subject_id") is not None:
        return str(example["subject_id"]), "patient"
    return str(example["record_id"]), "stay"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for cohort in CACHES:
        dense_ref = reference(cohort)
        native_ref = raw_reference(cohort)
        external_arms = _external_predictions(cohort, dense_ref)
        arms = {model: (predictions, dense_ref) for model, predictions in external_arms.items()}
        own, own_ref = _own_predictions(cohort, native_ref)
        if own:
            arms["This work"] = (own, own_ref)
        names = {name: index for index, name in enumerate(pickle.load((PROCESSED / CACHES[cohort][0]).open("rb"))["variable_names"])}
        for organ, variable, bounds, direction, support in COMPONENTS:
            variable_id = names[variable]
            for model, (predictions, model_ref) in arms.items():
                event_mask = model_ref["var"] == variable_id
                stay_event = model_ref["stay"][event_mask]
                cluster_event = model_ref["cluster"][event_mask]
                truth = destandardise(cohort, variable_id, model_ref["truth"][event_mask])
                last = destandardise(cohort, variable_id, model_ref["x_last"][event_mask])
                # Same support for observation, forecast and carry-forward.
                true_grade = grade(project(truth, support), bounds, direction)
                cf_grade = grade(project(last, support), bounds, direction)
                unique_stays = np.unique(stay_event)
                window_cluster = []
                for stay in unique_stays:
                    selected = stay_event == stay
                    clusters_for_stay = np.unique(cluster_event[selected])
                    if len(clusters_for_stay) != 1:
                        raise AssertionError(
                            f"{cohort}/{model}/{variable}: a stay maps to multiple bootstrap clusters"
                        )
                    window_cluster.append(clusters_for_stay[0])
                window_cluster = np.asarray(window_cluster, dtype=object)
                transition_seed_arrays: list[dict[str, np.ndarray]] = []
                for seed, prediction in enumerate(predictions):
                    pred_raw = destandardise(cohort, variable_id, prediction[event_mask])
                    pred_grade = grade(project(pred_raw, support), bounds, direction)
                    event_abs = np.abs(pred_grade.astype(int) - true_grade.astype(int))
                    event_wrong = pred_grade != true_grade
                    event_under = pred_grade < true_grade
                    event_over = pred_grade > true_grade
                    true_window, pred_window, cf_window = [], [], []
                    for stay in unique_stays:
                        selected = stay_event == stay
                        true_window.append(int(np.max(true_grade[selected])))
                        pred_window.append(int(np.max(pred_grade[selected])))
                        cf_window.append(int(np.max(cf_grade[selected])))
                    true_window = np.asarray(true_window, dtype=np.int8)
                    pred_window = np.asarray(pred_window, dtype=np.int8)
                    cf_window = np.asarray(cf_window, dtype=np.int8)
                    window_abs = np.abs(pred_window.astype(int) - true_window.astype(int))
                    window_wrong = pred_window != true_window
                    true_transition = np.sign(true_window.astype(int) - cf_window.astype(int))
                    pred_transition = np.sign(pred_window.astype(int) - cf_window.astype(int))
                    deterioration = _classification_metrics(true_transition == 1, pred_transition == 1)
                    improvement = _classification_metrics(true_transition == -1, pred_transition == -1)
                    transition_seed_arrays.append(
                        {
                            "deterioration_true": true_transition == 1,
                            "deterioration_pred": pred_transition == 1,
                            "improvement_true": true_transition == -1,
                            "improvement_pred": pred_transition == -1,
                        }
                    )
                    common = {
                        "analysis_plan_version": "JBI_analysis_plan_v1",
                        "cohort": cohort,
                        "organ": organ,
                        "variable": variable,
                        "model": model,
                        "seed": seed,
                        "n_events": len(true_grade),
                        "n_windows": len(true_window),
                        "window_inclusion_rate": len(true_window) / max(len(np.unique(model_ref["stay"])), 1),
                        "event_true_deterioration_rate": float(np.mean(true_grade > cf_grade)),
                        "window_true_deterioration_rate": float(np.mean(true_transition == 1)),
                        "window_true_improvement_rate": float(np.mean(true_transition == -1)),
                        "event_misclassification": float(np.mean(event_wrong)),
                        "event_mean_abs_grade_error": float(np.mean(event_abs)),
                        "event_undergrading": float(np.mean(event_under)),
                        "event_overgrading": float(np.mean(event_over)),
                        "window_misclassification": float(np.mean(window_wrong)),
                        "window_mean_abs_grade_error": float(np.mean(window_abs)),
                        "bootstrap_unit": model_ref["bootstrap_unit"],
                        **{f"deterioration_{key}": float(value) for key, value in deterioration.items()},
                        **{f"improvement_{key}": float(value) for key, value in improvement.items()},
                    }
                    rows.append(common)
                bootstrap = _cluster_bootstrap_transition_ci(
                    transition_seed_arrays,
                    window_cluster,
                    bootstrap_rng,
                    n_boot=N_BOOT,
                )
                bootstrap_rows.append(
                    {
                        "analysis_plan_version": "JBI_analysis_plan_v1",
                        "cohort": cohort,
                        "organ": organ,
                        "variable": variable,
                        "model": model,
                        "n_seeds": len(transition_seed_arrays),
                        "n_windows": len(window_cluster),
                        "n_bootstrap_clusters": int(np.unique(window_cluster).size),
                        "bootstrap_n": N_BOOT,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_unit": model_ref["bootstrap_unit"],
                        "split_unit": (
                            "registered_stay_split_patient_cluster_inference"
                            if cohort == "eicu"
                            else "registered_stay_split_patient_unverifiable"
                        ),
                        **bootstrap,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "e4_grade_transitions_by_seed.csv", index=False, float_format="%.17g")
    numeric = [
        column
        for column in frame.columns
        if column not in {"analysis_plan_version", "cohort", "organ", "variable", "model"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = frame.groupby(["cohort", "organ", "variable", "model"], as_index=False)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).strip("_") if isinstance(value, tuple) else value for value in summary.columns]
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    summary = summary.merge(
        bootstrap_frame,
        on=["cohort", "organ", "variable", "model"],
        how="left",
        validate="one_to_one",
    )
    bootstrap_frame.to_csv(
        OUT / "e4_grade_transitions_bootstrap.csv", index=False, float_format="%.17g"
    )
    summary.to_csv(OUT / "e4_grade_transitions_summary.csv", index=False, float_format="%.17g")
    print(summary.to_string(index=False))
    print(f"wrote {OUT / 'e4_grade_transitions_summary.csv'}")


if __name__ == "__main__":
    main()
