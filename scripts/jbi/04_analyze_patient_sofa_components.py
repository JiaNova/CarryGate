#!/usr/bin/env python
"""Evaluate SOFA-derived components on the patient-grouped eICU split.

This is the JBI extension counterpart of ``scripts/analyze_sofa_grades.py``.
The E-M0 runner stores native per-stay target order, whereas the external
harness stores dense row-major order.  This evaluator deliberately consumes
only the former and reconstructs the matching reference from the exact JBI
cache/split pair.  The primary uncertainty interval is a patient-cluster
bootstrap of the same seed-averaged event-level excess over carry-forward.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_sofa_grades import (  # noqa: E402
    COMPONENTS,
    N_BOOT,
    cluster_bootstrap_seed_average,
    grade,
    project,
)

PLAN = "JBI_analysis_plan_v1"
STRUCTURAL = frozenset({6, 19, 32})


def _load_reference(cache_path: Path, split_path: Path) -> dict[str, np.ndarray]:
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    variables: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    stays: list[np.ndarray] = []
    patients: list[np.ndarray] = []
    for stay_index, cache_index in enumerate(split["test"]):
        example = cache["examples"][int(cache_index)]
        target = example["target"]
        var = np.asarray(target["var"], dtype=np.int64)
        keep = ~np.isin(var, sorted(STRUCTURAL))
        variables.append(var[keep])
        truth.append(np.asarray(target["value"], dtype=np.float32)[keep])
        stays.append(np.full(int(keep.sum()), stay_index, dtype=np.int64))
        # eICU's grouped split uses uniquepid.  Fall back to subject_id only
        # for caches that expose no uniquepid; never fall back to stay index.
        patient = example.get("uniquepid", example.get("subject_id"))
        if patient is None:
            raise RuntimeError("JBI cache has no patient identifier")
        patients.append(np.full(int(keep.sum()), str(patient), dtype=object))
    return {
        "var": np.concatenate(variables),
        "truth": np.concatenate(truth),
        "stay": np.concatenate(stays),
        "patient": np.concatenate(patients),
        "test_stays": np.arange(len(split["test"]), dtype=np.int64),
        "variable_names": list(cache["variable_names"]),
    }


def _stats(cache_path: Path) -> dict:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)["standardizer"]


def _destandardise(values: np.ndarray, variable: int, stats: dict) -> np.ndarray:
    return values * float(stats["iqr"][variable]) + float(stats["median"][variable])


def _metric(true_positive: np.ndarray, predicted_positive: np.ndarray) -> dict[str, float]:
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


def _cluster_counts(
    true_positive: np.ndarray,
    predicted_positive: np.ndarray,
    inverse: np.ndarray,
    n_clusters: int,
) -> dict[str, np.ndarray]:
    return {
        "tp": np.bincount(
            inverse,
            weights=(true_positive & predicted_positive),
            minlength=n_clusters,
        ).astype(float),
        "tn": np.bincount(
            inverse,
            weights=(~true_positive & ~predicted_positive),
            minlength=n_clusters,
        ).astype(float),
        "fp": np.bincount(
            inverse,
            weights=(~true_positive & predicted_positive),
            minlength=n_clusters,
        ).astype(float),
        "fn": np.bincount(
            inverse,
            weights=(true_positive & ~predicted_positive),
            minlength=n_clusters,
        ).astype(float),
    }


def _counts_to_metrics(
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
    balanced_accuracy = np.divide(
        np.nansum(np.stack([sensitivity, specificity]), axis=0),
        np.isfinite(np.stack([sensitivity, specificity])).sum(axis=0),
        out=np.full_like(tp, np.nan),
        where=np.isfinite(np.stack([sensitivity, specificity])).sum(axis=0) > 0,
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": balanced_accuracy,
    }


def _transition_cluster_bootstrap(
    seed_arrays: list[dict[str, np.ndarray]],
    cluster: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    """Patient-cluster CI for the same seed-averaged transition estimand."""
    _, inverse = np.unique(np.asarray(cluster), return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    result: dict[str, float] = {}
    for transition in ("deterioration", "improvement"):
        by_metric = {
            key: []
            for key in ("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy")
        }
        for item in seed_arrays:
            counts = _cluster_counts(
                item[f"{transition}_true"],
                item[f"{transition}_pred"],
                inverse,
                n_clusters,
            )
            sampled = {key: values[picks].sum(axis=1) for key, values in counts.items()}
            metrics = _counts_to_metrics(
                sampled["tp"], sampled["tn"], sampled["fp"], sampled["fn"]
            )
            for key, values in metrics.items():
                by_metric[key].append(values)
        for key, values in by_metric.items():
            stack = np.stack(values, axis=0)
            valid_count = np.isfinite(stack).sum(axis=0)
            draws = np.divide(
                np.nansum(stack, axis=0),
                valid_count,
                out=np.full(valid_count.shape, np.nan, dtype=float),
                where=valid_count > 0,
            )
            finite = draws[np.isfinite(draws)]
            result[f"transition_{transition}_{key}_ci_lo"] = float(
                np.percentile(finite, 2.5) if len(finite) else np.nan
            )
            result[f"transition_{transition}_{key}_ci_hi"] = float(
                np.percentile(finite, 97.5) if len(finite) else np.nan
            )
    return result


def _window_arrays(
    stay: np.ndarray,
    true_grade: np.ndarray,
    pred_grade: np.ndarray,
    cf_grade: np.ndarray,
) -> dict[str, np.ndarray]:
    unique, inverse = np.unique(stay, return_inverse=True)
    true_window = np.zeros(len(unique), dtype=np.int8)
    pred_window = np.zeros(len(unique), dtype=np.int8)
    cf_window = np.zeros(len(unique), dtype=np.int8)
    for index in range(len(unique)):
        selected = inverse == index
        true_window[index] = np.max(true_grade[selected])
        pred_window[index] = np.max(pred_grade[selected])
        cf_window[index] = np.max(cf_grade[selected])
    true_transition = np.sign(true_window.astype(int) - cf_window.astype(int))
    pred_transition = np.sign(pred_window.astype(int) - cf_window.astype(int))
    deterioration = _metric(true_transition == 1, pred_transition == 1)
    improvement = _metric(true_transition == -1, pred_transition == -1)
    return {
        "true_window": true_window,
        "pred_window": pred_window,
        "cf_window": cf_window,
        "window_misclassification": pred_window != true_window,
        "window_mean_abs_grade_error": np.abs(
            pred_window.astype(int) - true_window.astype(int)
        ),
        "true_deterioration": true_transition == 1,
        "true_improvement": true_transition == -1,
        "pred_deterioration": pred_transition == 1,
        "pred_improvement": pred_transition == -1,
        **{f"deterioration_{key}": value for key, value in deterioration.items()},
        **{f"improvement_{key}": value for key, value in improvement.items()},
    }


def _prediction_files(prediction_dir: Path, prefix: str) -> list[Path]:
    paths = sorted(prediction_dir.glob(f"{prefix}-*_*.npz"))
    if not paths:
        raise SystemExit(f"no predictions matching {prefix}-*_*.npz in {prediction_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/eicu_events_jbi_v1.pkl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/processed/eicu_split_indices_jbi_v1.json")
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "results/jbi/e0_eicu_m0/predictions")
    parser.add_argument("--prefix", default="E-M0")
    parser.add_argument("--out", type=Path, default=ROOT / "results/jbi/e4_patient_sofa_components.csv")
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    args = parser.parse_args()

    reference = _load_reference(args.cache, args.split)
    stats = _stats(args.cache)
    paths = _prediction_files(args.prediction_dir, args.prefix)
    predictions: dict[str, list[dict[str, np.ndarray]]] = {}
    for path in paths:
        arm, seed_text = path.stem.rsplit("_", 1)
        seed = int(seed_text)
        z = np.load(path)
        for key in ("prediction", "target", "variable", "x_last"):
            if key not in z:
                raise SystemExit(f"{path} lacks {key}")
        if len(z["prediction"]) != len(reference["truth"]):
            raise SystemExit(
                f"{path}: {len(z['prediction'])} predictions vs "
                f"{len(reference['truth'])} reference events"
            )
        if not np.array_equal(z["variable"], reference["var"]):
            raise SystemExit(f"{path}: prediction variable order differs from JBI cache")
        if not np.allclose(z["target"], reference["truth"], atol=1e-5, rtol=0):
            raise SystemExit(f"{path}: saved targets differ from JBI cache")
        predictions.setdefault(arm, []).append(
            {
                "seed": int(seed),
                "prediction": np.asarray(z["prediction"], dtype=np.float32),
                "x_last": np.asarray(z["x_last"], dtype=np.float32),
            }
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict[str, object]] = []
    for organ, variable_name, bounds, direction, support in COMPONENTS:
        variable = int(reference["variable_names"].index(variable_name))
        event_mask = reference["var"] == variable
        patient = reference["patient"][event_mask]
        stay = reference["stay"][event_mask]
        true = _destandardise(reference["truth"][event_mask], variable, stats)
        base_x_last: np.ndarray | None = None
        # Same support for observation, forecast and carry-forward.
        true_grade = grade(project(true, support), bounds, direction)
        unique_stays = np.unique(stay)
        window_patient = []
        for stay_id in unique_stays:
            selected = stay == stay_id
            patient_ids = np.unique(patient[selected])
            if len(patient_ids) != 1:
                raise SystemExit(f"{variable_name}: a stay maps to multiple patients")
            window_patient.append(patient_ids[0])
        window_patient = np.asarray(window_patient, dtype=object)
        for arm, seed_items in sorted(predictions.items()):
            seed_items = sorted(seed_items, key=lambda item: int(item["seed"]))
            wrong_rows: list[np.ndarray] = []
            cf_wrong: np.ndarray | None = None
            per_seed: list[dict[str, object]] = []
            transition_seed_arrays: list[dict[str, np.ndarray]] = []
            for item in seed_items:
                x_last = item["x_last"][event_mask]
                if base_x_last is None:
                    base_x_last = x_last
                elif not np.allclose(base_x_last, x_last, atol=1e-6, rtol=0):
                    raise SystemExit(f"{arm}: carry-forward values differ across seeds")
                pred = _destandardise(item["prediction"][event_mask], variable, stats)
                last = _destandardise(x_last, variable, stats)
                pred_grade = grade(project(pred, support), bounds, direction)
                cf_grade = grade(project(last, support), bounds, direction)
                wrong = pred_grade != true_grade
                if cf_wrong is None:
                    cf_wrong = cf_grade != true_grade
                wrong_rows.append(wrong)
                window = _window_arrays(stay, true_grade, pred_grade, cf_grade)
                transition = {
                    key: float(value)
                    for key, value in window.items()
                    if key.startswith("deterioration_") or key.startswith("improvement_")
                }
                transition_seed_arrays.append(
                    {
                        "deterioration_true": window["true_deterioration"],
                        "deterioration_pred": window["pred_deterioration"],
                        "improvement_true": window["true_improvement"],
                        "improvement_pred": window["pred_improvement"],
                    }
                )
                per_seed.append(
                    {
                        "event_misclassification": float(np.mean(wrong)),
                        "event_mean_abs_grade_error": float(
                            np.mean(np.abs(pred_grade.astype(int) - true_grade.astype(int)))
                        ),
                        "event_undergrading": float(np.mean(pred_grade < true_grade)),
                        "event_overgrading": float(np.mean(pred_grade > true_grade)),
                        "window_misclassification": float(np.mean(window["window_misclassification"])),
                        "window_mean_abs_grade_error": float(np.mean(window["window_mean_abs_grade_error"])),
                        "window_true_deterioration_rate": float(np.mean(window["true_deterioration"])),
                        "window_true_improvement_rate": float(np.mean(window["true_improvement"])),
                        **transition,
                    }
                )
            if not wrong_rows or cf_wrong is None:
                continue
            wrong_matrix = np.stack(wrong_rows, axis=0)
            ci_lo, ci_hi, bootstrap_draws = cluster_bootstrap_seed_average(
                wrong_matrix,
                cf_wrong,
                patient,
                rng,
                n_boot=N_BOOT,
                return_draws=True,
            )
            p_raw = float(
                min(1.0, 2.0 * min(np.mean(bootstrap_draws <= 0.0), np.mean(bootstrap_draws >= 0.0)))
            )
            mean_metrics = {
                key: float(np.mean([float(item[key]) for item in per_seed]))
                for key in per_seed[0]
            }
            transition_ci = _transition_cluster_bootstrap(
                transition_seed_arrays,
                window_patient,
                rng,
                n_boot=N_BOOT,
            )
            rows.append(
                {
                    "analysis_plan_version": PLAN,
                    "cohort": "eicu",
                    "split_unit": "uniquepid",
                    "organ": organ,
                    "variable": variable_name,
                    "model": arm,
                    "n_seeds": len(seed_items),
                    "seed_list": json.dumps([int(item["seed"]) for item in seed_items]),
                    "n_events": int(event_mask.sum()),
                    "n_stays": int(np.unique(stay).size),
                    "n_patients": int(np.unique(patient).size),
                    "event_misclassification": mean_metrics["event_misclassification"],
                    "carryforward_misclassification": float(np.mean(cf_wrong)),
                    "excess_over_carryforward": mean_metrics["event_misclassification"] - float(np.mean(cf_wrong)),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_raw": p_raw,
                    "event_mean_abs_grade_error": mean_metrics["event_mean_abs_grade_error"],
                    "event_undergrading": mean_metrics["event_undergrading"],
                    "event_overgrading": mean_metrics["event_overgrading"],
                    "window_misclassification": mean_metrics["window_misclassification"],
                    "window_mean_abs_grade_error": mean_metrics["window_mean_abs_grade_error"],
                    "window_true_deterioration_rate": mean_metrics["window_true_deterioration_rate"],
                    "window_true_improvement_rate": mean_metrics["window_true_improvement_rate"],
                    **{key: mean_metrics[key] for key in mean_metrics if key.startswith(("deterioration_", "improvement_"))},
                    "bootstrap_n": N_BOOT,
                    "bootstrap_seed": args.bootstrap_seed,
                    "bootstrap_unit": "uniquepid",
                    "estimand": "mean_seed_event_misclassification_difference",
                    "transition_bootstrap_n": N_BOOT,
                    "transition_bootstrap_seed": args.bootstrap_seed,
                    "transition_bootstrap_unit": "uniquepid",
                    "transition_n_patients": int(np.unique(window_patient).size),
                    "source_cache": str(args.cache),
                    "source_split": str(args.split),
                    "prediction_dir": str(args.prediction_dir),
                    **transition_ci,
                }
            )

    frame = pd.DataFrame(rows)
    # The four components of each model form a predeclared clinical family.
    # Holm is reported for this arm-specific family; the main arm's four
    # values are the confirmatory eICU comparison, while other arms remain
    # exploratory mechanism/capacity comparisons.
    frame["p_holm"] = np.nan
    for model, positions in frame.groupby("model").groups.items():
        p = frame.loc[positions, "p_raw"].to_numpy(dtype=float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        m = len(p)
        for rank, index in enumerate(order):
            running = max(running, float((m - rank) * p[index]))
            adjusted[index] = min(running, 1.0)
        frame.loc[positions, "p_holm"] = adjusted
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.17g")
    summary = frame.drop(columns=["seed_list"], errors="ignore").copy()
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False, float_format="%.17g")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nwrote {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()
