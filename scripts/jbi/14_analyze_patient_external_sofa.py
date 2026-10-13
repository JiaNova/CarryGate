#!/usr/bin/env python
"""Run the preregistered SOFA-component audit on either Track-B objective arm.

The clinical endpoint is deliberately independent of the persistence-skill
numerator: a forecast is first destandardised, projected when the clinical
scale has discrete support, and then passed through a threshold-based SOFA
grade function.  The script is parameterised by cohort and objective arm, but
the frozen model tags are resolved only by
``scripts/jbi/23_adapter_vs_external_paired.py``.

Besides the component-level summary, a patient-by-seed table is written.  The
latter is the auditable input to the paired MSE versus MAE bootstrap in
``29_objective_effect_on_grading.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))

from analyze_sofa_grades import (  # noqa: E402
    COMPONENTS,
    N_BOOT,
    cluster_bootstrap_seed_average,
    grade,
    project,
)
from rescore_macro import cohort_events  # noqa: E402

PLAN = "JBI_analysis_plan_v1"
BOOTSTRAP_SEED = 20260913
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")
COHORTS = ("eicu", "mimiciv")
ARMS = ("macro", "mae_macro")


def _load_trackb_module():
    """Load the existing tag/reference implementation without duplicating it."""

    path = ROOT / "scripts" / "jbi" / "23_adapter_vs_external_paired.py"
    spec = importlib.util.spec_from_file_location("trackb_adapter_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Track-B reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACKB = _load_trackb_module()


def _load_identity(cohort: str) -> tuple[dict, dict[str, float], list[str]]:
    """Load the exact patient-filtered dense reference and its standardiser."""

    reference = TRACKB._reference(cohort)
    events_name, split_name = TRACKB.VARIANT_CACHES["jbi_patient"][cohort]
    with (ROOT / "data" / "processed" / events_name).open("rb") as handle:
        cache = pickle.load(handle)
    split = json.loads((ROOT / "data" / "processed" / split_name).read_text(encoding="utf-8"))
    if split.get("split_unit") != "patient_grouped":
        raise AssertionError(f"{cohort}: SOFA audit requires patient_grouped split")

    event_stay_dense = np.concatenate(
        [
            np.full(len(cache["examples"][int(index)]["target"]["var"]), position, dtype=np.int64)
            for position, index in enumerate(split["test"])
        ]
    )
    if event_stay_dense.size != reference["n_events_dense"]:
        raise AssertionError(
            f"{cohort}: stay identity length {event_stay_dense.size} != "
            f"dense events {reference['n_events_dense']}"
        )
    keep = np.asarray(reference["keep"], dtype=bool)
    stay = event_stay_dense[keep]
    if len(stay) != len(reference["truth"]):
        raise AssertionError(f"{cohort}: filtered stay identity is misaligned")

    return (
        {
            **reference,
            "stay": stay,
            "events_path": str((ROOT / "data" / "processed" / events_name).resolve()),
            "split_path": str((ROOT / "data" / "processed" / split_name).resolve()),
        },
        cache["standardizer"],
        list(cache["variable_names"]),
    )


def _load_predictions(eval_dir: Path, reference: dict) -> np.ndarray:
    prediction = np.load(eval_dir / "output_pred.npy")
    truth = np.load(eval_dir / "input_y.npy")
    mask = np.load(eval_dir / "input_y_mask.npy").astype(bool)
    sample_id = np.load(eval_dir / "input_sample_ID.npy").astype(np.int64)
    if sample_id.tolist() != reference["test_index"]:
        raise AssertionError(f"test split order mismatch: {eval_dir}")
    dense_truth = truth[mask]
    if dense_truth.size != reference["dense_truth"].size:
        raise AssertionError(f"dense target count mismatch: {eval_dir}")
    if not np.allclose(dense_truth, reference["dense_truth"], atol=1e-5, rtol=0):
        raise AssertionError(f"dense target values mismatch: {eval_dir}")
    return np.asarray(prediction[mask], dtype=np.float32)[reference["keep"]]


def _window_max(stay: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(stay, return_inverse=True)
    result = np.empty(len(unique), dtype=np.int8)
    for index in range(len(unique)):
        result[index] = np.max(values[inverse == index])
    return unique, result


def _bootstrap(
    wrong: np.ndarray,
    carry_wrong: np.ndarray,
    cluster: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    low, high, draws = cluster_bootstrap_seed_average(
        wrong,
        carry_wrong,
        cluster,
        rng,
        n_boot=N_BOOT,
        return_draws=True,
    )
    p_raw = float(
        min(
            1.0,
            2.0
            * min(
                np.mean(draws <= 0.0),
                np.mean(draws >= 0.0),
            ),
        )
    )
    return float(low), float(high), p_raw


def _patient_counts(
    wrong: np.ndarray, patient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return patient-level numerator and denominator for each seed."""

    n_seeds, _ = wrong.shape
    n_patients = int(np.max(patient)) + 1
    numerator = np.zeros((n_seeds, n_patients), dtype=np.int64)
    denominator = np.zeros(n_patients, dtype=np.int64)
    for seed in range(n_seeds):
        np.add.at(numerator[seed], patient, wrong[seed].astype(np.int64))
    np.add.at(denominator, patient, 1)
    return numerator, denominator


def _validate_config(eval_dir: Path, cohort: str, arm: str) -> dict:
    config_path = eval_dir.parent / "configs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("checkpoint_selection_metric") != "macro_persistence":
        raise AssertionError(f"non-macro checkpoint selection: {config_path}")
    if config.get("dataset_subset_name") != "jbi_patient":
        raise AssertionError(f"non-patient cache selection: {config_path}")
    expected_loss = "MAE" if arm == "mae_macro" else "MSE"
    if str(config.get("loss", "")).upper() != expected_loss:
        raise AssertionError(
            f"{cohort}/{arm}: expected loss {expected_loss}, found {config.get('loss')} "
            f"in {config_path}"
        )
    return config


def _detail_rows(
    *,
    cohort: str,
    arm: str,
    model: str,
    seed_values: list[int],
    event_patient: np.ndarray,
    window_patient: np.ndarray,
    event_wrong_matrix: np.ndarray,
    window_wrong_matrix: np.ndarray,
    carry_event_wrong: np.ndarray,
    carry_window_wrong: np.ndarray,
) -> list[dict[str, object]]:
    event_num, event_den = _patient_counts(event_wrong_matrix, event_patient)
    window_num, window_den = _patient_counts(window_wrong_matrix, window_patient)
    carry_event_num, _ = _patient_counts(carry_event_wrong[None, :], event_patient)
    carry_window_num, _ = _patient_counts(carry_window_wrong[None, :], window_patient)
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seed_values):
        for patient_id in range(event_num.shape[1]):
            rows.append(
                {
                    "analysis_plan_version": PLAN,
                    "cohort": cohort,
                    "arm": arm,
                    "model": model,
                    "organ": "",
                    "variable": "",
                    "seed": seed,
                    "patient_index": patient_id,
                    "event_wrong_count": int(event_num[seed_index, patient_id]),
                    "event_total": int(event_den[patient_id]),
                    "window_wrong_count": int(window_num[seed_index, patient_id]),
                    "window_total": int(window_den[patient_id]),
                    "carry_event_wrong_count": int(carry_event_num[0, patient_id]),
                    "carry_window_wrong_count": int(carry_window_num[0, patient_id]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    reference, stats, variable_names = _load_identity(args.cohort)
    rng = np.random.default_rng(args.bootstrap_seed)
    tags = TRACKB._tags(args.cohort, args.arm)
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    seed_values = list(range(2024, 2024 + 5))

    for model in args.models:
        if model not in tags:
            raise AssertionError(f"no preregistered Track-B tag for {args.cohort}/{args.arm}/{model}")
        directories = TRACKB._eval_dirs(model, tags[model], args.cohort)
        if len(directories) != 5:
            raise AssertionError(
                f"Track B requires five completed seeds for {args.cohort}/{args.arm}/{model}, "
                f"found {len(directories)}"
            )
        configs = [_validate_config(directory, args.cohort, args.arm) for directory in directories]
        predictions = [_load_predictions(directory, reference) for directory in directories]

        for organ, variable_name, boundaries, direction, support in COMPONENTS:
            variable = int(variable_names.index(variable_name))
            event_mask = reference["variable"] == variable
            if not bool(event_mask.any()):
                raise AssertionError(f"{args.cohort}: component variable missing: {variable_name}")

            scale = float(stats["iqr"][variable])
            location = float(stats["median"][variable])
            true = reference["truth"][event_mask] * scale + location
            carry = reference["x_last"][event_mask] * scale + location
            # Grade the observation on the scale's own support, exactly as
            # the forecast and carry-forward arms are graded below.
            true_grade = grade(project(true, support), boundaries, direction)
            carry_grade = grade(project(carry, support), boundaries, direction)
            carry_wrong = carry_grade != true_grade
            event_patient = reference["patient"][event_mask]
            event_stay = reference["stay"][event_mask]

            event_wrong_rows: list[np.ndarray] = []
            window_wrong_rows: list[np.ndarray] = []
            window_patient: np.ndarray | None = None
            per_seed_event: list[float] = []
            per_seed_window: list[float] = []
            per_seed_abs: list[float] = []
            carry_window = _window_max(event_stay, carry_grade)[1]
            true_window = _window_max(event_stay, true_grade)[1]
            carry_window_wrong = carry_window != true_window

            for prediction, config in zip(predictions, configs):
                predicted = prediction[event_mask] * scale + location
                predicted_grade = grade(project(predicted, support), boundaries, direction)
                event_wrong = predicted_grade != true_grade
                _, predicted_window = _window_max(event_stay, predicted_grade)
                unique_stays, _ = _window_max(event_stay, true_grade)
                window_wrong = predicted_window != true_window
                event_wrong_rows.append(event_wrong)
                window_wrong_rows.append(window_wrong)
                current_window_patient = np.asarray(
                    [event_patient[event_stay == stay_id][0] for stay_id in unique_stays],
                    dtype=np.int64,
                )
                if window_patient is None:
                    window_patient = current_window_patient
                elif not np.array_equal(window_patient, current_window_patient):
                    raise AssertionError("window patient order changed across seeds")
                per_seed_event.append(float(np.mean(event_wrong)))
                per_seed_window.append(float(np.mean(window_wrong)))
                per_seed_abs.append(
                    float(np.mean(np.abs(predicted_grade.astype(int) - true_grade.astype(int))))
                )

            if window_patient is None:
                raise AssertionError("no window-level component events")
            event_wrong_matrix = np.stack(event_wrong_rows, axis=0)
            window_wrong_matrix = np.stack(window_wrong_rows, axis=0)
            event_ci_lo, event_ci_hi, event_p = _bootstrap(
                event_wrong_matrix, carry_wrong, event_patient, rng
            )
            window_ci_lo, window_ci_hi, window_p = _bootstrap(
                window_wrong_matrix, carry_window_wrong, window_patient, rng
            )
            detail = _detail_rows(
                cohort=args.cohort,
                arm=args.arm,
                model=model,
                seed_values=seed_values,
                event_patient=event_patient,
                window_patient=window_patient,
                event_wrong_matrix=event_wrong_matrix,
                window_wrong_matrix=window_wrong_matrix,
                carry_event_wrong=carry_wrong,
                carry_window_wrong=carry_window_wrong,
            )
            for row in detail:
                row["organ"] = organ
                row["variable"] = variable_name
            detail_rows.extend(detail)

            rows.append(
                {
                    "analysis_plan_version": PLAN,
                    "cohort": args.cohort,
                    "arm": args.arm,
                    "training_objective": "MAE" if args.arm == "mae_macro" else "MSE",
                    "model": model,
                    "organ": organ,
                    "variable": variable_name,
                    "n_seeds": len(predictions),
                    "n_events": int(event_mask.sum()),
                    "n_stays": int(np.unique(event_stay).size),
                    "n_patients": int(np.unique(event_patient).size),
                    "event_misclassification": float(np.mean(per_seed_event)),
                    "carryforward_event_misclassification": float(np.mean(carry_wrong)),
                    "event_excess_over_carryforward": float(
                        np.mean(per_seed_event) - np.mean(carry_wrong)
                    ),
                    "event_ci_lo": event_ci_lo,
                    "event_ci_hi": event_ci_hi,
                    "event_p_raw": event_p,
                    "window_misclassification": float(np.mean(per_seed_window)),
                    "carryforward_window_misclassification": float(np.mean(carry_window_wrong)),
                    "window_excess_over_carryforward": float(
                        np.mean(per_seed_window) - np.mean(carry_window_wrong)
                    ),
                    "window_ci_lo": window_ci_lo,
                    "window_ci_hi": window_ci_hi,
                    "window_p_raw": window_p,
                    "event_mean_abs_grade_error": float(np.mean(per_seed_abs)),
                    "bootstrap_n": N_BOOT,
                    "bootstrap_seed": args.bootstrap_seed,
                    "bootstrap_unit": reference["group_column"],
                    "selection_metric": "macro_persistence",
                    "data_source": f"ICU-KAN/{args.cohort}",
                    "data_variant": "jbi_patient",
                    "split_unit": "patient_grouped",
                    "group_column": reference["group_column"],
                    "cache_path": reference["events_path"],
                    "split_path": reference["split_path"],
                    "loss": str(configs[0].get("loss")),
                    "tag": tags[model],
                }
            )

    frame = pd.DataFrame(rows)
    for p_column, adjusted_column in (
        ("event_p_raw", "event_p_holm"),
        ("window_p_raw", "window_p_holm"),
    ):
        frame[adjusted_column] = np.nan
        for model, positions in frame.groupby("model").groups.items():
            values = frame.loc[positions, p_column].to_numpy(dtype=float)
            order = np.argsort(values)
            adjusted = np.empty_like(values)
            running = 0.0
            m = len(values)
            for rank, index in enumerate(order):
                running = max(running, float((m - rank) * values[index]))
                adjusted[index] = min(running, 1.0)
            frame.loc[positions, adjusted_column] = adjusted

    target = args.out or (ROOT / "results" / "jbi" / f"e8_sofa_by_objective_{args.cohort}_{args.arm}.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format="%.17g")
    detail_path = target.with_name(target.stem + "_patient_seed.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, float_format="%.17g")
    audit_path = target.with_name(target.stem + "_audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "analysis_plan_version": PLAN,
                "cohort": args.cohort,
                "arm": args.arm,
                "models": list(args.models),
                "tags": {model: tags[model] for model in args.models},
                "summary_csv": str(target.resolve()),
                "patient_seed_csv": str(detail_path.resolve()),
                "bootstrap_n": N_BOOT,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_unit": reference["group_column"],
                "seed_average_inside_bootstrap": True,
                "cache_path": reference["events_path"],
                "split_path": reference["split_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nwrote {len(frame)} rows to {target}")
    print(f"wrote {len(detail_rows)} patient-seed rows to {detail_path}")


if __name__ == "__main__":
    main()
