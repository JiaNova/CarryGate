#!/usr/bin/env python
"""D3 E1: SOFA grade misclassification for this project and every baseline.

Spec: ICU_KAN_D3_clinical_endpoint_spec.md section 1.

Persistence skill says how much absolute error a model removes. It does not say
whether the model would put a patient in the right organ-dysfunction category.
SOFA supplies four components whose grade is a threshold function of a single
variable, so a forecast can be graded and compared with the grade computed from
the observed value.

Respiration and cardiovascular are excluded: SOFA needs ventilation status and
vasopressor dose for those, MechVent is one of the three variables dropped for a
degenerate denominator, and no vasopressor data exists in any of the caches.
Grading them on partial information would produce a number no reader could
falsify.
"""

from __future__ import annotations

import glob
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))

from rescore_macro import UNSCORED, cohort_events  # noqa: E402

OUT = ROOT / "results" / "d3_clinical"
COHORTS = ["p12", "mimiciv", "eicu"]
CACHES = {"p12": ("p12_events.pkl", "split_indices_12k.json"),
          "mimiciv": ("mimiciv_events.pkl", "mimiciv_split_indices_12k.json"),
          "eicu": ("eicu_events.pkl", "eicu_split_indices_12k.json")}
SEEDS = list(range(15))
N_BOOT = 2000

# Standard SOFA cut-points. Each entry is (organ, variable, boundaries, direction);
# "low" means smaller values are worse, "high" means larger values are worse.
# ``support`` projects a forecast onto the values the variable can actually take
# before it is graded. Only GCS needs it, and it needs it badly: the scale is
# integer-valued, SOFA CNS grade 0 requires a score of exactly 15, and 45% of
# observed values are exactly 15. Carry-forward copies an observed integer and
# lands on grade 0 for free, while a continuous forecast never does, so grading
# raw forecasts would measure the representation and not the model. This
# projection was added after the first run exposed the integer-scale issue.
# Observations, forecasts and carry-forward values now use the same projection;
# unprojected forecasts remain as a sensitivity column.
COMPONENTS = [
    ("Coagulation", "Platelets", [150.0, 100.0, 50.0, 20.0], "low", None),
    ("Liver", "Bilirubin", [1.2, 2.0, 6.0, 12.0], "high", None),
    ("CNS", "GCS", [15.0, 13.0, 10.0, 6.0], "low", (3.0, 15.0)),
    ("Renal", "Creatinine", [1.2, 2.0, 3.5, 5.0], "high", None),
]


def project(values: np.ndarray, support) -> np.ndarray:
    """Snap a forecast onto an integer-valued scale's support."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot project non-finite clinical values onto SOFA support")
    if support is None:
        return values
    low, high = support
    return np.clip(np.round(values), low, high)


def grade(values: np.ndarray, boundaries, direction: str) -> np.ndarray:
    """SOFA sub-score 0-4 for one component."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot assign a SOFA grade to a non-finite clinical value")
    out = np.zeros(values.shape, dtype=np.int8)
    if direction == "low":
        # grade 1 once below boundaries[0], rising as the value falls further
        out = np.where(values < boundaries[0], 1, 0)
        for level, bound in enumerate(boundaries[1:], start=2):
            out = np.where(values < bound, level, out)
    else:
        out = np.where(values >= boundaries[0], 1, 0)
        for level, bound in enumerate(boundaries[1:], start=2):
            out = np.where(values >= bound, level, out)
    return out.astype(np.int8)


_STATS: dict[str, dict] = {}
_STAYS: dict[str, np.ndarray] = {}


def standardiser(cohort: str) -> dict:
    """Load each cohort's robust statistics once; the pickles are hundreds of MB."""
    if cohort not in _STATS:
        events, _ = CACHES[cohort]
        with open(ROOT / "data" / "processed" / events, "rb") as handle:
            _STATS[cohort] = pickle.load(handle)["standardizer"]
    return _STATS[cohort]


def destandardise(cohort: str, variable_index: int, values: np.ndarray) -> np.ndarray:
    stats = standardiser(cohort)
    return values * float(stats["iqr"][variable_index]) + float(stats["median"][variable_index])


def stay_ids(cohort: str) -> np.ndarray:
    """Stay index for each scored target event, in cache order."""
    if cohort in _STAYS:
        return _STAYS[cohort]
    events, split = CACHES[cohort]
    cache = pickle.load(open(ROOT / "data" / "processed" / events, "rb"))["examples"]
    test = json.load(open(ROOT / "data" / "processed" / split))["test"]
    ids = []
    for position, index in enumerate(test):
        var = np.asarray(cache[index]["target"]["var"])
        ids.append(np.full(int((~np.isin(var, sorted(UNSCORED))).sum()), position))
    _STAYS[cohort] = np.concatenate(ids)
    return _STAYS[cohort]


def cluster_bootstrap_seed_average(
    wrong_a, wrong_b, stay, rng, n_boot=N_BOOT, return_draws=False
):
    """Bootstrap the same seed-averaged estimand used by the point estimate.

    ``wrong_a`` has shape ``(n_seeds, n_events)`` and contains one binary
    error vector per final training seed.  The point estimate is the mean of
    the seed-specific event error rates.  Each bootstrap draw resamples stays,
    computes that same rate for every seed, and then averages over seeds.  The
    previous implementation used only seed 0 for the interval while averaging
    all seeds for the point estimate; that estimand mismatch produced point
    estimates outside their own intervals.
    """
    wrong_a = np.asarray(wrong_a, dtype=float)
    if wrong_a.ndim == 1:
        wrong_a = wrong_a[None, :]
    wrong_b = np.asarray(wrong_b, dtype=float).reshape(-1)
    stay = np.asarray(stay).reshape(-1)
    if wrong_a.shape[1] != len(wrong_b) or len(stay) != len(wrong_b):
        raise ValueError("bootstrap arrays must share the event dimension")
    _, inverse = np.unique(stay, return_inverse=True)
    n_stays = int(inverse.max()) + 1
    n_seeds = int(wrong_a.shape[0])
    per_seed_diff = np.zeros((n_seeds, n_stays), dtype=float)
    for seed in range(n_seeds):
        per_seed_diff[seed] = np.bincount(
            inverse,
            weights=wrong_a[seed] - wrong_b,
            minlength=n_stays,
        )
    count = np.bincount(inverse, minlength=n_stays).astype(float)
    picks = rng.integers(0, n_stays, size=(n_boot, n_stays))
    numerator = per_seed_diff[:, picks].sum(axis=2)
    denominator = np.maximum(count[picks].sum(axis=1), 1.0)
    draws = (numerator / denominator[None, :]).mean(axis=0)
    interval = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    if return_draws:
        return interval[0], interval[1], draws
    return interval


def main(output_path: Path | None = None) -> None:
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {output_path}")
    names = yaml.safe_load(
        (ROOT / "configs" / "mimiciv_variable_map.yaml").read_text())["variable_names"]
    index_of = {n: i for i, n in enumerate(names)}
    matrix = pd.read_csv(ROOT / "results" / "d1_external" / "d1c_matrix.csv")
    rng = np.random.default_rng(0)
    rows = []

    for cohort in COHORTS:
        ref = cohort_events(cohort)                 # dense-grid order
        scored = ~np.isin(ref["var"], sorted(UNSCORED))
        var_dense = ref["var"][scored]
        truth_dense = ref["truth"][scored]
        last_dense = ref["x_last"][scored]
        stay_dense = stay_ids(cohort)               # contiguous per stay in both orders
        if stay_dense.size != var_dense.size:
            raise SystemExit(f"{cohort}: stay ids {stay_dense.size} vs events {var_dense.size}")

        # our predictions, averaged over seeds at the event level
        ours = []
        for seed in SEEDS:
            path = OUT / f"pred_{cohort}_seed{seed}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            keep = ~np.isin(z["variable"], sorted(UNSCORED))
            ours.append({"pred": z["prediction"][keep], "truth": z["target"][keep],
                         "var": z["variable"][keep], "x_last": z["x_last"][keep]})
        if not ours:
            raise SystemExit(f"no exported predictions for {cohort}; run the export first")

        arms = {"This work": ours}
        for model in sorted(matrix[matrix.cohort == cohort].model.unique()):
            seeds = []
            for e in matrix[(matrix.cohort == cohort) & (matrix.model == model)].itertuples():
                pred = np.load(Path(e.eval_dir) / "output_pred.npy")
                mask = np.load(Path(e.eval_dir) / "input_y_mask.npy").astype(bool)
                seeds.append({"pred": pred[mask][scored], "truth": truth_dense,
                              "var": var_dense, "x_last": last_dense})
            arms[model] = seeds

        for organ, variable, bounds, direction, support in COMPONENTS:
            v = index_of[variable]
            for name, seeds in arms.items():
                per_seed = []
                for s in seeds:
                    sel = s["var"] == v
                    if not sel.any():
                        continue
                    truth_c = destandardise(cohort, v, s["truth"][sel])
                    pred_raw = destandardise(cohort, v, s["pred"][sel])
                    last_c = destandardise(cohort, v, s["x_last"][sel])
                    pred_c = project(pred_raw, support)
                    # The observed value must be graded on the same support
                    # as the forecast and the carry-forward arm.  Without
                    # this, a scale whose standardiser does not round-trip
                    # exactly (eICU/P12 GCS) grades a true 15 as grade 1.
                    g_true = grade(project(truth_c, support), bounds, direction)
                    g_pred = grade(pred_c, bounds, direction)
                    g_raw = grade(pred_raw, bounds, direction)
                    g_last = grade(project(last_c, support), bounds, direction)
                    per_seed.append({
                        "wrong": (g_pred != g_true),
                        "wrong_unprojected": (g_raw != g_true),
                        # SOFA grade is monotone in severity for every
                        # component.  Undergrading therefore always means
                        # predicted grade < true grade, independent of the
                        # raw variable direction.
                        "under": (g_pred < g_true),
                        "abs": np.abs(g_pred.astype(int) - g_true.astype(int)),
                        "cf_wrong": (g_last != g_true),
                        "n": int(sel.sum()),
                    })
                if not per_seed:
                    continue
                wrong_matrix = np.stack([p["wrong"] for p in per_seed], axis=0)
                wrong_raw_matrix = np.stack(
                    [p["wrong_unprojected"] for p in per_seed], axis=0
                )
                under_matrix = np.stack([p["under"] for p in per_seed], axis=0)
                abs_matrix = np.stack([p["abs"] for p in per_seed], axis=0)
                cf_wrong = per_seed[0]["cf_wrong"]
                wrong = float(wrong_matrix.mean())
                wrong_raw = float(wrong_raw_matrix.mean())
                under = float(under_matrix.mean())
                abserr = float(abs_matrix.mean())
                cf = float(cf_wrong.mean())
                # Stays are contiguous and equally sized in both the cache
                # order and the dense-grid order, so element i of either
                # selection sits in the same stay.
                first = per_seed[0]
                stay_sel = stay_dense[var_dense == v]
                if stay_sel.size != first["wrong"].size:
                    raise SystemExit(
                        f"{cohort}/{variable}/{name}: {first['wrong'].size} events "
                        f"but {stay_sel.size} stay labels")
                lo, hi = cluster_bootstrap_seed_average(
                    wrong_matrix, cf_wrong, stay_sel, rng
                )
                rows.append({"analysis_plan_version": "JBI_analysis_plan_v1",
                             "cohort": cohort, "organ": organ, "variable": variable,
                             "model": name, "n_events": first["n"], "n_seeds": len(per_seed),
                             "misclassification": wrong,
                             "misclassification_unprojected": wrong_raw,
                             "carryforward_misclassification": cf,
                             "excess_over_carryforward": wrong - cf,
                             "undergrading": under, "mean_abs_grade_error": abserr,
                             "ci_lo": lo, "ci_hi": hi,
                             "bootstrap_n": N_BOOT,
                             "bootstrap_unit": "test_stay",
                             "estimand": "mean_seed_event_misclassification_difference"})

    frame = pd.DataFrame(rows)
    output = output_path or OUT / "sofa_grades.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.17g")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=None,
        help="write to a new path without overwriting an existing result",
    )
    args = parser.parse_args()
    main(output_path=args.output)
