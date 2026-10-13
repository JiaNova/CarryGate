#!/usr/bin/env python
"""Paired patient bootstrap *inside* the Track B protocol.

The headline comparison puts the 15-seed native ICU-KAN reference against
external models trained in the PyOmniTS harness.  Those two arms differ in the
training objective (masked L1 versus MSE), the seed count, the epoch cap and
the whole training pipeline, so a difference between them cannot be read as an
architecture effect.

Track B also ran this project's own architecture through the harness under the
externals' exact conditions (``ICUKANForecaster``).  Comparing that arm against
the externals holds everything but the architecture fixed, which is the
comparison an architecture claim actually needs.  It is the number a reviewer
will compute from Table Track B in about thirty seconds, so it belongs in the
manuscript rather than in a reviewer's reply.

Run from the repository root:
    python scripts/jbi/23_adapter_vs_external_paired.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "jbi"
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))

from rescore_macro import VARIANT_CACHES, cohort_events  # noqa: E402

N_VARIABLES = 37
UNSCORED = frozenset({6, 19, 32})
MIN_TEST_EVENTS = 50
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260914
ADAPTER = "ICUKANForecaster"
# The two comparison surfaces.  ``macro`` is the MSE-trained primary arm the
# matrix was first built on.  ``mae_macro`` is the factorial cell in which every
# model is trained on masked L1, the objective the evaluation metric is built
# from; the registered 2x2 shows MSE costs every model 0.056 to 0.065 macro
# skill, so an architecture comparison run on the MSE arm is being taken at the
# objective this paper's own decomposition shows to be misaligned.
ARMS = ("macro", "mae_macro")
PROTOCOL = {"macro": "trackb_macro_checkpoint", "mae_macro": "trackb_mae_macro"}


def _tags(cohort: str, arm: str = "macro") -> dict[str, str]:
    """Return the frozen tags for one isolated cohort and comparison arm."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "mae_macro":
        # GRU_D was not carried into the factorial, so the aligned-objective
        # arm holds the four models the factorial actually crossed.
        return {
            model: f"jbi_trackb_controls_factorial_{model}_mae_macro_v1"
            for model in ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")
        }
    hyper_tag = (
        "jbi_trackb_repaired_HyperIMTS_macro_v2"
        if cohort == "eicu"
        else "jbi_trackb_controls_macro_HyperIMTS_v1"
    )
    return {
        "ICUKANForecaster": "jbi_trackb_controls_macro_ICUKANForecaster_v1",
        "GraFITi": "jbi_trackb_controls_macro_GraFITi_v1",
        "HyperIMTS": hyper_tag,
        "Warpformer": "jbi_trackb_controls_macro_Warpformer_v1",
        "GRU_D": "jbi_trackb_controls_macro_GRU_D_v1",
    }


def destination(cohort: str, arm: str) -> Path:
    """Each arm writes its own file; neither may silently represent the other."""
    stem = "e2_trackb_adapter_vs_external_paired"
    if arm == "mae_macro":
        stem += "_mae"
    return OUT / (f"{stem}.csv" if cohort == "eicu" else f"{stem}_{cohort}.csv")


def _reference(cohort: str = "eicu") -> dict:
    events_name, split_name = VARIANT_CACHES["jbi_patient"][cohort]
    with (ROOT / "data" / "processed" / events_name).open("rb") as handle:
        examples = pickle.load(handle)["examples"]
    split = json.loads(
        (ROOT / "data" / "processed" / split_name).read_text(encoding="utf-8")
    )
    if split["split_unit"] != "patient_grouped":
        raise AssertionError("this comparison is only defined on the patient split")
    test_index = [int(i) for i in split["test"]]
    reference = cohort_events(cohort, data_variant="jbi_patient")
    group_column = str(
        split.get("group_column")
        or ("uniquepid" if cohort == "eicu" else "subject_id")
    )
    patients = sorted({str(examples[i][group_column]) for i in test_index})
    lookup = {p: k for k, p in enumerate(patients)}
    # cohort_events() uses the adapter's dense per-sample traversal.  The
    # patient label is constant within a sample, so repeating it by the
    # sample's target-event count preserves alignment after that traversal.
    patient = np.concatenate(
        [
            np.full(
                len(examples[i]["target"]["var"]),
                lookup[str(examples[i][group_column])],
                dtype=np.int64,
            )
            for i in test_index
        ]
    )
    truth = np.asarray(reference["truth"], dtype=np.float64)
    variable = np.asarray(reference["var"], dtype=np.int64)
    x_last = np.asarray(reference["x_last"], dtype=np.float64)
    if len(patient) != len(truth):
        raise AssertionError("patient/event alignment mismatch")
    keep = ~np.isin(variable, sorted(UNSCORED))
    return {
        "cohort": cohort,
        "test_index": test_index,
        "dense_truth": truth,
        "dense_var": variable,
        "dense_x_last": x_last,
        "truth": truth[keep],
        "variable": variable[keep],
        "x_last": x_last[keep],
        "patient": patient[keep],
        "keep": keep,
        "n_patients": len(patients),
        "n_events_dense": truth.size,
        "group_column": group_column,
    }


def _grid(values, ref):
    grid = np.zeros((ref["n_patients"], N_VARIABLES), dtype=np.float64)
    np.add.at(grid, (ref["patient"], ref["variable"]), values)
    return grid


def _eval_dirs(model: str, tag: str, cohort: str) -> list[Path]:
    root = (
        ROOT
        / "external"
        / "PyOmniTS"
        / "storage"
        / "results"
        / "ICUKAN"
        / cohort
        / model
        / f"{tag}_{cohort}"
        / "24_24"
    )
    return sorted(root.glob("*/iter*/eval_*/")) if root.exists() else []


def _arm(model: str, tag: str, ref: dict):
    directories = _eval_dirs(model, tag, ref["cohort"])
    if not directories:
        raise FileNotFoundError(f"no Track B runs for tag {tag}")
    total = np.zeros((ref["n_patients"], N_VARIABLES), dtype=np.float64)
    for directory in directories:
        config = yaml.safe_load((directory.parent / "configs.yaml").read_text()) or {}
        if config.get("dataset_subset_name") != "jbi_patient":
            raise AssertionError(f"non-patient cache: {directory}")
        sample_id = np.load(directory / "input_sample_ID.npy").astype(np.int64)
        if sample_id.tolist() != ref["test_index"]:
            raise AssertionError(f"test split order mismatch: {directory}")
        mask = np.load(directory / "input_y_mask.npy").astype(bool)
        truth = np.load(directory / "input_y.npy")[mask]
        if not np.allclose(truth, ref["dense_truth"], atol=1e-5, rtol=0):
            raise AssertionError(f"dense target mismatch: {directory}")
        prediction = np.load(directory / "output_pred.npy")[mask]
        if prediction.size != ref["n_events_dense"]:
            raise AssertionError(f"dense event count mismatch: {directory}")
        total += _grid(
            np.abs(prediction[ref["keep"]].astype(np.float64) - ref["truth"]), ref
        )
    return total / len(directories), len(directories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--cohort", choices=("eicu", "mimiciv"), default="eicu")
    parser.add_argument("--arm", choices=ARMS, default="macro")
    args = parser.parse_args()

    ref = _reference(args.cohort)
    baseline = _grid(
        np.abs(ref["x_last"] - ref["truth"]), ref
    )
    count = _grid(np.ones(ref["truth"].size), ref)
    scored = (
        (count.sum(axis=0) >= MIN_TEST_EVENTS)
        & (baseline.sum(axis=0) > 1e-9)
        & ~np.isin(np.arange(N_VARIABLES), sorted(UNSCORED))
    )

    def skills(selection, error):
        model_total = error[selection].sum(axis=0)
        base_total = baseline[selection].sum(axis=0)
        usable = scored & (base_total > 1e-9)
        return (float(np.mean(1.0 - model_total[usable] / base_total[usable])),
                float(1.0 - error[selection].sum() / baseline[selection].sum()))

    arms = {}
    tags = _tags(args.cohort, args.arm)
    for name, tag in tags.items():
        arms[name], n_seeds = _arm(name, tag, ref)
        print(f"{name:<18} {n_seeds} seeds")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = ref["n_patients"]
    replicates = [rng.integers(0, n, n) for _ in range(args.reps)]
    everyone = np.arange(n)
    ours = arms[ADAPTER]

    rows = []
    for name, error in arms.items():
        macro, pooled = skills(everyone, error)
        row = {"analysis_plan_version": "JBI_analysis_plan_v1",
               "cohort": args.cohort,
               "protocol": PROTOCOL[args.arm],
               "comparison": "architecture_only",
               "model": name, "macro": macro, "pooled": pooled}
        if name != ADAPTER:
            diffs = np.array([
                (skills(i, ours)[0] - skills(i, error)[0],
                 skills(i, ours)[1] - skills(i, error)[1]) for i in replicates
            ])
            ref_macro, ref_pooled = skills(everyone, error)
            our_macro, our_pooled = skills(everyone, ours)
            row.update({
                "paired_macro_difference": our_macro - ref_macro,
                "paired_macro_ci_low": float(np.percentile(diffs[:, 0], 2.5)),
                "paired_macro_ci_high": float(np.percentile(diffs[:, 0], 97.5)),
                "paired_pooled_difference": our_pooled - ref_pooled,
                "paired_pooled_ci_low": float(np.percentile(diffs[:, 1], 2.5)),
                "paired_pooled_ci_high": float(np.percentile(diffs[:, 1], 97.5)),
            })
        rows.append(row)

    frame = pd.DataFrame(rows)
    for estimator in ("macro", "pooled"):
        low, high = frame[f"paired_{estimator}_ci_low"], frame[f"paired_{estimator}_ci_high"]
        frame[f"paired_{estimator}_excludes_zero"] = (low > 0) | (high < 0)
    frame["bootstrap_reps"] = args.reps
    frame["bootstrap_seed"] = BOOTSTRAP_SEED
    frame["bootstrap_unit"] = ref["group_column"]
    frame["data_variant"] = "jbi_patient"
    frame["split_unit"] = "patient_grouped"
    frame["n_test_patients"] = n
    frame["arm"] = args.arm
    frame["training_objective"] = "MAE" if args.arm == "mae_macro" else "MSE"
    target = destination(args.cohort, args.arm)
    frame.to_csv(target, index=False)

    show = ["model", "macro", "paired_macro_difference", "paired_macro_ci_low",
            "paired_macro_ci_high", "paired_macro_excludes_zero"]
    print()
    print(frame[show].round(4).to_string(index=False))
    strong = frame[frame.model.isin(("GraFITi", "HyperIMTS", "Warpformer"))]
    if not strong.paired_macro_excludes_zero.any():
        print("\nEvery interval against a strong external model spans zero: under the "
              "Track B protocol this architecture is not distinguishable from them.")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
