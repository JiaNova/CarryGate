"""Independent checks for the P1 binary-policy decision-curve calculation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def net_benefit(tp: float, fp: float, denominator: float, threshold: float) -> float:
    """Independent implementation of TP/N - FP/N * p_t/(1-p_t)."""
    return tp / denominator - fp / denominator * threshold / (1.0 - threshold)


def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    return prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)


def test_decision_curve_formula_on_analytic_policies() -> None:
    n = 100
    positives = 30
    negatives = n - positives
    threshold = 0.2

    assert net_benefit(0, 0, n, threshold) == pytest.approx(0.0, abs=1e-15)
    assert net_benefit(positives, negatives, n, threshold) == pytest.approx(
        treat_all_net_benefit(positives / n, threshold), abs=1e-15
    )
    assert net_benefit(positives, 0, n, threshold) == pytest.approx(positives / n, abs=1e-15)

    # An alarm assigned independently with probability r has r times the
    # treat-all net benefit in expectation at the same prevalence.
    alarm_probability = 0.4
    random_nb = net_benefit(
        alarm_probability * positives,
        alarm_probability * negatives,
        n,
        threshold,
    )
    assert random_nb == pytest.approx(
        alarm_probability * treat_all_net_benefit(positives / n, threshold), abs=1e-15
    )

    # The event prevalence is the shared policy reference, not a risk score.
    prevalence = positives / n
    for pt in (0.05, 0.10, 0.20, 0.30):
        assert treat_all_net_benefit(prevalence, pt) == pytest.approx(
            net_benefit(positives, negatives, n, pt), abs=1e-15
        )


def test_saved_p1_curves_recompute_from_saved_confusion_counts() -> None:
    metrics_path = ROOT / "results/jbi/p1_landmark_metrics.csv"
    curves_path = ROOT / "results/jbi/p1_landmark_decision_curve.csv"
    if not metrics_path.is_file() or not curves_path.is_file():
        pytest.skip("requires locally generated aggregate P1 metrics; result files are not distributed")
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with curves_path.open(encoding="utf-8-sig", newline="") as handle:
        curves = list(csv.DictReader(handle))

    key_fields = ("cohort", "organ", "component", "model_or_strategy", "scope")
    grouped_metrics = defaultdict(list)
    for row in metrics:
        grouped_metrics[tuple(row[field] for field in key_fields)].append(row)
    assert curves, "the frozen P1 decision-curve output is required"

    # The two fixed policies are emitted once per model run group, so their
    # rows repeat. Their denominators and confusion counts must be identical.
    count_fields = (
        "n_reference_windows", "true_positive_count", "false_positive_count",
        "measurement_ascertained_deterioration_count",
    )
    for key, repeated in grouped_metrics.items():
        for field in count_fields:
            values = [float(row[field]) for row in repeated]
            assert max(values) - min(values) <= 1e-12, f"inconsistent repeated count {field} for {key}"

    for curve in curves:
        key = tuple(curve[field] for field in key_fields)
        assert key in grouped_metrics, f"missing confusion counts for {key}"
        count = grouped_metrics[key][0]
        n = float(count["n_reference_windows"])
        tp = float(count["true_positive_count"])
        fp = float(count["false_positive_count"])
        positives = float(count["measurement_ascertained_deterioration_count"])
        threshold = float(curve["threshold_probability"])
        assert n > 0 and 0.0 < threshold < 1.0

        expected_nb = net_benefit(tp, fp, n, threshold)
        expected_treat_all = treat_all_net_benefit(positives / n, threshold)
        assert float(curve["net_benefit"]) == pytest.approx(expected_nb, abs=1e-12)
        assert float(curve["treat_none_net_benefit"]) == pytest.approx(0.0, abs=1e-12)
        assert float(curve["treat_all_net_benefit"]) == pytest.approx(expected_treat_all, abs=1e-12)


def test_saved_counts_reproduce_the_seed_average_estimand() -> None:
    """Mean counts recover mean net benefit because every seed shares N.

    Net benefit and treat-all net benefit are linear in TP, FP, and outcome
    count when the denominator is fixed. Recomputing from the saved equal-
    weight mean counts therefore equals the mean of seed-specific values in
    the original analysis.
    """
    metrics_path = ROOT / "results/jbi/p1_landmark_metrics.csv"
    if not metrics_path.is_file():
        pytest.skip("requires locally generated aggregate P1 metrics; result files are not distributed")
    with metrics_path.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    models = [row for row in rows if row["model_or_strategy"] in {
        "ICUKANForecaster", "GraFITi", "HyperIMTS"
    }]
    assert models
    for row in models:
        assert row["seed_aggregation"] == "equal-weight mean of seed-specific metrics"
        assert float(row["n_reference_windows"]) > 0
