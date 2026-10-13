"""The observation must be graded on the same support as the forecast.

Each SOFA component compares three graded quantities: the observation, the
forecast and the carry-forward value.  Forecasts and carry-forward values have
always been projected onto an integer scale's support before grading.  The
observation was not, and observations do not survive the standardise /
de-standardise round trip exactly: on the eICU patient cache the Glasgow Coma
Scale is stored with median 14 and IQR 5, so a true 15 comes back as
14.99999979 and grades as 1 rather than 0.  Because grade 0 requires exactly
15, every true grade-0 event in that cohort was mis-graded, which inverted the
neurological component's comparison between training objectives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_sofa_grades import COMPONENTS, grade, project  # noqa: E402

GCS = next(c for c in COMPONENTS if c[1] == "GCS")

STANDARD_CUTPOINTS = {
    "Platelets": ("low", (150.0, 100.0, 50.0, 20.0), None),
    "Bilirubin": ("high", (1.2, 2.0, 6.0, 12.0), None),
    "GCS": ("low", (15.0, 13.0, 10.0, 6.0), (3.0, 15.0)),
    "Creatinine": ("high", (1.2, 2.0, 3.5, 5.0), None),
}


def test_round_trip_below_a_boundary_still_grades_as_the_integer() -> None:
    _, _, bounds, direction, support = GCS
    # The exact value produced by (15 - 14) / 5 stored as float32, times 5,
    # plus 14 -- the eICU patient cache's round trip for a true GCS of 15.
    observed = np.array([14.999999791384], dtype=np.float64)
    assert grade(observed, bounds, direction)[0] == 1, "the hazard is real"
    assert grade(project(observed, support), bounds, direction)[0] == 0


def test_projection_is_identity_on_clean_integers() -> None:
    _, _, bounds, direction, support = GCS
    clean = np.arange(3.0, 16.0)
    assert np.array_equal(project(clean, support), clean)
    assert np.array_equal(
        grade(project(clean, support), bounds, direction),
        grade(clean, bounds, direction),
    )


@pytest.mark.parametrize(("component", "threshold", "expected"), [
    ("Platelets", 150.0, (1, 0, 0)),
    ("Platelets", 100.0, (2, 1, 1)),
    ("Platelets", 50.0, (3, 2, 2)),
    ("Platelets", 20.0, (4, 3, 3)),
    ("Bilirubin", 1.2, (0, 1, 1)),
    ("Bilirubin", 2.0, (1, 2, 2)),
    ("Bilirubin", 6.0, (2, 3, 3)),
    ("Bilirubin", 12.0, (3, 4, 4)),
    ("GCS", 15.0, (0, 0, 0)),
    ("GCS", 13.0, (1, 1, 1)),
    ("GCS", 10.0, (2, 2, 2)),
    ("GCS", 6.0, (3, 3, 3)),
    ("Creatinine", 1.2, (0, 1, 1)),
    ("Creatinine", 2.0, (1, 2, 2)),
    ("Creatinine", 3.5, (2, 3, 3)),
    ("Creatinine", 5.0, (3, 4, 4)),
])
def test_standard_sofa_cutpoint_boundaries(component: str, threshold: float, expected: tuple[int, int, int]) -> None:
    direction, bounds, support = STANDARD_CUTPOINTS[component]
    epsilon = 1e-6
    values = np.asarray([threshold - epsilon, threshold, threshold + epsilon], dtype=np.float64)
    if support is not None:
        values = project(values, support)
    assert tuple(grade(values, bounds, direction).tolist()) == expected


def test_production_sofa_table_matches_hand_written_standard_table() -> None:
    production = {row[1]: (row[3], tuple(row[2]), row[4]) for row in COMPONENTS}
    assert production == STANDARD_CUTPOINTS


def test_gcs_support_extremes_and_standardisation_round_trip() -> None:
    _, _, bounds, direction, support = GCS
    values = np.asarray([3.0, 15.0, 14.999999791384], dtype=np.float64)
    assert np.array_equal(grade(project(values, support), bounds, direction), [4, 0, 0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected_instead_of_silently_graded(value: float) -> None:
    _, _, bounds, direction, support = GCS
    with pytest.raises(ValueError, match="non-finite"):
        grade(np.asarray([value]), bounds, direction)
    with pytest.raises(ValueError, match="non-finite"):
        project(np.asarray([value]), support)


@pytest.mark.parametrize("source", [
    "scripts/analyze_sofa_grades.py",
    "scripts/jbi/04_analyze_patient_sofa_components.py",
    "scripts/jbi/05_analyze_grade_transitions.py",
    "scripts/jbi/14_analyze_patient_external_sofa.py",
])
def test_no_pipeline_grades_an_unprojected_observation(source: str) -> None:
    text = (ROOT / source).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("g_true", "true_grade")):
            continue
        if "grade(" not in stripped:
            continue
        assert "project(" in stripped, f"{source}: {stripped}"
