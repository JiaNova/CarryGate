import collections

import numpy as np
import pytest

from src.data.eicu_extract import (
    EXCLUDED_LABNAMES,
    Stay,
    _convert_value,
    _load_map,
    _numeric_query,
    _stratified_sample,
)
from src.data.p12_events import TIME_SERIES_VARIABLES
from scripts.run_eicu_loho import build_hospital_folds


def _stay(stay_id: int, label: int) -> Stay:
    return Stay(
        stay_id=stay_id,
        subject_id=stay_id,
        hadm_id=stay_id,
        hospital_id=stay_id,
        uniquepid=f"u-{stay_id}",
        gender="",
        anchor_age=50,
        label=label,
        outtime=48.0,
        admission_weight=80.0,
    )


def test_map_keeps_exact_p12_order_and_registered_exclusions():
    mapping = _load_map()

    assert tuple(mapping["variable_names"]) == TIME_SERIES_VARIABLES
    assert tuple(mapping["variables"]) == TIME_SERIES_VARIABLES
    assert [mapping["variables"][name]["var_id"] for name in TIME_SERIES_VARIABLES] == list(range(37))
    assert set(mapping["excluded_variable_ids"]) == {6, 19, 32}
    assert all(spec["verified"] is True for spec in mapping["variables"].values())
    assert mapping["variables"]["Cholesterol"]["labname"] == "total cholesterol"
    assert mapping["variables"]["TroponinI"]["labname"] == "troponin - I"
    assert mapping["variables"]["FiO2"]["source_table"] == "pivoted_bg"
    assert not EXCLUDED_LABNAMES.intersection(
        {mapping["variables"][name].get("labname") for name in ("Cholesterol", "Mg", "TroponinI", "TroponinT")}
    )


def test_sampling_is_deterministic_and_preserves_mortality_quota():
    stays = [_stay(index, int(index % 5 == 0)) for index in range(100)]
    first = _stratified_sample(stays, sample_size=20, seed=42)
    second = _stratified_sample(stays, sample_size=20, seed=42)

    assert [stay.stay_id for stay in first] == [stay.stay_id for stay in second]
    assert len(first) == 20
    assert collections.Counter(stay.label for stay in first) == {0: 16, 1: 4}


def test_registered_unit_conversions_and_domain_checks():
    quality = collections.Counter()
    fio2 = _convert_value(
        "FiO2", 50.0, {"unit_rule": "fraction_or_percent"}, quality
    )
    assert fio2 == pytest.approx(0.5)
    assert quality["fio2_values_seen"] == 1
    assert quality["fio2_percent_to_fraction"] == 1

    assert _convert_value("Temp", 37.0, {"unit_rule": "celsius_verified"}, quality) == pytest.approx(37.0)
    assert _convert_value("Temp", 98.6, {"unit_rule": "celsius_verified"}, quality) is None
    assert _convert_value("GCS", 15.0, {"unit_rule": "gcs_3_to_15"}, quality) == pytest.approx(15.0)
    assert _convert_value("GCS", 2.0, {"unit_rule": "gcs_3_to_15"}, quality) is None


def test_numeric_query_uses_bound_stay_ids_and_registered_offsets():
    mapping = _load_map()
    specs = [("HR", mapping["variables"]["HR"]), ("MAP", mapping["variables"]["MAP"])]
    query, params, aliases = _numeric_query([_stay(11, 0), _stay(12, 1)], specs)

    # P1 extraction reads only the first 48 hours; stay IDs and horizon are
    # bound parameters rather than interpolated SQL text.
    assert params == [11, 12, 48 * 60]
    assert aliases == {"v14": "HR", "v18": "MAP"}
    assert "VALUES (%s),(%s)" in query
    assert "e.\"chartoffset\" >= 0" in query
    assert "e.\"chartoffset\" < %s" in query
    assert "e.\"patientunitstayid\"" in query
    assert "11" not in query and "12" not in query


def test_stay_contract_contains_hospital_and_binary_label():
    stay = _stay(9, 1)
    assert stay.hospital_id == 9
    assert stay.label in (0, 1)
    assert np.isfinite(stay.outtime)


def test_loho_fold_assignment_is_disjoint_balanced_and_reproducible():
    counts = {hospital: 100 + (hospital % 37) for hospital in range(1, 146)}
    first, first_rows = build_hospital_folds(counts, n_folds=5, seed=42)
    second, second_rows = build_hospital_folds(counts, n_folds=5, seed=42)

    assert first == second
    assert first_rows == second_rows
    assert len(first_rows) == 145
    assert set(first) == set(counts)
    assert len(set(first.values())) == 5
    assert all(sum(fold == assigned for assigned in first.values()) == 29 for fold in range(5))
    for layer in range(5):
        layer_rows = [row for row in first_rows if row["size_layer"] == layer]
        assert len(layer_rows) == 29
        assert sorted(collections.Counter(row["fold"] for row in layer_rows).values()) == [5, 6, 6, 6, 6]
