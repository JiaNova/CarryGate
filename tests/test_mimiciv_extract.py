from pathlib import Path

import numpy as np
import pytest

from src.data.mimiciv_extract import _convert_value, _load_map, _load_reference_ranges
from src.data.p12_events import (
    TIME_SERIES_VARIABLES,
    EventStandardizer,
    annotate_events,
    make_event_example,
)


def _record():
    events = [
        ("HR", 1.0, 60.0),
        ("HR", 4.0, 80.0),
        ("Temp", 23.5, 37.0),
        ("FiO2", 24.0, 50.0),
        ("PaO2", 47.999, 90.0),
    ]
    return {
        "record_id": 7,
        "sample_id": "mimiciv-7",
        "label": 0,
        "set_name": "mimiciv",
        "static": {"ICUType": 1},
        "events": events,
        "events_with_delta": annotate_events(events),
    }


def test_map_keeps_the_exact_p12_variable_contract():
    mapping = _load_map()

    assert tuple(mapping["variable_names"]) == TIME_SERIES_VARIABLES
    assert tuple(mapping["variables"]) == TIME_SERIES_VARIABLES
    assert [mapping["variables"][name]["var_id"] for name in TIME_SERIES_VARIABLES] == list(range(37))
    assert all(spec["verified"] is True for spec in mapping["variables"].values())


def test_cache_example_has_drop_in_keys_and_causal_windows():
    record = _record()
    standardizer = EventStandardizer.fit([record])
    example = make_event_example(record, standardizer, t_obs=24.0)

    assert set(example) == {
        "record_id",
        "sample_id",
        "label",
        "set_name",
        "icu_type",
        "t_obs",
        "obs",
        "target",
    }
    assert tuple(standardizer.variable_names) == TIME_SERIES_VARIABLES
    assert example["icu_type"] == 1
    assert example["obs"]["time"].tolist() == [1.0, 4.0, 23.5]
    assert example["target"]["time"].tolist() == pytest.approx([24.0, 47.999])
    assert example["obs"]["delta_t"].tolist()[:2] == [1.0, 3.0]
    assert np.all(example["obs"]["time"] < 24.0)
    assert np.all(example["target"]["time"] >= 24.0)
    assert np.all(example["target"]["time"] < 48.0)


def test_registered_unit_conversions_and_gcs_domain():
    temp_spec = {"unit_rule": "fahrenheit_above_50_to_celsius"}
    fio2_spec = {"unit_rule": "fio2_percent_to_fraction"}
    gcs_spec = {"unit_rule": "gcs_total_3_to_15"}

    temp, temp_hit = _convert_value("Temp", 98.6, temp_spec)
    fio2, fio2_hit = _convert_value("FiO2", 0.5, fio2_spec)
    fio2_pct, fio2_pct_hit = _convert_value("FiO2", 50.0, fio2_spec)

    assert temp == pytest.approx(37.0, abs=1e-6)
    assert temp_hit is True
    # P12 stores FiO2 as a fraction, so MIMIC-IV percentages convert down.
    assert fio2 == pytest.approx(0.5)
    assert fio2_hit is False
    assert fio2_pct == pytest.approx(0.5)
    assert fio2_pct_hit is True
    assert _convert_value("GCS", 3, gcs_spec)[0] == 3.0
    with pytest.raises(ValueError, match="GCS"):
        _convert_value("GCS", 2, gcs_spec)


def test_measured_reference_table_has_the_exact_p12_contract():
    path = Path("results/mimiciv/p12_reference_ranges.csv")
    if not path.is_file():
        pytest.skip("requires locally generated MIMIC-IV reference ranges; protected extraction outputs are not distributed")
    reference = _load_reference_ranges(path)

    assert tuple(reference) == TIME_SERIES_VARIABLES
    assert [reference[name]["var_id"] for name in TIME_SERIES_VARIABLES] == list(range(37))
    assert reference["FiO2"]["p50"] == pytest.approx(0.5)
    assert reference["FiO2"]["max"] == pytest.approx(1.0)
    assert reference["MechVent"]["min"] == pytest.approx(1.0)
    assert reference["MechVent"]["max"] == pytest.approx(1.0)


def test_pandas_csv_float_parsing_must_be_round_trip():
    """A bitwise gate is only meaningful if the reader round-trips the writer.

    pandas' default CSV float parser may not exactly round-trip values written
    with ``%.17g``; the explicit reader mode preserves the writer's value.
    """

    import io

    import pandas as pd

    written = "skill\n0.10862866005719737\n0.1129288718387224\n0.1161786553746879\n"
    exact = [float(line) for line in written.strip().split("\n")[1:]]

    default = pd.read_csv(io.StringIO(written))["skill"].tolist()
    assert any(a.hex() != b.hex() for a, b in zip(default, exact)), (
        "pandas now round-trips by default; the explicit float_precision may be droppable"
    )

    exactly = pd.read_csv(io.StringIO(written), float_precision="round_trip")["skill"].tolist()
    assert all(a.hex() == b.hex() for a, b in zip(exactly, exact))
