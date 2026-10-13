import numpy as np

from src.data.p12_hourly_agg import (
    N_HOURS,
    aggregate_record,
    build_hourly_features,
    build_static_features,
    compute_delta,
    fit_preprocessor,
    stratified_split_indices,
)
from src.data.p12_parse import TIME_SERIES_VARIABLES


def _record(events, static=None):
    return {
        "record_id": 1,
        "static": static or {"Age": 50, "Gender": 1, "Height": 170, "ICUType": 3, "Weight": 70},
        "events": events,
    }


def test_hourly_mean_mask_and_delta():
    record = _record(
        [
            ("HR", 0.1, 60.0),
            ("HR", 0.8, 80.0),
            ("HR", 2.0, 100.0),
            ("Urine", 1.2, 0.0),
        ]
    )
    observed, mask, static = aggregate_record(record)
    hr = TIME_SERIES_VARIABLES.index("HR")
    urine = TIME_SERIES_VARIABLES.index("Urine")
    assert observed[0, hr] == 70.0
    assert mask[0, hr]
    assert observed[1, urine] == 0.0
    assert mask[1, urine]
    assert compute_delta(mask)[0, hr] == 0
    assert compute_delta(mask)[1, hr] == 1
    assert compute_delta(mask)[2, hr] == 0
    assert static.shape == (5,)


def test_features_use_forward_fill_and_have_expected_dimensions():
    observed = np.zeros((2, N_HOURS, len(TIME_SERIES_VARIABLES)), dtype=np.float32)
    mask = np.zeros_like(observed, dtype=bool)
    hr = TIME_SERIES_VARIABLES.index("HR")
    observed[0, 0, hr] = 60
    observed[0, 2, hr] = 90
    mask[0, 0, hr] = True
    mask[0, 2, hr] = True
    mean = np.zeros(len(TIME_SERIES_VARIABLES), dtype=np.float32)
    std = np.ones(len(TIME_SERIES_VARIABLES), dtype=np.float32)
    features = build_hourly_features(observed, mask, mean, std)
    assert features.shape == (2, N_HOURS, 111)
    assert np.isclose(features[0, 1, hr], 60)
    assert features[0, 1, len(TIME_SERIES_VARIABLES) + hr] == 0
    assert features[0, 1, 2 * len(TIME_SERIES_VARIABLES) + hr] > 0


def test_static_encoding_and_train_only_stats():
    observed = np.zeros((2, N_HOURS, len(TIME_SERIES_VARIABLES)), dtype=np.float32)
    mask = np.zeros_like(observed, dtype=bool)
    observed[:, 0, 0] = [1, 3]
    mask[:, 0, 0] = True
    static = np.asarray([[50, 1, 170, 3, 70], [60, 0, 180, 4, 80]], dtype=np.float32)
    prep = fit_preprocessor(observed, mask, static)
    encoded = build_static_features(static, prep)
    assert encoded.shape == (2, 8)
    assert np.isclose(encoded[0, 4:8].sum(), 1.0)
    assert prep.input_dim == 111
    split = stratified_split_indices([0] * 10 + [1] * 10, seed=42)
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["val"]).isdisjoint(split["test"])
