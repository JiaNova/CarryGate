import numpy as np
import torch

from src.data.p12_events import EventStandardizer, make_event_example, split_events
from src.data.p12_forecast_dataset import P12ForecastDataset, forecast_collate


def _record():
    return {
        "record_id": 1,
        "sample_id": "a-0001",
        "label": 0,
        "events": [
            ("HR", 1.0, 60.0),
            ("HR", 4.0, 80.0),
            ("Urine", 23.5, 1.0),
            ("HR", 24.0, 90.0),
            ("Urine", 30.0, 2.0),
        ],
    }


def test_delta_and_window_split_are_strictly_causal():
    record = _record()
    obs, target = split_events(record, t_obs=24.0)
    assert [item["delta_t"] for item in obs if item["variable"] == "HR"] == [1.0, 3.0]
    assert obs[0]["is_first_observation"] is True
    assert target[0]["time"] == 24.0
    assert all(float(item["time"]) < 24.0 for item in obs)
    assert not any(float(item["time"]) >= 24.0 for item in obs)


def test_standardizer_is_train_only_and_clipping_is_recorded():
    record = _record()
    standardizer = EventStandardizer.fit([record])
    example = make_event_example(record, standardizer, t_obs=24.0)
    assert example["obs"]["value"].dtype == np.float32
    assert standardizer.clipping_ratio.shape == (37,)
    assert np.all(example["obs"]["value"] <= standardizer.upper_clip[example["obs"]["var"]])


def test_collate_keeps_future_events_out_of_encoder_tensor():
    record = _record()
    standardizer = EventStandardizer.fit([record])
    example = make_event_example(record, standardizer, t_obs=24.0)
    item = P12ForecastDataset([example], [0])[0]
    batch = forecast_collate([item])
    assert torch.all(batch["obs_time"][batch["obs_mask"]] < batch["t_obs"][0])
    assert torch.all(batch["target_time"][batch["target_mask"]] >= batch["t_obs"][0])


def test_event_example_preserves_icu_type_for_domain_split():
    record = _record()
    record["static"] = {"ICUType": 4}
    standardizer = EventStandardizer.fit([record])
    example = make_event_example(record, standardizer, t_obs=24.0)
    assert example["icu_type"] == 4


def test_dataset_can_leave_out_icu_types():
    example_a = make_event_example({**_record(), "static": {"ICUType": 1}}, EventStandardizer.fit([_record()]), t_obs=24.0)
    example_b = make_event_example({**_record(), "record_id": 2, "static": {"ICUType": 2}}, EventStandardizer.fit([_record()]), t_obs=24.0)
    dataset = P12ForecastDataset([example_a, example_b], [0, 1], held_out_icu_types=[2])
    assert len(dataset) == 1
    assert dataset[0]["icu_type"] == 1
