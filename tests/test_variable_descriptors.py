import torch

from src.data.p12_forecast_dataset import P12ForecastDataset, forecast_collate
from src.data.variable_descriptors import VariableDescriptors


def _batch(target_value: float):
    return {
        "record_id": torch.tensor([1]),
        "t_obs": torch.tensor([2.0]),
        "obs_var": torch.tensor([[0, 0]]),
        "obs_time": torch.tensor([[0.0, 1.0]]),
        "obs_value": torch.tensor([[1.0, 2.0]]),
        "obs_delta_t": torch.tensor([[0.0, 1.0]]),
        "obs_mask": torch.tensor([[True, True]]),
        "target_var": torch.tensor([[0]]),
        "target_time": torch.tensor([[3.0]]),
        "target_value": torch.tensor([[target_value]]),
        "target_mask": torch.tensor([[True]]),
    }


def test_descriptors_are_fit_from_the_loader_passed_by_the_caller():
    prior = torch.zeros(1)
    train = VariableDescriptors.fit([_batch(2.5)], 1, prior)
    repeated = VariableDescriptors.fit([_batch(2.5)], 1, prior)
    validation = VariableDescriptors.fit([_batch(99.0)], 1, prior)
    assert torch.equal(train.values, repeated.values)
    assert not torch.equal(train.raw, validation.raw)
    assert train.source == "train"
    assert tuple(train.FIELDS) == (
        "persist_ratio",
        "log_n_obs",
        "volatility",
        "sampling_irregularity",
    )


def test_held_out_variables_are_excluded_only_from_training_loss_mask():
    example = {
        "record_id": 1,
        "sample_id": "synthetic-1",
        "label": 0,
        "t_obs": 2.0,
        "obs": {
            "var": [0],
            "time": [1.0],
            "value": [1.0],
            "delta_t": [1.0],
            "is_first": [True],
        },
        "target": {
            "var": [0, 1],
            "time": [3.0, 4.0],
            "value": [1.0, 2.0],
            "delta_t": [2.0, 3.0],
            "is_first": [False, False],
        },
    }
    batch = forecast_collate(
        [P12ForecastDataset([example], [0], train_median=torch.zeros(2), held_out_variables=[1])[0]]
    )
    assert torch.equal(batch["target_mask"], torch.tensor([[True, True]]))
    assert torch.equal(batch["target_loss_mask"], torch.tensor([[True, False]]))
