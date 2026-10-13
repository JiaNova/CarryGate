import torch

from src.models.backbones.strats_lite import STraTSForecastModel
from src.training.forecast_train import _classification_logits


def _batch(padded: bool):
    return {
        "obs_var": torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]]),
        "obs_value": torch.randn(2, 4),
        "obs_delta_t": torch.ones(2, 4),
        "obs_mask": torch.tensor([[True, True, True, False], [True, True, False, False]]),
        "obs_pad_mask": torch.tensor([[False, False, False, True], [False, False, True, True]]),
        "target_var": torch.tensor([[0, 1], [1, 2]]),
        "target_time": torch.tensor([[24.0, 25.0], [24.0, 25.0]]),
        "target_mask": torch.ones(2, 2, dtype=torch.bool),
    }


def test_forecast_shapes_and_padding_are_finite():
    model = STraTSForecastModel(n_vars=3, group_of_var=[0, 0, 0], encoder_type="mlp")
    output = model(_batch(True))
    assert output["forecast"].shape == (2, 2)
    assert output["mortality"].shape == (2,)
    assert torch.isfinite(output["forecast"]).all()


def test_padding_does_not_change_unpadded_sample():
    torch.manual_seed(4)
    model = STraTSForecastModel(n_vars=3, group_of_var=[0, 0, 0], encoder_type="linear")
    model.eval()
    batch = _batch(True)
    with torch.no_grad():
        full = model(batch)["forecast"][0]
        single = {key: value[:1] for key, value in batch.items()}
        single_out = model(single)["forecast"][0]
    assert torch.allclose(full, single_out, atol=1e-5, rtol=1e-5)


def test_classification_branch_accepts_observation_window_without_targets():
    model = STraTSForecastModel(n_vars=3, group_of_var=[0, 0, 0], encoder_type="mlp")
    batch = _batch(True)
    observation_only = {
        key: batch[key]
        for key in ("obs_var", "obs_value", "obs_delta_t", "obs_mask", "obs_pad_mask")
    }
    logits = _classification_logits(model, observation_only)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
