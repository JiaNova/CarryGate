import torch

from src.models.baselines.naive import naive_1, naive_2, naive_3


def test_naive_baseline_strategies_follow_their_definitions():
    batch = {
        "obs_var": torch.tensor([[0, 0, 1]]),
        "obs_value": torch.tensor([[1.0, 3.0, 5.0]]),
        "obs_mask": torch.tensor([[True, True, True]]),
        "obs_pad_mask": torch.tensor([[False, False, False]]),
        "target_var": torch.tensor([[0, 1, 2]]),
    }
    # Variable 0 has last value 3, variable 1 has last value 5, and variable
    # 2 has no observation so all strategies fall back to the train mean.
    assert torch.allclose(naive_1(batch, [10.0, 20.0, 30.0]), torch.tensor([[3.0, 5.0, 30.0]]))
    assert torch.allclose(naive_2(batch, [10.0, 20.0, 30.0]), torch.tensor([[2.0, 5.0, 30.0]]))
    assert torch.allclose(naive_3(batch, [10.0, 20.0, 30.0]), torch.tensor([[10.0, 20.0, 30.0]]))
