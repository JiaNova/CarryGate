import torch

from src.data.persistence_baseline import build_persistence_baseline


def test_persistence_baseline_uses_only_events_before_t_obs():
    obs_var = torch.tensor([0, 0, 0, 1])
    obs_time = torch.tensor([1.0, 5.0, 25.0, 4.0])
    obs_value = torch.tensor([10.0, 20.0, 999.0, 7.0])
    tgt_var = torch.tensor([0, 1, 2])
    tgt_time = torch.tensor([30.0, 30.0, 30.0])

    x_last, staleness = build_persistence_baseline(
        obs_var,
        obs_time,
        obs_value,
        tgt_var,
        tgt_time,
        t_obs=24.0,
        train_median=torch.tensor([0.0, 0.0, -3.0]),
    )

    assert torch.equal(x_last, torch.tensor([20.0, 7.0, -3.0]))
    assert torch.equal(staleness, torch.tensor([25.0, 26.0, 30.0]))
