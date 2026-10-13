import torch

from src.models.heads.persistence_residual_head import PersistenceResidualHead


def test_zero_initialized_residual_head_is_persistence():
    torch.manual_seed(0)
    head = PersistenceResidualHead(d_model=8, n_vars=3, gate_type="none")
    head.eval()
    hidden = torch.randn(2, 4, 8)
    obs_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    target_var = torch.tensor([[0, 1, 2], [2, 1, 0]])
    target_time = torch.tensor([[24.0, 25.0, 26.0], [24.0, 25.0, 26.0]])
    x_last = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    prior = torch.zeros_like(x_last)
    staleness = torch.ones_like(x_last)

    prediction = head(
        hidden,
        obs_mask,
        target_var,
        target_time,
        x_last,
        prior,
        staleness,
    )

    assert torch.equal(prediction, x_last)
