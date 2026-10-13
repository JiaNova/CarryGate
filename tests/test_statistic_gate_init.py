import torch

from src.models.gates.statistic_conditioned_gate import StatisticConditionedGate
from src.models.heads.persistence_residual_head import PersistenceResidualHead


def test_statistic_gate_initialises_near_persistence():
    descriptors = torch.zeros(3, 4)
    gate = StatisticConditionedGate(descriptors, hidden=8, use_staleness=True)
    var_id = torch.tensor([[0, 1], [2, 1]])
    log_delta = torch.ones(2, 2)
    omega = gate(var_id, log_delta)
    assert torch.allclose(omega, torch.full_like(omega, torch.sigmoid(torch.tensor(4.0))))


def test_zero_residual_statistic_gate_returns_its_mixed_baseline():
    torch.manual_seed(0)
    head = PersistenceResidualHead(
        d_model=8,
        n_vars=3,
        gate_type="statistic_conditioned",
        descriptors=torch.zeros(3, 4),
    )
    head.eval()
    hidden = torch.randn(1, 2, 8)
    obs_mask = torch.tensor([[True, True]])
    target_var = torch.tensor([[0, 2]])
    target_time = torch.tensor([[24.0, 25.0]])
    x_last = torch.tensor([[1.0, 2.0]])
    prior = torch.tensor([[0.0, 0.5]])
    staleness = torch.ones_like(x_last)
    with torch.no_grad():
        output = head(hidden, obs_mask, target_var, target_time, x_last, prior, staleness)
    omega = torch.sigmoid(torch.tensor(4.0))
    expected = omega * x_last + (1.0 - omega) * prior
    assert torch.allclose(output, expected)
