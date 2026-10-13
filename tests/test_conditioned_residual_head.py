import torch

from src.models.heads.conditioned_residual_head import ConditionedResidualHead


def _inputs():
    torch.manual_seed(3)
    return (
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([[0, 1], [2, 1]]),
        torch.tensor([[24.0, 25.0], [24.0, 25.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.zeros(2, 2),
        torch.ones(2, 2),
    )


def test_zero_initialized_conditioned_head_keeps_the_baseline():
    head = ConditionedResidualHead(
        d_model=8,
        n_vars=3,
        gate_type="none",
        descriptors=torch.zeros(3, 4),
    ).eval()
    args = _inputs()
    with torch.no_grad():
        prediction = head(*args)
    assert torch.equal(prediction, args[4])


def test_scale_one_matches_parent_output():
    descriptors = torch.zeros(3, 4)
    conditioned = ConditionedResidualHead(
        d_model=8,
        n_vars=3,
        gate_type="none",
        descriptors=descriptors,
    ).eval()
    args = _inputs()
    with torch.no_grad():
        parent_output = super(type(conditioned), conditioned).forward(*args)
        conditioned.residual_gate[-1].bias.fill_(30.0)
        conditioned_output = conditioned(*args)
    assert torch.allclose(conditioned_output, parent_output, atol=1e-5, rtol=1e-5)
