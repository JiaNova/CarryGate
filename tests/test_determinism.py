import torch

from src.models.layers.grouped_kan import GroupedMLP
from src.utils.seed_utils import set_seed


def _run_once():
    set_seed(17, deterministic=True)
    model = GroupedMLP(2, 1, 3, hidden_features=4)
    group = torch.tensor([[0, 1, 0], [1, 0, 1]])
    x = torch.randn(2, 3, 1)
    loss = model(group, x).square().sum()
    loss.backward()
    return [value.detach().clone() for value in model.parameters()], [
        value.grad.detach().clone() for value in model.parameters()
    ]


def test_same_seed_forward_backward_is_bitwise_equal():
    first = _run_once()
    second = _run_once()
    for left, right in zip(first[0], second[0]):
        assert torch.equal(left, right)
    for left, right in zip(first[1], second[1]):
        assert torch.equal(left, right)


def test_grouped_deterministic_dispatch_matches_reference():
    set_seed(3, deterministic=False)
    model = GroupedMLP(3, 1, 2, hidden_features=3)
    group = torch.tensor([[0, 2, 1, 2]])
    x = torch.randn(1, 4, 1)
    flat_group, flat_x = model._check(group, x)
    reference = model._apply_grouped_reference(flat_group, flat_x, model.layers)
    deterministic = model._apply_grouped_deterministic(flat_group, flat_x, model.layers)
    assert torch.allclose(reference, deterministic, atol=1e-7, rtol=1e-7)

    model.zero_grad(set_to_none=True)
    reference.sum().backward()
    reference_grads = [value.grad.detach().clone() for value in model.parameters()]
    model.zero_grad(set_to_none=True)
    deterministic.sum().backward()
    deterministic_grads = [value.grad.detach().clone() for value in model.parameters()]
    for left, right in zip(reference_grads, deterministic_grads):
        assert torch.allclose(left, right, atol=1e-7, rtol=1e-7)
