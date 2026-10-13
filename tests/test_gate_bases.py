import torch

from src.models.gates.gate_bases import build_gate_basis
from src.training.param_matching import solve_gate_basis_match


def test_phase5_basis_counts_are_exact_and_extrapolation_has_gradient():
    for basis in ("rbf", "bspline", "fourier", "rational"):
        match = solve_gate_basis_match(basis, n_groups=3, target_params_per_edge=25)
        edge = build_gate_basis(
            basis,
            n_groups=3,
            x_range=(0.0, 1.0),
            **match["parameters"],
        )
        assert sum(parameter.numel() for parameter in edge.parameters()) == 75
        x = torch.tensor([[2.0]], requires_grad=True)
        group = torch.zeros_like(x, dtype=torch.long)
        value = edge(group, x).sum()
        gradient = torch.autograd.grad(value, x)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().item() > 0.0
