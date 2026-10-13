import torch

from src.models.gates.recency_gates import build_recency_gate


def test_monotone_gate_is_non_increasing_for_random_inputs():
    torch.manual_seed(11)
    for basis in ("rbf", "bspline", "fourier", "rational"):
        gate = build_recency_gate(
            "kan_monotone",
            n_vars=4,
            scope="per_variable",
            basis=basis,
            num_centers=24,
            x_range=(0.0, 4.0),
            grid_size=20,
            n_freq=12,
            num_num=12,
            num_den=12,
        )
        delta = torch.linspace(0.0, 10.0, 101).repeat(4, 1)
        variable = torch.arange(4).reshape(-1, 1).expand_as(delta)
        omega = gate(variable, delta)
        assert torch.all(omega[:, 1:] <= omega[:, :-1] + 1e-6)



def test_monotone_gate_builds_one_quadrature_table_per_group():
    """The integral depends only on the group, so it must not scale with events.

    Per-event integration costs ``numel(x) * n_grid`` inner calls, and the
    strict grouped dispatch multiplies that by ``n_groups`` again, which
    exhausts device memory on a realistic batch.
    """

    from src.models.gates.gate_bases import build_gate_basis
    from src.models.gates.monotone_gate import MonotoneDecreasingGate

    class CountingEdge(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.n_groups = inner.n_groups
            self.rows = 0

        def forward(self, group, x):
            self.rows += int(x.reshape(-1, x.shape[-1]).shape[0])
            return self.inner(group, x)

    counter = CountingEdge(build_gate_basis("rbf", 8, num_centers=24, x_range=(0.0, 4.0)))
    gate = MonotoneDecreasingGate(counter, x_range=(0.0, 4.0))
    delta = torch.rand(16, 40) * 4.0
    variable = torch.randint(0, 8, (16, 40))
    gate(variable, delta)

    quadrature = gate.n_groups * (gate.n_grid + 1)
    assert counter.rows == quadrature + delta.numel()
    assert counter.rows < delta.numel() * gate.n_grid


def test_monotone_gate_does_not_count_its_integration_grid_as_coverage():
    from src.models.gates.gate_bases import build_gate_basis
    from src.models.gates.monotone_gate import MonotoneDecreasingGate

    edge = build_gate_basis("rbf", 1, num_centers=24, x_range=(0.0, 2.0))
    gate = MonotoneDecreasingGate(edge, x_range=(0.0, 2.0))
    edge.reset_coverage()
    delta = torch.tensor([[0.5, 1.0, 3.0, 4.0]])
    gate(torch.zeros_like(delta, dtype=torch.long), delta)

    stats = edge.coverage_stats()
    assert stats["n_total"] == 4
    assert stats["n_in_range"] == 2
    assert stats["coverage"] == 0.5
