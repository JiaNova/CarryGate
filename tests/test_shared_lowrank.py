import torch

from src.models.encoders.shared_lowrank import SharedPlusLowRank


def test_lowrank_zero_gate_starts_at_shared_solution():
    torch.manual_seed(7)
    shared = SharedPlusLowRank(n_vars=4, d_in=1, d_out=5, rank=0, hidden=3)
    adapted = SharedPlusLowRank(n_vars=4, d_in=1, d_out=5, rank=2, hidden=3)
    adapted.shared.load_state_dict(shared.shared.state_dict())
    assert torch.equal(adapted.alpha, torch.zeros_like(adapted.alpha))
    var_id = torch.tensor([[0, 1, 3], [2, 1, 0]])
    x = torch.randn(2, 3, 1)
    assert torch.equal(shared(var_id, x), adapted(var_id, x))
