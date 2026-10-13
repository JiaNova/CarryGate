import torch

from src.models.layers.grouped_kan import GroupedLinear
from src.utils.seed_utils import set_seed


def test_strict_mode_uses_dense_path_for_large_group_count():
    set_seed(9, deterministic=True)
    layer = GroupedLinear(9, 2, 3)
    group = torch.tensor([[0, 8, 3, 1]])
    x = torch.randn(1, 4, 2)
    output = layer(group, x)
    output.sum().backward()
    assert layer.determinism == "strict"
    assert torch.isfinite(output).all()
