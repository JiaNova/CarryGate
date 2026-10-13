import torch

from src.models.layers.kan_bspline_head import BSplineKANLayer, BSplineKANHead
from src.models.layers.rbf_kan_head import RBFKANLayer, RBFKANHead


def test_bspline_layer_shape_and_gradient():
    layer = BSplineKANLayer(3, 5, grid_size=5, spline_order=3)
    x = torch.randn(7, 3, requires_grad=True)
    y = layer(x)
    assert y.shape == (7, 5)
    assert layer.b_splines(x).shape == (7, 3, 8)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_kan_heads_support_batched_inputs():
    x = torch.randn(4, 6)
    for head in (
        BSplineKANHead(6, 4, grid_size=5, spline_order=3),
        RBFKANHead(6, 4, num_centers=8),
    ):
        assert head(x).shape == (4, 1)


def test_rbf_layer_shape_and_gradient():
    layer = RBFKANLayer(2, 3, num_centers=8)
    x = torch.randn(5, 2, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 3)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()
