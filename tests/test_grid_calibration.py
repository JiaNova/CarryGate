import torch
from torch import nn

from src.models.layers.kan_bspline_head import BSplineKANLayer
from src.training.grid_calibration import audit_grid_coverage


def test_grid_range_clips_forward_but_keeps_outside_gradient():
    layer = BSplineKANLayer(1, 2, grid_size=5, spline_order=3, x_range=(-1.0, 1.0))
    layer.reset_coverage()
    x = torch.tensor([[4.0]], requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert layer.coverage == 0.0
    assert torch.isfinite(x.grad).all()
    assert bool((x.grad.abs() > 0).any())


def test_grid_coverage_tracks_in_range_inputs():
    layer = BSplineKANLayer(1, 1, grid_size=5, spline_order=3, x_range=(-1.0, 1.0))
    layer(torch.tensor([[-0.5], [0.5], [2.0]]))
    assert layer.coverage == 2 / 3
    layer.set_grid_range(-2.0, 2.0)
    layer.reset_coverage()
    layer(torch.tensor([[-0.5], [0.5], [2.0]]))
    assert layer.coverage == 1.0


def test_grid_underflow_is_a_warning_with_usable_metrics():
    layer = BSplineKANLayer(1, 1, grid_size=5, spline_order=3, x_range=(-1.0, 1.0))
    nn.Sequential(layer)(torch.tensor([[-0.5], [2.0]]))
    audit = audit_grid_coverage(nn.Sequential(layer), minimum=0.99)
    assert audit["status"] == "grid_underflow"
    assert audit["n_underflow_layers"] == 1
    assert audit["metrics_usable"] is True
    assert audit["policy"] == "warning_only_metrics_retained"
