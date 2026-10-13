import torch

from src.models.layers.fourier_kan_layer import FourierKANLayer


def test_fourier_parameter_count_matches_truncated_expansion():
    layer = FourierKANLayer(2, 3, n_freq=6)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 2 * 3 * (2 * 6 + 1)


def test_fourier_out_of_range_input_keeps_gradient_path():
    layer = FourierKANLayer(1, 1, n_freq=6, x_range=(-1.0, 1.0))
    with torch.no_grad():
        layer.base_weight.fill_(1.0)
        layer.coef.zero_()
    x = torch.tensor([[4.0]], requires_grad=True)
    layer(x).sum().backward()
    assert layer.coverage == 0.0
    assert torch.isfinite(x.grad).all()
    assert bool((x.grad.abs() > 0).any())
