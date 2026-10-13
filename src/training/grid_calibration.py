"""Data-driven basis-range calibration and coverage auditing."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping

import torch
from torch import nn

from ..models.layers.fourier_kan_layer import FourierKANLayer
from ..models.layers.kan_bspline_head import BSplineKANLayer
from ..models.layers.rbf_kan_layer import RBFKANLayer
from ..models.gates.gate_bases import RangeAwareEdge


_RANGE_LAYERS = (BSplineKANLayer, RBFKANLayer, FourierKANLayer, RangeAwareEdge)

# This is an audit threshold, not a training-loss threshold.  A run with a
# small amount of out-of-range input still has a valid prediction and must not
# be silently discarded; it is reported as a warning instead.
GRID_COVERAGE_MINIMUM = 0.99
GRID_UNDERFLOW_STATUS = "grid_underflow"
GRID_UNDERFLOW_POLICY = "warning_only_metrics_retained"


def _run_batch(model: nn.Module, batch, forward_fn: Callable | None):
    if forward_fn is not None:
        return forward_fn(model, batch)
    if isinstance(batch, Mapping):
        # Phase 1's model accepts the complete batch mapping.  Keeping this
        # fallback also makes the utility useful with a simple ``model(x)``
        # test fixture.
        try:
            return model(batch)
        except (TypeError, AttributeError):
            return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(item, device) for item in value)
    return value


@torch.no_grad()
def calibrate_grids(
    model: nn.Module,
    loader,
    device: torch.device | str = "cpu",
    n_batches: int = 20,
    lo: float = 0.001,
    hi: float = 0.999,
    pad_fraction: float = 0.10,
    forward_fn: Callable | None = None,
) -> dict[str, dict[str, float | int]]:
    """Set every range-aware KAN layer's range from training batches.

    Hooks observe the actual input to each layer, including the joint KAN's
    two-dimensional input.  The returned report is serializable and records
    both quantiles and the padded range used by the model.
    """

    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("lo and hi must satisfy 0 <= lo < hi <= 1")
    if n_batches < 1:
        raise ValueError("n_batches must be positive")
    if pad_fraction < 0.0:
        raise ValueError("pad_fraction must be non-negative")
    device = torch.device(device)
    buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []
    layer_names = {
        id(layer): name
        for name, layer in model.named_modules()
        if isinstance(layer, _RANGE_LAYERS)
    }

    for name, layer in model.named_modules():
        if not isinstance(layer, _RANGE_LAYERS):
            continue

        def capture(module, inputs, key=name):
            if isinstance(module, RangeAwareEdge):
                # A suspended edge is being evaluated on a synthetic grid by
                # the monotone wrapper.  Calibrating on those points would
                # mix a uniform grid into the data quantiles.
                if not module.coverage_enabled:
                    return
                candidate = inputs[1] if len(inputs) > 1 else None
            else:
                candidate = inputs[0] if inputs else None
            if torch.is_tensor(candidate):
                values = candidate.detach().reshape(-1, module.in_features).cpu()
                buffers[key].append(values)

        handles.append(layer.register_forward_pre_hook(capture))

    was_training = model.training
    model.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= int(n_batches):
            break
        _run_batch(model, _to_device(batch, device), forward_fn)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    report: dict[str, dict[str, float | int]] = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, _RANGE_LAYERS):
            continue
        if not buffers.get(name):
            # A very sparse variable/group may not occur in the first 20
            # batches.  Keep its explicit configured range and record the
            # absence; later coverage auditing still sees any real inputs.
            report[name] = {
                "quantile_low": float(layer.x_range[0]),
                "quantile_high": float(layer.x_range[1]),
                "range_min": float(layer.x_range[0]),
                "range_max": float(layer.x_range[1]),
                "n_values": 0,
            }
            layer.reset_coverage()
            continue
        values = torch.cat(buffers[name], dim=0)
        for feature in range(layer.in_features):
            feature_values = values[:, feature]
            finite = feature_values[torch.isfinite(feature_values)]
            if finite.numel() == 0:
                raise ValueError(f"no finite calibration inputs for {name}[{feature}]")
            q_low = float(torch.quantile(finite, lo).item())
            q_high = float(torch.quantile(finite, hi).item())
            span = max(q_high - q_low, 1e-3)
            # Each range-aware layer has one range shared by all input edges.
            # Use the extrema across input features so no feature is silently
            # squeezed out of the calibrated interval.
            if feature == 0:
                low = q_low
                high = q_high
            else:
                low = min(low, q_low)
                high = max(high, q_high)
        span = max(high - low, 1e-3)
        pad = pad_fraction * span
        layer.set_grid_range(low - pad, high + pad)
        layer.reset_coverage()
        report[name] = {
            "quantile_low": float(low),
            "quantile_high": float(high),
            "range_min": float(low - pad),
            "range_max": float(high + pad),
            "n_values": int(values.numel()),
        }
    return report


def coverage_report(model: nn.Module) -> dict[str, dict[str, float | int]]:
    """Collect the observed coverage of all range-aware basis layers."""

    return {
        name: layer.coverage_stats()
        for name, layer in model.named_modules()
        if isinstance(layer, _RANGE_LAYERS)
    }


def audit_grid_coverage(
    model: nn.Module,
    minimum: float = GRID_COVERAGE_MINIMUM,
) -> dict[str, object]:
    """Summarize range coverage without turning a warning into a failure.

    ``grid_underflow`` means at least one range-aware layer observed less than
    ``minimum`` in-range inputs.  The forward pass uses the layer's clipped
    value together with its straight-through gradient, so the run remains
    numerically valid and its metrics remain usable.  The detailed layer
    report is retained for the generated Phase 2 audit.
    """

    minimum = float(minimum)
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum must be in (0, 1]")
    layers = coverage_report(model)
    underflow_layers = sorted(
        name for name, data in layers.items() if float(data["coverage"]) < minimum
    )
    min_coverage = (
        min(float(data["coverage"]) for data in layers.values())
        if layers
        else None
    )
    return {
        "status": GRID_UNDERFLOW_STATUS if underflow_layers else "ok",
        "minimum": minimum,
        "n_layers": len(layers),
        "n_underflow_layers": len(underflow_layers),
        "min_coverage": min_coverage,
        "underflow_layers": underflow_layers,
        "layers": layers,
        "metrics_usable": True,
        "policy": GRID_UNDERFLOW_POLICY,
    }


def assert_grid_coverage(
    model: nn.Module,
    minimum: float = GRID_COVERAGE_MINIMUM,
) -> dict[str, dict[str, float | int]]:
    """Strict compatibility helper for callers that require full coverage."""

    audit = audit_grid_coverage(model, minimum=minimum)
    if audit["status"] == GRID_UNDERFLOW_STATUS:
        underflow = {
            name: audit["layers"][name] for name in audit["underflow_layers"]
        }
        raise RuntimeError(f"grid_underflow: {underflow}")
    return audit["layers"]
