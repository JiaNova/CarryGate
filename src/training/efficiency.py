"""Parameter, FLOP, latency, and memory measurements."""

from __future__ import annotations

import time
from typing import Dict

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _analytic_flops(model: nn.Module, x: torch.Tensor, static: torch.Tensor) -> int:
    """Conservative multiply-add estimate when fvcore cannot trace a GRU/KAN."""

    model = model.module if isinstance(model, nn.DataParallel) else model
    batch, steps, input_dim = x.shape
    hidden = getattr(getattr(model, "backbone", None), "hidden_size", 0)
    layers = getattr(getattr(model, "backbone", None), "num_layers", 1)
    if hidden:
        gru = batch * steps * layers * 2 * 3 * (input_dim * hidden + hidden * hidden + hidden)
        static_flops = batch * 2 * (static.shape[-1] * hidden + hidden)
    else:
        gru = 0
        static_flops = 0
    head = batch * 2 * sum(
        parameter.numel() for name, parameter in model.head.named_parameters() if parameter.ndim >= 1
    )
    return int(gru + static_flops + head)


def estimate_flops(model: nn.Module, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> int:
    x, static = sample_batch
    model_for_trace = model.module if isinstance(model, nn.DataParallel) else model
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model_for_trace, (x, static))
        total = int(analysis.total())
        if total > 0:
            return total
    except Exception:
        pass
    return _analytic_flops(model, x, static)


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    n_repeat: int = 100,
    warmup: int = 10,
) -> float:
    """Return median forward latency in milliseconds for one batch."""

    x, static = (tensor.to(device) for tensor in sample_batch)
    model.eval()
    for _ in range(max(0, warmup)):
        model(x, static)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(max(1, n_repeat)):
        start = time.perf_counter()
        model(x, static)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return float(timings[len(timings) // 2])


def measure_peak_vram(model: nn.Module, sample_batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(sample_batch[0].to(device), sample_batch[1].to(device))
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def measure_efficiency(
    model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    n_repeat: int = 100,
) -> Dict[str, float | int]:
    return {
        "n_params": count_parameters(model),
        "flops": estimate_flops(model, sample_batch),
        "latency_ms": benchmark_latency(model, sample_batch, device, n_repeat=n_repeat),
        "peak_vram_mb": measure_peak_vram(model, sample_batch, device),
    }
