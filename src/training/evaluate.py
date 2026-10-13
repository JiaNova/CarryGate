"""Model prediction and metric evaluation helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from ..utils.metrics import binary_metrics


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probabilities = []
    for x, static, y in loader:
        logits = model(x.to(device), static.to(device))
        labels.append(y.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(labels).astype(np.float32), np.concatenate(probabilities).astype(np.float32)


def evaluate_loader(model: nn.Module, loader: Iterable, device: torch.device) -> dict:
    y_true, probability = predict_loader(model, loader, device)
    return {**binary_metrics(y_true, probability), "y_true": y_true, "probability": probability}


@torch.no_grad()
def predict_arrays(
    model: nn.Module,
    x: np.ndarray,
    static: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    probabilities = []
    for start in range(0, len(x), batch_size):
        x_batch = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32))
        static_batch = torch.from_numpy(
            np.ascontiguousarray(static[start : start + batch_size], dtype=np.float32)
        )
        logits = model(x_batch.to(device), static_batch.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)
