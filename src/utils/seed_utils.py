"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without changing the split seed.

    ``CUBLAS_WORKSPACE_CONFIG`` has to be present before the first CUDA
    context is created.  Keeping the switch here (rather than in individual
    runners) makes the reproducibility contract apply to every experiment.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # CUDA reads this when its first BLAS handle is created.  Setting it
        # after torch.manual_seed() is too late on some PyTorch builds.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(value: str | None = "auto") -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
