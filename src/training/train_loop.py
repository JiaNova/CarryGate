"""Training protocol for the P12 H0 head comparison."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..data.p12_dataset import P12SplitBundle, load_p12_splits, make_loaders
from ..data.p12_hourly_agg import cache_features
from ..models.backbones.gru_backbone import GRUBackbone
from ..models.icu_model import ICUHead0Model, build_head, count_parameters
from ..utils.seed_utils import resolve_device, set_seed
from .efficiency import measure_efficiency
from .evaluate import evaluate_loader, predict_arrays
from .param_matching import mlp_target_params


def _cfg(cfg: Mapping[str, Any], name: str, default: Any) -> Any:
    return cfg[name] if name in cfg else default


def _flatten_spec(prefix: str, value: Any, output: dict) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_spec(f"{prefix}_{key}" if prefix else str(key), item, output)
    elif isinstance(value, (tuple, list)):
        output[prefix] = json.dumps(value)
    else:
        output[prefix] = value


def _train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for x, static, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.to(device), static.to(device))
        loss = criterion(logits, y.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_one_run(
    head_type: str,
    seed: int,
    cfg: Mapping[str, Any],
    bundle: P12SplitBundle | None = None,
) -> dict:
    """Train one head/seed pair and return a flat result record."""

    set_seed(seed)
    device = resolve_device(_cfg(cfg, "device", "auto"))
    if bundle is None:
        bundle = load_p12_splits(cfg["processed_path"], cfg["split_path"])
    loaders = make_loaders(
        bundle,
        batch_size=int(_cfg(cfg, "batch_size", 128)),
        seed=seed,
        num_workers=int(_cfg(cfg, "num_workers", 0)),
    )

    input_dim = int(_cfg(cfg, "input_dim", bundle.cache.preprocessor.input_dim))
    static_dim = int(_cfg(cfg, "static_dim", bundle.cache.preprocessor.static_dim))
    hidden_size = int(_cfg(cfg, "hidden_size", 64))
    # The budget compares prediction heads only.  The head receives the
    # fused GRU representation (hidden_size), not the 111-dimensional hourly
    # input; using input_dim here would silently give KAN a larger budget.
    head_target = mlp_target_params(hidden_size, hidden_dim=hidden_size)
    backbone = GRUBackbone(
        input_dim=input_dim,
        hidden_size=hidden_size,
        static_dim=static_dim,
        num_layers=int(_cfg(cfg, "num_layers", 1)),
        dropout=float(_cfg(cfg, "dropout", 0.1)),
    )
    head, head_spec = build_head(
        head_type,
        in_dim=hidden_size,
        target_params=head_target,
        hidden_dim=hidden_size,
        grid_size=int(_cfg(cfg, "grid_size", 5)),
        spline_order=int(_cfg(cfg, "spline_order", 3)),
        num_centers=int(_cfg(cfg, "num_centers", 8)),
        x_range=tuple(_cfg(cfg, "head_x_range", (-5.0, 5.0))),
    )
    model_core = ICUHead0Model(backbone, head)
    model: nn.Module = model_core
    if (
        bool(_cfg(cfg, "multi_gpu", False))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        requested_ids = _cfg(cfg, "gpu_ids", list(range(torch.cuda.device_count())))
        device_ids = [int(item) for item in requested_ids if int(item) < torch.cuda.device_count()]
        if len(device_ids) >= 2:
            model = nn.DataParallel(model_core, device_ids=device_ids)
            print(f"Using DataParallel on GPUs {device_ids}")
    model = model.to(device)

    train_labels = bundle.cache.labels[np.asarray(bundle.split_indices["train"], dtype=np.int64)]
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    if positive == 0:
        raise ValueError("training split has no positive mortality labels")
    pos_weight = torch.tensor(negative / positive, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_cfg(cfg, "learning_rate", 1e-3)),
        weight_decay=float(_cfg(cfg, "weight_decay", 1e-4)),
    )

    output_dir = Path(_cfg(cfg, "output_dir", "results/stage1"))
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{head_type}_{seed}.pt"

    best_val_auprc = -np.inf
    best_epoch = 0
    wait = 0
    best_state = None
    start_time = time.perf_counter()
    max_epochs = int(_cfg(cfg, "epochs", 100))
    patience = int(_cfg(cfg, "patience", 10))
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(model, loaders["train"], optimizer, criterion, device)
        val = evaluate_loader(model, loaders["val"], device)
        val_auprc = float(val["auprc"])
        if np.isfinite(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model_state": best_state, "epoch": epoch, "head_spec": head_spec},
                checkpoint_path,
            )
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{head_type} seed {seed} never produced a finite validation AUPRC")
    model.load_state_dict(best_state)

    clean = evaluate_loader(model, loaders["test"], device)
    test_indices = np.asarray(bundle.split_indices["test"], dtype=np.int64)
    robust_rng = np.random.default_rng(int(_cfg(cfg, "robust_seed", 2024)) + int(seed))
    dropout_probability = float(_cfg(cfg, "robust_dropout", 0.1))
    original_mask = bundle.cache.mask[test_indices]
    dropped = robust_rng.random(original_mask.shape) < dropout_probability
    robust_mask = original_mask & ~dropped
    robust_x, robust_static, robust_y = cache_features(bundle.cache, test_indices, robust_mask)
    robust_probability = predict_arrays(
        model,
        robust_x,
        robust_static,
        device,
        batch_size=int(_cfg(cfg, "batch_size", 128)),
    )
    from ..utils.metrics import binary_metrics

    robust = {
        **binary_metrics(robust_y, robust_probability),
        "probability": robust_probability,
        "y_true": robust_y,
    }

    test_probability = np.asarray(clean["probability"], dtype=np.float32)
    prediction_path = prediction_dir / f"{head_type}_{seed}.npz"
    np.savez_compressed(
        prediction_path,
        record_ids=bundle.cache.record_ids[test_indices],
        y_true=np.asarray(clean["y_true"], dtype=np.int64),
        probability=test_probability,
        robust_probability=robust_probability,
    )

    sample_x, sample_static, _ = next(iter(loaders["test"]))
    efficiency = measure_efficiency(
        model,
        (sample_x, sample_static),
        device,
        n_repeat=int(_cfg(cfg, "latency_repeats", 100)),
    )
    elapsed = time.perf_counter() - start_time
    result = {
        "head_type": head_type,
        "seed": int(seed),
        "status": "ok",
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "train_loss_last": train_loss,
        "val_auprc": best_val_auprc,
        "auprc": float(clean["auprc"]),
        "auroc": float(clean["auroc"]),
        "brier": float(clean["brier"]),
        "robust_auprc": float(robust["auprc"]),
        "robust_auroc": float(robust["auroc"]),
        "robust_brier": float(robust["brier"]),
        "robust_auprc_drop": float(clean["auprc"] - robust["auprc"]),
        "n_params_total": count_parameters(model),
        "n_params_backbone": count_parameters(model_core.backbone),
        "n_params_head": count_parameters(model_core.head),
        "head_target_params": head_target,
        "checkpoint": str(checkpoint_path),
        "prediction_file": str(prediction_path),
        "backend": getattr(head, "backend", "standard"),
        "duration_sec": elapsed,
        **{f"efficiency_{key}": value for key, value in efficiency.items()},
    }
    _flatten_spec("head", head_spec, result)
    return result
