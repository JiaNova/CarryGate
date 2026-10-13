"""Minimal training/evaluation loop for Phase 1 forecasting experiments."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from ..data.p12_forecast_dataset import ForecastSplitBundle, make_forecast_loaders
from ..data.persistence_baseline import build_persistence_batch
from ..models.backbones.strats_lite import STraTSForecastModel
from ..training.forecast_loss import (
    masked_l1_loss,
    masked_mse_loss,
    masked_variable_balanced_l1_loss,
)
from ..utils.forecast_metrics import normalized_forecast_metrics
from ..utils.provenance import provenance_fingerprint
from ..utils.skill_metrics import persistence_skill, stratified_report
from ..utils.seed_utils import resolve_device, set_seed
from .grid_calibration import audit_grid_coverage, calibrate_grids
from .param_matching import solve_matched_pair


def move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def determinism_status(module: nn.Module, requested: bool) -> str:
    """Return the strongest determinism claim supported by dispatched layers."""

    modes = [
        str(getattr(child, "determinism"))
        for child in module.modules()
        if hasattr(child, "determinism")
    ]
    if "best_effort" in modes:
        return "best_effort"
    return "strict" if requested else "off"


def _standardized_train_median(n_variables: int, device: torch.device) -> torch.Tensor:
    """The robust transform maps each train-variable median to zero."""

    return torch.zeros(int(n_variables), dtype=torch.float32, device=device)


def _prepare_persistence_fields(
    batch: Mapping[str, object],
    train_median: torch.Tensor,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    prepared = dict(batch)
    target_var = prepared["target_var"].to(dtype=torch.long)
    if "x_last" in prepared and "staleness" in prepared:
        x_last = prepared["x_last"].to(dtype=torch.float32)
        staleness = prepared["staleness"].to(dtype=torch.float32)
    else:
        x_last, staleness = build_persistence_batch(prepared, train_median)
        prepared["x_last"] = x_last
        prepared["staleness"] = staleness
    prepared["prior"] = train_median[target_var]
    return prepared, x_last, staleness


@torch.no_grad()
def measure_forecast_efficiency(model: nn.Module, batch, device: torch.device, repeats: int = 10) -> dict[str, float | int]:
    """Measure the same efficiency fields for the dict-shaped event model."""

    model.eval()
    sample = move_batch(batch, device)
    if getattr(model, "head_type", "direct") == "residual":
        median = _standardized_train_median(
            int(model.backbone.event_encoder.n_vars), device
        )
        sample, _, _ = _prepare_persistence_fields(sample, median)
    for _ in range(2):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak = 0.0
    if device.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    # A conservative auditable proxy: two scalar operations per parameter per
    # batch.  Exact Transformer/KAN FLOP tracing with variable-length dict
    # inputs is not stable across PyTorch versions, so the report labels this
    # explicitly instead of presenting a false precision.
    return {
        "efficiency_n_params": count_parameters(model),
        "efficiency_flops_proxy": int(2 * count_parameters(model)),
        "efficiency_latency_ms": float(np.median(timings)),
        "efficiency_peak_vram_mb": peak,
    }


def build_forecast_model(
    bundle: ForecastSplitBundle,
    *,
    encoder_type: str,
    cfg: Mapping[str, object],
) -> STraTSForecastModel:
    n_vars = bundle.standardizer.n_variables
    from ..models.encoders.functional_event_encoder import build_group_mapping

    group, _ = build_group_mapping(
        bundle.standardizer.variable_names,
        str(cfg.get("grouping", "physiological")),
    )
    return STraTSForecastModel(
        n_vars=n_vars,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        d_ff=int(cfg.get("d_ff", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        encoder_type=encoder_type,
        group_of_var=group,
        grid_size=int(cfg.get("grid_size", 8)),
        spline_order=int(cfg.get("spline_order", 3)),
        spline_x_range=tuple(cfg.get("spline_x_range", (-3.0, 3.0))),
        mlp_hidden=int(cfg.get("mlp_hidden", 16)),
        decay_type=str(cfg.get("decay_type", "additive")),
        basis=str(cfg.get("basis", "bspline")),
        num_centers=int(cfg.get("num_centers", 8)),
        n_freq=int(cfg.get("n_freq", 6)),
        lowrank_rank=int(cfg.get("lowrank_rank", 0)),
        lowrank_hidden=(
            None
            if cfg.get("lowrank_hidden") is None
            else int(cfg.get("lowrank_hidden"))
        ),
        head_type=str(cfg.get("head_type", "residual")),
        gate_type=str(cfg.get("gate_type", "none")),
        gate_scope=str(cfg.get("gate_scope", "shared")),
        gate_hidden=int(cfg.get("gate_hidden", 16)),
        recency_num_centers=int(cfg.get("recency_num_centers", 8)),
        recency_x_range=tuple(cfg.get("recency_x_range", (-3.0, 3.0))),
        gate_descriptors=cfg.get("variable_descriptors"),
        gate_omega_per_var=cfg.get("omega_per_var"),
        gate_use_staleness=bool(cfg.get("gate_use_staleness", True)),
        gate_descriptor_hidden=int(cfg.get("gate_descriptor_hidden", 32)),
        gate_descriptor_mean=cfg.get("descriptor_mean"),
        gate_descriptor_std=cfg.get("descriptor_std"),
        residual_scale_scope=cfg.get("residual_scale_scope"),
        recency_basis=str(cfg.get("recency_basis", "rbf")),
        recency_grid_size=(
            None if cfg.get("recency_grid_size") is None else int(cfg.get("recency_grid_size"))
        ),
        recency_spline_order=int(cfg.get("recency_spline_order", 3)),
        recency_n_freq=(
            None if cfg.get("recency_n_freq") is None else int(cfg.get("recency_n_freq"))
        ),
        recency_num_num=(
            None if cfg.get("recency_num_num") is None else int(cfg.get("recency_num_num"))
        ),
        recency_num_den=(
            None if cfg.get("recency_num_den") is None else int(cfg.get("recency_num_den"))
        ),
        recency_parameterization=str(cfg.get("recency_parameterization", "fixed")),
        gate_monotone=bool(cfg.get("gate_monotone", False)),
        conditioned_residual=bool(cfg.get("conditioned_residual", False)),
        residual_descriptors=cfg.get("residual_descriptors"),
        residual_descriptor_hidden=int(cfg.get("residual_descriptor_hidden", 32)),
    )


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    mortality_criterion=None,
    mortality_weight: float = 0.0,
    train_median: torch.Tensor | None = None,
    training_objective: str = "event_weighted_l1",
) -> float:
    model.train()
    losses = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if getattr(model, "head_type", "direct") == "residual":
            if train_median is None:
                train_median = _standardized_train_median(
                    int(model.backbone.event_encoder.n_vars), device
                )
            batch, _, _ = _prepare_persistence_fields(batch, train_median)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss_mask = batch.get("target_loss_mask", batch["target_mask"])
        if str(training_objective) == "event_weighted_l1":
            loss = masked_l1_loss(output["forecast"], batch["target_value"], loss_mask)
        elif str(training_objective) == "event_weighted_mse":
            loss = masked_mse_loss(output["forecast"], batch["target_value"], loss_mask)
        elif str(training_objective) == "variable_balanced_l1":
            loss = masked_variable_balanced_l1_loss(
                output["forecast"],
                batch["target_value"],
                loss_mask,
                batch["target_var"],
            )
        else:
            raise ValueError(
                "training_objective must be event_weighted_l1, event_weighted_mse, "
                "or variable_balanced_l1"
            )
        if mortality_criterion is not None and mortality_weight > 0.0:
            loss = loss + float(mortality_weight) * mortality_criterion(
                output["mortality"], batch["label"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Phase 1 forecast loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_forecast(
    model,
    loader,
    device,
    train_mean,
    train_median=None,
    variable_filter=None,
) -> dict[str, object]:
    model.eval()
    losses = []
    pooled_mse_numerator = 0.0
    pooled_mse_denominator = 0.0
    predictions = []
    targets = []
    variables = []
    persistence_values = []
    staleness_values = []
    labels = []
    mortality_scores = []
    if train_median is None:
        train_median = _standardized_train_median(len(train_mean), device)
    else:
        train_median = torch.as_tensor(
            train_median, dtype=torch.float32, device=device
        ).reshape(-1)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch, x_last, staleness = _prepare_persistence_fields(batch, train_median)
        output = model(batch)
        loss = masked_l1_loss(output["forecast"], batch["target_value"], batch["target_mask"])
        losses.append(float(loss.detach().cpu()))
        mse_mask = batch["target_mask"]
        if variable_filter is not None:
            allowed_for_mse = torch.as_tensor(
                list(variable_filter), dtype=torch.long, device=device
            )
            mse_mask = mse_mask & torch.isin(
                batch["target_var"].to(dtype=torch.long), allowed_for_mse
            )
        squared = (output["forecast"] - batch["target_value"]).square()
        pooled_mse_numerator += float((squared * mse_mask).sum().detach().cpu())
        pooled_mse_denominator += float(mse_mask.sum().detach().cpu())
        mask = batch["target_mask"]
        if variable_filter is not None:
            allowed = torch.as_tensor(
                list(variable_filter), dtype=torch.long, device=device
            )
            mask = mask & torch.isin(batch["target_var"].to(dtype=torch.long), allowed)
        predictions.append(output["forecast"][mask].detach().cpu().numpy())
        targets.append(batch["target_value"][mask].detach().cpu().numpy())
        variables.append(batch["target_var"][mask].detach().cpu().numpy())
        persistence_values.append(x_last[mask].detach().cpu().numpy())
        staleness_values.append(staleness[mask].detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        mortality_scores.append(output["mortality"].detach().cpu().numpy())
    prediction = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.float32)
    variable = np.concatenate(variables) if variables else np.empty(0, dtype=np.int64)
    x_last = (
        np.concatenate(persistence_values)
        if persistence_values
        else np.empty(0, dtype=np.float32)
    )
    staleness = (
        np.concatenate(staleness_values)
        if staleness_values
        else np.empty(0, dtype=np.float32)
    )
    metrics = normalized_forecast_metrics(
        prediction,
        target,
        variable,
        train_mean,
        n_variables=len(train_mean),
    )
    skill, skill_frame, skill_excluded = persistence_skill(
        prediction, target, variable, x_last
    )
    by_dynamics, by_staleness, skill_frame = stratified_report(
        skill_frame,
        staleness,
        variable,
        prediction,
        target,
        x_last,
        train_mean=train_mean,
    )
    from ..utils.metrics import binary_metrics

    classification = binary_metrics(
        np.concatenate(labels) if labels else np.empty(0, dtype=np.float32),
        1.0 / (1.0 + np.exp(-np.concatenate(mortality_scores)))
        if mortality_scores
        else np.empty(0, dtype=np.float32),
    ) if labels else {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan")}
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "pooled_mse": (
            pooled_mse_numerator / pooled_mse_denominator
            if pooled_mse_denominator > 0.0
            else float("nan")
        ),
        "nmae": float(metrics["nmae"]),
        "nrmse": float(metrics["nrmse"]),
        "skill": float(skill),
        "metrics": metrics,
        "prediction": prediction,
        "target": target,
        "variable": variable,
        "x_last": x_last,
        "staleness": staleness,
        "skill_frame": skill_frame,
        "skill_excluded": skill_excluded,
        "by_dynamics": by_dynamics,
        "by_staleness": by_staleness,
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=np.float32),
        "mortality_score": np.concatenate(mortality_scores)
        if mortality_scores
        else np.empty(0, dtype=np.float32),
        "classification": classification,
    }


def _classification_logits(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Run the mortality branch without exposing future target events.

    The forecasting model contains a cross-attention head whose queries are
    built from future target variables.  That is appropriate for the forecast
    task, but S4-T is a classification-only replication and must use only the
    observation window.  Pooling the encoded observations directly keeps the
    task boundary explicit and prevents target-window leakage.
    """

    hidden = model.encode(batch)
    valid = batch["obs_mask"].to(dtype=hidden.dtype).unsqueeze(-1)
    pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return model.mortality_head(pooled).squeeze(-1)


def _run_classification_epoch(model, loader, optimizer, device, criterion) -> float:
    model.train()
    losses = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _classification_logits(model, batch)
        loss = criterion(logits, batch["label"])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite S4-T classification loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_classification(model, loader, device) -> dict[str, object]:
    """Evaluate the observation-only mortality branch for S4-T."""

    model.eval()
    losses = []
    probabilities = []
    labels = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = _classification_logits(model, batch)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    probability = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    label = np.concatenate(labels) if labels else np.empty(0, dtype=np.float32)
    from ..utils.metrics import binary_metrics

    metrics = (
        binary_metrics(label, probability)
        if len(label)
        else {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan")}
    )
    return {
        "auprc": float(metrics["auprc"]),
        "auroc": float(metrics["auroc"]),
        "brier": float(metrics["brier"]),
        "probability": probability,
        "label": label,
    }


def train_classification_run(
    bundle: ForecastSplitBundle,
    *,
    encoder_type: str,
    seed: int,
    cfg: Mapping[str, object],
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train one observation-only mortality classifier for S4-T.

    The encoder, grouping, calibration and optimizer protocol match the
    forecasting runs.  Early stopping is based on validation AUPRC, which is
    the primary metric for this imbalanced task.
    """

    set_seed(int(seed), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(str(cfg.get("device", "auto")))
    loaders = make_forecast_loaders(
        bundle,
        batch_size=int(cfg.get("batch_size", 16)),
        seed=int(seed),
        num_workers=int(cfg.get("num_workers", 0)),
        train_fraction=float(cfg.get("train_fraction", 1.0)),
        train_fraction_seed=int(cfg.get("train_fraction_seed", cfg.get("split_seed", 42))),
        held_out_icu_types=cfg.get("held_out_icu_types"),
    )
    model = build_forecast_model(bundle, encoder_type=encoder_type, cfg=cfg).to(device)
    calibration = calibrate_grids(
        model,
        loaders["train"],
        device,
        n_batches=int(cfg.get("calibration_batches", 20)),
        forward_fn=lambda current, batch: current.encode(batch),
    )

    train_examples = bundle.cache["examples"]
    train_indices = list(loaders["train"].dataset.indices)
    train_labels = np.asarray(
        [float(train_examples[int(index)]["label"]) for index in train_indices],
        dtype=np.float32,
    )
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("S4-T training split must contain both mortality classes")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    best_state = None
    best_val = float("-inf")
    best_epoch = 0
    wait = 0
    start = time.perf_counter()
    for epoch in range(1, int(cfg.get("epochs", 20)) + 1):
        train_loss = _run_classification_epoch(model, loaders["train"], optimizer, device, criterion)
        validation = evaluate_classification(model, loaders["val"], device)
        if np.isfinite(validation["auprc"]) and validation["auprc"] > best_val:
            best_val = float(validation["auprc"])
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= int(cfg.get("patience", 5)):
                break
    if best_state is None:
        raise RuntimeError(f"{encoder_type} seed {seed} produced no finite S4-T validation result")
    model.load_state_dict(best_state)
    test = evaluate_classification(model, loaders["test"], device)
    efficiency = measure_forecast_efficiency(
        model,
        next(iter(loaders["test"])),
        device,
        repeats=int(cfg.get("latency_repeats", 10)),
    )
    grid_audit = audit_grid_coverage(model)
    grid_status = str(grid_audit["status"])
    grid_report = grid_audit["layers"]
    result: dict[str, object] = {
        "task": "classification",
        "encoder_type": str(encoder_type),
        "seed": int(seed),
        "status": grid_status,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "train_loss": float(train_loss),
        "val_auprc": float(best_val),
        "auprc": float(test["auprc"]),
        "auroc": float(test["auroc"]),
        "n_params": count_parameters(model),
        "determinism": determinism_status(model, bool(cfg.get("deterministic", True))),
        **efficiency,
        "duration_sec": float(time.perf_counter() - start),
        "grid_status": grid_status,
        "grid_report": grid_report,
        "grid_min_coverage": grid_audit["min_coverage"],
        "grid_underflow_count": grid_audit["n_underflow_layers"],
        "grid_underflow_layers": grid_audit["underflow_layers"],
        "grid_audit_policy": grid_audit["policy"],
        **provenance_fingerprint(
            cfg,
            project_root=Path(__file__).resolve().parents[2],
            device=device,
        ),
        "calibration": calibration,
        "model": model,
        "test": test,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": best_state, "encoder_type": encoder_type, "seed": int(seed), "task": "classification"},
            output_dir / f"{encoder_type}_{int(seed)}.pt",
        )
    return result


def train_forecast_run(
    bundle: ForecastSplitBundle,
    *,
    encoder_type: str,
    seed: int,
    cfg: Mapping[str, object],
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train one encoder/seed pair and return a flat auditable result."""

    set_seed(int(seed), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(str(cfg.get("device", "auto")))
    loaders = make_forecast_loaders(
        bundle,
        batch_size=int(cfg.get("batch_size", 16)),
        seed=int(seed),
        num_workers=int(cfg.get("num_workers", 0)),
        train_fraction=float(cfg.get("train_fraction", 1.0)),
        train_fraction_seed=int(cfg.get("train_fraction_seed", cfg.get("split_seed", 42))),
        held_out_variables=cfg.get("held_out_variables"),
        held_out_icu_types=cfg.get("held_out_icu_types"),
    )
    model = build_forecast_model(bundle, encoder_type=encoder_type, cfg=cfg).to(device)
    train_median = _standardized_train_median(
        bundle.standardizer.n_variables, device
    )
    n_groups = len(model.backbone.event_encoder.group_names)
    n_functions = 1 if str(encoder_type) == "kan_joint" else 2
    alignment = solve_matched_pair(
        n_groups,
        d_in=2 if str(encoder_type) == "kan_joint" else 1,
        d_out=int(cfg.get("d_model", 64)),
        n_functions=n_functions,
        reference_hidden=int(cfg.get("mlp_hidden", 16)),
        grid_size=int(cfg.get("grid_size", 8)),
        spline_order=int(cfg.get("spline_order", 3)),
        basis=str(cfg.get("basis", "bspline")),
        num_centers=int(cfg.get("num_centers", 8)),
        n_freq=int(cfg.get("n_freq", 6)),
    )
    calibration = calibrate_grids(
        model,
        loaders["train"],
        device,
        n_batches=int(cfg.get("calibration_batches", 20)),
        forward_fn=(
            lambda current, batch: current(
                _prepare_persistence_fields(batch, train_median)[0]
            )
            if getattr(model, "head_type", "direct") == "residual"
            else None
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    mortality_weight = float(cfg.get("mortality_weight", 0.0))
    mortality_criterion = None
    if mortality_weight > 0.0:
        train_examples = bundle.cache["examples"]
        train_indices = list(loaders["train"].dataset.indices)
        train_labels = np.asarray(
            [float(train_examples[int(index)]["label"]) for index in train_indices],
            dtype=np.float32,
        )
        positives = float(train_labels.sum())
        negatives = float(len(train_labels) - positives)
        if positives > 0.0:
            mortality_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(negatives / positives, device=device)
            )
    best_state = None
    default_selection = (
        "macro_persistence"
        if getattr(model, "head_type", "direct") == "residual"
        else "pooled_mse"
    )
    selection_metric = str(cfg.get("checkpoint_selection_metric", default_selection))
    if selection_metric not in {"macro_persistence", "pooled_mse"}:
        raise ValueError(
            "checkpoint_selection_metric must be macro_persistence or pooled_mse"
        )
    use_skill_selection = selection_metric == "macro_persistence"
    best_val = float("-inf") if use_skill_selection else float("inf")
    best_val_loss = float("nan")
    best_epoch = 0
    wait = 0
    start = time.perf_counter()
    for epoch in range(1, int(cfg.get("epochs", 20)) + 1):
        train_loss = _run_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            mortality_criterion=mortality_criterion,
            mortality_weight=mortality_weight,
            train_median=train_median,
            training_objective=str(
                cfg.get("training_objective", "event_weighted_l1")
            ),
        )
        validation = evaluate_forecast(
            model,
            loaders["val"],
            device,
            bundle.standardizer.train_mean,
            train_median=train_median,
            variable_filter=cfg.get("selection_variables"),
        )
        validation_value = (
            float(validation["skill"])
            if use_skill_selection
            else float(validation["pooled_mse"])
        )
        is_better = (
            validation_value > best_val if use_skill_selection else validation_value < best_val
        )
        if np.isfinite(validation_value) and is_better:
            best_val = validation_value
            best_val_loss = float(validation["loss"])
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= int(cfg.get("patience", 5)):
                break
    if best_state is None:
        raise RuntimeError(f"{encoder_type} seed {seed} produced no finite validation result")
    model.load_state_dict(best_state)
    test = evaluate_forecast(
        model,
        loaders["test"],
        device,
        bundle.standardizer.train_mean,
        train_median=train_median,
        variable_filter=cfg.get("test_variables", cfg.get("eval_variables")),
    )
    efficiency = measure_forecast_efficiency(
        model,
        next(iter(loaders["test"])),
        device,
        repeats=int(cfg.get("latency_repeats", 10)),
    )
    grid_audit = audit_grid_coverage(model)
    grid_status = str(grid_audit["status"])
    grid_report = grid_audit["layers"]
    result: dict[str, object] = {
        "encoder_type": str(encoder_type),
        "seed": int(seed),
        "status": grid_status,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": best_val_loss,
        "val_selection_metric": selection_metric,
        "training_objective": str(
            cfg.get("training_objective", "event_weighted_l1")
        ),
        "val_selection_value": float(best_val),
        "nmae": float(test["nmae"]),
        "nrmse": float(test["nrmse"]),
        "skill": float(test["skill"]),
        "auprc": float(test["classification"]["auprc"]),
        "auroc": float(test["classification"]["auroc"]),
        "n_events_test": int(len(test["target"])),
        "n_params": count_parameters(model),
        "determinism": determinism_status(model, bool(cfg.get("deterministic", True))),
        "parameter_alignment": alignment,
        **efficiency,
        "duration_sec": float(time.perf_counter() - start),
        "grid_status": grid_status,
        "grid_report": grid_report,
        "grid_min_coverage": grid_audit["min_coverage"],
        "grid_underflow_count": grid_audit["n_underflow_layers"],
        "grid_underflow_layers": grid_audit["underflow_layers"],
        "grid_audit_policy": grid_audit["policy"],
        **provenance_fingerprint(
            cfg,
            project_root=Path(__file__).resolve().parents[2],
            device=device,
        ),
        "calibration": calibration,
        "skill_frame": test["skill_frame"],
        "skill_excluded": test["skill_excluded"],
        "by_dynamics": test["by_dynamics"],
        "by_staleness": test["by_staleness"],
        "model": model,
        "test": test,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": best_state, "encoder_type": encoder_type, "seed": int(seed)},
            output_dir / f"{encoder_type}_{int(seed)}.pt",
        )
    return result
