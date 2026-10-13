"""P1 training loop with exact masked-L1 accumulation across padded pieces.

This is a project-owned counterpart to PyOmniTS ``Exp_Main.train``. It keeps
the original loader, batches, sample order, optimizer, scheduler, checkpoint
selection, and validation path. Only an over-budget batch is partitioned; its
piece gradients are accumulated against the original full-batch denominator
before one optimizer step.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from p1_padded_batch_split import plan_padded_batch, slice_padded_batch


class SingleSampleCudaOutOfMemory(RuntimeError):
    """A complete singleton forward/backward raised a real CUDA OOM."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        self.evidence = evidence
        super().__init__(message)


# A rescue attempt re-plans the same batch at this fraction of the budget.
OOM_RESCUE_BUDGET_DIVISOR = 4


class _PieceOutOfMemory(Exception):
    """Internal: one piece hit CUDA OOM; the caller decides whether to rescue."""

    def __init__(self, cause: BaseException, evidence: dict[str, Any]):
        self.cause = cause
        self.evidence = evidence
        super().__init__(str(cause))

    def raise_terminal(self) -> None:
        if self.evidence.get("piece_sample_count") == 1:
            raise SingleSampleCudaOutOfMemory(
                "single complete P1 sample failed during forward/backward: "
                f"sample_ID={self.evidence['sample_ids'][0]}, "
                f"padded_size={self.evidence['piece_padded_size']}, "
                f"budget={self.evidence['padded_size_budget']}, "
                f"offloaded={self.evidence['piece_offloaded']}; {self.cause}",
                self.evidence,
            ) from self.cause
        raise self.cause


def run_batch_plan(
    exp: Any,
    model_train: Any,
    model_optim: Any,
    criterion: Any,
    raw_batch: dict[str, Tensor],
    plan: Any,
    *,
    train_stage: int,
    epoch: int,
    batch_index: int,
    offload_all: bool,
    rescue: bool,
) -> dict[str, Any]:
    """Forward/backward every piece of one original batch; no optimizer step.

    Module-level so the gradient-equivalence proof can call the exact code
    that trains, rather than a re-implementation.

    Pieces above the budget (only singletons can be, and only when
    ``offload_oversized_pieces``) and every piece of a rescue attempt run
    with autograd's saved tensors in pinned host memory. This trades
    transfer time for VRAM on the few batches that need it and leaves the
    arithmetic unchanged.
    """
    from utils.globals import accelerator, logger

    model_optim.zero_grad()
    audit = {
        "train_stage": int(train_stage),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "full_batch_measured_target_count": int(float(raw_batch["y_mask"].sum().item())),
        "p1_observation_padding_width": int(max(plan.observation_counts)),
        "normalizer": "measured targets in the original full DataLoader batch",
        "optimizer_step_count_for_original_batch": 1,
        "rescue_attempt": bool(rescue),
        **plan.to_dict(),
        "piece_target_counts": [],
        "piece_mae_losses": [],
        "piece_normalized_losses": [],
        "piece_offloaded": [],
    }
    full_target_count = float(raw_batch["y_mask"].sum().item())
    scaled = 0.0
    target_total = 0
    offloaded = 0
    for piece_number, piece_positions in enumerate(plan.piece_positions):
        piece_size = int(plan.piece_padded_sizes[piece_number])
        offload_piece = offload_all or piece_size > plan.padded_size_budget
        piece = slice_padded_batch(raw_batch, piece_positions)
        piece_ids = [int(value) for value in piece["sample_ID"].tolist()]
        evidence = {
            "model_id": str(exp.configs.model_id),
            "train_stage": int(train_stage),
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "piece_number": int(piece_number),
            "sample_ids": piece_ids,
            "piece_sample_count": len(piece_positions),
            "piece_padded_size": piece_size,
            "padded_size_budget": int(plan.padded_size_budget),
            "piece_offloaded": bool(offload_piece),
            "rescue_attempt": bool(rescue),
            "original_batch_padded_size": int(plan.original_padded_size),
            "original_batch_sample_ids": list(plan.original_sample_ids),
            "input_time_row_counts": [plan.input_time_row_counts[i] for i in piece_positions],
            "outcome_time_row_counts": [plan.outcome_time_row_counts[i] for i in piece_positions],
            "observation_counts": [plan.observation_counts[i] for i in piece_positions],
        }
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if offload_piece else contextlib.nullcontext()
        )
        try:
            with context:
                piece = {key: value.to(exp.device) for key, value in piece.items()}
                outputs: dict[str, Tensor] = model_train(
                    **piece,
                    exp_stage="train",
                    train_stage=train_stage,
                    current_epoch=epoch,
                    p1_observation_padding_width=max(plan.observation_counts),
                )
                if not torch.equal(outputs.get("mask"), piece["y_mask"]):
                    raise AssertionError(
                        "model output mask differs from this piece's real measured-target mask"
                    )
                if batch_index == 0 and epoch == 0 and piece_number == 0 and not rescue:
                    exp._check_model_outputs(piece, outputs)

                piece_loss: Tensor = criterion(
                    **outputs,
                    exp_stage="train",
                    train_stage=train_stage,
                    current_epoch=epoch,
                    model=model_train,
                )["loss"]
                piece_target_count = int(piece["y_mask"].sum().item())
                target_total += piece_target_count
                # MAE and MSE return each piece's masked mean. Multiplying by
                # n_piece/n_full recovers sum(piece errors)/n_full.
                normalized_piece_loss = piece_loss * (piece_target_count / full_target_count)

                if torch.any(torch.isnan(normalized_piece_loss)):
                    logger.exception(
                        "P1 microbatch loss is NaN at batch %s piece %s",
                        batch_index, piece_number,
                    )
                    for key, value in outputs.items():
                        if isinstance(value, Tensor) and torch.any(torch.isnan(value)):
                            logger.error("NaN model output %s with shape %s", key, tuple(value.shape))
                    return {"nan_found": True, "scaled_batch_loss": scaled, "piece_target_total": target_total,
                            "offloaded_piece_count": offloaded, "audit": audit}

                scaled += float(normalized_piece_loss.detach().cpu())
                audit["piece_target_counts"].append(piece_target_count)
                audit["piece_mae_losses"].append(float(piece_loss.detach().cpu()))
                audit["piece_normalized_losses"].append(float(normalized_piece_loss.detach().cpu()))
                audit["piece_offloaded"].append(bool(offload_piece))
                if exp.configs.sweep:
                    normalized_piece_loss.backward(retain_graph=exp.configs.retain_graph)
                else:
                    accelerator.backward(normalized_piece_loss, retain_graph=exp.configs.retain_graph)
            offloaded += int(offload_piece)
        except torch.cuda.OutOfMemoryError as exc:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(exp.device)
                free_mib = int(free_bytes // (1024 * 1024))
                total_mib = int(total_bytes // (1024 * 1024))
            except Exception:
                free_mib = total_mib = None
            evidence.update({
                "cuda_memory_free_mib_after_oom": free_mib,
                "cuda_memory_total_mib": total_mib,
                "cuda_peak_allocated_mib": int(torch.cuda.max_memory_allocated(exp.device) // (1024 * 1024)),
                "cuda_peak_reserved_mib": int(torch.cuda.max_memory_reserved(exp.device) // (1024 * 1024)),
                "cuda_device_name": torch.cuda.get_device_name(exp.device),
                "underlying_error": f"{type(exc).__name__}: {exc}",
            })
            outputs = None
            piece = None
            raise _PieceOutOfMemory(exc, evidence) from exc
        finally:
            if isinstance(piece, dict):
                del piece
    return {"nan_found": False, "scaled_batch_loss": scaled, "piece_target_total": target_total,
            "offloaded_piece_count": offloaded, "audit": audit}


def train_with_padded_microbatches(
    exp: Any,
    padded_size_budget: int,
    offload_oversized_pieces: bool = False,
    oom_rescue: bool = False,
) -> dict[str, Any]:
    """Train one P1 run, returning an auditable summary of actual splitting.

    ``offload_oversized_pieces`` runs any single sample above the budget with
    its autograd tensors in host memory instead of refusing it. ``oom_rescue``
    retries a batch that hits CUDA OOM once, from zeroed gradients, with
    pieces planned at a quarter of the budget and all of them offloaded.
    Neither changes the samples, the normalizer or the optimizer-step count.
    """

    from utils.globals import accelerator, logger
    from utils.tools import EarlyStopping

    if bool(exp.configs.use_multi_gpu):
        raise RuntimeError("P1 exact microbatch path currently requires the frozen single-GPU setup")
    if int(accelerator.gradient_accumulation_steps) != 1:
        raise RuntimeError(
            "P1 piece gradients must accumulate without Accelerator loss scaling; "
            f"got gradient_accumulation_steps={accelerator.gradient_accumulation_steps}"
        )
    # Both losses are (sum of masked element errors) / (mask count), so scaling
    # each piece by its target share recovers the full-batch loss exactly.
    # P1 uses MAE only; P2 also trains MSE (work order §20.2).
    if str(exp.configs.loss).upper() not in {"MAE", "MSE"}:
        raise RuntimeError("exact microbatch path is defined only for the masked MAE and MSE losses")
    if int(exp.configs.batch_size) != 128:
        raise RuntimeError(f"P1 microbatch training requires batch_size=128, got {exp.configs.batch_size}")
    padded_size_budget = int(padded_size_budget)
    if padded_size_budget < 1:
        raise ValueError("padded-size budget must be positive")

    logger.info(">>>>>>> training start (P1 padded microbatches) <<<<<<<")
    train_start = time.perf_counter()
    path = (
        Path(exp.configs.checkpoints)
        / exp.configs.dataset_name
        / exp.configs.dataset_id
        / exp.configs.model_name
        / exp.configs.model_id
        / f"{exp.configs.seq_len}_{exp.configs.pred_len}"
        / exp.configs.subfolder_train
        / f"iter{exp.configs.itr_i}"
    )
    if (exp.configs.wandb and accelerator.is_main_process) or exp.configs.sweep:
        import wandb

    train_data, train_loader = exp._get_data(flag="train")
    vali_data, vali_loader = exp._get_data(flag="val")
    model_train = exp._build_model()
    batch_norm_layers = [
        name for name, module in model_train.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if batch_norm_layers:
        raise RuntimeError(
            "P1 exact microbatch accumulation assumes independent samples; "
            f"BatchNorm layers found: {batch_norm_layers}"
        )
    model_optim = exp._select_optimizer(model_train)
    lr_scheduler = exp._select_lr_scheduler(model_optim)
    criterion = exp._select_criterion()

    if not exp.configs.sweep:
        # Keep collated full batches on CPU until the planner has isolated any
        # oversized window. In single-process P1 this retains the same sampler
        # and batches; only device placement differs from Accelerate's default.
        train_loader = accelerator.prepare_data_loader(
            train_loader, device_placement=False
        )
        vali_loader = accelerator.prepare_data_loader(
            vali_loader, device_placement=False
        )
        model_train, model_optim = accelerator.prepare(model_train, model_optim)
        accelerator.register_for_checkpointing(model_optim)
    else:
        model_train, model_optim = accelerator.prepare(model_train, model_optim)

    initial_optimizer_state = model_optim.state_dict()
    initial_scheduler_state = lr_scheduler.state_dict()
    model_train = model_train.to(exp.device)

    audit_path = path / "p1_padded_batch_split_audit.jsonl"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite microbatch audit: {audit_path}")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if_nan_loss = False
    stage_summaries: list[dict[str, Any]] = []
    split_batch_count = 0
    optimizer_step_count = 0
    max_original_padded_size = 0
    max_piece_count = 1
    original_batch_count = 0

    offloaded_piece_count = 0
    oom_rescue_count = 0

    with audit_path.open("x", encoding="utf-8") as audit_file:
        for train_stage in range(1, exp.configs.n_train_stages + 1):
            early_stopping = EarlyStopping(patience=exp.configs.patience, verbose=True)
            best_epoch = None
            logger.info("Train stage %s/%s starts.", train_stage, exp.configs.n_train_stages)
            for epoch in tqdm(range(exp.configs.train_epochs), desc="Epochs"):
                train_loss: list[float] = []
                model_train.train()
                with tqdm(total=len(train_loader), leave=False, desc="Training") as progress:
                    for batch_index, raw_batch in enumerate(train_loader):
                        batch_size = int(raw_batch["sample_ID"].shape[0])
                        if batch_size != exp.configs.batch_size and epoch == 0:
                            logger.warning(
                                "P1 train: batch %s has %s samples, config batch_size=%s",
                                batch_index, batch_size, exp.configs.batch_size,
                            )
                        if torch.sum(raw_batch["y_mask"]).item() == 0:
                            if epoch == 0:
                                logger.warning(
                                    "P1 train: batch %s has no measured target and is skipped",
                                    batch_index,
                                )
                            continue

                        # Planning and the full-batch normalizer are computed
                        # from the untouched DataLoader batch on CPU.
                        plan = plan_padded_batch(
                            raw_batch, padded_size_budget,
                            allow_oversized_singletons=offload_oversized_pieces,
                        )
                        full_target_count = float(raw_batch["y_mask"].sum().item())
                        if full_target_count <= 0:
                            raise AssertionError("nonempty P1 batch has no measured-target denominator")
                        try:
                            attempt = run_batch_plan(
                                exp, model_train, model_optim, criterion, raw_batch, plan,
                                train_stage=train_stage, epoch=epoch, batch_index=batch_index,
                                offload_all=False, rescue=False,
                            )
                        except _PieceOutOfMemory as first_failure:
                            if not oom_rescue:
                                first_failure.raise_terminal()
                            # Discard every partial gradient of this batch and
                            # retry the same 128 samples once: smaller pieces,
                            # all offloaded. Normalizer and the single
                            # optimizer step are unchanged.
                            model_optim.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            rescue_plan = plan_padded_batch(
                                raw_batch,
                                max(1, padded_size_budget // OOM_RESCUE_BUDGET_DIVISOR),
                                allow_oversized_singletons=True,
                            )
                            try:
                                attempt = run_batch_plan(
                                    exp, model_train, model_optim, criterion, raw_batch, rescue_plan,
                                    train_stage=train_stage, epoch=epoch, batch_index=batch_index,
                                    offload_all=True, rescue=True,
                                )
                            except _PieceOutOfMemory as second_failure:
                                second_failure.raise_terminal()
                            attempt["audit"]["oom_rescue"] = {
                                "first_attempt_piece_number": first_failure.evidence.get("piece_number"),
                                "first_attempt_piece_sample_count": first_failure.evidence.get("piece_sample_count"),
                                "first_attempt_error": first_failure.evidence.get("underlying_error"),
                                "rescue_budget": int(rescue_plan.padded_size_budget),
                            }
                            oom_rescue_count += 1
                            plan = rescue_plan

                        max_original_padded_size = max(
                            max_original_padded_size, plan.original_padded_size
                        )
                        max_piece_count = max(max_piece_count, len(plan.piece_positions))
                        offloaded_piece_count += attempt["offloaded_piece_count"]
                        if plan.was_split:
                            split_batch_count += 1
                        scaled_batch_loss = attempt["scaled_batch_loss"]
                        if attempt["nan_found"]:
                            if_nan_loss = True
                            break
                        if attempt["piece_target_total"] != int(full_target_count):
                            raise AssertionError(
                                "P1 split pieces do not conserve the original batch's measured-target count: "
                                f"pieces={attempt['piece_target_total']}, original={int(full_target_count)}"
                            )
                        if plan.was_split or "oom_rescue" in attempt["audit"] or attempt["offloaded_piece_count"]:
                            attempt["audit"].update({
                                "accumulated_batch_loss": float(scaled_batch_loss),
                                "optimizer_step_applied": True,
                            })
                            audit_file.write(json.dumps(attempt["audit"], ensure_ascii=False) + "\n")
                            audit_file.flush()

                        # This is the only optimizer update for the original
                        # DataLoader batch, regardless of piece count.
                        if exp.configs.task_name == "classification":
                            clip_grad_norm_(model_train.parameters(), max_norm=4.0)
                        model_optim.step()
                        optimizer_step_count += 1
                        original_batch_count += 1
                        if optimizer_step_count != original_batch_count:
                            raise AssertionError("P1 exact microbatch loop changed optimizer-step cardinality")
                        train_loss.append(scaled_batch_loss)

                        if accelerator.is_main_process:
                            progress.update()
                            progress.set_postfix(loss=f"{scaled_batch_loss:.2e}")

                if if_nan_loss:
                    accelerator.set_trigger()
                    if accelerator.check_trigger():
                        accelerator.wait_for_everyone()
                        break

                if epoch % exp.configs.val_interval == 0:
                    validation = exp.vali(
                        model_train=model_train,
                        vali_loader=vali_loader,
                        criterion=criterion,
                        current_epoch=epoch,
                        train_stage=train_stage,
                        return_metrics=True,
                    )
                    val_loss = float(validation["loss"])
                    selection_metric = getattr(
                        exp.configs, "checkpoint_selection_metric", "pooled_mse"
                    )
                    if selection_metric == "macro_persistence":
                        selection_value = -float(validation["macro_persistence_skill"])
                    else:
                        selection_value = float(validation["pooled_mse"])
                    previous_best = early_stopping.best_score
                    early_stopping(selection_value, model_train, path)
                    if early_stopping.best_score is not None and early_stopping.best_score != previous_best:
                        best_epoch = int(epoch)
                    if (exp.configs.wandb and accelerator.is_main_process) or exp.configs.sweep:
                        wandb.log({
                            "loss_train": np.mean(train_loss),
                            "loss_val": val_loss,
                            "loss_val_best": -early_stopping.best_score,
                            "checkpoint_selection_value": selection_value,
                            **({"val_macro_persistence_skill": validation["macro_persistence_skill"]}
                               if "macro_persistence_skill" in validation else {}),
                        })
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        accelerator.set_trigger()
                elif (exp.configs.wandb and accelerator.is_main_process) or exp.configs.sweep:
                    wandb.log({"loss_train": np.mean(train_loss)})

                lr_scheduler.step()
                logger.debug("Updating learning rate to %.6e", lr_scheduler.get_last_lr()[0])
                if accelerator.check_trigger():
                    accelerator.wait_for_everyone()
                    break

            best_selection_value = (
                float(-early_stopping.best_score)
                if early_stopping.best_score is not None else None
            )
            stage_summaries.append({
                "train_stage": int(train_stage),
                "checkpoint_selection_metric": str(selection_metric),
                "best_epoch": best_epoch,
                "best_selection_value": best_selection_value,
                "best_macro_persistence_skill": (
                    float(-best_selection_value)
                    if selection_metric == "macro_persistence" and best_selection_value is not None
                    else None
                ),
                "early_stopped": bool(early_stopping.early_stop),
            })
            model_optim.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)

    if accelerator.is_main_process:
        if optimizer_step_count != original_batch_count:
            raise AssertionError(
                f"P1 microbatch optimizer steps ({optimizer_step_count}) do not match "
                f"nonempty original DataLoader batches ({original_batch_count})"
            )
        (path / "training_summary.json").write_text(
            json.dumps({
                "checkpoint_selection_metric": str(
                    getattr(exp.configs, "checkpoint_selection_metric", "pooled_mse")
                ),
                "train_epochs": int(exp.configs.train_epochs),
                "patience": int(exp.configs.patience),
                "batch_size_effective": int(exp.configs.batch_size),
                "duration_sec": float(time.perf_counter() - train_start),
                "stages": stage_summaries,
                "p1_padded_microbatch": {
                    "padded_size_budget": padded_size_budget,
                    "padded_size_definition": "B * (max valid input rows + max valid outcome rows) * max observed measurements",
                    "original_batch_count": original_batch_count,
                    "optimizer_step_count": optimizer_step_count,
                    "split_batch_count_total": split_batch_count,
                    "max_original_padded_size": max_original_padded_size,
                    "max_piece_count": max_piece_count,
                    "split_audit_path": str(audit_path),
                    "offload_oversized_pieces": bool(offload_oversized_pieces),
                    "oom_rescue_enabled": bool(oom_rescue),
                    "offloaded_piece_count_total": offloaded_piece_count,
                    "oom_rescue_count": oom_rescue_count,
                },
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "padded_size_budget": padded_size_budget,
        "original_batch_count": original_batch_count,
        "optimizer_step_count": optimizer_step_count,
        "split_batch_count": split_batch_count,
        "max_original_padded_size": max_original_padded_size,
        "max_piece_count": max_piece_count,
        "split_audit_path": str(audit_path),
        "split_audit_exists": audit_path.is_file(),
        "offload_oversized_pieces": bool(offload_oversized_pieces),
        "oom_rescue_enabled": bool(oom_rescue),
        "offloaded_piece_count": offloaded_piece_count,
        "oom_rescue_count": oom_rescue_count,
    }
