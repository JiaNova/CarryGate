#!/usr/bin/env python
"""Apply the project-owned Track B patch to a pinned PyOmniTS checkout.

The external repository is intentionally cloned afresh by ``setup_harness.sh``.
Keeping this patch in the project makes the task-aligned validation criterion
reproducible instead of relying on edits in a particular vendored directory.
The operation is idempotent and fails loudly if the pinned upstream layout has
changed in a way that could invalidate the protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"cannot apply {label}: anchor missing in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_expconfigs(root: Path) -> None:
    path = root / "utils" / "ExpConfigs.py"
    replace_once(
        path,
        "    loss: str\n    lr_scheduler: str\n",
        "    loss: str\n    checkpoint_selection_metric: str\n    lr_scheduler: str\n",
        "ExpConfigs checkpoint criterion",
    )


def patch_configs(root: Path) -> None:
    path = root / "utils" / "configs.py"
    replace_once(
        path,
        "    parser.add_argument('--loss', type=str, default='MSE', help='loss function. PyOmniTS will try to construct the loss function from class Loss in loss_fns/LOSS.py')\n",
        "    parser.add_argument('--loss', type=str, default='MSE', help='loss function. PyOmniTS will try to construct the loss function from class Loss in loss_fns/LOSS.py')\n    parser.add_argument('--checkpoint_selection_metric', type=str, choices=['pooled_mse', 'macro_persistence'], default='pooled_mse', help='validation criterion used for early stopping and checkpoint selection')\n",
        "argparse checkpoint criterion",
    )


def patch_exp_main(root: Path) -> None:
    path = root / "exp" / "exp_main.py"
    text = path.read_text(encoding="utf-8")
    if "def _validation_persistence_counts(" not in text:
        helper = r'''    def _validation_persistence_counts(
        self, pred: Tensor, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return model error, carry-forward error and event counts by variable.

        Track B selects checkpoints with the same macro persistence skill used
        by ICU-KAN's registered analysis. The adapter feeds standardized event
        values whose training median is zero, so the no-history fallback is
        also zero. Counts are accumulated over the complete validation set;
        the score is not an average of batch-level scores.
        """
        target = batch["y"]
        mask = batch["y_mask"] > 0
        if pred.shape[-1] != target.shape[-1]:
            target = target[..., : pred.shape[-1]]
            mask = mask[..., : pred.shape[-1]]

        x = batch["x"][..., : pred.shape[-1]]
        x_mask = batch["x_mask"][..., : pred.shape[-1]] > 0
        x_mark = batch["x_mark"]
        if x_mark.ndim == 3:
            x_mark = x_mark[..., 0]
        cutoff = float(self.configs.seq_len) / (
            float(self.configs.seq_len) + float(self.configs.pred_len)
        )
        times = x_mark.unsqueeze(-1).expand_as(x_mask)
        valid = x_mask & (times < cutoff)
        neg_inf = torch.finfo(times.dtype).min
        latest = torch.where(valid, times, torch.full_like(times, neg_inf)).max(
            dim=1, keepdim=True
        ).values
        at_latest = valid & (times == latest)
        positions = torch.arange(
            x.shape[1], device=x.device, dtype=torch.long
        ).view(1, x.shape[1], 1).expand_as(at_latest)
        chosen = torch.where(
            at_latest, positions, torch.full_like(positions, x.shape[1])
        ).min(dim=1).values
        observed = chosen < x.shape[1]
        gathered = x.gather(
            1, chosen.clamp(max=x.shape[1] - 1).unsqueeze(1)
        ).squeeze(1)
        x_last = torch.where(observed, gathered, torch.zeros_like(gathered))

        error = (torch.abs(pred - target) * mask).sum(dim=(0, 1))
        baseline_error = (
            torch.abs(x_last.unsqueeze(1) - target) * mask
        ).sum(dim=(0, 1))
        counts = mask.sum(dim=(0, 1)).to(dtype=error.dtype)
        return error, baseline_error, counts

'''
        text = text.replace("    def vali(\n", helper + "    def vali(\n", 1)

    text = text.replace(
        "        train_stage: int\n    ) -> np.ndarray:\n        total_loss = []\n        model_train.eval()\n",
        "        train_stage: int,\n        return_metrics: bool = False,\n    ) -> np.ndarray | dict[str, float]:\n        total_loss = []\n        model_error = None\n        baseline_error = None\n        event_counts = None\n        track_macro = (\n            return_metrics\n            and getattr(self.configs, \"checkpoint_selection_metric\", \"pooled_mse\")\n            == \"macro_persistence\"\n        )\n        model_train.eval()\n",
        1,
    )
    text = text.replace(
        "                    total_loss.append(loss.item())\n\n                    if accelerator.is_main_process:\n",
        "                    total_loss.append(loss.item())\n\n                    if track_macro:\n                        current_error, current_baseline, current_counts = (\n                            self._validation_persistence_counts(outputs[\"pred\"], batch)\n                        )\n                        packed = torch.stack(\n                            [current_error, current_baseline, current_counts], dim=0\n                        )\n                        if accelerator.num_processes > 1:\n                            packed = accelerator.gather(packed)\n                            packed = packed.reshape(-1, 3, packed.shape[-1]).sum(dim=0)\n                        if model_error is None:\n                            model_error, baseline_error, event_counts = packed\n                        else:\n                            model_error = model_error + packed[0]\n                            baseline_error = baseline_error + packed[1]\n                            event_counts = event_counts + packed[2]\n\n                    if accelerator.is_main_process:\n",
        1,
    )
    text = text.replace(
        "        total_loss = np.average(total_loss)\n        model_train.train()\n        return total_loss\n",
        "        total_loss = float(np.average(total_loss))\n        model_train.train()\n        if return_metrics:\n            result = {\"loss\": total_loss}\n            if track_macro:\n                if model_error is None:\n                    raise RuntimeError(\"validation produced no scored batches\")\n                valid = (event_counts >= 50) & (baseline_error > 1e-9)\n                if not bool(valid.any()):\n                    raise RuntimeError(\"validation has no non-degenerate macro variables\")\n                skill = 1.0 - model_error[valid] / baseline_error[valid]\n                result[\"macro_persistence_skill\"] = float(skill.mean().item())\n            return result\n        return total_loss\n",
        1,
    )
    text = text.replace(
        "                    vali_loss = self.vali(\n                        model_train=model_train, \n                        vali_loader=vali_loader, \n                        criterion=criterion, \n                        current_epoch=epoch,\n                        train_stage=train_stage\n                    )\n                    early_stopping(vali_loss, model_train, path)\n",
        "                    validation = self.vali(\n                        model_train=model_train, \n                        vali_loader=vali_loader, \n                        criterion=criterion, \n                        current_epoch=epoch,\n                        train_stage=train_stage,\n                        return_metrics=True,\n                    )\n                    vali_loss = float(validation[\"loss\"])\n                    selection_metric = getattr(\n                        self.configs, \"checkpoint_selection_metric\", \"pooled_mse\"\n                    )\n                    if selection_metric == \"macro_persistence\":\n                        selection_value = -float(validation[\"macro_persistence_skill\"])\n                    else:\n                        selection_value = vali_loss\n                    early_stopping(selection_value, model_train, path)\n",
        1,
    )
    text = text.replace(
        '                            "loss_val_best": -early_stopping.best_score\n',
        '                            "loss_val_best": -early_stopping.best_score,\n                            "checkpoint_selection_value": selection_value,\n                            **({"val_macro_persistence_skill": validation["macro_persistence_skill"]}\n                               if "macro_persistence_skill" in validation else {})\n',
        1,
    )
    path.write_text(text, encoding="utf-8")
    required = (
        "def _validation_persistence_counts(",
        "return_metrics: bool = False",
        "checkpoint_selection_value",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"Track B exp_main patch incomplete: {missing}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_task_aligned_patch.py /path/to/PyOmniTS")
    root = Path(sys.argv[1]).resolve()
    patch_expconfigs(root)
    patch_configs(root)
    patch_exp_main(root)
    print(f"task-aligned validation patch ready: {root}")


if __name__ == "__main__":
    main()
