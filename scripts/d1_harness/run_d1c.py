"""Run one external baseline on one ICU-KAN cohort inside the local harness checkout.

Hyper-parameters are taken verbatim from the model's shipped ``scripts/<model>/P12.sh``.
Only the arguments that name the dataset and the window are overridden, and the
overrides are printed so they end up in the run log.

Usage:
    python scripts/d1_harness/run_d1c.py <model> <cohort> [--epochs N] [--itr N] [--gpu N]
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "external" / "PyOmniTS"
DEFAULT_PYTHON = sys.executable

# Arguments whose value the shipped script derives from the P12 benchmark rather
# than from the model, and which therefore must be replaced for our task.
OVERRIDDEN = {
    "dataset_root_path", "dataset_name", "dataset_id", "dataset_subset_name",
    "seq_len", "pred_len", "enc_in", "dec_in", "c_out", "checkpoint_selection_metric",
    "model_id", "itr", "train_epochs", "save_arrays", "is_training", "model_name",
}


# NAIVE-1 is ours, so it has no upstream script to copy settings from. It has no
# hyper-parameters either; these only make the loop run once over the data.
OWN_MODEL_FLAGS = {
    "collate_fn": "collate_fn", "loss": "MSE", "features": "M",
    "batch_size": "32", "learning_rate": "1e-3", "patience": "10", "val_interval": "1",
}


def shipped_flags(model: str) -> dict[str, str]:
    script = HARNESS / "scripts" / model / "P12.sh"
    if not script.exists():
        if model.startswith("ICUKAN_") or model == "ICUKANForecaster":
            return dict(OWN_MODEL_FLAGS)
        raise FileNotFoundError(f"no shipped P12 script for {model}")
    text = script.read_text()
    # Flags appear as `--name value \` inside one long command.
    flags = dict(re.findall(r"--([a-zA-Z_0-9]+)\s+(\"[^\"]*\"|'[^']*'|\$?[^\s\\]+)", text))
    return {k: v.strip("\"'") for k, v in flags.items() if k not in OVERRIDDEN}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("cohort", choices=["p12", "mimiciv", "eicu"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--itr", type=int, default=5)
    parser.add_argument(
        "--iteration-index",
        type=int,
        default=-1,
        help="run one isolated seed index for recovery; preserves seed 2024+index",
    )
    parser.add_argument(
        "--hyperimts-force-low-memory",
        action="store_true",
        help="start HyperIMTS in its built-in low-memory message-passing mode",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", default="d1c")
    parser.add_argument("--python", dest="python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--data-variant", choices=("registered", "jbi_patient"), default="registered")
    parser.add_argument("--selection-metric", choices=("pooled_mse", "macro_persistence"), default="pooled_mse")
    parser.add_argument(
        "--loss",
        dest="loss_name",
        choices=("MSE", "VariableBalancedMSE", "MAE"),
        default=None,
        help="override the shipped loss for a controlled objective arm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command without launching the external harness",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a shipped model hyperparameter; repeat for a frozen search trial",
    )
    args = parser.parse_args()

    kept = shipped_flags(args.model)
    # `$n_variables` and friends are shell lookups in the shipped script.
    kept = {k: v for k, v in kept.items() if not v.startswith("$")}
    if args.loss_name is not None:
        kept["loss"] = args.loss_name
    allowed_overrides = {
        "batch_size",
        "learning_rate",
        "patience",
        "val_interval",
        "d_model",
        "d_ff",
        "n_layers",
        "n_heads",
        "dropout",
    }
    for raw_override in args.override:
        if "=" not in raw_override:
            raise SystemExit(f"--override must be KEY=VALUE, got {raw_override!r}")
        key, value = raw_override.split("=", 1)
        key = key.strip()
        if not key or not value:
            raise SystemExit(f"--override must contain non-empty KEY and VALUE: {raw_override!r}")
        if key in allowed_overrides:
            kept[key] = value
        else:
            raise SystemExit(
                f"unsupported search override {key!r}; allowed keys are "
                f"{sorted(allowed_overrides)}"
            )
    loss_name = kept.pop("loss", "MSE")

    command = [args.python_bin, "main.py",
               "--is_training", "1",
               "--model_name", args.model,
               "--model_id", f"{args.tag}_{args.cohort}",
               "--dataset_name", "ICUKAN",
               "--dataset_root_path", str(ROOT / "data" / "processed"),
               "--dataset_id", args.cohort,
               "--dataset_subset_name", args.data_variant,
               "--seq_len", "24", "--pred_len", "24",
               "--enc_in", "37", "--dec_in", "37", "--c_out", "37",
               "--checkpoint_selection_metric", args.selection_metric,
               "--loss", loss_name,
               "--train_epochs", str(args.epochs),
               "--itr", str(args.itr),
               "--save_arrays", "1"]
    if args.iteration_index >= 0:
        command += ["--iteration_index", str(args.iteration_index)]
    if args.hyperimts_force_low_memory:
        command += ["--hyperimts_force_low_memory", "1"]
    for key, value in sorted(kept.items()):
        command += [f"--{key}", value]

    print(f"[d1c] {args.model} on {args.cohort}, gpu {args.gpu}")
    print(f"[d1c] kept from shipped P12.sh: {kept}")
    print(f"[d1c] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)}
    import os
    full = dict(os.environ, **env)
    return subprocess.call(command, cwd=HARNESS, env=full)


if __name__ == "__main__":
    sys.exit(main())
