#!/usr/bin/env python
"""CPU feasibility check for the frozen P1 landmark query contract.

This check instantiates the selected Track-B model configurations without
loading checkpoints or training.  A synthetic 48-hour irregular context is
queried at the 24 fixed hourly midpoints, the exact landmark boundary, and two
off-grid times.  It verifies output shapes, finite requested predictions,
query masks, and CarryGate's decoded continuous query coordinates.

The check intentionally does not use P1 test examples, labels, predictions,
or clinical measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
PYOMNITS = ROOT / "external" / "PyOmniTS"
sys.path.insert(0, str(PYOMNITS))
sys.path.insert(0, str(ROOT))

SOFA_VARIABLES = (5, 7, 10, 27)  # Bilirubin, Creatinine, GCS, Platelets in P12 order.
N_VARIABLES = 37
INPUT_HOURS = 48
FORECAST_HOURS = 24
TOTAL_HOURS = INPUT_HOURS + FORECAST_HOURS
OFF_GRID_HOURS = (49.13, 70.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def selected_config(cohort: str, config_dir: str) -> Path:
    return (
        PYOMNITS
        / "configs"
        / config_dir
        / f"jbi_trackb_controls_factorial_{config_dir}_mae_macro_v1_{cohort}"
        / "ICUKAN.yaml"
    )


def make_synthetic_batch() -> tuple[torch.Tensor, ...]:
    """Return sparse 48h input events and 27 future query rows."""

    batch, n_input = 1, 128
    x = torch.zeros(batch, n_input, N_VARIABLES, dtype=torch.float32)
    x_mask = torch.zeros_like(x)
    x_mark = torch.zeros(batch, n_input, 1, dtype=torch.float32)

    # Irregularity is explicit in value/mask density; event times span [0,48).
    # Multiple event rows may share a variable, as they do in real event streams.
    for row in range(n_input):
        hour = 0.25 + 47.5 * row / (n_input - 1)
        variable = row % N_VARIABLES
        x[0, row, variable] = float(np.sin(row / 9.0))
        x_mask[0, row, variable] = 1.0
        if row % 7 == 0:
            second_variable = (variable + 11) % N_VARIABLES
            x[0, row, second_variable] = float(np.cos(row / 11.0))
            x_mask[0, row, second_variable] = 1.0
        x_mark[0, row, 0] = hour / TOTAL_HOURS

    # Model-local time uses u = 48 + (s - t); hence t is u=48, and the
    # fixed hourly forecast grid is 48.5, ..., 71.5 on a [0,72) scale.
    grid = [INPUT_HOURS + k - 0.5 for k in range(1, FORECAST_HOURS + 1)]
    query_hours = sorted({float(INPUT_HOURS), *grid, *OFF_GRID_HOURS})
    y = torch.zeros(batch, len(query_hours), N_VARIABLES, dtype=torch.float32)
    y_mask = torch.zeros_like(y)
    y_mark = torch.zeros(batch, len(query_hours), 1, dtype=torch.float32)
    for row, hour in enumerate(query_hours):
        y_mark[0, row, 0] = hour / TOTAL_HOURS
        y_mask[0, row, list(SOFA_VARIABLES)] = 1.0

    if not (float(x_mark.min()) >= 0 and float(x_mark.max()) < INPUT_HOURS / TOTAL_HOURS):
        raise AssertionError("synthetic input query coordinates escaped [0,48)")
    if not (float(y_mark.min()) >= INPUT_HOURS / TOTAL_HOURS and float(y_mark.max()) < 1):
        raise AssertionError("synthetic forecast coordinates escaped [48,72)")
    if not np.allclose(grid, np.arange(48.5, 72.0, 1.0)):
        raise AssertionError("hourly query grid does not match the frozen P1 definition")

    return x, x_mark, x_mask, y, y_mark, y_mask, query_hours


def run_check() -> dict[str, Any]:
    # Delay upstream imports until execution so --help and static inspection
    # work before the user rebuilds the external-baseline harness.
    from models.GraFITi import Model as GraFITi
    from models.HyperIMTS import Model as HyperIMTS
    from models.ICUKANForecaster import Model as CarryGate

    models = {
        "CarryGate": (CarryGate, "ICUKANForecaster"),
        "GraFITi": (GraFITi, "GraFITi"),
        "HyperIMTS": (HyperIMTS, "HyperIMTS"),
    }
    torch.set_num_threads(2)
    torch.manual_seed(20261002)
    x, x_mark, x_mask, y, y_mark, y_mask, query_hours = make_synthetic_batch()
    checks: list[dict[str, Any]] = []

    for cohort in ("eicu", "mimiciv"):
        for model_name, (model_cls, config_dir) in models.items():
            config_path = selected_config(cohort, config_dir)
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            # Change only the frozen task window lengths: P1 defines 48h input
            # and 24h output. Architecture and selected model hyperparameters
            # remain those of the selected masked-L1 / macro-persistence arm.
            config_data.update(
                seq_len=INPUT_HOURS,
                pred_len=FORECAST_HOURS,
                seq_len_max_irr=None,
                pred_len_max_irr=None,
                enc_in=N_VARIABLES,
            )
            model = model_cls(SimpleNamespace(**config_data)).eval()

            with torch.no_grad():
                if model_name == "CarryGate":
                    batch = model.build_batch(x, x_mark, x_mask, y, y_mark, y_mask)
                    expected_times = torch.as_tensor(
                        np.repeat(query_hours, len(SOFA_VARIABLES)), dtype=torch.float32
                    )
                    actual_times = batch["target_time"][0, : expected_times.numel()]
                    if not torch.allclose(actual_times, expected_times, atol=1e-5, rtol=0):
                        raise AssertionError(
                            f"CarryGate query-time mapping mismatch for {cohort}: "
                            f"{actual_times.tolist()} != {expected_times.tolist()}"
                        )
                    expected_variables = torch.as_tensor(
                        np.tile(SOFA_VARIABLES, len(query_hours)), dtype=torch.long
                    )
                    actual_variables = batch["target_var"][0, : expected_variables.numel()]
                    if not torch.equal(actual_variables, expected_variables):
                        raise AssertionError("CarryGate target-variable mapping mismatch")

                output = model(
                    x,
                    x_mark,
                    x_mask,
                    y,
                    y_mark,
                    y_mask,
                    exp_stage="test",
                )

            pred = output["pred"]
            output_mask = output["mask"]
            if tuple(pred.shape) != tuple(y.shape):
                raise AssertionError(
                    f"{model_name}/{cohort} output shape {tuple(pred.shape)} != {tuple(y.shape)}"
                )
            if output_mask.shape != y_mask.shape or not torch.equal(output_mask, y_mask):
                raise AssertionError(f"{model_name}/{cohort} returned a mismatched query mask")
            finite_count = int(torch.isfinite(pred[y_mask.bool()]).sum().item())
            requested_count = int(y_mask.sum().item())
            if finite_count != requested_count:
                raise AssertionError(
                    f"{model_name}/{cohort}: only {finite_count}/{requested_count} "
                    "requested outputs are finite"
                )

            checks.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    "configuration": str(config_path.relative_to(ROOT)),
                    "configuration_sha256": sha256(config_path),
                    "seq_len_hours": INPUT_HOURS,
                    "pred_len_hours": FORECAST_HOURS,
                    "synthetic_input_rows": int(x.shape[1]),
                    "fixed_grid_query_count": FORECAST_HOURS,
                    "off_grid_query_count": len(OFF_GRID_HOURS),
                    "boundary_query_included": True,
                    "query_time_count": len(query_hours),
                    "requested_component_outputs": requested_count,
                    "finite_component_outputs": finite_count,
                    "output_shape_matches_query_tensor": True,
                    "output_mask_matches_query_mask": True,
                    "status": "PASS",
                }
            )

    return {
        "schema_version": "p1_landmark_model_query_feasibility_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_commit": git_value(
            "log", "-1", "--format=%H", "--", "paper/预设用例分析计划_P1.md"
        ),
        "current_commit": git_value("rev-parse", "HEAD"),
        "analysis_plan_sha256": sha256(ROOT / "paper/预设用例分析计划_P1.md"),
        "checker_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "checker_script_sha256": sha256(Path(__file__).resolve()),
        "model_code_sha256": {
            "CarryGate adapter": sha256(PYOMNITS / "models/ICUKANForecaster.py"),
            "CarryGate backbone": sha256(ROOT / "src/models/backbones/strats_lite.py"),
            "GraFITi": sha256(PYOMNITS / "models/GraFITi.py"),
            "GraFITi encoder": sha256(PYOMNITS / "layers/GraFITi/GraFITi_layers.py"),
            "HyperIMTS": sha256(PYOMNITS / "models/HyperIMTS.py"),
        },
        "device": "cpu",
        "training_started": False,
        "checkpoint_loaded": False,
        "test_data_or_labels_used": False,
        "input_coordinate_hours": "[0,48), normalized by 72",
        "outcome_query_coordinate_hours": "[48,72), normalized by 72",
        "fixed_grid_hours": [48.5 + i for i in range(24)],
        "additional_query_hours": [48.0, *OFF_GRID_HOURS],
        "model_sequence_limit_review": {
            "CarryGate": "Continuous event triplets; STraTS encoder has no positional-index table limit.",
            "GraFITi": "Dynamic event and query tensors; continuous time embedding; no fixed query grid or positional-index table limit.",
            "HyperIMTS": "Dynamic event and query tensors; continuous time marks; time_length sets attention scale and is not a hard sequence cap.",
            "task_window": "The 48-hour input equals configured seq_len=48; output is pred_len=24.",
        },
        "existing_dataset_adapter_requirement": {
            "status": "P1-specific adapter required before training",
            "evidence": "external/PyOmniTS/data/data_provider/datasets/ICUKAN.py fixes L_TOTAL_HOURS=48; tsdmDataset._check_lengths rejects seq_len+pred_len > L_TOTAL. P1's 48+24=72 coordinate system therefore needs a dedicated adapter that maps u=48+(s-t) and normalizes by 72.",
            "model_architecture_changes": False,
        },
        "checks": checks,
        "summary": {
            "passed": sum(check["status"] == "PASS" for check in checks),
            "total": len(checks),
            "status": "PASS" if len(checks) == 6 else "FAIL",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/jbi/p1_landmark_model_query_feasibility.json",
    )
    args = parser.parse_args()
    report = run_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"wrote {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")


if __name__ == "__main__":
    main()
