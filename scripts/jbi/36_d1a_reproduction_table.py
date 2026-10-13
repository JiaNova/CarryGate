#!/usr/bin/env python
"""Build the D1a fidelity table from the immutable reproduction ledger."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.style import display  # noqa: E402
from output_safety import prepare_generation  # noqa: E402

CSV = ROOT / "results/d1_external/d1a_reproduction.csv"
RAW = ROOT / "results/d1_external/d1a_recheck_seed_metrics.csv"
OUTPUT = ROOT / "paper/tables/T10_d1a_reproduction.tex"

REGISTERED = ["GRU_D", "tPatchGNN", "GraFITi", "APN"]
SUPPLEMENTARY = ["Warpformer", "mTAN", "NeuralFlows"]
PUBLISHED = {
    "GRU_D": (0.3419, 0.0029, 0.3992, 0.0011),
    "tPatchGNN": (0.3133, 0.0053, 0.3697, 0.0049),
    "GraFITi": (0.3075, 0.0015, 0.3637, 0.0036),
    "APN": (0.3093, 0.0011, 0.3650, 0.0026),
    "Warpformer": (0.3056, 0.0011, 0.3661, 0.0016),
    "mTAN": (0.3809, 0.0043, 0.4291, 0.0035),
    "NeuralFlows": (0.4056, 0.0033, 0.4466, 0.0027),
}


def _nonfinite_selection_runs() -> dict[str, tuple[int, int]]:
    """Return models with seeds whose official pooled-MSE selector was invalid."""
    if not RAW.exists():
        return {}
    raw = pd.read_csv(RAW)
    affected: dict[str, tuple[int, int]] = {}
    for model in SUPPLEMENTARY:
        runs = raw[raw.model == model]
        nonfinite = 0
        for _, row in runs.iterrows():
            training_path = ROOT / str(row.checkpoint_dir) / "training_summary.json"
            if not training_path.is_file():
                raise FileNotFoundError(training_path)
            training = json.loads(training_path.read_text(encoding="utf-8"))
            stages = training.get("stages", [])
            if not stages:
                raise ValueError(f"{model} seed {row.seed}: no training stages in summary")
            values = [stage.get("best_selection_value") for stage in stages]
            if any(value is None or not math.isfinite(float(value)) for value in values):
                nonfinite += 1
        if nonfinite:
            affected[model] = (nonfinite, len(runs))
    return affected


def _metric(mean: float | None, sd: float | None) -> str:
    if mean is None or sd is None:
        return "--"
    return f"{mean:.4f} $\\pm$ {sd:.4f}"


def _rows() -> tuple[list[list[str]], list[list[str]]]:
    data = pd.read_csv(CSV)
    selection_caveats = _nonfinite_selection_runs()
    if data.model.duplicated().any():
        raise ValueError("d1a_reproduction.csv contains duplicate model rows")
    indexed = data.set_index("model", drop=False)
    missing_original = sorted(set(REGISTERED) - set(indexed.index))
    if missing_original:
        raise ValueError(f"original D1a rows missing: {missing_original}")

    groups: list[list[list[str]]] = [[], []]
    for group_index, models in enumerate((REGISTERED, SUPPLEMENTARY)):
        for model in models:
            ref_mse, ref_mse_sd, ref_mae, ref_mae_sd = PUBLISHED[model]
            if model not in indexed.index:
                groups[group_index].append([
                    display(model),
                    "Original" if group_index == 0 else "Added 2026-10",
                    "--", _metric(ref_mse, ref_mse_sd), "--",
                    _metric(ref_mae, ref_mae_sd),
                    "--" if group_index == 0 else "Pending",
                ])
                continue

            row = indexed.loc[model]
            for column, expected in (
                ("mse_published", ref_mse),
                ("mse_published_std", ref_mse_sd),
                ("mae_published", ref_mae),
                ("mae_published_std", ref_mae_sd),
            ):
                if abs(float(row[column]) - expected) > 1e-12:
                    raise ValueError(f"{model}: {column} differs from the frozen published value")
            verdict = str(row["mse_verdict"])
            verdict = "PASS" if verdict == "PASS" else "NOT REPRODUCED"
            if verdict == "PASS" and model in selection_caveats:
                verdict += r"$^{\dagger}$"
            groups[group_index].append([
                display(model),
                "Original" if group_index == 0 else "Added 2026-10",
                _metric(float(row.mse_mean), float(row.mse_std)),
                _metric(ref_mse, ref_mse_sd),
                _metric(float(row.mae_mean), float(row.mae_std)),
                _metric(ref_mae, ref_mae_sd),
                verdict,
            ])
    unexpected = sorted(set(indexed.index) - set(REGISTERED) - set(SUPPLEMENTARY))
    if unexpected:
        raise ValueError(f"unexpected D1a model rows: {unexpected}")
    return groups[0], groups[1]


def build_tex() -> str:
    registered, supplementary = _rows()
    selection_caveats = _nonfinite_selection_runs()
    has_pending = any(row[-1] == "Pending" for row in registered + supplementary)
    caption = (
        r"D1a reproduction fidelity on P12 with the official 36-to-3-hour scripts "
        r"on the PhysioNet record-level split (patient identifiers unavailable). "
        r"The original four-model gate was fixed before execution; three rows are supplementary "
        r"checks added in October 2026. Values are mean $\pm$ SD over five runs. "
        r"Published references are from the PhysioNet column of APN (AAAI'26), Table 2. "
        r"The MSE gate is $|\bar{x}-\mu_{pub}|\leq\max(3\sigma_{pub},0.05|\mu_{pub}|)$; "
        r"MAE is descriptive."
    )
    if has_pending:
        caption += " Pending rows indicate runs not yet completed."
    if selection_caveats:
        caveat_models = ", ".join(
            f"{display(model)} ({nonfinite}/{total} seeds)"
            for model, (nonfinite, total) in selection_caveats.items()
        )
        caption += (
            r" A dagger marks models with non-finite official pooled-MSE validation "
            r"selection (" + caveat_models + r"); their numerical MSE passes do not "
            r"establish valid validation-selected reproductions."
        )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{" + caption + "}",
        r"\label{tab:d1a-repro}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Model & Scope & MSE reproduced & MSE published & MAE reproduced & MAE published & MSE gate\\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Original D1a fidelity sample}}\\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r"\\" for row in registered)
    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Supplementary checks; do not change the original gate}}\\",
        r"\midrule",
    ])
    lines.extend(" & ".join(row) + r"\\" for row in supplementary)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def write_tex(output_path: Path = OUTPUT) -> Path:
    content = build_tex()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"wrote {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the current D1a table before replacing it")
    args = parser.parse_args()
    prepare_generation(ROOT, (OUTPUT,), refresh=args.refresh_generated,
                       label="d1a_reproduction_table")
    write_tex()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
