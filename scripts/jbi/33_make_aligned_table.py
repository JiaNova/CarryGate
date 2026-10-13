#!/usr/bin/env python
"""Emit the Track B tables: the full factorial (S1) and the aligned-objective
architecture comparison (S4).

The Results section used to enumerate every macro skill, paired difference and
TOST verdict in prose, twice over, once per cohort.  That is table material:
the numbers are a grid of cohort by model by quantity, and a reader comparing
GraFITi across cohorts should not have to hold two paragraphs in their head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from output_safety import prepare_generation

ROOT = Path(__file__).resolve().parents[2]
JBI = ROOT / "results" / "jbi"
OUT = ROOT / "paper" / "tables" / "T8_aligned_architecture.tex"
FACTORIAL_OUT = ROOT / "paper" / "tables" / "T7_factorial.tex"
FACTORIAL = (("eICU-CRD", "trackb_controls"), ("MIMIC-IV", "trackb_controls_mimiciv"))
OBJECTIVES = (("mse", "MSE"), ("balanced", "VariableBalancedMSE"),
              ("mae", "masked L1 (aligned)"))
COHORTS = (("eICU-CRD", "e2_trackb_adapter_vs_external_paired_mae.csv", "e6_tost_eicu_mae.csv"),
           ("MIMIC-IV", "e2_trackb_adapter_vs_external_paired_mae_mimiciv.csv",
            "e6_tost_mimiciv_mae.csv"))
DISPLAY = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi",
           "HyperIMTS": "HyperIMTS", "Warpformer": "Warpformer"}
ORDER = ("ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer")


def make_factorial(output_dir: Path | None = None) -> None:
    """Every model x objective cell under the macro checkpoint criterion."""
    output_dir = output_dir or OUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    head = " & ".join(DISPLAY[m] for m in ORDER)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\caption{Complete patient-grouped Track B factorial under the macro checkpoint "
        r"criterion. Each cell is the five-seed patient-cluster macro persistence skill; "
        r"CarryGate is trained inside the same harness as the external models. "
        r"The aligned-objective comparison is the masked-L1 row; MSE and "
        r"VariableBalancedMSE are sensitivity arms. No model or cohort cell is "
        r"omitted.}",
        r"\label{tab:t7}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        f"cohort & objective & {head} \\\\",
        r"\midrule",
    ]
    for label, folder in FACTORIAL:
        summary = pd.read_csv(JBI / folder / "e2_trackb_controls_summary.csv")
        summary = summary[summary.phase == "factorial"]
        macro = summary.set_index(["model", "cell"]).macro_mean
        assert macro.index.is_unique, folder
        for k, (loss, name) in enumerate(OBJECTIVES):
            cells = " & ".join(f"{macro.loc[(m, f'{loss}_macro')]:.4f}" for m in ORDER)
            lines.append(f"{label if k == 0 else ''} & {name} & {cells} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    output_path = output_dir / FACTORIAL_OUT.name
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def make_aligned(output_dir: Path | None = None) -> None:
    output_dir = output_dir or OUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{\textbf{Architecture comparison at the aligned objective.} All arms "
        r"trained with masked L1 in the Track B harness on patient-grouped splits, "
        r"five seeds each. Differences are CarryGate minus external from a paired "
        r"bootstrap over the same patients (2{,}000 resamples), with 95\% intervals; "
        r"the TOST uses the 90\% interval against the "
        r"pre-specified margins $\delta_1=0.010$ and $\delta_2=0.020$; the eICU test is "
        r"post hoc and the MIMIC-IV test is prospective under the frozen analysis "
        r"order.}\label{tab:aligned}",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Cohort & Arm & Macro skill & Paired difference (95\% CI) & "
        r"$\delta_1$ & $\delta_2$ \\",
        r"\midrule",
    ]
    for label, paired_name, tost_name in COHORTS:
        paired = pd.read_csv(JBI / paired_name).set_index("model")
        tost = pd.read_csv(JBI / tost_name)
        for k, model in enumerate(ORDER):
            row = paired.loc[model]
            cohort_cell = label if k == 0 else ""
            if model == "ICUKANForecaster":
                diff, d1, d2 = r"\textemdash", r"\textemdash", r"\textemdash"
            else:
                diff = (f"${row.paired_macro_difference:+.4f}$ "
                        f"(${row.paired_macro_ci_low:+.4f}$, ${row.paired_macro_ci_high:+.4f}$)")
                verdict = {float(r.delta): bool(r.tost_pass)
                           for r in tost[tost.model == model].itertuples()}
                d1 = r"\checkmark" if verdict.get(0.010) else "--"
                d2 = r"\checkmark" if verdict.get(0.020) else "--"
            lines.append(f"{cohort_cell} & {DISPLAY[model]} & {row.macro:.4f} & "
                         f"{diff} & {d1} & {d2} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    output_path = output_dir / OUT.name
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive existing Track B tables before replacing them")
    args = parser.parse_args()
    prepare_generation(ROOT, (OUT, FACTORIAL_OUT), refresh=args.refresh_generated,
                       label="trackb_tables")
    make_factorial()
    make_aligned()


if __name__ == "__main__":
    main()
