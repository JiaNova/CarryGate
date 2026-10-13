#!/usr/bin/env python
"""Generate Supplementary Table T9 from pre/post GCS grading results.

The historical CSVs are read from their last pre-correction commits. The
corrected D3 output is kept under a new filename so the archived result is
never overwritten. Rebuild that corrected D3 CSV first with
``scripts/analyze_sofa_grades.py --output .../gcs_truth_projected/sofa_grades.csv``.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.style import display  # noqa: E402
from output_safety import prepare_generation  # noqa: E402

OLD_E8_COMMIT = "39e3d8c"
OLD_D3_COMMIT = "b2e097e"
OLD_E8_PRIMARY_PATH = "results/jbi/e8_objective_effect_on_grading.csv"
OLD_E8_PATH = "results/jbi/e8_objective_effect_on_grading_all.csv"
OLD_D3_PATH = "results/d3_clinical/sofa_grades.csv"
NEW_E8_PATH = ROOT / "results/jbi/e8_objective_effect_on_grading_all.csv"
NEW_E8_PRIMARY_PATH = ROOT / "results/jbi/e8_objective_effect_on_grading.csv"
NEW_D3_PATH = ROOT / "results/d3_clinical/gcs_truth_projected/sofa_grades.csv"
OUTPUT = ROOT / "paper/tables/T9_gcs_correction.tex"


def git_csv(commit: str, path: str) -> pd.DataFrame:
    history_root = os.environ.get("ICU_KAN_GCS_HISTORY_ROOT")
    if history_root:
        historical_path = Path(history_root) / path
        if not historical_path.is_file():
            raise FileNotFoundError(
                f"historical GCS input not found under ICU_KAN_GCS_HISTORY_ROOT: {path}"
            )
        raw = historical_path.read_bytes()
    else:
        raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw))


def tex_escape(text: object) -> str:
    value = str(text)
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def fmt_percent(value: object) -> str:
    return f"{100 * float(value):.2f}\\%"


def fmt_pp(value: object) -> str:
    return f"{float(value):+.2f}"


def arrow(before: str, after: str) -> str:
    return f"{before} $\\rightarrow$ {after}"


def _e8_rows() -> tuple[list[list[str]], list[str]]:
    before = git_csv(OLD_E8_COMMIT, OLD_E8_PATH)
    after = pd.read_csv(NEW_E8_PATH)
    keys = ["cohort", "model", "organ", "variable"]
    old = before[(before.cohort == "eicu") & (before.variable == "GCS")]
    new = after[(after.cohort == "eicu") & (after.variable == "GCS")]
    joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    expected_models = {"ICUKANForecaster", "GraFITi", "HyperIMTS", "Warpformer"}
    if len(joined) != 4 or set(joined.model) != expected_models:
        raise ValueError("P0-A before/after inputs must contain the four eICU GCS rows")
    rows = []
    order = {"ICUKANForecaster": 0, "GraFITi": 1, "HyperIMTS": 2, "Warpformer": 3}
    for row in joined.assign(_order=joined.model.map(order)).sort_values("_order").itertuples():
        old_delta = float(row.delta_pp_old)
        new_delta = float(row.delta_pp_new)
        rows.append([
            tex_escape(display(row.model)),
            arrow(fmt_percent(row.misclassification_mse_old),
                  fmt_percent(row.misclassification_mse_new)),
            arrow(fmt_percent(row.misclassification_mae_old),
                  fmt_percent(row.misclassification_mae_new)),
            arrow(fmt_pp(old_delta), fmt_pp(new_delta)) + " pp",
            "Yes" if (old_delta > 0) != (new_delta > 0) else "No",
        ])
    old_primary = git_csv(OLD_E8_COMMIT, OLD_E8_PRIMARY_PATH)
    new_primary = pd.read_csv(NEW_E8_PRIMARY_PATH)
    old_positive = int((old_primary.delta_pp > 0).sum())
    new_positive = int((new_primary.delta_pp > 0).sum())
    if (old_positive, len(old_primary), new_positive, len(new_primary)) != (21, 24, 24, 24):
        raise ValueError(
            "P0-A scope differs from the audited ISSUE-001 transition: "
            f"{old_positive}/{len(old_primary)} -> {new_positive}/{len(new_primary)}"
        )
    old_ci = int(((old_primary.delta_ci_low > 0) | (old_primary.delta_ci_high < 0)).sum())
    new_ci = int(((new_primary.delta_ci_low > 0) | (new_primary.delta_ci_high < 0)).sum())
    summary = (
        rf"\multicolumn{{5}}{{l}}{{P0-A primary cells with positive $\Delta_{{pp}}$: "
        rf"{old_positive}/{len(old_primary)} $\rightarrow$ {new_positive}/{len(new_primary)}; "
        rf"95\% CIs excluding zero: {old_ci}/{len(old_primary)} $\rightarrow$ "
        rf"{new_ci}/{len(new_primary)}.}}"
    )
    return rows, [summary]


def _d3_rows() -> tuple[list[list[str]], pd.DataFrame]:
    before = git_csv(OLD_D3_COMMIT, OLD_D3_PATH)
    if not NEW_D3_PATH.exists():
        raise FileNotFoundError(
            f"corrected D3 result missing; run analyze_sofa_grades.py --output {NEW_D3_PATH.relative_to(ROOT)}"
        )
    after = pd.read_csv(NEW_D3_PATH)
    keys = ["cohort", "organ", "variable", "model"]
    old_gcs = before[before.variable == "GCS"]
    new_gcs = after[after.variable == "GCS"]
    joined = old_gcs.merge(new_gcs, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(joined) != 30:
        raise ValueError(f"expected 30 historical D3 GCS cells, found {len(joined)}")
    changed_mask = (
        ((joined.misclassification_old - joined.misclassification_new).abs() > 1e-12)
        | ((joined.excess_over_carryforward_old
            - joined.excess_over_carryforward_new).abs() > 1e-12)
    )
    changed = joined[changed_mask].copy()
    changed = changed.sort_values(["cohort", "model"])
    rows = []
    for row in changed.itertuples(index=False):
        old_excess = float(row.excess_over_carryforward_old)
        new_excess = float(row.excess_over_carryforward_new)
        rows.append([
            tex_escape({"eicu": "eICU", "mimiciv": "MIMIC-IV", "p12": "P12"}.get(row.cohort, row.cohort)),
            tex_escape(display(row.model)),
            arrow(fmt_percent(row.misclassification_old), fmt_percent(row.misclassification_new)),
            arrow(fmt_pp(100 * old_excess), fmt_pp(100 * new_excess)) + " pp",
            "Yes" if (old_excess > 0) != (new_excess > 0) else "No",
        ])
    return rows, after


def _scope_summary(corrected_d3: pd.DataFrame) -> str:
    external = corrected_d3[~corrected_d3.model.isin({"This work", "ICUKAN_Naive"})]
    own = corrected_d3[corrected_d3.model == "This work"]
    n_positive = int((external.excess_over_carryforward > 0).sum())
    n_ci_positive = int((external.ci_lo > 0).sum())
    own_excludes = own[(own.ci_lo > 0) | (own.ci_hi < 0)]
    own_favourable = int((own_excludes.excess_over_carryforward < 0).sum())
    own_adverse = int((own_excludes.excess_over_carryforward > 0).sum())
    maximum = external.loc[external.excess_over_carryforward.idxmax()]
    max_pp = 100 * float(maximum.excess_over_carryforward)
    unchanged = (
        len(external) == 96 and n_positive == 96 and n_ci_positive == 89
        and len(own) == 12 and len(own_excludes) == 3
        and own_favourable == 1 and own_adverse == 2
        and abs(max_pp - 55.5) <= 0.1
        and maximum.model == "NeuralFlows" and maximum.organ == "Liver"
        and maximum.cohort == "mimiciv"
    )
    status = "Conclusions unchanged outside the corrected P0-A cells" if unchanged else "Broader conclusions to update after rescoring"
    cohort = {"eicu": "eICU", "mimiciv": "MIMIC-IV", "p12": "P12"}.get(
        str(maximum.cohort), str(maximum.cohort)
    )
    return (
        rf"\multicolumn{{5}}{{p{{0.96\linewidth}}}}{{\textit{{{status}:}} "
        rf"rescore of the corrected D3 file gives external models with positive point excess in "
        rf"{n_positive}/{len(external)} cells and lower confidence bounds above zero in "
        rf"{n_ci_positive}/{len(external)}. CarryGate has {len(own_excludes)}/{len(own)} "
        rf"intervals excluding zero ({own_favourable} favourable, {own_adverse} adverse). "
        rf"The largest excess is {max_pp:.2f} pp ({display(maximum.model)}, "
        rf"{tex_escape(maximum.organ)}, {tex_escape(cohort)}). These claims were listed as "
        rf"unaffected in ISSUE-001 and were recomputed from the corrected output.}}"
    )


def build_tex() -> str:
    e8, e8_summary = _e8_rows()
    d3, corrected_d3 = _d3_rows()
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering\scriptsize",
        r"\caption{GCS grading correction: estimates before and after projecting observed values onto the integer support. P0-A values are from the patient-grouped objective comparison; D3 values are the historical stay-level analysis. All corrections were post hoc.}",
        r"\label{tab:gcs-correction}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textbf{P0-A: eICU GCS cells (primary models plus full-table Warpformer)}}\\",
        r"Model & MSE misclassification & MAE misclassification & $\Delta_{pp}$ & Sign changed?\\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r"\\" for row in e8)
    lines.extend(line + r"\\" for line in e8_summary)
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Historical D3: GCS before/after point-estimate check}}\\",
        r"Cohort & Model & Misclassification & Excess over carry-forward & Sign changed?\\",
        r"\midrule",
    ])
    if d3:
        lines.extend(" & ".join(row) + r"\\" for row in d3)
    else:
        lines.append(r"None & \multicolumn{4}{l}{No point-estimate cell changed in the archived D3 output.}\\")
    lines.extend([
        r"\midrule",
        _scope_summary(corrected_d3) + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
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
                        help="archive the current GCS correction table before replacing it")
    args = parser.parse_args()
    prepare_generation(ROOT, (OUTPUT,), refresh=args.refresh_generated,
                       label="gcs_correction_table")
    write_tex()
    print(OUTPUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
