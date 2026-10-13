#!/usr/bin/env python
"""Build the paper assets for the external-baseline comparison and the capacity control.

Tables and the figure are generated from the run CSVs so that the numbers in the
manuscript cannot drift from the numbers in the matrices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from style import COHORT_LABEL, MODEL_NAME, display  # noqa: E402
from output_safety import prepare_generation  # noqa: E402
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "results" / "paper"
COHORTS = ["p12", "mimiciv", "eicu"]
PRETTY = COHORT_LABEL
LEVELS = ["C1", "C2", "C3", "C4", "C5"]


def table_external(output_dir: Path | None = None) -> None:
    """Table 5: every arm on both estimators, plus the per-variable win count."""
    output_dir = output_dir or TAB
    output_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "results" / "d1_external" / "d1_both_metrics.csv")
    d["model"] = d.model.map(display)
    agg = d.groupby(["model", "cohort"]).agg(
        macro=("macro", "mean"), pooled=("pooled", "mean"),
        below=("n_below", "mean"), n_vars=("n_vars", "max"))
    year = {"HyperIMTS": "2025", "GraFITi": "2024", "Warpformer": "2023",
            "APN": "2026", "t-PatchGNN": "2024", "GRU-D": "2018",
            "mTAN": "2021", "Neural Flows": "2021"}

    lines = [r"\begin{table}[htbp]\centering\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{" + MODEL_NAME + r" against eight published models, each run from its "
             r"authors' released code through one harness (PyOmniTS, commit "
             r"\texttt{47d8a2f}) on identical target events, five seeds per arm (fifteen "
             r"for " + MODEL_NAME + r"). eICU-CRD and MIMIC-IV use original stay-level "
             r"splits; PhysioNet-2012 uses its official record-level split because patient "
             r"identifiers are unavailable. \emph{macro} is the original primary metric, the "
             r"unweighted mean of per-variable persistence skill; \emph{pool} is the "
             r"event-weighted ratio; zero is carry-forward under both. \emph{blw} counts "
             r"the variables on which the model is worse than carry-forward, of 34 (32 on "
             r"eICU-CRD). Rows are ordered by the original metric on MIMIC-IV. The two "
             r"estimators do not agree on the ordering.}\label{tab:t5}",
             r"\begin{tabular}{l r rrr rrr rrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{PhysioNet-2012} & \multicolumn{3}{c}{MIMIC-IV}"
             r" & \multicolumn{3}{c}{" + PRETTY["eicu"] + r"} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"model & yr & macro & pool & blw & macro & pool & blw"
             r" & macro & pool & blw \\", r"\midrule"]
    order = sorted(agg.index.get_level_values(0).unique(),
                   key=lambda m: -agg.loc[(m, "mimiciv"), "macro"])
    for name in order:
        cells = []
        for c in COHORTS:
            r = agg.loc[(name, c)]
            cells += [f"${r.macro:+.3f}$", f"${r.pooled:+.3f}$", f"{r.below:.1f}"]
        label = r"\textbf{" + name + "}" if name == MODEL_NAME else name
        lines.append(f"{label} & {year.get(name, '---')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "T5_external_baselines.tex").write_text("\n".join(lines))
    print("wrote T5")


def table_capacity(output_dir: Path | None = None) -> None:
    output_dir = output_dir or TAB
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "results" / "d2_capacity" / "d2_1_matrix.csv")
    probe = pd.read_csv(ROOT / "results" / "d2_capacity" / "d2_probe_verdicts.csv")
    params = df.groupby("level").n_params.max().reindex(LEVELS)

    lines = [r"\begin{table}[htbp]\centering\scriptsize",
             r"\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Capacity control on the original stay-level splits; PhysioNet-2012 "
             r"is record-level because patient identifiers are unavailable. Five backbone sizes spanning a 41-fold "
             r"parameter range, each with its own learning rate selected on validation "
             r"data before the matrix was run. \emph{Skill} is the MLP arm, mean over "
             r"five seeds. \emph{Headroom} is the Gate~P statistic "
             r"$\mathrm{val}(\text{linear})-\mathrm{val}(\text{MLP})$ and \emph{2sd} is "
             r"its prespecified threshold; the position is discriminative only where "
             r"headroom exceeds the threshold, which happens in none of the fifteen "
             r"cells.}\label{tab:t6}",
             r"\begin{tabular}{lrr" + "rrr" * 3 + r"}", r"\toprule",
             r" & & & \multicolumn{3}{c}{PhysioNet-2012} & \multicolumn{3}{c}{MIMIC-IV}"
             r" & \multicolumn{3}{c}{" + PRETTY["eicu"] + r"} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
             r"level & params & lr & skill & headroom & 2sd & skill & headroom & 2sd"
             r" & skill & headroom & 2sd \\", r"\midrule"]
    import json
    lrs = json.loads((ROOT / "results" / "d2_capacity" / "d2_0_lr_gate.json").read_text())
    lrs = lrs["learning_rate_by_level"]
    for lv in LEVELS:
        cells = []
        for c in COHORTS:
            sk = df[(df.cohort == c) & (df.level == lv) & (df.encoder == "mlp")].skill.mean()
            p = probe[(probe.cohort == c) & (probe.level == lv)].iloc[0]
            cells += [f"{sk:.4f}", f"{p.headroom:+.5f}", f"{p.threshold:.5f}"]
        tag = r"\textbf{" + lv + r"}" if lv == "C2" else lv
        lines.append(f"{tag} & {params[lv]:,} & {lrs[lv]:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output_dir / "T6_capacity_control.tex").write_text("\n".join(lines))
    print("wrote T6")


def figure_capacity() -> None:
    df = pd.read_csv(ROOT / "results" / "d2_capacity" / "d2_1_matrix.csv")
    d1 = pd.read_csv(ROOT / "results" / "d1_external" / "d1c_matrix.csv")
    params = df.groupby("level").n_params.max().reindex(LEVELS)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, c in zip(axes, COHORTS):
        sub = df[df.cohort == c]
        for enc, style in [("identity", "o-"), ("linear", "s--"), ("mlp", "^-")]:
            g = sub[sub.encoder == enc].groupby("level").skill.agg(["mean", "std"]).reindex(LEVELS)
            ax.errorbar(params.values, g["mean"], yerr=g["std"], fmt=style,
                        capsize=2, ms=4, lw=1.2, label=enc)
        best = d1[d1.cohort == c].groupby("model").persistence_skill.mean().max()
        ax.axhline(best, color="crimson", lw=1.0, ls=":")
        ax.text(params.values[0], best, " best external baseline", color="crimson",
                va="bottom", ha="left", fontsize=7)
        ax.set_xscale("log")
        ax.set_title(PRETTY[c], fontsize=10)
        ax.set_xlabel("backbone parameters")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("persistence skill")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "F7_capacity_control.pdf", bbox_inches="tight")
    print("wrote F7")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the existing tables and capacity figure before replacing them")
    args = parser.parse_args()
    prepare_generation(ROOT, (TAB / "T5_external_baselines.tex",
                              TAB / "T6_capacity_control.tex",
                              FIG / "F7_capacity_control.pdf"),
                       refresh=args.refresh_generated, label="baseline_capacity_assets")
    TAB.mkdir(parents=True, exist_ok=True)
    table_external()
    table_capacity()
    figure_capacity()
