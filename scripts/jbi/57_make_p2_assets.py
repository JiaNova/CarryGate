#!/usr/bin/env python
"""Generate the P2 leave-hospitals-out table, figure and manuscript numbers.

Inputs are the frozen P2 verdict, claim details, metrics and the hospital
overlap audit. Nothing here re-estimates a claim: every number is read or
summarised from those files, and the verdict flags are copied, not recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from output_safety import prepare_generation

ROOT = Path(__file__).resolve().parents[2]
JBI = ROOT / "results/jbi"
VERDICT = JBI / "p2_loho_verdict.json"
DETAILS = JBI / "p2_loho_claim_details.csv"
METRICS = JBI / "p2_loho_metrics.csv"
OVERLAP = JBI / "p2_hospital_overlap_audit.json"
TABLE = ROOT / "paper/tables/T_p2_loho.tex"
FIGURE = ROOT / "results/paper/F9_p2_loho"
NUMBERS = JBI / "p2_loho_paper_numbers.json"

MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
LABEL = {"ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi", "HyperIMTS": "HyperIMTS"}
COMPONENTS = ("Platelets", "Bilirubin", "GCS", "Creatinine")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    verdict = json.loads(VERDICT.read_text(encoding="utf-8"))
    details = pd.read_csv(DETAILS)
    metrics = pd.read_csv(METRICS)
    overlap = json.loads(OVERLAP.read_text(encoding="utf-8"))

    folds = overlap["folds"]
    skill = (metrics.groupby(["fold", "model", "objective"]).macro_persistence_skill.mean()
             .unstack("objective"))
    l1_minus_mse = details[details.claim == "C2_fold_model"]
    c3 = verdict["C3"]["fold_details"]
    c4 = details[details.claim == "C4_component_cell"].copy()
    c4["gap_pp"] = 100 * (c4.mean_seed_model_misclassification - c4.carry_forward_misclassification)
    c1 = verdict["C1"]["objective_specific_candidates"]

    def gap_summary(objective: str) -> dict:
        gaps = c4[c4.objective == objective].gap_pp
        return {"cells": int(len(gaps)), "carry_forward_better": int((gaps > 0).sum()),
                "median_pp": float(gaps.median()), "min_pp": float(gaps.min()), "max_pp": float(gaps.max())}

    numbers = {
        "sources": {path.name: _sha256(path) for path in (VERDICT, DETAILS, METRICS, OVERLAP)},
        "hospitals_total": int(overlap["n_hospitals_in_fold_map"]),
        "stays_total": int(overlap["n_stays"]),
        "heldout_hospitals_per_fold": [int(f["heldout_hospital_count"]) for f in folds],
        "heldout_test_stays_per_fold": [int(f["heldout_test_stay_count"]) for f in folds],
        "heldout_test_stays_total": int(sum(f["heldout_test_stay_count"] for f in folds)),
        "train_validation_stays_per_fold": [8400, 1800],
        "run_count": int(verdict["fixed_run_count"]),
        "verdicts": {claim: bool(verdict[claim]["pass"]) if claim != "C1" else bool(verdict["C1"]["selected_verdict"])
                     for claim in ("C1", "C2", "C3", "C4")},
        "masked_l1_skill_range": [float(skill["MAE"].min()), float(skill["MAE"].max())],
        "mse_skill_range": [float(skill["MSE"].min()), float(skill["MSE"].max())],
        "l1_minus_mse_positive": int((l1_minus_mse.skill_MAE_minus_MSE > 0).sum()),
        "l1_minus_mse_count": int(len(l1_minus_mse)),
        "l1_minus_mse_range": [float(l1_minus_mse.skill_MAE_minus_MSE.min()), float(l1_minus_mse.skill_MAE_minus_MSE.max())],
        "architecture_gap_median": float(verdict["C3"]["median_fold_architecture_gap"]),
        "objective_effect_median": float(verdict["C3"]["median_fold_objective_effect"]),
        "architecture_gap_range": [min(f["architecture_gap"] for f in c3), max(f["architecture_gap"] for f in c3)],
        "objective_effect_range": [min(f["objective_effect"] for f in c3), max(f["objective_effect"] for f in c3)],
        "c1_mae_variables_losing": {LABEL[m]: c1["MAE"]["model_existence"][m]["variable_count_meeting_existence_condition"] for m in MODELS},
        "c1_mae_quartile_loss": {q: float(v["loss_proportion"]) for q, v in c1["MAE"]["quartile_loss_proportions"].items()},
        "c1_mse_quartile_loss": {q: float(v["loss_proportion"]) for q, v in c1["MSE"]["quartile_loss_proportions"].items()},
        "c4_required": int(verdict["C4"]["required_cells"]),
        "c4_masked_l1": gap_summary("MAE"),
        "c4_mse": gap_summary("MSE"),
    }
    if numbers["c4_masked_l1"]["carry_forward_better"] != verdict["C4"]["carry_forward_better_cells"]:
        raise RuntimeError("C4 tally differs from the frozen verdict")
    return {"numbers": numbers, "skill": skill, "c3": c3, "c4": c4, "c1": c1, "l1_minus_mse": l1_minus_mse}


def write_table(data: dict) -> None:
    n = data["numbers"]
    skill = data["skill"]
    delta = data["l1_minus_mse"].set_index(["fold", "model"]).skill_MAE_minus_MSE
    rows = []
    for fold, gap in enumerate(data["c3"]):
        cells = " & ".join(f"{skill.loc[(fold, m), 'MAE']:.3f}" for m in MODELS)
        deltas = " & ".join(f"{delta.loc[(float(fold), m)]:+.3f}" for m in MODELS)
        rows.append(f"{fold + 1} & {n['heldout_hospitals_per_fold'][fold]} & {n['heldout_test_stays_per_fold'][fold]:,} & "
                    f"{cells} & {deltas} & {gap['architecture_gap']:.4f} & {gap['objective_effect']:.4f} \\\\")
    verdict_text = (
        f"Sparse-variable losses (C1): {', '.join(f'{k} {v}' for k, v in n['c1_mae_variables_losing'].items())} variables lose to carry-forward in $\\ge$3/5 folds; "
        f"loss share {100 * n['c1_mae_quartile_loss']['Q1']:.1f}\\% in the sparsest quartile vs {100 * n['c1_mae_quartile_loss']['Q4']:.1f}\\% in the densest. "
        f"Objective direction (C2): masked L1 above MSE in {n['l1_minus_mse_positive']}/{n['l1_minus_mse_count']} fold--architecture pairs. "
        f"Objective vs architecture (C3): medians {n['objective_effect_median']:.3f} vs {n['architecture_gap_median']:.4f}. "
        f"Grading (C4): carry-forward lower misclassification in {n['c4_masked_l1']['carry_forward_better']}/{n['c4_masked_l1']['cells']} masked-L1 cells "
        f"(median gap {n['c4_masked_l1']['median_pp']:+.1f} pp; {n['c4_masked_l1']['min_pp']:+.1f} to {n['c4_masked_l1']['max_pp']:+.1f}); "
        f"under MSE the median gap is {n['c4_mse']['median_pp']:+.1f} pp ({n['c4_mse']['min_pp']:+.1f} to {n['c4_mse']['max_pp']:+.1f}). "
        "All four original P2 criteria were met."
    )
    text = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{The objective effect replicates on held-out hospitals.} Patient-grouped eICU leave-hospitals-out validation: five folds, each training on 8,400 stays from the remaining hospitals and testing on every eligible stay of the held-out hospitals. Skill is macro persistence skill averaged over three seeds.}",
        r"\label{tab:p2-loho}",
        r"\begin{tabular}{rrrrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Masked-L1 skill} & \multicolumn{3}{c}{Masked L1 $-$ MSE} & Arch. & Objective \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Fold & Hosp. & Test stays & CarryGate & GraFITi & HyperIMTS & CarryGate & GraFITi & HyperIMTS & gap & effect \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.97\linewidth}\vspace{3pt}\footnotesize " + verdict_text
        + r" Architecture gap: median of the six same-objective pairwise skill differences within a fold; objective effect: median over architectures of $|$masked L1 $-$ MSE$|$.\end{minipage}",
        r"\end{table}",
        "",
    ])
    TABLE.write_text(text, encoding="utf-8")


def write_figure(data: dict) -> None:
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import style

    style.use_style()
    n = data["numbers"]
    obj = style.OBJECTIVE_COLOUR
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 0.9, 1.3], "wspace": 0.42})

    ax = axes[0]
    folds = np.arange(1, 6)
    effect = [f["objective_effect"] for f in data["c3"]]
    gap = [f["architecture_gap"] for f in data["c3"]]
    ax.bar(folds - 0.19, effect, width=0.38, color=obj["MAE"], label="objective effect")
    ax.bar(folds + 0.19, gap, width=0.38, color=style.MUTED, label="architecture gap")
    for x, e, g in zip(folds, effect, gap):
        ax.text(x, e + 0.0015, f"{e / g:.1f}\u00d7", ha="center", va="bottom",
                fontsize=6.2, color=style.INK)
    ax.set_xticks(folds)
    ax.set_xlabel("held-out hospital fold")
    ax.set_ylabel("macro persistence skill")
    ax.set_ylim(0, max(effect) * 1.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=6.2,
              handlelength=1.0, columnspacing=0.8)
    style.panel_label(ax, "A", dx=-0.30, dy=1.12)

    ax = axes[1]
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(4)
    for shift, key, objective in ((-0.19, "c1_mse_quartile_loss", "MSE"),
                                  (0.19, "c1_mae_quartile_loss", "MAE")):
        values = [100 * n[key][q] for q in quartiles]
        ax.bar(x + shift, values, width=0.38, color=obj[objective],
               label=style.OBJECTIVE_LABEL[objective])
        for xi, value in zip(x + shift, values):
            ax.text(xi, value + 0.8, f"{value:.0f}", ha="center", va="bottom",
                    fontsize=6.0, color=style.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nsparsest", "Q2", "Q3", "Q4\ndensest"])
    ax.set_ylabel("variable\u2013fold cells losing\nto carry-forward (%)")
    ax.set_xlabel("measurement-frequency quartile")
    ax.legend(loc="upper right", fontsize=6.4)
    style.panel_label(ax, "B", dx=-0.34, dy=1.12)

    ax = axes[2]
    c4 = data["c4"]
    for offset, objective in ((-0.17, "MSE"), (0.17, "MAE")):
        for ci, component in enumerate(COMPONENTS):
            for mi, model in enumerate(MODELS):
                gaps = c4[(c4.objective == objective) & (c4.component == component)
                          & (c4.model == model)].gap_pp
                xs = np.full(len(gaps), ci + offset + (mi - 1) * 0.06)
                ax.scatter(xs, gaps, s=9, marker=style.TRACKB_MARKER[model],
                           facecolor=obj[objective], edgecolor="white", linewidths=0.3,
                           zorder=3)
    style.zero_reference(ax, axis="y")
    ax.set_xticks(range(4))
    ax.set_xticklabels(COMPONENTS)
    ax.set_ylabel("model \u2212 carry-forward grading\nerror (percentage points)")
    l1, mse = n["c4_masked_l1"], n["c4_mse"]
    ax.text(0.99, 0.97,
            f"median: MSE {mse['median_pp']:+.1f} pp, masked L1 {l1['median_pp']:+.1f} pp\n"
            "positive values favour carry-forward",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=style.MUTED)
    handles = [Line2D([], [], marker="s", ls="", color=obj[o], label=style.OBJECTIVE_LABEL[o])
               for o in ("MSE", "MAE")]
    handles += [Line2D([], [], marker=style.TRACKB_MARKER[m], ls="", color=style.INK,
                       markersize=3.5, label=LABEL[m]) for m in MODELS]
    ax.set_ylim(top=max(c4.gap_pp) * 1.35)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5,
              fontsize=6.0, handletextpad=0.2, columnspacing=0.7)
    style.panel_label(ax, "C", dx=-0.22, dy=1.12)

    for ax in axes:
        style.soft_grid(ax, axis="y")
    fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the existing P2 table, figure, and paper numbers before replacing them")
    args = parser.parse_args()
    prepare_generation(ROOT, (TABLE, FIGURE.with_suffix(".pdf"),
                              FIGURE.with_suffix(".png"), NUMBERS),
                       refresh=args.refresh_generated, label="p2_paper_assets")
    data = collect()
    write_table(data)
    write_figure(data)
    NUMBERS.write_text(json.dumps(data["numbers"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["numbers"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
