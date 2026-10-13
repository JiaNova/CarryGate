#!/usr/bin/env python
"""Figure 1: the reference model, and the study design around three evaluation choices.

Panel a draws the registered model. The event-encoder position is an identity
map, which is not a simplification for the diagram: it is what the saturation
probe selected, and drawing anything richer there would misrepresent the model.
Panel b lays out the cohorts and one column per evaluation choice -- the
analysis, its headline finding and where it is reported. Every number printed
there is read from a generated result file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (MODEL_NAME, BETTER, CARRY, CATEGORICAL, GRID, INK, MODERN, MUTED, OURS,
                   panel_label, save, tint, use_style)

BOX = dict(boxstyle="round,pad=0.42,rounding_size=0.16", linewidth=1.1)


def box(ax, x, y, w, h, text, face, edge, tcolour=INK, fontsize=7.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge,
                                mutation_scale=1.0, zorder=3, **BOX))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tcolour, zorder=4, linespacing=1.45, fontweight=weight)


def arrow(ax, x0, y0, x1, y1, colour=MUTED, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, color=colour, lw=lw, zorder=2,
                                 shrinkA=1, shrinkB=1))


def panel_model(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    ax.text(1, 32.6, f"{MODEL_NAME}: carry-forward, gated by staleness", fontsize=8.2,
            color=OURS, fontweight="bold", ha="left", va="center")
    box(ax, 1, 20, 17, 10,
        "irregular events\n$(v,\\ t,\\ x)$\nno grid, no imputation", tint(MUTED), GRID)
    box(ax, 22, 20, 16, 10,
        "event encoding\n$\\mathbf{identity}$\n(probe verdict)", tint(OURS), OURS,
        tcolour=INK)
    box(ax, 42, 20, 18, 10,
        "transformer backbone\n$d$=64, 2 layers\n90,455 parameters", tint(MODERN), MODERN)
    box(ax, 64, 20, 17, 10, "per-variable\nrecency gate\n$\\omega_v(\\delta)$",
        tint(CATEGORICAL[2]), CATEGORICAL[2])
    box(ax, 85, 20, 14, 10, "forecast\n$\\hat{x}$", tint(MUTED), GRID)
    for x0, x1 in [(18, 22), (38, 42), (60, 64), (81, 85)]:
        arrow(ax, x0, 25, x1, 25)

    ax.text(50, 14.2,
            r"$\hat{x}_j \;=\; \omega_v(\delta_j)\, x^{\mathrm{last}}_v \;+\; "
            r"(1-\omega_v(\delta_j))\, \mu_v \;+\; \Delta_\theta$",
            ha="center", va="center", fontsize=9.4, color=INK)
    ax.text(50, 10.2,
            "staleness-weighted blend of the last observation and the training median, "
            "plus a learned residual",
            ha="center", va="center", fontsize=6.9, color=MUTED)
    box(ax, 15, 0.8, 70, 6.2,
        "$\\Delta_\\theta$ starts at zero and $\\omega_v\\approx 0.5$, so the untrained forecast sits "
        "halfway between the last value and the median;\ntraining learns, per variable, how "
        "quickly a stale value should give way to the median",
        tint(CARRY, 0.94), CARRY, fontsize=7.2)


def _design_numbers() -> dict:
    """Numbers printed in panel b, read from the generated result files."""
    import json
    import pandas as pd
    root = Path(__file__).resolve().parents[2] / "results" / "jbi"
    p2 = json.loads((root / "p2_loho_paper_numbers.json").read_text(encoding="utf-8"))
    p1 = json.loads((root / "p1_landmark_verdict.json").read_text(encoding="utf-8"))
    gaps = pd.read_csv(root / "p1_grading_gap_decomposition.csv", encoding="utf-8-sig")
    cells = json.loads((root / "e5_protocol_gap_decomposition.json").read_text(
        encoding="utf-8"))["cells_by_cohort"]
    objective = [c["harness_L1"] - c["harness_MSE"] for c in cells.values()]
    return {
        "objective_lo": min(objective), "objective_hi": max(objective),
        "loho_objective": p2["objective_effect_median"],
        "loho_architecture": p2["architecture_gap_median"],
        "loho_pairs": f"{p2['l1_minus_mse_positive']}/{p2['l1_minus_mse_count']}",
        "loho_hospitals": p2["hospitals_total"],
        "loho_stays": p2["stays_total"],
        "grading": " \u2192 ".join(
            f"{int(r.carry_forward_better_cells)}/{int(r.cell_count)}" for r in gaps.itertuples()),
        "alerting": f"{p1['I2']['passing_cells']}/{p1['I2']['cell_count']}",
    }


def panel_design(ax):
    """Study design: the cohorts, then one column per evaluation choice."""
    n = _design_numbers()
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    box(ax, 0.5, 6.0, 17.5, 24.0,
        "Data\n\nPhysioNet-2012\nMIMIC-IV\neICU-CRD\n12,000 ICU stays each\n\n"
        f"eICU-CRD held-out\nhospital cohort\n{n['loho_stays']:,} stays,\n"
        f"{n['loho_hospitals']} hospitals",
        tint(CARRY, 0.94), CARRY, fontsize=6.9)
    columns = [
        ("1  How errors are aggregated",
         "8 published models (released code),\nCarryGate and carry-forward;\n"
         "per-variable vs pooled skill",
         "rankings reorder; every published\nmodel loses to carry-forward on\nsome variables, mostly sparse labs",
         "Figure 2"),
        ("2  Which objective is trained",
         "4 architectures \u00d7 MSE / masked L1,\npatient-grouped splits;\n"
         "replicated on 5 held-out-hospital folds",
         f"objective {n['objective_lo']:.3f}\u2013{n['objective_hi']:.3f} vs small\n"
         f"architecture gaps; held-out: {n['loho_objective']:.3f}\n"
         f"vs {n['loho_architecture']:.3f}, masked L1 ahead {n['loho_pairs']}",
         "Figures 3\u20135, Table 1"),
        ("3  How the clinical evaluation is designed",
         "SOFA-derived component grades;\nfirst-day vs day 1\u20137 landmarks;\n"
         "alerts scored by net benefit",
         f"carry-forward grades better in\n{n['grading']} comparisons*;\n"
         f"alerts beat both defaults in {n['alerting']} settings",
         "Figure 6, Table 2"),
    ]
    x0, width, gap = 21.5, 24.0, 2.0
    for k, (title, method, finding, where) in enumerate(columns):
        x = x0 + k * (width + gap)
        box(ax, x, 26.0, width, 4.0, title, tint(BETTER, 0.80), BETTER,
            fontsize=7.2, weight="bold")
        box(ax, x, 15.6, width, 8.2, method, "white", GRID, fontsize=6.8)
        box(ax, x, 5.4, width, 8.2, finding, tint(INK, 0.93), INK, fontsize=6.8)
        ax.text(x + width / 2, 3.4, where, ha="center", va="center", fontsize=6.6,
                color=MUTED, style="italic")
        arrow(ax, x + width / 2, 26.0, x + width / 2, 23.9)
        arrow(ax, x + width / 2, 15.6, x + width / 2, 13.7)
    arrow(ax, 18.0, 18.0, 21.5, 18.0)
    ax.text(x0, 33.6, "Each comparison carried a decision rule written before it ran",
            fontsize=6.9, color=MUTED, ha="left", va="center", style="italic")
    ax.text(99.5, 0.6, "*first-day MSE \u2192 first-day masked L1 \u2192 day 1\u20137 landmarks",
            fontsize=6.3, color=MUTED, ha="right", va="center")


def main() -> None:
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.6),
                             gridspec_kw={"height_ratios": [0.92, 1.08], "hspace": 0.10})
    panel_model(axes[0]); panel_label(axes[0], "a", dx=-0.01, dy=1.02)
    panel_design(axes[1]); panel_label(axes[1], "b", dx=-0.01, dy=1.02)
    save(fig, "F1_overview")


if __name__ == "__main__":
    main()
