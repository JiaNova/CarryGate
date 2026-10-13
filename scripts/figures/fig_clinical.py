#!/usr/bin/env python
"""Figure: SOFA grade misclassification relative to carry-forward.

A forest layout is the right form here. The quantity is a signed difference
against a fixed reference, the reference is exactly zero, and the claim being
made is about unanimity across cells -- so the eye needs to check whether any
marker falls on the wrong side of a single vertical line, which is what this
layout makes trivial and a grouped bar chart makes hard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (MODEL_NAME, display, CARRY, COHORT_LABEL, COHORT_MARKER, GRID, INK, MODEL_COLOURS,
                   MUTED, OURS, panel_label, save, soft_grid, use_style)

ORGANS = ["Coagulation", "Liver", "CNS", "Renal"]
ORGAN_VAR = {"Coagulation": "Platelets", "Liver": "Bilirubin",
             "CNS": "GCS", "Renal": "Creatinine"}
ORDER = [MODEL_NAME, "HyperIMTS", "Warpformer", "GraFITi", "APN", "t-PatchGNN",
         "mTAN", "GRU-D", "Neural Flows"]


def main() -> None:
    use_style()
    d = pd.read_csv(ROOT / "results" / "d3_clinical" / "sofa_grades.csv")
    d["model"] = d.model.map(display)
    d = d[d.model != "ICUKAN_Naive"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.1), sharey=True)
    y = {m: len(ORDER) - 1 - i for i, m in enumerate(ORDER)}

    for ax, organ in zip(axes, ORGANS):
        sub = d[d.organ == organ]
        soft_grid(ax, axis="x")
        ax.axvline(0, color=CARRY, lw=1.4, zorder=3)
        for model in ORDER:
            rows = sub[sub.model == model]
            colour = MODEL_COLOURS.get(model, MUTED)
            row_y = y[model]
            if model == MODEL_NAME:
                ax.axhspan(row_y - 0.5, row_y + 0.5, color=OURS, alpha=0.08, zorder=1)
            vals = [float(rows[rows.cohort == c].excess_over_carryforward.iloc[0])
                    for c in ["p12", "mimiciv", "eicu"] if (rows.cohort == c).any()]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [row_y, row_y], color=colour,
                        lw=1.1, alpha=0.55, zorder=3, solid_capstyle="round")
            for cohort in ["p12", "mimiciv", "eicu"]:
                r = rows[rows.cohort == cohort]
                if r.empty:
                    continue
                ax.scatter(float(r.excess_over_carryforward.iloc[0]), row_y,
                           marker=COHORT_MARKER[cohort], s=26, color=colour,
                           edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_title(f"{organ}   ·   {ORGAN_VAR[organ]}", color=INK, pad=6)
        ax.set_ylim(-0.8, len(ORDER) - 0.2)
        ax.tick_params(axis="y", length=0)
        left = min(0.0, float(sub.excess_over_carryforward.min()))
        right = float(sub.excess_over_carryforward.max())
        pad = 0.06 * (right - left)
        ax.set_xlim(left - pad, right + 3.2 * pad)   # room for the legend

    axes[0].set_yticks([y[m] for m in ORDER])
    axes[0].set_yticklabels(ORDER)
    for label in axes[0].get_yticklabels():
        if label.get_text() == MODEL_NAME:
            label.set_color(OURS)
            label.set_fontweight("bold")

    handles = [plt.Line2D([], [], marker=COHORT_MARKER[c], color=MUTED, ls="none",
                          ms=5, label=COHORT_LABEL[c]) for c in ["p12", "mimiciv", "eicu"]]
    axes[-1].legend(handles=handles, loc="upper right", handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.25, bbox_to_anchor=(1.02, 1.0))
    fig.supxlabel("excess SOFA-derived component misclassification relative to carry-forward"
                  "      (right of the line = worse than carry-forward)",
                  fontsize=8, color=INK, y=0.005)
    fig.subplots_adjust(wspace=0.10, bottom=0.18, top=0.88)
    save(fig, "F2_sofa_grades")


if __name__ == "__main__":
    main()
