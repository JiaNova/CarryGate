#!/usr/bin/env python
"""Figure: this project against eight published models, under both estimators.

Rows are ordered by the registered per-variable metric. Plotting the pooled
value on the same row makes the disagreement between the two estimators visible
without a second panel: where the pooled markers stop descending monotonically,
the two estimators have swapped that pair of models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (LEGACY, MODERN, MODEL_NAME, display, CARRY, COHORT_LABEL, GRID, INK, MODEL_COLOURS, MUTED, OURS,
                   panel_label, save, soft_grid, use_style)

COHORTS = ["p12", "mimiciv", "eicu"]


def main() -> None:
    use_style()
    d = pd.read_csv(ROOT / "results" / "d1_external" / "d1_both_metrics.csv")
    d["model"] = d.model.map(display)
    agg = d.groupby(["model", "cohort"]).agg(
        macro=("macro", "mean"), pooled=("pooled", "mean"),
        below=("n_below", "mean"), n_vars=("n_vars", "max")).reset_index()

    order = (agg[agg.cohort == "mimiciv"].sort_values("macro", ascending=False)
             .model.tolist())
    y = {m: len(order) - 1 - i for i, m in enumerate(order)}

    fig = plt.figure(figsize=(11.0, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.50, wspace=0.10)

    for col, cohort in enumerate(COHORTS):
        ax = fig.add_subplot(gs[0, col])
        sub = agg[agg.cohort == cohort].set_index("model")
        soft_grid(ax, axis="x")
        ax.axvline(0, color=CARRY, lw=1.3, zorder=3)
        for model in order:
            row_y, colour = y[model], MODEL_COLOURS.get(model, MUTED)
            if model == MODEL_NAME:
                ax.axhspan(row_y - 0.5, row_y + 0.5, color=OURS, alpha=0.08, zorder=1)
            m, p = float(sub.loc[model, "macro"]), float(sub.loc[model, "pooled"])
            ax.plot([m, p], [row_y, row_y], color=colour, lw=1.0, alpha=0.5, zorder=3)
            ax.scatter(m, row_y, s=30, color=colour, edgecolor="white", lw=0.5, zorder=5)
            ax.scatter(p, row_y, s=30, facecolor="white", edgecolor=colour,
                       lw=1.2, zorder=5)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_title(COHORT_LABEL[cohort], pad=5)
        ax.tick_params(axis="y", length=0)
        if col == 0:
            ax.set_yticks([y[m] for m in order])
            ax.set_yticklabels(order)
            for label in ax.get_yticklabels():
                if label.get_text() == MODEL_NAME:
                    label.set_color(OURS); label.set_fontweight("bold")
            panel_label(ax, "a", dx=-0.42)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("persistence skill")

    handles = [plt.Line2D([], [], marker="o", color=MUTED, ls="none", ms=5,
                          label="per-variable (registered)"),
               plt.Line2D([], [], marker="o", mfc="white", mec=MUTED, color=MUTED,
                          ls="none", ms=5, label="pooled over events")]
    # Hue carries publication era (see style.py); name it on the figure itself.
    handles += [plt.Line2D([], [], color=c, lw=2.4, label=l) for c, l in
                ((MODERN, "published 2023+"), (LEGACY, "published \u2264 2021"),
                 (OURS, MODEL_NAME))]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 0.975))

    # Lower row: a point chart keeps the three cohort counts readable for every
    # model without asking the reader to compare thirty adjacent bar lengths.
    ax2 = fig.add_subplot(gs[1, :])
    soft_grid(ax2, axis="x")
    cohort_markers = {"p12": "o", "mimiciv": "s", "eicu": "^"}
    for i, cohort in enumerate(COHORTS):
        sub = agg[agg.cohort == cohort].set_index("model")
        for model in order:
            ax2.scatter(float(sub.loc[model, "below"]), y[model] + (1 - i) * 0.20,
                        s=37, marker=cohort_markers[cohort],
                        color=MODEL_COLOURS.get(model, MUTED), edgecolor="white",
                        linewidth=0.5, zorder=3)
    ax2.set_yticks([y[m] for m in order])
    ax2.set_yticklabels(order)
    for label, model in zip(ax2.get_yticklabels(), order):
        if model == MODEL_NAME:
            label.set_color(OURS); label.set_fontweight("bold")
    ax2.set_ylim(-0.7, len(order) - 0.3)
    ax2.set_xlim(-0.5, 25)
    ax2.set_xlabel("variables worse than carry-forward (of 34; 32 on eICU-CRD)")
    panel_label(ax2, "b", dx=-0.055, dy=1.08)
    shapes = [plt.Line2D([], [], marker=cohort_markers[c], color=MUTED, ls="none",
                         ms=5, label=COHORT_LABEL[c]) for c in COHORTS]
    ax2.legend(handles=shapes, loc="upper right", ncol=3, handlelength=1.1)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.07)
    save(fig, "F3_external_comparison")


if __name__ == "__main__":
    main()
