#!/usr/bin/env python
"""Figure: the encoder position stays saturated across a 41-fold capacity range.

Panel a plots skill against parameter count on a log axis, because the ladder is
geometric; the three encoder arms are drawn together so that their overlap is the
visible fact. Panel b reduces each of the fifteen cells to the single quantity
the registered rule tests -- the probe gap divided by its own threshold -- so
that "nothing clears" is one glance rather than fifteen comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (MODEL_NAME, display, BETTER, CARRY, COHORT_COLOUR, COHORT_LABEL, COHORT_MARKER, GRID,
                   INK, MUTED, PROBE_ARMS, panel_label, save, soft_grid, tint,
                   use_style)

LEVELS = ["C1", "C2", "C3", "C4", "C5"]
COHORTS = ["p12", "mimiciv", "eicu"]
ENCODER_STYLE = {name: (colour, marker, label)
                 for (name, (colour, marker)), label
                 in zip(PROBE_ARMS.items(), ("identity", "linear", "MLP"))}


def main() -> None:
    use_style()
    d = pd.read_csv(ROOT / "results" / "d2_capacity" / "d2_1_matrix.csv")
    probe = pd.read_csv(ROOT / "results" / "d2_capacity" / "d2_probe_verdicts.csv")
    ext = pd.read_csv(ROOT / "results" / "d1_external" / "d1_both_metrics.csv")
    ext["model"] = ext.model.map(display)
    best_ext = (ext[ext.model != MODEL_NAME].groupby(["cohort", "model"]).macro.mean()
                .groupby("cohort").max())

    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.30)

    for col, cohort in enumerate(COHORTS):
        ax = fig.add_subplot(gs[0, col])
        soft_grid(ax)
        sub = d[d.cohort == cohort]
        for encoder, (colour, marker, label) in ENCODER_STYLE.items():
            g = (sub[sub.encoder == encoder].groupby("level")
                 .agg(skill=("skill", "mean"), sd=("skill", "std"),
                      params=("n_params", "max")).reindex(LEVELS))
            ax.errorbar(g.params, g.skill, yerr=g.sd, fmt=marker + "-", color=colour,
                        ms=4.2, lw=1.4, capsize=2, elinewidth=0.9, mec="white",
                        mew=0.5, label=label, zorder=4)
        # The best published model sits far below this range on every cohort; it is
        # quoted in the caption rather than drawn, because including it would
        # compress the ladder itself, which is what this panel is for.
        ax.set_xscale("log")
        ax.set_title(COHORT_LABEL[cohort], pad=5)
        ax.set_xlabel("backbone parameters")
        if col == 0:
            ax.set_ylabel("persistence skill")
            ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.2)
            panel_label(ax, "a", dx=-0.30)

    ax = fig.add_subplot(gs[0, 3])
    soft_grid(ax, axis="y")
    probe["ratio"] = probe.headroom / probe.threshold
    params = d.groupby("level").n_params.max().reindex(LEVELS)
    for cohort in COHORTS:
        sub = probe[probe.cohort == cohort].set_index("level").reindex(LEVELS)
        ax.plot(range(len(LEVELS)), sub.ratio, marker=COHORT_MARKER[cohort],
                color=COHORT_COLOUR[cohort],
                ms=4.5, lw=1.3, mec="white", mew=0.5, label=COHORT_LABEL[cohort], zorder=4)
    ax.axhline(1.0, color=CARRY, lw=1.4, zorder=5)
    ax.axhspan(1.0, 1.6, color=tint(BETTER, 0.90), zorder=1)
    ax.text(0.03, 1.42, "position would be discriminative", fontsize=6.4, color=INK,
            va="center")
    ax.text(0.97, -0.30, "saturated in 15 of 15 cells", fontsize=6.4, color=MUTED,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"{l}\n{params[l]/1000:.0f}k" for l in LEVELS])
    ax.set_ylim(-0.35, 1.6)
    ax.set_ylabel("probe gap / registered threshold")
    ax.set_xlabel("backbone capacity")
    ax.legend(loc="lower left", labelspacing=0.2, handlelength=1.3)
    panel_label(ax, "b", dx=-0.30)
    save(fig, "F4_capacity_control")


if __name__ == "__main__":
    main()
