#!/usr/bin/env python
"""Plot the patient-level eICU aggregation comparison for the JBI main text.

The left panel keeps the same predictions and changes only the aggregation
rule.  The right panel uses the paired patient-bootstrap difference, which is
the appropriate uncertainty statement for comparing models on the same test
patients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from style import (GRID, INK, MODEL_COLOURS, MODEL_NAME, MUTED, OURS, display, save,  # noqa: E402
                   soft_grid, use_style)


def main() -> None:
    use_style()
    frame = pd.read_csv(ROOT / "results" / "jbi" / "e2_paired_patient_comparison.csv")
    order = ["This work", "HyperIMTS", "Warpformer", "GraFITi", "GRU_D"]
    frame["model"] = pd.Categorical(frame["model"], categories=order, ordered=True)
    frame = frame.sort_values("model")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.35), gridspec_kw={"wspace": 0.42})

    # Same test predictions; only the aggregation rule changes.
    soft_grid(left, axis="y")
    x_macro, x_pooled = 0.0, 1.0
    for row in frame.itertuples():
        colour = OURS if row.model == "This work" else MODEL_COLOURS.get(str(row.model), MUTED)
        left.plot([x_macro, x_pooled], [row.macro, row.pooled], color=colour,
                  lw=1.15, alpha=0.7, zorder=2)
        left.scatter(x_macro, row.macro, color=colour, s=32,
                     edgecolor="white", linewidth=0.5, zorder=3)
        left.scatter(x_pooled, row.pooled, facecolor="white", edgecolor=colour,
                     linewidth=1.2, s=32, zorder=3)
    left.axhline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
    left.set_xticks([x_macro, x_pooled])
    left.set_xticklabels(["macro\n(variable-balanced)", "pooled\n(event-weighted)"])
    left.set_ylabel("persistence skill")
    left.set_title("Same predictions, different aggregation", pad=8)
    left.set_xlim(-0.62, 1.2)
    left.set_ylim(min(frame.macro.min(), frame.pooled.min()) - 0.06,
                 max(frame.macro.max(), frame.pooled.max()) + 0.04)
    left.text(0.98, 0.04, "filled = macro\nopen = pooled", transform=left.transAxes,
              ha="right", va="bottom", fontsize=6.5, color=MUTED)
    # Direct labels.  The three modern models trace the same path, so they are
    # named once as a group rather than stacked on top of one another.
    ours = frame[frame.model == "This work"].iloc[0]
    left.text(x_macro - 0.06, ours.macro, f"{MODEL_NAME}\n(native)", ha="right",
              va="center", fontsize=6.4, color=OURS, fontweight="bold")
    modern = frame[frame.model.isin(["HyperIMTS", "Warpformer", "GraFITi"])]
    left.text(x_macro - 0.06, modern.macro.mean() - 0.035, "HyperIMTS,\nWarpformer,\nGraFITi",
              ha="right", va="top", fontsize=6.4, color=MODEL_COLOURS["HyperIMTS"])
    gru = frame[frame.model == "GRU_D"].iloc[0]
    left.text(x_macro - 0.06, gru.macro, "GRU-D", ha="right", va="center", fontsize=6.4,
              color=MODEL_COLOURS["GRU-D"])

    # Paired patient bootstrap: same patient resample for both arms.
    ext = frame[frame["model"] != "This work"].copy()
    ys = np.arange(len(ext))
    right.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    soft_grid(right, axis="x")
    for offset, metric, marker, filled in [
        (0.12, "macro", "o", True),
        (-0.12, "pooled", "s", False),
    ]:
        for y, row in zip(ys, ext.itertuples()):
            colour = MODEL_COLOURS.get(str(row.model), MUTED)
            right.errorbar(
                row.__getattribute__(f"paired_{metric}_difference"),
                y + offset,
                xerr=[[row.__getattribute__(f"paired_{metric}_difference") - row.__getattribute__(f"paired_{metric}_ci_low")],
                      [row.__getattribute__(f"paired_{metric}_ci_high") - row.__getattribute__(f"paired_{metric}_difference")]],
                fmt=marker,
                color=colour,
                markerfacecolor=colour if filled else "white",
                markeredgecolor=colour,
                markersize=4.5,
                capsize=2.2,
                linewidth=1.0,
                zorder=3,
            )
    right.set_yticks(ys)
    right.set_yticklabels([display(str(value)) for value in ext["model"]])
    right.invert_yaxis()
    right.set_xlabel(f"{MODEL_NAME} (native) $-$ external model (harness)\n"
                     "paired over the same 1,752 patients, 95% interval")
    right.set_title("Aggregation changes the strength of the claim", pad=8)
    right.legend(
        handles=[
            plt.Line2D([], [], marker="o", color=INK, markerfacecolor=INK, ls="none", ms=4.5,
                       label="macro"),
            plt.Line2D([], [], marker="s", color=INK, markerfacecolor="white", ls="none", ms=4.5,
                       label="pooled"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6.5,
    )

    fig.text(0.5, 0.005,
             "Primary estimand: average event errors across seeds, then compute skill; "
             "2,000 patient-cluster bootstrap replicates.",
             ha="center", va="bottom", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.20)
    save(fig, "F7_jbi_paired_aggregation")


if __name__ == "__main__":
    main()
