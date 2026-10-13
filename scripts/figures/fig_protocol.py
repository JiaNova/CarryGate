#!/usr/bin/env python
"""Figure: the same model changes position when the evaluation protocol changes.

Panel a is the field's standard protocol and its own metric; panel b is the
per-variable protocol used throughout this paper. The model and the harness are
identical in both. The panels deliberately do not share an axis, because the two
metrics are not commensurable -- that is the point being made.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (MODEL_NAME, display, CARRY, INK, MODEL_COLOURS, MUTED, OURS, panel_label, save,
                   soft_grid, use_style)


def main() -> None:
    use_style()
    bench = pd.read_csv(ROOT / "results" / "d3_clinical" / "benchmark_standard.csv")
    repro = pd.read_csv(ROOT / "results" / "d1_external" / "d1a_reproduction.csv")
    repro["model"] = repro.model.map(display)
    ours_mse, ours_sd = bench.mse.mean(), bench.mse.std(ddof=1)

    standard = pd.concat([
        repro[["model", "mse_mean"]].rename(columns={"mse_mean": "mse"}),
        pd.DataFrame([{"model": MODEL_NAME, "mse": ours_mse}])], ignore_index=True)
    standard = standard.sort_values("mse")                      # lower is better

    both = pd.read_csv(ROOT / "results" / "d1_external" / "d1_both_metrics.csv")
    both["model"] = both.model.map(display)
    ours_protocol = (both[both.cohort == "p12"].groupby("model").macro.mean()
                     .loc[list(standard.model) + ["carry-forward"]]
                     .sort_values(ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))

    ax = axes[0]
    soft_grid(ax, axis="x")
    ys = np.arange(len(standard))[::-1]
    colours = [MODEL_COLOURS.get(m, MUTED) for m in standard.model]
    # MSE has no meaningful zero and the models differ by under 0.04, so this axis
    # has to be truncated. Bars on a truncated axis would misstate the ratios;
    # points do not.
    for yy, value, colour in zip(ys, standard.mse, colours):
        ax.plot([standard.mse.min() - 0.004, value], [yy, yy], color=colour, lw=1.0,
                alpha=0.35, zorder=3)
        ax.scatter(value, yy, s=42, color=colour, edgecolor="white", lw=0.6, zorder=5)
    # The seed standard deviation (about 0.0004) is narrower than the marker;
    # drawn as an error bar its caps collapse into an "H" glyph.  The caption
    # states it instead.
    ax.set_yticks(ys); ax.set_yticklabels(standard.model)
    for label in ax.get_yticklabels():
        if label.get_text() == MODEL_NAME:
            label.set_color(OURS); label.set_fontweight("bold")
    ax.set_xlim(standard.mse.min() - 0.006, standard.mse.max() + 0.004)
    ax.set_xlabel("MSE  (lower is better)")
    ax.set_title("standard protocol\nPhysioNet-2012, 36 h $\\rightarrow$ 3 h, pooled", pad=6)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a", dx=-0.36)

    ax = axes[1]
    soft_grid(ax, axis="x")
    ys = np.arange(len(ours_protocol))[::-1]
    colours = [MODEL_COLOURS.get(m, MUTED) for m in ours_protocol.index]
    ax.barh(ys, ours_protocol.values, color=colours, height=0.62, edgecolor="white",
            lw=0.6, zorder=3)
    ax.axvline(0, color=CARRY, lw=1.2, zorder=4)
    ax.set_yticks(ys); ax.set_yticklabels(ours_protocol.index)
    for label in ax.get_yticklabels():
        if label.get_text() == MODEL_NAME:
            label.set_color(OURS); label.set_fontweight("bold")
    ax.set_xlabel("persistence skill  (higher is better)")
    ax.set_title("per-variable protocol\nPhysioNet-2012, 24 h $\\rightarrow$ 24 h", pad=6)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "b", dx=-0.36)

    fig.subplots_adjust(wspace=0.62, bottom=0.20, top=0.82)
    save(fig, "F5_protocol_dependence")


if __name__ == "__main__":
    main()
