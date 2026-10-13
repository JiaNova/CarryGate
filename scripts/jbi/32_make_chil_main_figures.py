#!/usr/bin/env python
"""Make the two evidence-focused composite figures for the CHIL main text.

The source data are already frozen in the JBI audit outputs.  This script only
renders them; it does not select checkpoints, refit models, or change any
analysis rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style as _style  # noqa: E402
from output_safety import prepare_generation  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
JBI = ROOT / "results" / "jbi"
FIG = ROOT / "results" / "paper"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


_DISPLAY = {"ICUKANForecaster": f"{_style.MODEL_NAME} (harness)"}


def make_protocol_objective() -> None:
    decomposition = json.loads((JBI / "e5_protocol_gap_decomposition.json").read_text())
    e7 = pd.read_csv(JBI / "e7_objective_effect_by_variable.csv")
    colors = dict(_style.TRACKB_COLOUR)
    _style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.5), constrained_layout=True)
    for ax, cohort, title in zip(axes[0], ("eicu", "mimiciv"), ("eICU-CRD", "MIMIC-IV")):
        cells = decomposition["cells_by_cohort"][cohort]
        matrix = np.array(
            [[cells["native_MSE"], cells["native_L1"]],
             [cells["harness_MSE"], cells["harness_L1"]]], dtype=float
        )
        im = ax.imshow(matrix, cmap=_style.SEQUENTIAL, vmin=0.0, vmax=0.115, aspect="auto")
        ax.set_xticks([0, 1], ["MSE", "masked L1 / MAE"])
        ax.set_yticks([0, 1], ["native pipeline", "controlled harness"])
        ax.set_title(f"{title}: protocol × objective")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="white" if matrix[i, j] > 0.075 else _style.INK,
                        fontsize=9.5, fontweight="bold")
        ax.set_xlabel("training objective")
        ax.set_ylabel("evaluation pipeline")
    for ax, cohort, title in zip(axes[1], ("eicu", "mimiciv"), ("eICU-CRD", "MIMIC-IV")):
        part = e7[e7.cohort == cohort]
        for model, group in part.groupby("model", sort=False):
            if model not in colors:
                continue
            summary = group.groupby("frequency_quartile", as_index=False).objective_gain.mean()
            ax.plot(summary.frequency_quartile, summary.objective_gain,
                    marker=_style.TRACKB_MARKER[model], lw=1.8, ms=5,
                    label=_DISPLAY.get(model, model), color=colors[model],
                    mec="white", mew=0.6)
        _style.zero_reference(ax, 0.0, axis="y", label="no objective effect")
        ax.set_xticks([1, 2, 3, 4], ["Q1\nrare", "Q2", "Q3", "Q4\ndense"])
        ax.set_ylabel("MAE skill − MSE skill")
        ax.set_xlabel("variable measurement-frequency quartile")
        ax.set_title(f"{title}: objective gain by frequency")
        _style.soft_grid(ax)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    bar = fig.colorbar(im, ax=axes[0], fraction=0.030, pad=0.02)
    bar.set_label("macro persistence skill", fontsize=7.5)
    bar.ax.tick_params(labelsize=6.5)

    _save(fig, FIG / "F3_protocol_objective.pdf")


def make_aligned_architecture() -> None:
    """Paired differences against the equivalence margins.

    The earlier version plotted each external model's marginal skill with an
    interval back-derived from the paired contrast.  A reader then compares
    overlapping marginal intervals -- the practice recommendation five of the
    paper tells people to stop.  The quantity that was tested is the paired
    difference, and the test is whether its 90% interval sits inside the
    pre-specified margin, so that is what this draws: differences, their
    intervals, and the margins, with the numbers printed beside each point.
    """
    _style.use_style()
    cohorts = (
        ("eicu", "eICU-CRD", "e2_trackb_adapter_vs_external_paired_mae.csv", "e6_tost_eicu_mae.csv"),
        ("mimiciv", "MIMIC-IV", "e2_trackb_adapter_vs_external_paired_mae_mimiciv.csv",
         "e6_tost_mimiciv_mae.csv"),
    )
    models = ("HyperIMTS", "GraFITi", "Warpformer")
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for half, colour in ((0.020, "#F2F2F2"), (0.010, "#E1E1E1")):
        ax.axvspan(-half, half, color=colour, zorder=0, lw=0)
    ax.axvline(0, color=_style.CARRY, lw=0.8, zorder=1)
    ax.text(0.0, 1.02, "$\\pm\\delta_1{=}0.010$", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color=_style.MUTED)
    ax.text(0.020, 1.02, "$\\pm\\delta_2{=}0.020$", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color=_style.MUTED)

    rows = []
    for k, model in enumerate(models):
        centre = len(models) - 1 - k
        for j, (key, label, paired_name, tost_name) in enumerate(cohorts):
            paired = pd.read_csv(JBI / paired_name).set_index("model").loc[model]
            tost = pd.read_csv(JBI / tost_name)
            tost = tost[(tost.model == model)].iloc[0]
            y = centre + (0.17 if j == 0 else -0.17)
            colour, marker = _style.COHORT_COLOUR[key], _style.COHORT_MARKER[key]
            # thin: 95% interval; thick: the 90% interval the TOST uses
            ax.plot([paired.paired_macro_ci_low, paired.paired_macro_ci_high], [y, y],
                    color=colour, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot([tost.tost_ci90_low, tost.tost_ci90_high], [y, y],
                    color=colour, lw=3.2, zorder=3, solid_capstyle="butt")
            ax.plot(paired.paired_macro_difference, y, marker, color=colour, ms=5.5,
                    mec="white", mew=0.7, zorder=4)
            ax.text(paired.paired_macro_ci_high + 0.003, y,
                    f"{paired.paired_macro_difference:+.4f}", va="center", ha="left",
                    fontsize=6.4, color=_style.INK)
        rows.append((centre, model))
    ax.set_yticks([c for c, _ in rows]); ax.set_yticklabels([m for _, m in rows])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(-0.03, 0.15)
    ax.set_xlabel(f"{_style.MODEL_NAME} $-$ external model, macro persistence skill "
                  "(paired over the same test patients)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=_style.COHORT_COLOUR[k], marker=_style.COHORT_MARKER[k],
                      lw=3.2, ms=5, mec="white", label=lab) for k, lab, *_ in cohorts]
    handles += [Line2D([], [], color=_style.MUTED, lw=3.2, label="90% interval (TOST)"),
                Line2D([], [], color=_style.MUTED, lw=1.0, label="95% interval")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.3, handlelength=1.6,
              labelspacing=0.3, borderaxespad=0.2)
    _save(fig, FIG / "F4_aligned_architecture.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the existing figure generation before replacing it")
    args = parser.parse_args()
    generated = [FIG / name for stem in ("F3_protocol_objective", "F4_aligned_architecture")
                 for name in (stem + ".pdf", stem + ".png")]
    prepare_generation(ROOT, generated, refresh=args.refresh_generated,
                       label="trackb_composite_figures")
    make_protocol_objective()
    make_aligned_architecture()
    print("wrote F3_protocol_objective.pdf and F4_aligned_architecture.pdf")
