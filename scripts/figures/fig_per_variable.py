#!/usr/bin/env python
"""Figure: where the difference between the methods actually lives.

Variables are ordered by how often they are measured, because that ordering is
what the result turns on: the methods agree on densely monitored signals and
separate on sparsely sampled laboratory values. The six SOFA components are
marked, since they are selected on clinical grounds rather than by us.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))
sys.path.insert(0, str(ROOT))
from style import (MODEL_NAME, CARRY, COHORT_LABEL, GRID, INK, MODERN, MUTED, OURS,
                   panel_label, save, soft_grid, use_style)

from rescore_macro import UNSCORED, cohort_events           # noqa: E402
from src.utils.skill_metrics import persistence_skill        # noqa: E402

OURS_GLOB = {"p12": "results/phase5_ladder/stratified/L-K16_*_per_variable.csv",
             "mimiciv": "results/mimiciv_m1/stratified/M1-K16_*_per_variable.csv",
             "eicu": "results/phase3_eicu/e_m1/stratified/E-M1-K16_*_per_variable.csv"}
SOFA = {"PaO2", "Platelets", "Bilirubin", "MAP", "GCS", "Creatinine"}
COHORTS = ["p12", "mimiciv", "eicu"]


def main() -> None:
    use_style()
    names = yaml.safe_load(
        (ROOT / "configs" / "mimiciv_variable_map.yaml").read_text())["variable_names"]
    matrix = pd.read_csv(ROOT / "results" / "d1_external" / "d1c_matrix.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        ref = cohort_events(cohort)
        scored = ~np.isin(ref["var"], sorted(UNSCORED))
        ours = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / OURS_GLOB[cohort]))]
                         ).groupby("variable")[["skill", "n_test"]].mean()
        best = None
        for model in ["HyperIMTS", "GraFITi", "Warpformer", "APN", "tPatchGNN"]:
            seeds = []
            for e in matrix[(matrix.cohort == cohort) & (matrix.model == model)].itertuples():
                pred = np.load(Path(e.eval_dir) / "output_pred.npy")
                mask = np.load(Path(e.eval_dir) / "input_y_mask.npy").astype(bool)
                _, frame, _ = persistence_skill(pred[mask][scored], ref["truth"][scored],
                                                ref["var"][scored], ref["x_last"][scored])
                seeds.append(frame.set_index("variable").skill)
            mean = pd.concat(seeds, axis=1).mean(axis=1)
            best = mean if best is None else np.maximum(best, mean)

        table = pd.DataFrame({"ours": ours.skill, "best": best, "n": ours.n_test})
        table = table.sort_values("n")
        xs = np.arange(len(table))

        soft_grid(ax, axis="y")
        ax.axhspan(-1.4, 0, color=GRID, alpha=0.55, zorder=1)
        ax.axhline(0, color=CARRY, lw=1.2, zorder=4)
        ax.vlines(xs, table.best, table.ours, color=MUTED, lw=0.8, alpha=0.5, zorder=3)
        ax.scatter(xs, table.best, s=17, color=MODERN, zorder=5,
                   label="best published model")
        ax.scatter(xs, table.ours, s=17, color=OURS, zorder=5, label=MODEL_NAME)
        for i, v in enumerate(table.index):
            if names[v] in SOFA:
                ax.annotate(names[v], (i, min(table.best.iloc[i], table.ours.iloc[i])),
                            textcoords="offset points", xytext=(0, -7), ha="center",
                            fontsize=5.6, color=INK, rotation=90, va="top")
        ax.set_title(COHORT_LABEL[cohort], pad=5)
        ax.set_xlabel("variables, ordered by how often they are measured\n"
                      "(sparse laboratory values ← → densely monitored signals)",
                      fontsize=6.8)
        ax.set_xticks([])
        ax.set_ylim(-0.95, 0.33)
    axes[0].set_ylabel("persistence skill")
    axes[0].legend(loc="lower right", labelspacing=0.25)
    axes[0].text(0.02, 0.06, "worse than carry-forward", transform=axes[0].transAxes,
                 fontsize=6.2, color=MUTED)
    panel_label(axes[0], "", dx=-0.12)
    fig.subplots_adjust(wspace=0.08, bottom=0.26)
    save(fig, "F6_per_variable")


if __name__ == "__main__":
    main()
