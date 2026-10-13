#!/usr/bin/env python
"""Build the prespecified historical deterioration frontier and decision-curve audit.

This is a descriptive downstream analysis of the frozen window-transition
results in ``e4_grade_transitions_summary.csv``.  It does not retrain a model
or choose a new alert threshold.  Each model's existing binary transition
operating point is placed on sensitivity versus false-positive rate, while a
decision-curve calculation evaluates that point over the frozen probability
threshold range 0.05--0.30.

The source transition analysis treats a window's maximum SOFA-derived grade as
the window state and compares it with the maximum carry-forward grade.  The
carry-forward arm therefore has sensitivity 0, false-positive rate 0 and no
positive predictive value; its NNA is undefined rather than zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from output_safety import prepare_generation  # noqa: E402

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style as _style  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "jbi"
FIGURES = ROOT / "results" / "paper"
PLAN = "JBI_analysis_plan_v1"
THRESHOLDS = np.round(np.arange(0.05, 0.301, 0.01), 2)
MODEL_ORDER = (
    "This work",
    "APN",
    "GraFITi",
    "HyperIMTS",
    "Warpformer",
    "tPatchGNN",
    "GRU_D",
    "mTAN",
    "NeuralFlows",
    "ICUKAN_Naive",
)
DISPLAY = _style.DISPLAY
PALETTE = {
    name: _style.model_colour(name)
    for name in ("This work", "APN", "GraFITi", "HyperIMTS", "Warpformer",
                 "tPatchGNN", "GRU_D", "mTAN", "NeuralFlows", "ICUKAN_Naive")
}
MARKERS = {"p12": "o", "mimiciv": "s", "eicu": "^"}


def _decision_curve(prevalence: float, sensitivity: float, specificity: float):
    fpr = 1.0 - specificity
    values = []
    for threshold in THRESHOLDS:
        odds = threshold / (1.0 - threshold)
        values.append(prevalence * sensitivity - (1.0 - prevalence) * fpr * odds)
    return np.asarray(values, dtype=float)


def _metric_ci(row: pd.Series, metric: str) -> tuple[float, float]:
    lo = float(row[f"deterioration_{metric}_ci_lo"])
    hi = float(row[f"deterioration_{metric}_ci_hi"])
    return lo, hi


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = RESULTS / "e4_grade_transitions_summary.csv"
    bootstrap_path = RESULTS / "e4_grade_transitions_bootstrap.csv"
    if not summary_path.exists() or not bootstrap_path.exists():
        raise FileNotFoundError("P1 requires e4_grade_transitions_summary.csv and bootstrap.csv")
    summary = pd.read_csv(summary_path)
    bootstrap = pd.read_csv(bootstrap_path)
    required = {
        "cohort",
        "organ",
        "variable",
        "model",
        "window_true_deterioration_rate_mean",
        "deterioration_sensitivity_mean",
        "deterioration_specificity_mean",
        "deterioration_ppv_mean",
        "n_windows",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise AssertionError(f"summary missing columns: {missing}")
    return summary, bootstrap


def _build(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    merged = summary.merge(
        bootstrap,
        on=["analysis_plan_version", "cohort", "organ", "variable", "model"],
        suffixes=("", "_boot"),
        validate="one_to_one",
    )
    for row in merged.itertuples(index=False):
        values = row._asdict()
        cohort, organ, variable, model = (
            values["cohort"], values["organ"], values["variable"], values["model"]
        )
        prevalence = float(values["window_true_deterioration_rate_mean"])
        sensitivity = float(values["deterioration_sensitivity_mean"])
        specificity = float(values["deterioration_specificity_mean"])
        ppv = float(values["deterioration_ppv_mean"])
        if model == "ICUKAN_Naive":
            # Keep this explicit even if the source file changes: it is a
            # structural property of the no-change prediction.
            sensitivity = 0.0
            specificity = 1.0
            ppv = np.nan
        sens_lo, sens_hi = _metric_ci(pd.Series(values), "sensitivity")
        spec_lo, spec_hi = _metric_ci(pd.Series(values), "specificity")
        if model == "ICUKAN_Naive":
            sens_lo = sens_hi = 0.0
            spec_lo = spec_hi = 1.0
        fpr = 1.0 - specificity
        fpr_lo, fpr_hi = 1.0 - spec_hi, 1.0 - spec_lo
        nna = float(1.0 / ppv) if np.isfinite(ppv) and ppv > 0 else np.nan
        ppv_lo, ppv_hi = _metric_ci(pd.Series(values), "ppv")
        nna_lo = float(1.0 / ppv_hi) if np.isfinite(ppv_hi) and ppv_hi > 0 else np.nan
        nna_hi = float(1.0 / ppv_lo) if np.isfinite(ppv_lo) and ppv_lo > 0 else np.inf
        curve = _decision_curve(prevalence, sensitivity, specificity)
        curve_lo = _decision_curve(prevalence, sens_lo, spec_lo)
        curve_hi = _decision_curve(prevalence, sens_hi, spec_hi)
        # Using the marginal sensitivity/specificity intervals gives an
        # intentionally conservative envelope; no joint bootstrap draws are
        # available in the archived transition summary.
        curve_lower = np.minimum(curve_lo, curve_hi)
        curve_upper = np.maximum(curve_lo, curve_hi)
        treat_all = prevalence - (1.0 - prevalence) * (THRESHOLDS / (1.0 - THRESHOLDS))
        majority_positive = int(np.sum((curve > 0.0) & (curve > 0.0)))
        rows.append(
            {
                "analysis_plan_version": PLAN,
                "cohort": cohort,
                "organ": organ,
                "variable": variable,
                "model": model,
                "model_display": DISPLAY.get(model, model),
                "n_windows": int(values["n_windows"]),
                "window_true_deterioration_rate": prevalence,
                "sensitivity": sensitivity,
                "sensitivity_ci_low": sens_lo,
                "sensitivity_ci_high": sens_hi,
                "false_positive_rate": fpr,
                "false_positive_rate_ci_low": fpr_lo,
                "false_positive_rate_ci_high": fpr_hi,
                "specificity": specificity,
                "ppv": ppv,
                "ppv_ci_low": ppv_lo if model != "ICUKAN_Naive" else np.nan,
                "ppv_ci_high": ppv_hi if model != "ICUKAN_Naive" else np.nan,
                "nna": nna,
                "nna_ci_low": nna_lo if model != "ICUKAN_Naive" else np.nan,
                "nna_ci_high": nna_hi if model != "ICUKAN_Naive" else np.nan,
                "carryforward_operating_point": bool(model == "ICUKAN_Naive"),
                "bootstrap_n": int(values["bootstrap_n"]),
                "bootstrap_seed": int(values["bootstrap_seed"]),
                "bootstrap_unit": values["bootstrap_unit"],
                "split_unit": values["split_unit"],
                "decision_threshold_min": float(THRESHOLDS.min()),
                "decision_threshold_max": float(THRESHOLDS.max()),
                "decision_curve_ci_method": "marginal_sensitivity_specificity_envelope",
                "decision_curve_majority_above_treat_none": bool(majority_positive > len(THRESHOLDS) / 2),
                "decision_value_interpretation": (
                    "not_shown" if majority_positive <= len(THRESHOLDS) / 2 else "descriptive_only"
                ),
            }
        )
        for threshold, nb, nb_lo, nb_hi, all_nb in zip(
            THRESHOLDS, curve, curve_lower, curve_upper, treat_all
        ):
            curves.append(
                {
                    "analysis_plan_version": PLAN,
                    "cohort": cohort,
                    "organ": organ,
                    "variable": variable,
                    "model": model,
                    "model_display": DISPLAY.get(model, model),
                    "threshold_probability": float(threshold),
                    "window_true_deterioration_rate": prevalence,
                    "net_benefit": float(nb),
                    "net_benefit_ci_low": float(nb_lo),
                    "net_benefit_ci_high": float(nb_hi),
                    "treat_none_net_benefit": 0.0,
                    "treat_all_net_benefit": float(all_nb),
                    "carryforward_net_benefit": 0.0,
                    "bootstrap_n": int(values["bootstrap_n"]),
                    "bootstrap_seed": int(values["bootstrap_seed"]),
                    "decision_curve_ci_method": "marginal_sensitivity_specificity_envelope",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def _plot(frontier: pd.DataFrame, curve: pd.DataFrame, output: Path) -> None:
    """Two claims, two forms.

    The top row is a frontier: it exists to show that carry-forward sits exactly
    at the origin, because it never raises an alarm.  The bottom row used to be
    thirty overlapping decision curves per component, which cannot show a
    120-cell result -- the reader cannot find carry-forward's line, let alone
    count how many arms clear it.  The count is the claim, so the bottom row
    counts.  The full curves remain in the supplementary CSVs.
    """
    _style.use_style()
    components = (("CNS", "GCS"), ("Coagulation", "Platelets"),
                  ("Liver", "Bilirubin"), ("Renal", "Creatinine"))
    fig = plt.figure(figsize=(11.0, 5.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.30)

    for col, (organ, variable) in enumerate(components):
        ax = fig.add_subplot(gs[0, col])
        _style.soft_grid(ax)
        subset = frontier[(frontier.organ == organ) & (frontier.variable == variable)]
        for row in subset.itertuples(index=False):
            carry = row.model == "ICUKAN_Naive"
            ax.errorbar(
                row.false_positive_rate, row.sensitivity,
                xerr=[[row.false_positive_rate - row.false_positive_rate_ci_low],
                      [row.false_positive_rate_ci_high - row.false_positive_rate]],
                yerr=[[row.sensitivity - row.sensitivity_ci_low],
                      [row.sensitivity_ci_high - row.sensitivity]],
                fmt="X" if carry else MARKERS[row.cohort],
                color=PALETTE[row.model], ms=7 if carry else 4.2,
                lw=0.8, capsize=0, alpha=0.9, zorder=5 if carry else 4,
                mec="white", mew=0.4)
        top = max(0.12, float(subset.sensitivity_ci_high.max()) * 1.10)
        right = max(0.12, float(subset.false_positive_rate_ci_high.max()) * 1.10)
        ax.set_xlim(-right * 0.06, right); ax.set_ylim(-top * 0.06, top)
        ax.set_title(f"{organ} · {variable}")
        ax.set_xlabel("false-alarm rate")
        if col == 0:
            ax.set_ylabel("deterioration sensitivity")
            _style.panel_label(ax, "a", dx=-0.30)
            ax.annotate("carry-forward: never alarms", xy=(0, 0), xytext=(26, -2),
                        textcoords="offset points", fontsize=6.4, color=_style.CARRY,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=_style.CARRY, lw=0.7))

    # Bottom: how many of each arm's 12 cohort-by-component cells clear the
    # prespecified rule (above treat-none and carry-forward over a majority of
    # the frozen range), and how many still clear it once treat-all is added.
    # Treat-all is the third standard comparator in decision-curve analysis;
    # the original rule omitted it, so it is shown beside the original
    # count rather than replacing it.
    win_reg = ((curve.net_benefit > curve.treat_none_net_benefit)
               & (curve.net_benefit > curve.carryforward_net_benefit))
    win_all = win_reg & (curve.net_benefit > curve.treat_all_net_benefit)
    per_cell = (curve.assign(reg=win_reg, full=win_all)
                .groupby(["model", "cohort", "organ"])[["reg", "full"]].mean() > 0.5)
    counts = per_cell.groupby("model").sum().astype(int)
    order = [m for m in MODEL_ORDER if m in counts.index]
    order.sort(key=lambda m: (counts.loc[m, "reg"], counts.loc[m, "full"],
                              m == "This work"), reverse=True)

    ax = fig.add_subplot(gs[1, :])
    _style.soft_grid(ax, axis="x")
    y = np.arange(len(order))[::-1]
    for pos, model in zip(y, order):
        colour = PALETTE[model]
        reg, full = counts.loc[model, "reg"], counts.loc[model, "full"]
        ax.plot([full, reg], [pos, pos], color=colour, lw=2.2, alpha=0.45,
                solid_capstyle="round", zorder=3)
        ax.plot([reg], [pos], "o", color="white", mec=colour, mew=1.6, ms=6.5, zorder=4)
        ax.plot([full], [pos], "o", color=colour, mec="white", mew=0.8, ms=6.5, zorder=5)
        ax.text(reg + 0.25, pos, f"{reg}", va="center", fontsize=6.8, color=_style.MUTED)
        if full != reg:
            ax.text(full - 0.25, pos, f"{full}", va="center", ha="right", fontsize=6.8,
                    color=_style.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([_style.display(m) for m in order])
    for tick, model in zip(ax.get_yticklabels(), order):
        if model == "This work":
            tick.set_color(_style.OURS); tick.set_fontweight("bold")
    ax.set_xlim(-0.6, 13.2); ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("cohort-by-component cells (of 12) whose net benefit is above the "
                  "comparators over a majority of threshold probabilities 0.05–0.30")
    _style.panel_label(ax, "b", dx=-0.085, dy=1.10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="white", mec=_style.MUTED, mew=1.6, ms=6.5,
               label="above treat-none and carry-forward (original rule)"),
        Line2D([], [], marker="o", ls="", color=_style.MUTED, mec="white", ms=6.5,
               label="above treat-all as well"),
    ], loc="lower right", fontsize=6.6, handletextpad=0.3, borderaxespad=0.3)
    ax.text(0.9, y[order.index("ICUKAN_Naive")],
            "never raises an alarm, so its net benefit is exactly treat-none",
            va="center", ha="left", fontsize=6.6, color=_style.CARRY)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-frontier", type=Path, default=RESULTS / "e9_deterioration_frontier.csv")
    parser.add_argument("--out-curve", type=Path, default=RESULTS / "e9_decision_curve.csv")
    parser.add_argument("--figure", type=Path, default=FIGURES / "F8_deterioration_frontier.pdf")
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the existing frontier, curve, audit, and figure before replacement")
    args = parser.parse_args()
    generated = (args.out_frontier, args.out_curve, args.figure,
                 args.figure.with_suffix(".png"),
                 args.out_frontier.parent / "e9_deterioration_frontier_audit.json")
    prepare_generation(ROOT, generated, refresh=args.refresh_generated,
                       label="historical_deterioration_frontier")
    summary, bootstrap = _load()
    frontier, curve = _build(summary, bootstrap)
    args.out_frontier.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(args.out_frontier, index=False, float_format="%.17g")
    curve.to_csv(args.out_curve, index=False, float_format="%.17g")
    _plot(frontier, curve, args.figure)
    carry = frontier[frontier.model == "ICUKAN_Naive"]
    audit = {
        "analysis_plan_version": PLAN,
        "frontier_csv": str(args.out_frontier.resolve()),
        "decision_curve_csv": str(args.out_curve.resolve()),
        "figure": str(args.figure.resolve()),
        "threshold_probability_range": [float(THRESHOLDS.min()), float(THRESHOLDS.max())],
        "threshold_probability_step": 0.01,
        "carryforward_rows": int(len(carry)),
        "carryforward_sensitivity_all_zero": bool(np.allclose(carry.sensitivity, 0.0)),
        "carryforward_false_positive_rate_all_zero": bool(np.allclose(carry.false_positive_rate, 0.0)),
        "carryforward_ppv_all_undefined": bool(carry.ppv.isna().all()),
        "carryforward_nna_all_undefined": bool(carry.nna.isna().all()),
        "decision_value_rule": "only call decision value shown when point net benefit exceeds treat-none and carry-forward on a majority of frozen thresholds; otherwise report not shown",
    }
    (args.out_frontier.parent / "e9_deterioration_frontier_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(frontier.head(12).round(4).to_string(index=False))
    print(f"\nwrote {len(frontier)} frontier rows to {args.out_frontier}")
    print(f"wrote {len(curve)} decision-curve rows to {args.out_curve}")
    print(f"wrote figure to {args.figure}")


if __name__ == "__main__":
    main()
