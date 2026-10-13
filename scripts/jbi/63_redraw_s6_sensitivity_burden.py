#!/usr/bin/env python
"""Redraw Supplementary Figure S6 from the frozen P1 metrics table."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import argparse
import shutil
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results/jbi/p1_landmark_metrics.csv"
OUTPUT_PDF = ROOT / "results/paper/F8_p1_sensitivity_alarm_burden_supplement.pdf"
OUTPUT_PNG = ROOT / "results/paper/F8_p1_sensitivity_alarm_burden_supplement.png"
PLOT_FIELDS = (
    "window_sensitivity", "window_sensitivity_ci_low", "window_sensitivity_ci_high",
    "alarms_per_reference_eligible_patient_day",
    "alarms_per_reference_eligible_patient_day_ci_low",
    "alarms_per_reference_eligible_patient_day_ci_high",
)
COHORTS = ("eicu", "mimiciv")
MODELS = ("ICUKANForecaster", "GraFITi", "HyperIMTS")
STRATEGIES = (*MODELS, "no-alarm", "alert-all")
COMPONENTS = ("Platelets", "Bilirubin", "GCS", "Creatinine")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plot_rows() -> list[dict[str, Any]]:
    with METRICS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"cohort", "component", "model_or_strategy", "scope", *PLOT_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"P1 metric table is empty or missing fields: {sorted(required - set(rows[0] if rows else []))}")
    for row in rows:
        for field in PLOT_FIELDS:
            value = row[field].strip()
            row[field] = float(value) if value else math.nan
    selected = [row for row in rows if row["scope"] == "all_reference_windows"]
    expected_cohorts = {"eicu", "mimiciv"}
    expected_components = {"Platelets", "Bilirubin", "GCS", "Creatinine"}
    expected_strategies = {"ICUKANForecaster", "GraFITi", "HyperIMTS", "no-alarm", "alert-all"}
    if ({row["cohort"] for row in selected} != expected_cohorts
            or {row["component"] for row in selected} != expected_components
            or not expected_strategies.issubset({row["model_or_strategy"] for row in selected})):
        raise ValueError("P1 metrics do not cover the S6 cohorts, components, and strategies")
    return rows


def plot_s6(rows: list[dict[str, Any]]) -> None:
    """Adapt the original 41-script sensitivity/burden layout to shared styling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import style

    style.use_style()
    colors = {
        **{model: style.TRACKB_COLOUR[model] for model in MODELS},
        "no-alarm": style.CARRY,
        "alert-all": style.CATEGORICAL[1],
    }
    labels = {
        "ICUKANForecaster": "CarryGate", "GraFITi": "GraFITi", "HyperIMTS": "HyperIMTS",
        "no-alarm": "No alarm", "alert-all": "Alert all",
    }
    fig, axes = plt.subplots(
        2, 4, figsize=(7.4, 4.8), sharex="col", sharey="row",
        gridspec_kw={"wspace": 0.22, "hspace": 0.34},
    )
    for column, component in enumerate(COMPONENTS):
        for row_index, metric in enumerate(("window_sensitivity", "alarms_per_reference_eligible_patient_day")):
            axis = axes[row_index, column]
            style.soft_grid(axis, axis="y")
            for strategy_index, strategy in enumerate(STRATEGIES):
                for cohort_index, cohort in enumerate(COHORTS):
                    item = next((entry for entry in rows
                                 if entry["cohort"] == cohort
                                 and entry["component"] == component
                                 and entry["model_or_strategy"] == strategy
                                 and entry["scope"] == "all_reference_windows"), None)
                    if item is None:
                        continue
                    value = float(item.get(metric, math.nan))
                    if not math.isfinite(value):
                        continue
                    low = float(item.get(f"{metric}_ci_low", value))
                    high = float(item.get(f"{metric}_ci_high", value))
                    x = strategy_index + (-0.14 if cohort_index == 0 else 0.14)
                    axis.errorbar(
                        x, value,
                        yerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                        fmt=style.COHORT_MARKER[cohort], color=colors[strategy],
                        markerfacecolor=colors[strategy], markeredgecolor="white",
                        markeredgewidth=0.35, markersize=3.6, linewidth=0.75,
                        capsize=1.2, alpha=0.95, zorder=3,
                    )
            axis.set_title(component, color=style.INK, pad=3)
            if row_index == 0:
                axis.set_ylim(0.0, 1.05)
                if column == 0:
                    axis.set_ylabel("Sensitivity for measurement-\nascertained deterioration")
            else:
                axis.set_ylim(0.0, 1.25)
                if column == 0:
                    axis.set_ylabel("Alarms per reference-eligible patient-day")
            axis.set_xticks(range(len(STRATEGIES)))
            axis.set_xticklabels([labels[name] for name in STRATEGIES], rotation=24, ha="right")
            if row_index == 0:
                axis.tick_params(axis="x", labelbottom=False)
            axis.tick_params(axis="both", labelsize=6.1)
            axis.set_xlim(-0.48, len(STRATEGIES) - 0.52)

    strategy_handles = [
        Line2D([], [], color=colors[name], lw=1.5, label=labels[name])
        for name in STRATEGIES
    ]
    cohort_handles = [
        Line2D([], [], marker=style.COHORT_MARKER[cohort], ls="", color=style.INK,
               markersize=4, label=style.COHORT_LABEL[cohort])
        for cohort in COHORTS
    ]
    fig.suptitle("P1 held-out ICU use case · measurement-ascertained deterioration",
                 y=0.985, fontsize=9, color=style.INK)
    fig.legend(handles=strategy_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=5, handlelength=1.0, columnspacing=0.9, fontsize=6.2)
    fig.legend(handles=cohort_handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=2, handletextpad=0.35, columnspacing=1.2, fontsize=6.2)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.79, bottom=0.19)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive and replace the existing S6 PDF/PNG")
    args = parser.parse_args()
    if not METRICS.is_file():
        raise FileNotFoundError(METRICS)
    outputs = (OUTPUT_PDF, OUTPUT_PNG)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.refresh_generated:
        raise FileExistsError("S6 already exists; use --refresh-generated to archive it before replacement")
    if args.refresh_generated and (len(existing) != len(outputs) or not all(path.is_file() for path in outputs)):
        raise RuntimeError("--refresh-generated requires the complete existing S6 PDF/PNG pair")

    archive_path = None
    if args.refresh_generated:
        archive_root = ROOT / "results/jbi/_archive" / (
            "r11_s6_previous_generation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        archive_root.mkdir(parents=True, exist_ok=False)
        archive_entries = []
        for path in outputs:
            archived = archive_root / path.name
            shutil.copy2(path, archived)
            archive_entries.append({
                "path": str(path.relative_to(ROOT)),
                "archive_path": archived.name,
                "sha256": sha256(archived),
            })
        (archive_root / "archive_manifest.json").write_text(json.dumps({
            "reason": "preserve previous S6 PDF/PNG before shared-style redraw",
            "files": archive_entries,
        }, indent=2) + "\n", encoding="utf-8")
        archive_path = archive_root

    input_hash = sha256(METRICS)
    rows = load_plot_rows()
    plot_s6(rows)

    if sha256(METRICS) != input_hash:
        raise RuntimeError("P1 metrics changed during the figure redraw")
    print(json.dumps({
        "input": str(METRICS.relative_to(ROOT)),
        "input_sha256_unchanged": input_hash,
        "rows": len(rows),
        "output_pdf": str(OUTPUT_PDF.relative_to(ROOT)),
        "output_pdf_sha256": sha256(OUTPUT_PDF),
        "output_png": str(OUTPUT_PNG.relative_to(ROOT)),
        "output_png_sha256": sha256(OUTPUT_PNG),
        "archived_previous_generation": str(archive_path.relative_to(ROOT)) if archive_path else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
