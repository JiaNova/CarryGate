#!/usr/bin/env python
"""Generate every paper figure and table in one pass.

Follows ICU_KAN_figure_generation_spec.md: one script, final CSV/JSON sources
only, no intermediate files, and a hard failure rather than a partial asset when
a source is missing.  Writes to results/paper/ plus a source manifest recording
the sha256 and row count of every input.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper"
sys.path.insert(0, str(ROOT / "scripts" / "jbi"))
from output_safety import prepare_generation  # noqa: E402
SOURCES: dict[str, list[str]] = {}

SINGLE, DOUBLE = 3.50, 7.20  # inches: 89 mm and 183 mm
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MLP_C, KAN_C, REF_C = "#3B6EA5", "#D1682B", "#8A8A8A"


def src(asset: str, path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"[{asset}] missing source: {path.relative_to(ROOT)}")
    SOURCES.setdefault(asset, []).append(str(path.relative_to(ROOT)))
    return path


def read(asset: str, path: Path) -> pd.DataFrame:
    return pd.read_csv(src(asset, path), float_precision="round_trip")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def arm_means(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("config")["skill"].mean()


# ---------------------------------------------------------------- F1 ---------
def figure1() -> None:
    a = "F1"
    panels = [
        ("PhysioNet-2012", read(a, ROOT / "results/phase1/saturation_probe.csv")),
        ("MIMIC-IV", read(a, ROOT / "results/mimiciv_m2/f1/saturation_probe.csv")),
        ("eICU", read(a, ROOT / "results/phase3_eicu/c3/saturation_probe.csv")),
    ]
    order = ["identity", "linear", "mlp"]
    for name, frame in panels:
        if len(frame) != 9 or set(frame["encoder_type"]) != set(order):
            raise SystemExit(f"[F1] {name} probe must be 3 encoders x 3 seeds, got {len(frame)} rows")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), sharey=False)
    for ax, (name, frame) in zip(axes, panels):
        g = frame.groupby("encoder_type")["val_loss"]
        means, sds = g.mean().reindex(order), g.std().reindex(order)
        colours = [KAN_C if e == "identity" else REF_C for e in order]
        ax.bar(order, means, yerr=sds, capsize=2.5, color=colours, width=0.62,
               error_kw={"lw": 0.8})
        headroom = float(means["linear"] - means["mlp"])
        threshold = 2 * float(max(sds["linear"], sds["mlp"]))
        ax.set_title(f"{name}\nheadroom {headroom:+.5f} vs 2σ {threshold:.5f}", fontsize=7)
        ax.set_ylim(means.min() - 3 * sds.max(), means.max() + 3 * sds.max())
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].set_ylabel("validation MAE (lower is better)")
    fig.suptitle("Saturation probe: PhysioNet-2012 clears Gate P by a thin margin, "
                 "MIMIC-IV and eICU do not", fontsize=8, y=1.06)
    save(fig, "F1_saturation_probe")


# ---------------------------------------------------------------- F2 ---------
def figure2() -> None:
    a = "F2"
    panels = [
        ("PhysioNet-2012", read(a, ROOT / "results/phase5_h5b/raw_results.csv")),
        ("MIMIC-IV", read(a, ROOT / "results/mimiciv_m1/raw_results.csv")),
        ("eICU", read(a, ROOT / "results/phase3_eicu/e_m1/raw_results.csv")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    for ax, (name, frame) in zip(axes, panels):
        g = frame.groupby("config")[["skill", "nmae"]].mean()
        if len(g) < 5:
            raise SystemExit(f"[F2] {name} needs >= 5 configurations, got {len(g)}")
        rank_nmae = g["nmae"].rank(ascending=True)      # lower nMAE = better
        rank_skill = g["skill"].rank(ascending=False)   # higher skill = better
        rho = float(pd.Series(rank_nmae).corr(pd.Series(rank_skill), method="spearman"))
        ax.scatter(rank_nmae, rank_skill, s=26, color=MLP_C, zorder=3)
        for label, x, y in zip(g.index, rank_nmae, rank_skill):
            ax.annotate(label, (x, y), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points", color="#444444")
        lim = [0.4, len(g) + 0.6]
        ax.plot(lim, lim, ls=":", lw=0.8, color=REF_C, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{name}   Spearman ρ = {rho:+.2f}", fontsize=7)
        ax.set_xlabel("rank by nMAE")
    axes[0].set_ylabel("rank by persistence skill")
    fig.suptitle("The two metrics do not order the same models", fontsize=8, y=1.04)
    save(fig, "F2_metric_disagreement")


# ---------------------------------------------------------------- F3 ---------
def figure3() -> None:
    a = "F3"
    mech = arm_means(read(a, ROOT / "results/phase4_mechanism/final/raw_results.csv"))
    scope = arm_means(read(a, ROOT / "results/phase4_gate_scope_deterministic/final/raw_results.csv"))
    steps = [
        ("no gate", mech["G2-b0"]),
        ("residual scaling", mech["G2-b3"]),
        ("learned constant ω", scope["G1-scalar-pv"]),
        ("frozen learned ω", mech["G2-b2"]),
        ("frozen analytic ω*", mech["G2-b1"]),
        ("+ Δt, MLP edge", scope["G1-mlp-pv"]),
        ("+ adaptive basis", scope["G1-kan-pv"]),
    ]
    labels = [s[0] for s in steps]
    values = np.array([s[1] for s in steps])
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    ax.bar(labels, values, color=[REF_C] * 5 + [MLP_C, KAN_C], width=0.6)
    for i, (x, v) in enumerate(zip(labels, values)):
        ax.text(i, v + 0.0025, f"{v:.4f}", ha="center", fontsize=6.5)
        if i:
            ax.annotate(f"{v - values[i-1]:+.4f}", (i - 0.5, max(v, values[i-1]) + 0.010),
                        ha="center", fontsize=6, color="#666666")
    ax.set_ylabel("persistence skill")
    ax.set_ylim(0, values.max() * 1.28)
    ax.tick_params(axis="x", labelrotation=18)
    ax.set_title("Decomposing the gate on PhysioNet-2012 (Phase 4, 20 epochs, 5 seeds). "
                 "A freely learned constant gate buys exactly what residual scaling already buys.",
                 fontsize=7)
    save(fig, "F3_mechanism_waterfall")


# ---------------------------------------------------------------- F4 ---------
def _ladder_p12(a: str) -> pd.DataFrame:
    frame = read(a, ROOT / "results/phase5_ladder/ladder.csv")
    frame = frame[frame["converged"].astype(bool)]
    return frame.rename(columns={"P": "P", "mean_skill": "mean", "seed_sd": "sd"})[
        ["family", "P", "mean", "sd", "effective_capacity_low"]]


def _ladder_from_runs(a: str, spec: dict[tuple[str, int], tuple[Path, str]],
                      collapsed: set[int]) -> pd.DataFrame:
    rows = []
    for (family, P), (path, config) in spec.items():
        frame = read(a, path)
        sub = frame[frame["config"] == config]["skill"]
        if len(sub) != 15:
            raise SystemExit(f"[F4] {path.name}:{config} must have 15 seeds, got {len(sub)}")
        rows.append({"family": family, "P": P, "mean": float(sub.mean()),
                     "sd": float(sub.std(ddof=1)),
                     "effective_capacity_low": family == "adaptive_kan" and P in collapsed})
    return pd.DataFrame(rows)


def figure4() -> None:
    a = "F4"
    mimic_m1 = ROOT / "results/mimiciv_m1/raw_results.csv"
    mimic_lad = ROOT / "results/mimiciv_m1_ladder/raw_results.csv"
    eicu_m1 = ROOT / "results/phase3_eicu/e_m1/raw_results.csv"
    # epsilon is each cohort's own MLP capacity step, exactly as the frozen
    # reports computed it; the arrow must use the same rule as the ratio label.
    panels = [
        ("PhysioNet-2012", _ladder_p12(a), 7.72, 0.003609),
        ("MIMIC-IV", _ladder_from_runs(a, {
            ("mlp", 25): (mimic_m1, "M1-M8"), ("mlp", 193): (mimic_m1, "M1-M64"),
            ("mlp", 385): (mimic_lad, "M1-M128"),
            ("adaptive_kan", 25): (mimic_m1, "M1-K8"), ("adaptive_kan", 49): (mimic_m1, "M1-K16"),
            ("adaptive_kan", 385): (mimic_lad, "M1-K128"),
        }, collapsed={385}), 7.72, 0.005761957652670957),
        ("eICU", _ladder_from_runs(a, {
            ("mlp", 25): (eicu_m1, "E-M1-M8"), ("mlp", 193): (eicu_m1, "E-M1-M64"),
            ("mlp", 385): (ROOT / "results/phase3_eicu/c2/raw_results.csv", "C2-M128"),
            ("mlp", 769): (ROOT / "results/phase3_eicu/c2b/raw_results.csv", "C2-M256"),
            ("adaptive_kan", 25): (eicu_m1, "E-M1-K8"),
            ("adaptive_kan", 49): (eicu_m1, "E-M1-K16"),
        }, collapsed=set()), 15.40, 0.003829253827183965),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.4), sharey=False)
    for ax, (name, frame, ratio, epsilon) in zip(axes, panels):
        for family, colour, label in (("mlp", MLP_C, "MLP edge"),
                                      ("adaptive_kan", KAN_C, "adaptive basis")):
            sub = frame[frame["family"] == family].sort_values("P")
            x = np.log2(sub["P"].to_numpy(dtype=float))
            ax.plot(x, sub["mean"], "-", color=colour, lw=1.2, label=label, zorder=2)
            ax.fill_between(x, sub["mean"] - sub["sd"], sub["mean"] + sub["sd"],
                            color=colour, alpha=0.18, lw=0, zorder=1)
            # Mixed object column (bool / numpy.bool_ / NaN): `is True` misses
            # numpy.bool_, `astype(bool)` turns NaN into True.  Normalise once.
            hollow = sub["effective_capacity_low"].map(
                lambda v: str(v).lower() == "true").to_numpy()
            ax.scatter(x[~hollow], sub["mean"].to_numpy()[~hollow], s=22, color=colour, zorder=3)
            ax.scatter(x[hollow], sub["mean"].to_numpy()[hollow], s=26, facecolors="none",
                       edgecolors=colour, lw=1.0, zorder=3)
        mlp = frame[frame["family"] == "mlp"]
        peak = mlp.loc[mlp["mean"].idxmax()]
        kan = frame[frame["family"] == "adaptive_kan"]
        equivalent = kan[(kan["mean"] - peak["mean"]).abs() < epsilon]["P"]
        if equivalent.empty:
            raise SystemExit(f"[F4] {name}: no adaptive capacity is within epsilon of the MLP peak")
        kan_min = float(equivalent.min())
        if not np.isclose(peak["P"] / kan_min, ratio, atol=0.01):
            raise SystemExit(f"[F4] {name}: arrow {peak['P']}/{kan_min:.0f} disagrees with "
                             f"the registered ratio {ratio}")
        ax.annotate("", xy=(np.log2(kan_min), peak["mean"]), xytext=(np.log2(peak["P"]), peak["mean"]),
                    arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 0.9})
        ax.text((np.log2(kan_min) + np.log2(peak["P"])) / 2, peak["mean"] + 0.0016,
                f"{ratio:.2f}×", ha="center", fontsize=7)
        ticks = sorted(frame["P"].unique())
        ax.set_xticks(np.log2(ticks)); ax.set_xticklabels([str(int(t)) for t in ticks])
        ax.set_xlabel("gate parameters per variable")
        ax.set_title(f"{name}   MLP peak at P={int(peak['P'])}", fontsize=7)
    axes[0].set_ylabel("persistence skill")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("The adaptive basis reaches the MLP capacity peak with far fewer parameters "
                 "(hollow markers: collapsed centres)", fontsize=8, y=1.04)
    save(fig, "F4_capacity_ladder")


# ---------------------------------------------------------------- F5 ---------
def _curves(a: str, directory: Path, config: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(src(a, directory).glob(f"{config}_*.npz"))
    if len(paths) != 15:
        raise SystemExit(f"[F5] {directory.name}/{config} needs 15 seeds, got {len(paths)}")
    stack, hours = [], None
    for path in paths:
        with np.load(path) as data:
            hours = np.asarray(data["hours"], dtype=float)
            stack.append(np.asarray(data["omega"], dtype=float))
    return hours, np.stack(stack)


def _delta_by_variable(a: str, directory: Path, kan: str, mlp: str) -> pd.Series:
    frames = {}
    for config in (kan, mlp):
        parts = [read(a, p) for p in sorted(src(a, directory).glob(f"{config}_*_per_variable.csv"))]
        frames[config] = pd.concat(parts).groupby("variable")["skill"].mean()
    return (frames[kan] - frames[mlp]).dropna()


def figure5() -> None:
    a = "F5"
    hours, kan = _curves(a, ROOT / "results/phase5_h5b/gate_curves", "H-5b")
    _, mlp = _curves(a, ROOT / "results/phase5_h5b/gate_curves", "H-3b")
    delta = _delta_by_variable(a, ROOT / "results/phase5_h5b/stratified", "H-5b", "H-3b")
    ordered = delta.sort_values()
    chosen = list(ordered.index[-3:][::-1]) + list(ordered.index[:3])
    # Staleness is a property of the task, identical across arms, so it is read
    # from an arm already in this directory rather than a second matrix.
    stale = np.load(src(a, ROOT / "results/phase5_h5b/predictions/H-3b_0.npz"))["staleness"]
    lo, hi = np.quantile(stale[np.isfinite(stale)], [0.001, 0.999])
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.4), sharex=True, sharey=True)
    for ax, variable in zip(axes.ravel(), chosen):
        for stackk, colour, label in ((mlp, MLP_C, "MLP"), (kan, KAN_C, "adaptive")):
            curve = stackk[:, int(variable), :]
            ax.plot(hours, curve.mean(0), color=colour, lw=1.1, label=label, zorder=3)
            ax.fill_between(hours, curve.min(0), curve.max(0), color=colour, alpha=0.20, lw=0)
        if lo > hours.min():
            ax.axvspan(hours.min(), lo, color=REF_C, alpha=0.16, lw=0)
        if hi < hours.max():
            ax.axvspan(hi, hours.max(), color=REF_C, alpha=0.16, lw=0)
        ax.set_title(f"v{int(variable)}   ΔS = {delta[variable]:+.3f}", fontsize=7)
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.supxlabel("staleness δ (hours)", fontsize=8)
    fig.supylabel("ω(δ)", fontsize=8)
    fig.suptitle("Recency functions at identical parameter budgets, PhysioNet-2012 "
                 "(band = 15-seed range; grey = outside the observed staleness range)",
                 fontsize=8, y=1.02)
    save(fig, "F5_gate_shapes")


# ---------------------------------------------------------------- F6 ---------
def figure6() -> None:
    a = "F6"
    raw = read(a, ROOT / "results/phase3_eicu/e_loho/raw_results.csv")
    meta = json.loads(src(a, ROOT / "results/phase3_eicu/e_loho/loho_summary.json").read_text())
    held = {int(f["fold"]): int(f["n_held_hospitals"]) for f in meta["folds"]}
    contrasts = [("K-M8", "K-0", "gate vs no gate"), ("K-K8", "K-M8", "adaptive vs MLP edge")]
    # sharey would let the right panel's empty labels wipe the left panel's too;
    # both panels get identical ticks anyway, so alignment needs no sharing.
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharex=True)
    lo, hi = 0.0, 0.0
    stats = []
    for left, right, title in contrasts:
        row = []
        for fold in sorted(raw["fold"].unique()):
            pivot = raw[raw["fold"] == fold].pivot_table(index="seed", columns="config", values="skill")
            diff = (pivot[left] - pivot[right]).dropna().to_numpy()
            mean = float(diff.mean())
            half = 1.96 * float(diff.std(ddof=1)) / np.sqrt(len(diff))
            row.append((int(fold), mean, half, int((diff > 0).sum()), len(diff)))
            lo, hi = min(lo, mean - half), max(hi, mean + half)
        stats.append((title, row))
    for index, (ax, (title, row)) in enumerate(zip(axes, stats)):
        ys = np.arange(len(row))[::-1]
        for y, (fold, mean, half, wins, n) in zip(ys, row):
            ax.errorbar(mean, y, xerr=half, fmt="o", ms=4, lw=1.1, capsize=2.5,
                        color=KAN_C if "adaptive" in title else MLP_C)
            # Axes-fraction x keeps the seed counts inside their own panel
            # instead of colliding with the neighbouring panel's labels.
            ax.text(0.985, y, f"{wins}/{n}", va="center", ha="right", fontsize=6,
                    color="#555555", transform=ax.get_yaxis_transform())
        ax.axvline(0.0, color="#222222", lw=1.0, zorder=0)
        ax.set_yticks(ys)
        if index == 0:
            ax.set_yticklabels([f"fold {fold} ({held[fold]} hosp.)" for fold, *_ in row])
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Δ persistence skill")
    # Shared scale is the whole point of this figure: it is what shows the two
    # contrasts differ by more than an order of magnitude.
    for ax in axes:
        ax.set_xlim(lo - 0.006, hi * 1.18)
    fig.suptitle("Leave-one-hospital-out on eICU (145 hospitals, 5 folds, shared x-axis)",
                 fontsize=8, y=1.05)
    save(fig, "F6_loho_forest")


# ---------------------------------------------------------------- tables -----
def tables(output_dir: Path | None = None) -> None:
    output_dir = output_dir or OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    a = "T"
    # PhysioNet-2012 has no 40-epoch no-op or shared arm: those live in the
    # Phase 4 / Line H matrices at 20 epochs.  The budget is therefore carried
    # as a column rather than hidden, and every row names its own matrix.
    P12_H = ROOT / "results/phase5_h5b/raw_results.csv"
    P12_LADDER = ROOT / "results/phase5_ladder/raw_results.csv"
    P12_LINE_H = ROOT / "results/phase5_gate_basis/raw_results.csv"
    P12_SCOPE = ROOT / "results/phase4_gate_scope_deterministic/final/raw_results.csv"
    MIMIC = ROOT / "results/mimiciv_m1/raw_results.csv"
    EICU = ROOT / "results/phase3_eicu/e_m1/raw_results.csv"
    spec = {
        "PhysioNet-2012": [
            # Both from the same Phase 4 matrix so the two rows are directly
            # comparable; Line H's H-0 is the identical configuration at 15 seeds.
            ("no gate", P12_SCOPE, "F3-0", 20), ("shared MLP@25", P12_SCOPE, "G1-mlp-sh", 20),
            ("MLP@25", P12_H, "H-3b", 40), ("MLP@193", P12_H, "H-4b", 40),
            ("adaptive@25", P12_H, "H-5b", 40), ("adaptive@49", P12_LADDER, "L-K16", 40),
        ],
        "MIMIC-IV": [
            ("no gate", MIMIC, "M1-noop", 80), ("shared MLP@25", MIMIC, "M1-sh", 80),
            ("MLP@25", MIMIC, "M1-M8", 80), ("MLP@193", MIMIC, "M1-M64", 80),
            ("adaptive@25", MIMIC, "M1-K8", 80), ("adaptive@49", MIMIC, "M1-K16", 80),
        ],
        "eICU": [
            ("no gate", EICU, "E-M1-noop", 80), ("shared MLP@25", EICU, "E-M1-sh", 80),
            ("MLP@25", EICU, "E-M1-M8", 80), ("MLP@193", EICU, "E-M1-M64", 80),
            ("adaptive@25", EICU, "E-M1-K8", 80), ("adaptive@49", EICU, "E-M1-K16", 80),
        ],
    }
    rows = []
    for dataset, entries in spec.items():
        for label, path, config, budget in entries:
            frame = read(a, path)
            sub = frame[frame["config"] == config]
            if sub.empty:
                raise SystemExit(f"[T1] {dataset}/{label}: {config} absent from {path.name}")
            if "epochs" in sub and "patience" in sub:
                early = int((sub["best_epoch"] <= sub["epochs"] - sub["patience"]).sum())
            else:
                early = int((sub["best_epoch"] <= budget - 5).sum())
            rows.append({"dataset": dataset, "arm": label, "config": config,
                         "epoch_budget": budget, "n_seeds": int(len(sub)),
                         "mean_skill": round(float(sub["skill"].mean()), 6),
                         "seed_sd": round(float(sub["skill"].std(ddof=1)), 6),
                         "early_stopped": early,
                         "source": str(path.relative_to(ROOT))})
    table1 = pd.DataFrame(rows)
    for dataset, entries in spec.items():
        got = set(table1[table1["dataset"] == dataset]["arm"])
        if got != {label for label, *_ in entries}:
            raise SystemExit(f"[T1] {dataset} is missing arms: {sorted({l for l,*_ in entries} - got)}")
    table1.to_csv(output_dir / "T1_main_results.csv", index=False)

    reports = {
        "PhysioNet-2012": ROOT / "results/phase5_h5b/h5b_report.json",
        "MIMIC-IV": ROOT / "results/mimiciv_m1/mimiciv_m1_report.json",
        "eICU": ROOT / "results/phase3_eicu/eicu_external_validation_report.json",
    }
    # The three reports nest their registered comparisons differently; pull the
    # entries out explicitly rather than guessing, and assert every dataset
    # actually contributed -- a table missing a cohort is worse than no table.
    getters = {
        "PhysioNet-2012": lambda d: d["registered"],
        "MIMIC-IV": lambda d: list(d["comparisons"].values()),
        "eICU": lambda d: list(d["e_m1"]["comparisons"].values()),
    }
    t2 = []
    for dataset, path in reports.items():
        payload = json.loads(src(a, path).read_text())
        entries = getters[dataset](payload)
        if not entries:
            raise SystemExit(f"[T2] {dataset}: no registered comparisons found in {path.name}")
        for item in entries:
            t2.append({"dataset": dataset, "id": item.get("id"),
                       "left": item.get("left"), "right": item.get("right"),
                       "delta": item.get("delta_skill", item.get("mean_difference")),
                       "wins": item.get("wins"), "n_seeds": item.get("n_seeds"),
                       "sigma_seed": item.get("sigma_seed"), "epsilon": item.get("epsilon"),
                       "p_holm": item.get("p_holm"),
                       "pass": item.get("passes", item.get("pass"))})
    table2 = pd.DataFrame(t2)
    missing = set(reports) - set(table2["dataset"])
    if missing:
        raise SystemExit(f"[T2] datasets absent from the table: {sorted(missing)}")
    table2.to_csv(output_dir / "T2_registered_comparisons.csv", index=False)

    audit = [
        ("Phase 0", "KAN head is worse", "equal-parameter matching squeezed the KAN head to 64->5->1",
         "three-track parameter matching"),
        ("Phase 1", "the KAN encoder helps", "the matrix hard-coded one grouping; the control tied (0.68523 vs 0.68307)",
         "mandatory no-op / ablation control"),
        ("Phase 2", "a basis rescue closed H1", "the arm was only compared with an MLP, never with no component at all",
         "mandatory no-op anchor"),
        ("Phase 3", "a 4.5x scalar-gate advantage", "the scalar gate was the only per-variable arm; the rest were forced shared",
         "orthogonal scope axis"),
        ("Phase 5", "KAN direction closes (NO-GO)", "the tested gate had frozen centres, not the adaptive ones the claim came from; basis arms were also epoch-truncated",
         "bitwise replay table, config fingerprints, convergence criterion"),
    ]
    pd.DataFrame(audit, columns=["phase", "apparent conclusion", "actual cause",
                                 "control that caught it"]).to_csv(output_dir / "T3_audit.csv", index=False)

    def shared_minus_noop(frame: pd.DataFrame, shared: str, noop: str) -> tuple[float, float]:
        means = arm_means(frame)
        sd = float(frame[frame["config"] == shared]["skill"].std(ddof=1))
        return float(means[shared] - means[noop]), sd

    p12 = read(a, ROOT / "results/phase4_gate_scope_deterministic/final/raw_results.csv")
    mimic = read(a, ROOT / "results/mimiciv_m1/raw_results.csv")
    eicu = read(a, ROOT / "results/phase3_eicu/e_m1/raw_results.csv")
    diffs = [
        {"property": "shared gate minus no gate", "PhysioNet-2012": "%+.6f" % shared_minus_noop(p12, "G1-mlp-sh", "F3-0")[0],
         "MIMIC-IV": "%+.6f" % shared_minus_noop(mimic, "M1-sh", "M1-noop")[0],
         "eICU": "%+.6f" % shared_minus_noop(eicu, "E-M1-sh", "E-M1-noop")[0]},
        {"property": "shared-gate seed SD", "PhysioNet-2012": "%.6f" % shared_minus_noop(p12, "G1-mlp-sh", "F3-0")[1],
         "MIMIC-IV": "%.6f" % shared_minus_noop(mimic, "M1-sh", "M1-noop")[1],
         "eICU": "%.6f" % shared_minus_noop(eicu, "E-M1-sh", "E-M1-noop")[1]},
        {"property": "MLP capacity peak (P)", "PhysioNet-2012": "193", "MIMIC-IV": "193", "eICU": "385"},
        {"property": "parameter-efficiency ratio E", "PhysioNet-2012": "7.72", "MIMIC-IV": "7.72", "eICU": "15.40"},
        {"property": "model-side staleness median (h)", "PhysioNet-2012": "14.00", "MIMIC-IV": "13.00", "eICU": "13.25"},
        {"property": "standardization fitted on", "PhysioNet-2012": "own train split",
         "MIMIC-IV": "own train split", "eICU": "MIMIC-IV train split (C1 shows this is conservative)"},
        {"property": "GCS p1/p50", "PhysioNet-2012": "3 / 13", "MIMIC-IV": "7 / 15", "eICU": "3 / 14"},
        {"property": "MechVent events per stay", "PhysioNet-2012": "7.6", "MIMIC-IV": "0.5", "eICU": "0.8"},
        {"property": "TroponinI train events", "PhysioNet-2012": "1175", "MIMIC-IV": "0 (structurally missing)", "eICU": "7285"},
        {"property": "Weight semantics", "PhysioNet-2012": "longitudinal", "MIMIC-IV": "longitudinal",
         "eICU": "single admission weight"},
    ]
    pd.DataFrame(diffs).to_csv(output_dir / "T4_dataset_differences.csv", index=False)
    for name in ("T1_main_results", "T2_registered_comparisons", "T3_audit", "T4_dataset_differences"):
        print(f"  {name}.csv")


TEX = ROOT / "paper/tables"
CAPTIONS = {
    "T1_main_results": ("Main results on the three cohorts using the original stay-level "
                        "splits for MIMIC-IV and eICU and the record-level PhysioNet-2012 split "
                        "(patient identifiers are unavailable there). Every row names its own "
                        "matrix and epoch budget; PhysioNet-2012 has no 40-epoch no-op "
                        "or shared arm, so those two rows come from the 20-epoch "
                        "matrices and are labelled as such.", "tab:t1"),
    "T2_registered_comparisons": ("Prespecified comparisons on the original stay-level splits; "
                                  "PhysioNet-2012 is record-level because patient identifiers are unavailable. "
                                  "Holm correction is applied "
                                  "within each cohort's family of four; X4/C2 is a paired "
                                  "two one-sided test against that cohort's own capacity "
                                  "step.", "tab:t2"),
    "T3_audit": ("Prespecified controls and the errors they caught during the study.", "tab:t3"),
    "T4_dataset_differences": ("Differences between the three cohorts that are reported "
                               "rather than harmonised away.", "tab:t4"),
}


# Tables whose natural width exceeds the text block.
WIDE_TABLES = {"T2_registered_comparisons"}
# Full float repr is both unreadable and far too wide; four significant decimals
# is more than the seed noise in any of these quantities supports.
FLOAT_DECIMALS = 4
# Column specifications for tables carrying free text that has to wrap.
TEXT_COLUMN_WIDTHS = {
    "T3_audit": "p{0.10\\textwidth}p{0.27\\textwidth}p{0.29\\textwidth}p{0.24\\textwidth}",
    "T4_dataset_differences": ("p{0.19\\textwidth}p{0.24\\textwidth}"
                               "p{0.24\\textwidth}p{0.24\\textwidth}"),
}


def _format_cell(value) -> str:
    """Render one cell: floats to a readable precision, everything else as text."""
    if isinstance(value, float):
        if value != value:
            return "--"
        # counts arrive as floats; do not print them with four decimals
        if value.is_integer() and abs(value) >= 1:
            return str(int(value))
        text = f"{value:.{FLOAT_DECIMALS}f}"
        # keep a value that would round away to zero informative
        if float(text) == 0.0 and value != 0.0:
            text = f"{value:.1e}"
        return text
    return str(value)


def latex_tables(csv_dir: Path | None = None, output_dir: Path | None = None) -> None:
    """Emit booktabs versions of T1-T4 for the manuscript."""
    csv_dir = csv_dir or OUT
    output_dir = output_dir or TEX
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (caption, label) in CAPTIONS.items():
        frame = pd.read_csv(csv_dir / f"{name}.csv", float_precision="round_trip")
        if name == "T1_main_results":
            frame = frame.drop(columns=["source"])
        body = frame.astype(object).where(frame.notna(), "--")
        # Free-text columns must wrap or the table runs off the page; numeric
        # columns stay right-aligned. T3 is almost entirely prose, T2 is wide
        # enough to need the smaller size and tighter column separation.
        text_cols = {name: widths for name, widths in TEXT_COLUMN_WIDTHS.items()}
        if name in text_cols:
            columns = text_cols[name]
        else:
            columns = "l" * 2 + "r" * (len(frame.columns) - 2)
        size = "\\scriptsize" if name in WIDE_TABLES else "\\footnotesize"
        lines = [
            "\\begin{table}[htbp]\\centering" + size,
            "\\setlength{\\tabcolsep}{4pt}" if name in WIDE_TABLES else "",
            "\\caption{" + caption + "}\\label{" + label + "}",
            "\\begin{tabular}{" + columns + "}",
            "\\toprule",
            " & ".join(c.replace("_", " ") for c in frame.columns) + " \\\\",
            "\\midrule",
        ]
        for _, row in body.iterrows():
            cells = [_format_cell(v).replace("_", "\\_").replace("%", "\\%")
                     .replace("&", "\\&") for v in row]
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (output_dir / f"{name}.tex").write_text(
            "\n".join(x for x in lines if x) + "\n", encoding="utf-8")
        print(f"  paper/tables/{name}.tex")


def manifest() -> None:
    entries = {}
    for asset, paths in SOURCES.items():
        rows = []
        for rel in sorted(set(paths)):
            path = ROOT / rel
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None
            n = None
            if path.is_file() and path.suffix == ".csv":
                n = int(len(pd.read_csv(path, float_precision="round_trip")))
            rows.append({"path": rel, "sha256": digest, "rows": n})
        entries[asset] = rows
    (OUT / "asset_sources.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  asset_sources.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true",
                        help="generate T1-T4 without touching the legacy figures")
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive any existing generated files before replacing them")
    args = parser.parse_args()
    table_names = ("T1_main_results", "T2_registered_comparisons", "T3_audit",
                   "T4_dataset_differences")
    outputs = [OUT / f"{name}.csv" for name in table_names]
    outputs += [TEX / f"{name}.tex" for name in table_names]
    if args.tables_only:
        prepare_generation(ROOT, outputs, refresh=args.refresh_generated,
                           label="paper_tables_t1_t4")
        OUT.mkdir(parents=True, exist_ok=True)
        tables()
        latex_tables()
        return

    figure_names = ("F1_saturation_probe", "F2_metric_disagreement",
                    "F3_mechanism_waterfall", "F4_capacity_ladder",
                    "F5_gate_shapes", "F6_loho_forest")
    outputs += [OUT / f"{name}.{extension}" for name in figure_names
                for extension in ("pdf", "png")]
    outputs.append(OUT / "asset_sources.json")
    prepare_generation(ROOT, outputs, refresh=args.refresh_generated,
                       label="legacy_paper_assets")
    OUT.mkdir(parents=True, exist_ok=True)
    print("figures:")
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6()
    print("tables:")
    tables()
    latex_tables()
    manifest()


if __name__ == "__main__":
    main()
