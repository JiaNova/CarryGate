"""Shared figure style for the manuscript.

Colour is assigned by the job it does, not by which series happens to be first.
Four jobs appear in this paper:

*Reference.*  Carry-forward is the zero point of every metric in this study, not
a competitor, so it is drawn in neutral grey and never receives a hue.

*Identity.*  Ten arms is far more than any categorical palette can separate --
the previous palette put GraFITi and HyperIMTS 7.9 OKLab units apart, below the
floor at which full-colour readers can tell two marks apart.  In every figure
here the arms already sit on a labelled axis, so identity is carried by
position and the direct label.  Hue is freed to carry the *era* split that the
results keep reproducing, with this project's own arm reserved its own accent.

*Polarity.*  Persistence skill and excess misclassification are signed against
carry-forward, so they take a diverging ramp with a neutral midpoint at zero --
never a categorical hue, which would hide the sign.

*Magnitude.*  Measurement frequency and parameter count take a single-hue
sequential ramp, light to dark.

The categorical anchors are the Okabe-Ito set, which is designed for colour
vision deficiency.  Validated with the dataviz validator: lightness band, chroma
floor, CVD separation and normal-vision floor all pass; three hues fall below
3:1 contrast on white, which is why every figure carries direct labels or an
accompanying table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "jbi"))
from output_safety import prepare_generation, refresh_from_argv  # noqa: E402

INK = "#1A1A1A"
MUTED = "#6E6E6E"
GRID = "#E3E3E6"
PAPER = "#FFFFFF"

# --- reserved roles -------------------------------------------------------
OURS = "#D55E00"        # vermillion: this project's reference implementation
CARRY = "#4D4D4D"       # neutral grey: carry-forward, the zero point
MODERN = "#0072B2"      # blue: models published 2023 and later
LEGACY = "#E69F00"      # amber: models published 2021 and earlier

# --- categorical, for the few panels needing true identity (<= 5) ---------
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

# --- polarity: signed against carry-forward -------------------------------
WORSE = "#B2182B"
BETTER = "#2166AC"
NEUTRAL = "#F2F2F2"
DIVERGING = LinearSegmentedColormap.from_list(
    "skill_diverging", [WORSE, "#E8B4AE", NEUTRAL, "#9FC0DC", BETTER])

# --- magnitude ------------------------------------------------------------
SEQUENTIAL_STEPS = ["#D6E6F2", "#A8CBE3", "#6BA3CE", "#3C7FB4", "#0F4C81"]
SEQUENTIAL = LinearSegmentedColormap.from_list("frequency", SEQUENTIAL_STEPS)

ERA = {
    "HyperIMTS": "2023+", "GraFITi": "2023+", "Warpformer": "2023+",
    "APN": "2023+", "tPatchGNN": "2023+", "t-PatchGNN": "2023+",
    "GRU-D": "pre-2022", "GRU_D": "pre-2022", "mTAN": "pre-2022",
    "NeuralFlows": "pre-2022", "Neural Flows": "pre-2022",
}
ERA_COLOUR = {"2023+": MODERN, "pre-2022": LEGACY}

MODEL_NAME = "CarryGate"   # carry-forward, gated by staleness

_RESERVED = {
    "This work": OURS, "this work": OURS, "reference model": OURS,
    "ICUKANForecaster": OURS, MODEL_NAME: OURS,
    "NAIVE-1": CARRY, "NAIVE-1 (carry-forward)": CARRY,
    "carry-forward": CARRY, "Carry-forward": CARRY, "ICUKAN_Naive": CARRY,
}

# One spelling per arm, in every figure and table.  The result files use the
# harness's internal identifiers; readers should never see those.
DISPLAY = {
    "This work": MODEL_NAME, "this work": MODEL_NAME, "reference model": MODEL_NAME,
    "ICUKANForecaster": MODEL_NAME, "ICUKAN": MODEL_NAME,
    "ICUKAN_Naive": "carry-forward", "NAIVE-1": "carry-forward",
    "NAIVE-1 (carry-forward)": "carry-forward",
    "GRU_D": "GRU-D", "tPatchGNN": "t-PatchGNN", "NeuralFlows": "Neural Flows",
}


def display(name: str) -> str:
    """The one reader-facing spelling of an arm's name."""
    return DISPLAY.get(name, name)


def model_colour(name: str) -> str:
    """Hue carries era; this project and carry-forward keep reserved roles."""
    if name in _RESERVED:
        return _RESERVED[name]
    return ERA_COLOUR.get(ERA.get(name, ""), MUTED)


# Kept as a mapping for call sites that index it directly.
MODEL_COLOURS = {name: model_colour(name)
                 for name in list(_RESERVED) + list(ERA)}

COHORT_LABEL = {"p12": "PhysioNet-2012", "mimiciv": "MIMIC-IV", "eicu": "eICU-CRD"}
COHORT_MARKER = {"p12": "o", "mimiciv": "s", "eicu": "^"}
# Cohorts are a true categorical identity and appear as series in several
# panels.  They deliberately avoid the era hues: a reader who has just learned
# that blue means "published 2023+" must not meet blue again meaning MIMIC-IV.
# The trio's worst deuteranope separation is 7.6, inside the band that is legal
# only with secondary encoding -- hence COHORT_MARKER is applied wherever
# COHORT_COLOUR is, never colour alone.
COHORT_COLOUR = {"p12": "#009E73", "mimiciv": "#CC79A7", "eicu": "#56B4E9"}

# The two Track B panels compare four named arms and nothing else -- three
# strong baselines plus this project's.  Era hue cannot separate them (all
# three baselines are modern), and steps of one hue cannot either: three blues
# inside the lightness band come out 11.7 apart, under the floor at which full
# colour vision distinguishes them.  So these four arms get four hues, each one
# directly labelled.  HyperIMTS and Warpformer sit at 7.6 for deuteranopes,
# which is legal only with secondary encoding, hence TRACKB_MARKER.
TRACKB_COLOUR = {"ICUKANForecaster": OURS, "GraFITi": "#0072B2",
                 "HyperIMTS": "#009E73", "Warpformer": "#CC79A7"}
TRACKB_MARKER = {"ICUKANForecaster": "o", "GraFITi": "s",
                 "HyperIMTS": "^", "Warpformer": "D"}

# Training objective is the factor the paper's later figures turn on.  MSE is
# drawn as the light, de-emphasised state and masked L1 (the objective aligned
# with the L1-based metric) in the darkest step of the magnitude ramp, so the
# objective reads the same way in every panel that splits by it.  Architecture
# identity, where it is needed alongside, moves to TRACKB_MARKER.
OBJECTIVE_COLOUR = {"MSE": "#B4B4B4", "MAE": SEQUENTIAL_STEPS[-1]}
OBJECTIVE_LABEL = {"MSE": "MSE", "MAE": "masked L1"}

# The saturation-probe arms are meant to look alike: the registered finding is
# that the three encoder positions are not distinguishable.  Three steps of one
# neutral, separated by marker, show that instead of hiding it behind hue.
PROBE_ARMS = {"identity": (INK, "o"), "linear": (MUTED, "s"),
              "mlp": ("#A8A8A8", "^")}


def use_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.facecolor": PAPER,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "figure.facecolor": PAPER,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # editable text in the PDF, as journals require
        "ps.fonttype": 42,
    })


def tint(colour: str, amount: float = 0.88) -> tuple[float, float, float]:
    """A pale fill derived from a role colour, so fills never drift from hues."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    # Upper-case panel letters throughout, as the journal's figures use.
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, va="top", ha="left")


def soft_grid(ax, axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, lw=0.6, zorder=0)


def zero_reference(ax, value: float = 0.0, axis: str = "x", label: str | None = None) -> None:
    """Draw the carry-forward zero line: the one baseline every panel shares."""
    line = ax.axvline if axis == "x" else ax.axhline
    line(value, color=CARRY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    if label:
        if axis == "x":
            ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -8), textcoords="offset points",
                        color=CARRY, fontsize=6.5, ha="left", va="top")
        else:
            ax.annotate(label, xy=(0.0, value), xycoords=("axes fraction", "data"),
                        xytext=(3, 3), textcoords="offset points",
                        color=CARRY, fontsize=6.5, ha="left", va="bottom")


def era_legend(ax, include_ours: bool = True, **kwargs):
    """One legend for the whole paper: two eras, the reference, and this work."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=MODERN, lw=2.4, label="published 2023+"),
        Line2D([], [], color=LEGACY, lw=2.4, label="published ≤ 2021"),
        Line2D([], [], color=CARRY, lw=2.4, label="carry-forward"),
    ]
    if include_ours:
        handles.append(Line2D([], [], color=OURS, lw=2.4, label=MODEL_NAME))
    return ax.legend(handles=handles, **kwargs)


def save(fig, name: str, outdir="results/paper") -> None:
    out = Path(outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / f"{name}.{ext}" for ext in ("pdf", "png")]
    prepare_generation(ROOT, outputs, refresh=refresh_from_argv(),
                       label=f"figure_{name.lower()}")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf")
