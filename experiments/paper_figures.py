"""Draws the paper's figures at print size, typeset to match the paper body.

Usage:
    python -m experiments.paper_figures

Every figure is exactly 6.5 in wide, the paper's text width, so it is
included at \\linewidth with no scaling and 9 pt text prints as 9 pt.
Text is set in Times to match the body. Writes vector PDFs to
paper/figures/ and PNG previews to the scratchpad folder.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure

from experiments.evaluate import RESULTS, SETTINGS, load_runs
from experiments.figures import AQUA, BLUE, MUTED, ORANGE
from experiments.grid import ARCH_ORDER
from lhd.replay import replay

PAPER_FIGURES = RESULTS.parent / "paper" / "figures"
PREVIEWS = Path("/private/tmp/claude-501/-Users-sohanb-code/0d352244-cf6f-4b3c-a8d3-bdfac0853660/"
                "scratchpad/paperfigs")
TIMES = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
WIDTH = 6.5
SIZE = 9
TITLE_SIZE = 9.5
TEXT = "#222222"
SOFT = "#555555"
WHITE = "#ffffff"
GRIDLINE = "#e6e6e6"
DATA_WIDTH = 1.4
RULE_WIDTH = 0.7


def _deepen(color: str, amount: float) -> str:
    """The same hue a little darker, for contrast on white paper."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return to_hex(colorsys.hls_to_rgb(h, l * (1 - amount), s))


LHD_BLUE = _deepen(BLUE, 0.08)
ACC_ORANGE = _deepen(ORANGE, 0.06)
GREEN = _deepen(AQUA, 0.1)
RULE_GRAY = _deepen(MUTED, 0.06)
ARCH_COLORS = dict(zip(ARCH_ORDER, (LHD_BLUE, ACC_ORANGE, GREEN)))
ARCH_LABEL = {"mlp": "Fully connected", "transformer": "Transformer", "cnn": "CNN"}
PROPOSAL = ("frozen", "dead10", "dead25", "dead50")
DEAD = ("dead10", "dead25", "dead50")
DEAD_PCT = {"dead10": 10, "dead25": 25, "dead50": 50}
ACC3 = "Accuracy curve, 3 epochs below healthy"
TRADEOFF_LABELS = {
    "LHD": "LHD",
    ACC3: "Accuracy curve",
    "Accuracy curve, 1 epoch below healthy": "Accuracy curve, 1 epoch",
    "Loss curve, 3 epochs above healthy": "Loss curve",
    "Dead units check, 25%": "25% dead rule",
    "Dead units check, 50% (SageMaker DeadRelu default)": "SageMaker (50%)",
    "Dead units check, 70% (DeepDiagnosis Dead Node)": "DeepDiagnosis (70%)",
    "Dead units check, 80% (wandb-callbacks default)": "wandb-callbacks (80%)",
    "Any dead unit (wandb-callbacks verbose)": "Any dead unit",
}
HARM_GROUPS = (("under 0.5 pts", "Lost under 0.5 points"), ("0.5 to 2 pts", "Lost 0.5 to 2 points"),
               ("2 pts or more", "Lost 2 points or more"))
_RAW: dict[str, list[dict]] = {}


def setup() -> None:
    """One shared style for every paper figure: Times at 9 pt on white,
    thin dark axes with outward ticks, and a faint y grid."""
    font_manager.fontManager.addfont(str(TIMES))
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": SIZE,
        "text.color": TEXT,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "normal",
        "axes.titlecolor": TEXT,
        "axes.titlelocation": "left",
        "axes.labelsize": SIZE,
        "axes.labelcolor": TEXT,
        "axes.facecolor": WHITE,
        "axes.edgecolor": SOFT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.unicode_minus": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.5,
        "xtick.labelsize": SIZE,
        "ytick.labelsize": SIZE,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.fontsize": SIZE,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "lines.linewidth": DATA_WIDTH,
        "lines.solid_capstyle": "round",
        "figure.facecolor": WHITE,
        "figure.edgecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "pdf.fonttype": 42,
    })


def _figure(height: float, columns: int = 1, left: float = 0.5, right: float = 0.08,
            gaps: float | list[float] = 0.4, bottom: float = 0.45, top: float = 0.3,
            ratios: list[float] | None = None) -> tuple[Figure, list]:
    """A 6.5 in wide figure with panels placed by margins given in inches,
    so every panel's size is known before anything is drawn."""
    fig = Figure(figsize=(WIDTH, height))
    gaps = [gaps] * (columns - 1) if not isinstance(gaps, list) else gaps
    ratios = ratios or [1] * columns
    usable = WIDTH - left - right - sum(gaps)
    axes, x = [], left
    for i, ratio in enumerate(ratios):
        width = usable * ratio / sum(ratios)
        axes.append(fig.add_axes((x / WIDTH, bottom / height, width / WIDTH,
                                  (height - bottom - top) / height)))
        x += width + (gaps[i] if i < len(gaps) else 0)
    return fig, axes


def _panel(ax, letter: str, title: str) -> None:
    """Panel letter at the top left, with a short plain title beside it."""
    ax.annotate(f"({letter})", (0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=SIZE, color=TEXT, ha="left", va="baseline")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(14, 9), textcoords="offset points",
                fontsize=TITLE_SIZE, color=TEXT, ha="left", va="baseline")


def _points_per_unit(ax, axis: str) -> float:
    """How many printed points one data unit spans on an axis."""
    fig = ax.figure
    box = ax.get_position()
    if axis == "y":
        low, high = ax.get_ylim()
        return box.height * fig.get_figheight() * 72 / (high - low)
    low, high = ax.get_xlim()
    return box.width * fig.get_figwidth() * 72 / (high - low)


def _spread(targets: list[float], step: float, low: float, high: float) -> list[float]:
    """Positions at least step apart, each as close as it can be to its target."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    spots = [targets[i] for i in order]
    for _ in range(200):
        moved = False
        for k in range(1, len(spots)):
            overlap = step - (spots[k] - spots[k - 1])
            if overlap > 1e-9:
                spots[k - 1] -= overlap / 2
                spots[k] += overlap / 2
                moved = True
        if spots and spots[-1] > high:
            shift = spots[-1] - high
            spots = [s - shift for s in spots]
            moved = True
        if spots and spots[0] < low:
            shift = low - spots[0]
            spots = [s + shift for s in spots]
        if not moved:
            break
    result = [0.0] * len(targets)
    for k, i in enumerate(order):
        result[i] = spots[k]
    return result


def _end_labels(ax, items: list[tuple], x: float, gap_pt: float = 10.5, pad_pt: float = 5,
                low: float | None = None, high: float | None = None) -> None:
    """Label line ends just right of x, nudged apart so no two touch.

    items holds (y, text, color). A label moved off its line gets a
    thin leader in the line's color. Text is always dark.
    """
    per_point = 1 / _points_per_unit(ax, "y")
    y_low, y_high = ax.get_ylim()
    spots = _spread([item[0] for item in items], gap_pt * per_point,
                    y_low if low is None else low, y_high if high is None else high)
    dx = pad_pt / _points_per_unit(ax, "x")
    for (y, text, color), spot in zip(items, spots):
        leader = abs(spot - y) * _points_per_unit(ax, "y") > 2
        ax.annotate(text, (x, y), xytext=(x + dx * (2.4 if leader else 1), spot), textcoords="data",
                    fontsize=SIZE, color=TEXT, va="center", ha="left", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6, "relpos": (0, 0.5),
                                "shrinkA": 1.5, "shrinkB": 1.5} if leader else None)


def _rule(ax, x: float, text: str, ha: str = "left", y: float = 1.0) -> None:
    """A thin vertical reference rule with a small label at its top."""
    ax.axvline(x, color=SOFT, linewidth=RULE_WIDTH, zorder=1.5)
    ax.annotate(text, (x, y), xycoords=("data", "axes fraction"), xytext=(3 if ha == "left" else -3, -1),
                textcoords="offset points", fontsize=SIZE, color=SOFT, ha=ha, va="top")


def _percent_axis(ax, top: float = 100) -> None:
    ax.set_yticks([v for v in (0, 25, 50, 75, 100) if v <= top])


def _save(fig: Figure, name: str) -> None:
    """Write the vector PDF for the paper and a PNG preview, both at true size."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / f"{name}.pdf")
    fig.savefig(PREVIEWS / f"{name}.png", dpi=220)
    print(f"wrote {name}", flush=True)


def _raw(arch: str) -> list[dict]:
    if arch not in _RAW:
        _RAW[arch] = load_runs(arch)
    return _RAW[arch]


def _scored(arch: str, rows: list[dict]) -> list[dict]:
    """Saved runs for an architecture: every calibration run, plus only the
    fresh and broken runs already scored in runs.csv."""
    ids = {r["run_id"] for r in rows if r["arch"] == arch}
    return [run for run in _raw(arch)
            if run["metadata"]["role"] == "calibration" or run["metadata"]["run_id"] in ids]


def _widened(band: dict) -> tuple[np.ndarray, np.ndarray]:
    """The healthy range LHD checks against, widened by margin and floor."""
    return (band["low"] * (1 - SETTINGS["margin"]),
            band["high"] * (1 + SETTINGS["margin"]) + SETTINGS["floor"])


def pick_example(rows: list[dict], groups: dict, ref: dict) -> dict | None:
    """A CNN run with a frozen layer that LHD caught in the right layer with
    the right cause, whose accuracy at the alarm epoch was still inside the
    calibration range. Among those, the one the accuracy curve noticed
    latest (or never), then the one that lost the most accuracy."""
    by_id = {run["metadata"]["run_id"]: run for run in groups["fault"]}
    picks = []
    for row in rows:
        if not (row["arch"] == "cnn" and row["family"] == "frozen" and row["detected"]
                and row["layer_ok"] and row["cause_ok"] and row["run_id"] in by_id):
            continue
        run = by_id[row["run_id"]]
        report = replay(run["history"], ref["band"], **SETTINGS)
        if report is None:
            continue
        i = report["epoch"] - 1
        value = run["history"][i]["val_accuracy"]
        if not ref["accuracy"]["low"][i] <= value <= ref["accuracy"]["high"][i]:
            continue
        picks.append((-(row["acc_censored"] - row["lhd_censored"]), -(row["acc_drop_vs_twin"] or 0),
                      row["run_id"], run, report))
    if not picks:
        return None
    _, _, run_id, run, report = min(picks, key=lambda p: p[:3])
    return {"run": run, "report": report}
