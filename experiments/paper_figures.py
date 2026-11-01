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
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure

from experiments.evaluate import RESULTS, SETTINGS, load_runs, reference, split_runs
from experiments.figures import AQUA, BAND, BLUE, MUTED, ORANGE, read_csv
from experiments.grid import ARCH_ORDER
from lhd.detector import band_position
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


def how_lhd_works(rows: list[dict]) -> dict:
    """One CNN run with a frozen layer: the layer's weight change leaves the
    healthy range at once, while validation accuracy still looks normal."""
    groups = split_runs(_scored("cnn", rows))
    ref = reference(groups)
    example = pick_example(rows, groups, ref)
    if example is None:
        raise ValueError("no CNN frozen run fits")
    run, report = example["run"], example["report"]
    layer = run["metadata"]["fault"]["layer"]
    alarm = report["epoch"]
    history = run["history"]
    epochs = np.arange(1, len(history) + 1)
    stop = alarm - 1

    fig, (left, right) = _figure(3.0, 2, left=0.5, right=0.12, gaps=0.62, bottom=0.42, top=0.3)

    band = ref["band"][layer]["weight_change"]
    low, high = _widened(band)
    left.fill_between(epochs, low, high, color=BAND, linewidth=0, zorder=0.5)
    for cal in groups["calibration"]:
        left.plot(epochs, [e["layers"][layer]["weight_change"] for e in cal["history"]],
                  color=RULE_GRAY, linewidth=0.5, alpha=0.75, zorder=1)
    values = np.array([e["layers"][layer]["weight_change"] for e in history])
    left.plot(epochs[:alarm], values[:alarm], color=LHD_BLUE, linewidth=1.8, zorder=3)
    left.plot(epochs[stop:], values[stop:], color=LHD_BLUE, linewidth=1.2, alpha=0.3, zorder=3)
    outside = [i for i in range(alarm)
               if band_position(values[i], band["low"][i], band["high"][i], SETTINGS["margin"],
                                SETTINGS["floor"]) != "inside"]
    left.plot(epochs[outside], values[outside], "o", color=LHD_BLUE, markersize=4.2,
              markeredgecolor=WHITE, markeredgewidth=0.8, zorder=4)
    span = high.max()
    left.set_ylim(-0.17 * span, 1.12 * span)
    left.set_yticks([0, 0.5, 1.0, 1.5] if span > 1.2 else [0, 0.25, 0.5, 0.75, 1.0])
    left.set_ylabel("Relative weight change")
    _panel(left, "a", f"Weight change of layer {layer}")
    _rule(left, alarm, "LHD stops the run")
    left.annotate(f"{len(outside)} epochs outside", (np.mean(epochs[outside]), 0), xytext=(0, -6),
                  textcoords="offset points", ha="center", va="top", fontsize=SIZE, color=TEXT, zorder=5,
                  bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.5})
    left.annotate(f"Broken run, {layer} frozen", (epochs[-1], 0), xytext=(0, -6), textcoords="offset points",
                  ha="right", va="top", fontsize=SIZE, color=TEXT)
    curves = np.array([[e["layers"][layer]["weight_change"] for e in cal["history"]]
                       for cal in groups["calibration"]])
    leader = {"arrowstyle": "-", "color": SOFT, "linewidth": 0.6, "shrinkA": 1, "shrinkB": 1}
    x_band = alarm + 0.7
    y_band = (np.interp(x_band, epochs, curves.max(axis=0)) + np.interp(x_band, epochs, high)) / 2
    left.annotate("Healthy range", (x_band, y_band), xytext=(12, 22), textcoords="offset points",
                  fontsize=SIZE, color=TEXT, va="bottom", ha="left", arrowprops=leader)
    x_runs = alarm + 2.2
    y_runs = np.interp(x_runs, epochs, np.median(curves, axis=0))
    left.annotate(f"{len(curves)} healthy runs", (x_runs, y_runs), xytext=(12, 20), textcoords="offset points",
                  fontsize=SIZE, color=TEXT, va="bottom", ha="left", arrowprops=leader)

    acc_low, acc_high = 100 * ref["accuracy"]["low"], 100 * ref["accuracy"]["high"]
    accuracy = np.array([100 * e["val_accuracy"] for e in history])
    inside = acc_low[stop] <= accuracy[stop] <= acc_high[stop]
    if not inside:
        raise ValueError("picked run is not inside the accuracy range at the alarm")
    right.fill_between(epochs, acc_low, acc_high, color=BAND, linewidth=0, zorder=0.5)
    right.plot(epochs[:alarm], accuracy[:alarm], color=LHD_BLUE, linewidth=1.8, zorder=3)
    right.plot(epochs[stop:], accuracy[stop:], color=LHD_BLUE, linewidth=1.2, alpha=0.3, zorder=3)
    right.plot([alarm], [accuracy[stop]], "o", color=LHD_BLUE, markersize=4.2, markeredgecolor=WHITE,
               markeredgewidth=0.8, zorder=4)
    lowest = min(acc_low.min(), accuracy.min())
    right.set_ylim(lowest - 10, max(acc_high.max(), accuracy.max()) + 9)
    right.set_ylabel("Validation accuracy (%)")
    _panel(right, "b", "Validation accuracy of the same run")
    _rule(right, alarm, "LHD stops the run")
    right.annotate("Still inside the healthy range", (alarm, accuracy[stop]), xytext=(alarm + 1.6, accuracy[stop] - 17),
                   textcoords="data", fontsize=SIZE, color=TEXT, va="center",
                   arrowprops={"arrowstyle": "-|>", "color": SOFT, "linewidth": 0.6, "mutation_scale": 6,
                               "shrinkA": 2, "shrinkB": 4})
    right.annotate("Healthy range", (epochs[-1], acc_high[-1]), xytext=(0, 4), textcoords="offset points",
                   ha="right", va="bottom", fontsize=SIZE, color=TEXT)
    right.annotate("Broken run, if not stopped", (epochs[-1] - 0.4, np.interp(epochs[-1] - 0.4, epochs, accuracy)),
                   xytext=(epochs[-1], acc_low[-1] - 7), textcoords="data", ha="right", va="center",
                   fontsize=SIZE, color=TEXT,
                   arrowprops={"arrowstyle": "-", "color": SOFT, "linewidth": 0.6, "relpos": (0.95, 1),
                               "shrinkA": 1, "shrinkB": 1})
    for ax in (left, right):
        ax.set_xlim(0.6, epochs[-1] + 0.4)
        ax.set_xticks([e for e in epochs if e % 2 == 1])
        ax.set_xlabel("Epoch")
    _save(fig, "how_lhd_works")
    return {"run_id": run["metadata"]["run_id"], "layer": layer, "alarm": alarm,
            "accuracy": accuracy[stop], "range": (acc_low[stop], acc_high[stop]),
            "final": accuracy[-1], "final_range": (acc_low[-1], acc_high[-1]), "outside": len(outside)}


def detection_over_time(rows: list[dict]) -> dict:
    """Share of broken runs flagged by each epoch, LHD against the accuracy curve."""
    fig, axes = _figure(2.7, 3, left=0.5, right=0.1, gaps=0.22, bottom=0.42, top=0.3)
    found = {}
    for k, (ax, arch) in enumerate(zip(axes, ARCH_ORDER)):
        faults = [r for r in rows if r["arch"] == arch and r["role"] == "fault"
                  and r["family"] in PROPOSAL and r["onset"] == 1]
        last = max(r["epochs"] for r in faults)
        epochs = np.arange(0, last + 1)
        end = last + 0.5
        shares = {}
        for name, key, color in (("LHD", "lhd_epoch", LHD_BLUE), ("Accuracy curve", "acc_epoch", ACC_ORANGE)):
            share = np.array([100 * np.mean([r[key] is not None and r["onset"] <= r[key] <= e for r in faults])
                              for e in epochs])
            ax.step(np.append(epochs, end), np.append(share, share[-1]), where="post", color=color,
                    linewidth=DATA_WIDTH, zorder=3)
            ax.annotate(f"{name} {share[-1]:.0f}%", (end, share[-1]), xytext=(-1, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=SIZE, color=TEXT)
            shares[name] = share
        ax.set_xlim(0, end)
        ax.set_ylim(-2, 114)
        _percent_axis(ax)
        ax.set_xticks(range(0, last + 1, 5))
        ax.set_xlabel("Epoch")
        _panel(ax, "abc"[k], f"{ARCH_LABEL[arch]}, {len(faults)} runs")
        if k:
            ax.tick_params(labelleft=False)
        found[arch] = {name: s for name, s in shares.items()} | {"runs": len(faults)}
    axes[0].set_ylabel("Broken runs flagged (%)")
    first = found[ARCH_ORDER[0]]["LHD"]
    jump = int(np.argmax(first > 0))
    axes[0].annotate(f"{first[jump]:.0f}% by epoch {jump}", (jump, first[jump]),
                     xytext=(jump + 3.2, first[jump] - 26), textcoords="data", fontsize=SIZE, color=TEXT,
                     va="center", arrowprops={"arrowstyle": "-|>", "color": SOFT, "linewidth": 0.6,
                                              "mutation_scale": 6, "shrinkA": 2, "shrinkB": 2})
    _save(fig, "detection_over_time")
    return found


def detectors_tradeoff() -> dict:
    """Every detector as one point: healthy runs flagged against dead unit faults caught."""
    table = [r for r in read_csv("detectors.csv") if r["arch"] == "pooled"]
    points = {}
    for detector, short in TRADEOFF_LABELS.items():
        healthy = next(r for r in table if r["detector"] == detector and r["family"] == "proposal")
        dead = [r for r in table if r["detector"] == detector and r["family"] in DEAD]
        caught, broken = sum(r["caught"] for r in dead), sum(r["broken_runs"] for r in dead)
        points[detector] = (100 * healthy["false_alarm_rate"], 100 * caught / broken, short)
    fig, (ax,) = _figure(3.8, 1, left=0.5, right=0.15, bottom=0.42, top=0.12)
    ax.set_xlim(-3, 82)
    ax.set_ylim(-6, 106)
    ax.set_xticks(range(0, 81, 20))
    _percent_axis(ax)
    offsets = {
        "LHD": (8, 0, "left"),
        ACC3: (6, 2, "left"),
        "Loss curve, 3 epochs above healthy": (6, 0, "left"),
        "Dead units check, 70% (DeepDiagnosis Dead Node)": (6, -1, "left"),
        "Dead units check, 80% (wandb-callbacks default)": (6, -5, "left"),
        "Any dead unit (wandb-callbacks verbose)": (-7, 0, "right"),
    }
    for detector, (x, y, short) in points.items():
        lhd = detector == "LHD"
        ax.plot(x, y, "o", color=LHD_BLUE if lhd else RULE_GRAY, markersize=8 if lhd else 5.5,
                markeredgecolor=WHITE, markeredgewidth=0.9, zorder=4 if lhd else 3)
        dx, dy, ha = offsets.get(detector, (6, 0, "left"))
        ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=SIZE, color=TEXT)
    ax.annotate("", (8, 96), xytext=(19, 76), textcoords="data",
                arrowprops={"arrowstyle": "-|>", "color": SOFT, "linewidth": 0.7, "mutation_scale": 7,
                            "shrinkA": 0, "shrinkB": 0})
    ax.annotate("better", (13.5, 86), xytext=(9, 3), textcoords="offset points", fontsize=SIZE, color=SOFT,
                ha="left", va="center")
    ax.set_xlabel("New healthy runs flagged (%)")
    ax.set_ylabel("Dead unit faults caught (%)")
    _save(fig, "detectors_tradeoff")
    return points


def dose_response() -> dict:
    """Share of dead unit faults caught as more of a layer's units are killed."""
    table = [r for r in read_csv("detectors.csv") if r["family"] in DEAD and r["arch"] in ARCH_COLORS]
    fig, axes = _figure(2.8, 2, left=0.5, right=0.95, gaps=1.1, bottom=0.42, top=0.3)
    found = {}
    for k, (ax, detector, title) in enumerate(zip(axes, ("LHD", ACC3), ("LHD", "Accuracy curve"))):
        ax.set_xlim(6, 54)
        ax.set_ylim(-4, 106)
        ends = []
        for arch in ARCH_ORDER:
            picked = {r["family"]: r for r in table if r["arch"] == arch and r["detector"] == detector}
            fams = [f for f in DEAD if f in picked]
            xs = [DEAD_PCT[f] for f in fams]
            ys = [100 * picked[f]["caught_rate"] for f in fams]
            ax.plot(xs, ys, color=ARCH_COLORS[arch], linewidth=DATA_WIDTH, zorder=3)
            ax.plot(xs, ys, "o", color=ARCH_COLORS[arch], markersize=4.5, markeredgecolor=WHITE,
                    markeredgewidth=0.8, zorder=4)
            ends.append((ys[-1], ARCH_LABEL[arch], ARCH_COLORS[arch]))
            found[(title, arch)] = ys
        ax.set_xticks([10, 25, 50], ["10", "25", "50"])
        _percent_axis(ax)
        ax.set_xlabel("Units killed in one layer (%)")
        _panel(ax, "ab"[k], title)
        _end_labels(ax, ends, 50, gap_pt=11, low=-2, high=108)
    axes[0].set_ylabel("Dead unit faults caught (%)")
    _save(fig, "dose_response")
    return found


def accuracy_curves(rows: list[dict]) -> dict:
    """Validation accuracy of healthy runs and of two kinds of broken run."""
    fig, axes = _figure(2.8, 3, left=0.45, right=1.0, gaps=0.38, bottom=0.42, top=0.3)
    found = {}
    for k, (ax, arch) in enumerate(zip(axes, ARCH_ORDER)):
        groups = split_runs(_scored(arch, rows))
        ref = reference(groups)
        fam = {r["run_id"]: r["family"] for r in rows if r["arch"] == arch and r["role"] == "fault"
               and r["onset"] == 1}
        epochs = np.arange(1, len(groups["calibration"][0]["history"]) + 1)
        low, high = 100 * ref["accuracy"]["low"], 100 * ref["accuracy"]["high"]
        ax.fill_between(epochs, low, high, color=BAND, linewidth=0, zorder=0.5)
        series = [("New healthy runs", TEXT, groups["fresh"]),
                  ("Frozen layer", ACC_ORANGE,
                   [r for r in groups["fault"] if fam.get(r["metadata"]["run_id"]) == "frozen"]),
                  ("50% units dead", LHD_BLUE,
                   [r for r in groups["fault"] if fam.get(r["metadata"]["run_id"]) == "dead50"])]
        ends = [(high[-1], "Healthy range", BAND)]
        for label, color, group in series:
            mean = 100 * np.mean([[e["val_accuracy"] for e in r["history"]] for r in group], axis=0)
            ax.plot(epochs, mean, color=color, linewidth={TEXT: 1.2, ACC_ORANGE: 1.7}.get(color, 1.15),
                    zorder={TEXT: 4, ACC_ORANGE: 2}.get(color, 3))
            ends.append((mean[-1], label, color))
            found[(arch, label)] = (mean[-1], len(group))
        found[(arch, "range")] = (low[-1], high[-1])
        ax.set_xlim(1, epochs[-1])
        ax.set_xticks([1] + list(range(5, epochs[-1] + 1, 5)))
        ax.set_xlabel("Epoch")
        _panel(ax, "abc"[k], ARCH_LABEL[arch])
        if k == len(axes) - 1:
            _end_labels(ax, ends, epochs[-1], gap_pt=10.5)
    axes[0].set_ylabel("Validation accuracy (%)")
    _save(fig, "accuracy_curves")
    return found


def harm() -> dict:
    """Share of broken runs caught, grouped by how much final accuracy the fault cost."""
    table = {r["harm"]: r for r in read_csv("harm.csv") if r["scope"] == "proposal" and r["arch"] == "pooled"}
    groups = [(key, label) for key, label in HARM_GROUPS if key in table]
    fig, (ax,) = _figure(3.0, 1, left=0.5, right=0.1, bottom=0.5, top=0.12)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 108)
    _percent_axis(ax)
    width = 0.34
    gap = 1.5 / _points_per_unit(ax, "x")
    found = {}
    for j, (name, key, color) in enumerate((("LHD", "lhd", LHD_BLUE), ("Accuracy\ncurve", "accuracy", ACC_ORANGE))):
        for i, (group, _) in enumerate(groups):
            x = i + (j - 0.5) * (width + gap)
            value = 100 * table[group][f"{key}_rate"]
            ax.bar(x, value, width=width, color=color, linewidth=0, zorder=3)
            text = f"{value:.0f}%" if i else f"{name}\n{value:.0f}%"
            ax.annotate(text, (x, value), xytext=(0, 2.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=SIZE, color=TEXT, linespacing=1.05)
            found[(key, group)] = (value, table[group][f"{key}_caught"], table[group]["broken_runs"])
    ax.set_xticks(range(len(groups)), [f"{label}\n{table[g]['broken_runs']} broken runs" for g, label in groups])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("Broken runs caught (%)")
    _save(fig, "harm")
    return found


def calibration_size() -> dict:
    """Healthy runs flagged as the number of calibration runs grows."""
    rows = read_csv("sweep_calibration.csv")
    fig, (ax,) = _figure(2.9, 1, left=0.5, right=1.15, bottom=0.42, top=0.12)
    ks = sorted({r["k"] for r in rows})
    used = max(ks)
    ax.set_xlim(0.8, used + 0.6)
    ax.set_ylim(-1.5, 30)
    ends, found = [], {}
    for arch in ARCH_ORDER:
        by_k = defaultdict(list)
        for r in rows:
            if r["arch"] == arch:
                by_k[r["k"]].append(100 * r["false_alarm"])
        mean = [float(np.mean(by_k[k])) for k in ks]
        ax.plot(ks, mean, color=ARCH_COLORS[arch], linewidth=DATA_WIDTH, zorder=3)
        ax.plot(ks, mean, "o", color=ARCH_COLORS[arch], markersize=4.5, markeredgecolor=WHITE,
                markeredgewidth=0.8, zorder=4)
        ends.append((mean[-1], ARCH_LABEL[arch], ARCH_COLORS[arch]))
        found[arch] = dict(zip(ks, mean))
    ax.set_xticks(ks)
    ax.set_yticks(range(0, 31, 10))
    _rule(ax, used, f"{used} used in the study", ha="right")
    _end_labels(ax, ends, used + 0.6, gap_pt=10.5, pad_pt=3)
    ax.set_xlabel("Healthy calibration runs used to build the range")
    ax.set_ylabel("New healthy runs flagged (%)")
    _save(fig, "calibration_size")
    return found


def margin() -> dict:
    """Broken runs caught and healthy runs flagged as the healthy range is widened."""
    rows = [r for r in read_csv("sweep_settings.csv")
            if r["floor"] == SETTINGS["floor"] and r["patience"] == SETTINGS["patience"]]
    margins = sorted({r["margin"] for r in rows})
    caught, flagged = [], []
    for m in margins:
        picked = [r for r in rows if r["margin"] == m]
        caught.append(100 * sum(r["detection"] * r["faults"] for r in picked) / sum(r["faults"] for r in picked))
        flagged.append(100 * sum(r["false_alarm"] * r["fresh"] for r in picked) / sum(r["fresh"] for r in picked))
    xs = [100 * m for m in margins]
    fig, (ax,) = _figure(2.9, 1, left=0.5, right=1.45, bottom=0.42, top=0.12)
    ax.set_xlim(-2, max(xs) + 2)
    ax.set_ylim(-3, 106)
    for ys, color in ((caught, LHD_BLUE), (flagged, ACC_ORANGE)):
        ax.plot(xs, ys, color=color, linewidth=DATA_WIDTH, zorder=3)
        ax.plot(xs, ys, "o", color=color, markersize=4.5, markeredgecolor=WHITE, markeredgewidth=0.8, zorder=4)
    _percent_axis(ax)
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    _rule(ax, 100 * SETTINGS["margin"], "used in the study")
    _end_labels(ax, [(caught[-1], "Broken runs caught", LHD_BLUE),
                     (flagged[-1], "Healthy runs flagged", ACC_ORANGE)], max(xs) + 2, pad_pt=3)
    ax.set_xlabel("Margin: how far the healthy range is widened (%)")
    ax.set_ylabel("Runs (%)")
    _save(fig, "margin")
    return {"margin": margins, "caught": caught, "flagged": flagged}


def main() -> None:
    setup()
    rows = read_csv("runs.csv")
    results = {
        "how_lhd_works": how_lhd_works(rows),
        "detection_over_time": detection_over_time(rows),
        "detectors_tradeoff": detectors_tradeoff(),
        "dose_response": dose_response(),
        "accuracy_curves": accuracy_curves(rows),
        "harm": harm(),
        "calibration_size": calibration_size(),
        "margin": margin(),
    }
    for name, values in results.items():
        print(name, values if name != "detection_over_time" else
              {a: {k: (v[-1] if hasattr(v, "__len__") else v) for k, v in d.items()} for a, d in values.items()})
    print(f"figures in {PAPER_FIGURES} and previews in {PREVIEWS}")


if __name__ == "__main__":
    main()
