"""Draws the study's figures from saved runs and the analysis tables.

Usage:
    python -m experiments.figures

Run experiments.analyze first. Writes PNGs to results/figures/ and
vector PDFs to results/figures/pdf/.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from experiments.evaluate import RESULTS, SETTINGS, load_runs, reference, split_runs, wilson  # noqa: E402
from experiments.grid import ARCH_ORDER  # noqa: E402
from lhd.baseline import SIGNALS  # noqa: E402
from lhd.replay import replay  # noqa: E402

FIGURES = RESULTS / "figures"
FONT_DIR = RESULTS.parent / "data" / "fonts"
AVENIR = Path("/System/Library/Fonts/Avenir Next.ttc")
TABLES = RESULTS / "tables"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#e8e7e1"
RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
ARCH_NAMES = {"cnn": "CNN on GTSRB", "mlp": "Fully connected on Fashion-MNIST",
              "transformer": "Transformer on Tiny Shakespeare"}
SIGNAL_NAMES = {"weight_change": "Weight change", "grad_norm": "Gradient norm",
                "weight_norm": "Weight norm", "dead_unit_fraction": "Zero-activation fraction"}
FAMILY_NAMES = {"frozen": "Frozen layer", "dead10": "Dead units 10%", "dead25": "Dead units 25%",
                "dead50": "Dead units 50%", "latefrozen": "Frozen at epoch 6",
                "latedead50": "Dead units 50% at epoch 6", "detach": "Gradient cut",
                "lr10": "Learning rate x10 at epoch 6", "lr100": "Learning rate x100 at epoch 6"}
CAUSES = ["frozen_layer", "dead_units", "vanishing_gradient", "runaway_weights",
          "diverged", "unrecognized", "not detected"]
ARCH_COLORS = dict(zip(ARCH_ORDER, (BLUE, ORANGE, AQUA)))
ARCH_SHORT = {"mlp": "Fully connected", "transformer": "Transformer", "cnn": "CNN"}
BAND_LAYERS = {"mlp": "fc2", "cnn": "conv2", "transformer": "b0.ff1"}
PROPOSAL = ("frozen", "dead10", "dead25", "dead50")
DEAD = ("dead10", "dead25", "dead50")
DEAD_PCT = {"dead10": 10, "dead25": 25, "dead50": 50}
DEAD_STEPS = {"dead10": RAMP(0.42), "dead25": RAMP(0.68), "dead50": RAMP(0.93)}
ACC3 = "Accuracy curve, 3 epochs below healthy"
SIMPLE = "Zero update or dead units 50%"
DETECTORS = ["LHD", ACC3, "Accuracy curve, 1 epoch below healthy", "Loss curve, 3 epochs above healthy",
             "Zero update check", "Dead units check, 25%", "Dead units check, 50% (SageMaker DeadRelu default)",
             "Dead units check, 70% (DeepDiagnosis Dead Node)", "Dead units check, 80% (wandb-callbacks default)",
             "Any dead unit (wandb-callbacks verbose)", SIMPLE]
HARM_ORDER = {"under 0.5 pts": "Under 0.5 points", "0.5 to 2 pts": "0.5 to 2 points",
              "2 pts or more": "2 points or more"}
_RUNS: dict[str, list[dict]] = {}


def _register_fonts() -> bool:
    """Make the regular and demi bold faces of Avenir Next usable.

    matplotlib only reads the first face of a .ttc collection, which
    for Avenir Next is bold, so the two faces are copied out once.
    """
    if not AVENIR.exists():
        return False
    from fontTools.ttLib import TTCollection

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    collection = None
    for index, weight in ((7, "Regular"), (2, "DemiBold")):
        path = FONT_DIR / f"AvenirNext-{weight}.ttf"
        if not path.exists():
            collection = collection or TTCollection(str(AVENIR))
            face = collection.fonts[index]
            for record in face["name"].names:
                if record.nameID in (1, 16):
                    record.string = "Avenir Next"
            face.save(str(path))
        font_manager.fontManager.addfont(str(path))
    return True


def style() -> None:
    """Shared look: one sans face, hairline grid, recessive axes."""
    family = "Avenir Next" if _register_fonts() else "DejaVu Sans"
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "demibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
    })


def read_csv(name: str) -> list[dict]:
    """A results table as a list of dicts with numbers parsed."""
    with open(TABLES / name) as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _parse(value: str):
    if value in ("", "None"):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value else number


def save(fig, name: str) -> None:
    """Write a figure as a PNG and as a vector PDF with its fonts embedded."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    (FIGURES / "pdf").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(FIGURES / "pdf" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _runs(arch: str) -> list[dict]:
    """Every saved run for an architecture, read from disk once."""
    if arch not in _RUNS:
        _RUNS[arch] = load_runs(arch)
    return _RUNS[arch]


def _scored(arch: str, rows: list[dict]) -> list[dict]:
    """Saved runs for an architecture: all calibration runs, plus only the
    fresh and broken runs already scored in runs.csv, so per-epoch figures
    match the tables while new runs are still being trained."""
    ids = {r["run_id"] for r in rows if r["arch"] == arch}
    return [run for run in _runs(arch)
            if run["metadata"]["role"] == "calibration" or run["metadata"]["run_id"] in ids]


def _widened(band: dict) -> tuple[np.ndarray, np.ndarray]:
    """The healthy range LHD checks against: calibration low and high,
    widened by the margin, with the floor added on top."""
    low = band["low"] * (1 - SETTINGS["margin"])
    high = band["high"] * (1 + SETTINGS["margin"]) + SETTINGS["floor"]
    return low, high


def _epoch_axis(ax, last: int, first: int = 1) -> None:
    ax.set_xlabel("Epoch")
    ax.set_xlim(first, last)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _dot(ax, x, y, color, **kwargs) -> None:
    ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, **kwargs)


def _spread(targets: list[float], step: float, top: float) -> list[float]:
    """Heights for labels at least step apart and as close as possible to
    their sorted targets: labels that crowd are merged into a block
    centred on its targets, and the stack is kept below top."""
    blocks: list[list] = []
    for i, target in enumerate(targets):
        blocks.append([i, 1, target])
        while len(blocks) > 1 and blocks[-2][2] + blocks[-2][1] * step > blocks[-1][2]:
            first, count = blocks[-2][0], blocks[-2][1] + blocks[-1][1]
            start = float(np.mean([targets[first + k] - k * step for k in range(count)]))
            blocks[-2:] = [[first, count, start]]
    spots = [start + k * step for _, count, start in blocks for k in range(count)]
    spots[-1] = min(spots[-1], top)
    for i in range(len(spots) - 2, -1, -1):
        spots[i] = min(spots[i], spots[i + 1] - step)
    return spots


def _end_labels(ax, items: list[tuple], x: float, gap: float = 0.065) -> None:
    """Label line ends at x, nudged apart so no two labels overlap.

    items holds (y, text, color). gap is the smallest distance between
    two labels as a share of the y range. A label moved off its line
    gets a short leader in the line's color.
    """
    low, high = ax.get_ylim()
    step = gap * (high - low)
    items = sorted((item for item in items if np.isfinite(item[0])), key=lambda item: item[0])
    if not items:
        return
    spots = _spread([item[0] for item in items], step, high)
    left, right = ax.get_xlim()
    dx = 0.035 * (right - left)
    for (y, text, color), spot in zip(items, spots):
        leader = abs(spot - y) > 0.3 * step
        ax.annotate(text, (x, y), xytext=(x + dx, spot), textcoords="data", fontsize=8, color=INK_2,
                    va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8, "relpos": (0, 0.5),
                                "shrinkA": 2, "shrinkB": 1} if leader else None)


def anatomy(arch: str, run_id: str, name: str) -> None:
    """One broken run's flagged layer against the healthy band, signal by signal,
    with the accuracy curve last."""
    runs = load_runs(arch)
    groups = split_runs(runs)
    ref = reference(groups)
    run = next(r for r in runs if r["metadata"]["run_id"] == run_id)
    layer = run["metadata"]["fault"]["layer"]
    report = replay(run["history"], ref["band"], **SETTINGS)
    epochs = np.arange(1, len(run["history"]) + 1)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, signal in zip(axes, SIGNALS):
        band = ref["band"][layer][signal]
        low = band["low"] * (1 - SETTINGS["margin"])
        high = band["high"] * (1 + SETTINGS["margin"]) + SETTINGS["floor"]
        ax.fill_between(epochs, low, high, color=BAND, linewidth=0, label="Healthy range, widened")
        values = [e["layers"][layer][signal] for e in run["history"]]
        ax.plot(epochs, values, color=BLUE, label="Broken run")
        ax.set_title(SIGNAL_NAMES[signal])
        _mark_fire(ax, report)
    acc = [e["val_accuracy"] * 100 for e in run["history"]]
    axes[4].fill_between(epochs, ref["accuracy"]["low"] * 100, ref["accuracy"]["high"] * 100,
                         color=BAND, linewidth=0)
    axes[4].plot(epochs, acc, color=BLUE)
    axes[4].set_title("Validation accuracy (%)")
    _mark_fire(axes[4], report)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, epochs[-1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fault = run["metadata"]["fault"]
    fig.suptitle(f"{ARCH_NAMES[arch]}: {FAMILY_NAMES.get(_family(fault), fault['type'])} "
                 f"planted in {layer}", x=0.01, y=1.1, ha="left", color=INK, fontweight="demibold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2,
               bbox_to_anchor=(1, 1.11), fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    save(fig, name)


def _family(fault: dict) -> str:
    from experiments.evaluate import family
    return family(fault)


def _mark_fire(ax, report) -> None:
    """A thin vertical rule at the epoch LHD stops the run, with headroom
    above the data for its label."""
    if report:
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.14 * (high - low))
        ax.axvline(report["epoch"], color=INK_2, linewidth=1)
        ax.annotate("LHD stops", (report["epoch"], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_2)


def cumulative_detection(rows: list[dict]) -> None:
    """Share of broken runs flagged by each epoch: LHD, accuracy curve, loss curve."""
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 3.6), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        faults = [r for r in rows if r["arch"] == arch and r["role"] == "fault"
                  and r["family"] in ("frozen", "dead10", "dead25", "dead50")]
        if not faults:
            continue
        epochs = np.arange(0, faults[0]["epochs"] + 1)
        series = [("LHD", "lhd_epoch", BLUE, "-"), ("Accuracy curve", "acc_epoch", ORANGE, "-"),
                  ("Loss curve", "loss_epoch", AQUA, "-"), (SIMPLE, "simple_epoch", MUTED, "--")]
        ends = []
        for label, key, color, line in series:
            share = [100 * np.mean([r[key] is not None and r[key] <= e
                                    and (key != "simple_epoch" or r[key] >= r["onset"]) for r in faults])
                     for e in epochs]
            ax.step(epochs, share, where="post", color=color, linestyle=line, label=label)
            ends.append((share[-1], f"{share[-1]:.0f}%", color))
        ax.set_ylim(-2, 104)
        ax.set_xlim(0, epochs[-1])
        _end_labels(ax, ends, epochs[-1], gap=0.055)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{ARCH_NAMES[arch]} ({len(faults)} broken runs)")
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("Broken runs flagged (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=4,
               bbox_to_anchor=(1, 1.1), fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "cumulative_detection")


def false_alarms(summary: dict) -> None:
    """False alarm rate on fresh healthy runs, with 95% Wilson intervals."""
    archs = [a for a in ARCH_ORDER if a in summary["archs"]]
    detectors = [("LHD", "lhd_false_alarm", BLUE), ("Accuracy curve", "acc_false_alarm", ORANGE),
                 ("Loss curve", "loss_false_alarm", AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, arch in enumerate(archs):
        healthy = summary["archs"][arch]["healthy"]
        for j, (label, key, color) in enumerate(detectors):
            stat = healthy[key]
            y = i + (j - 1) * 0.22
            ax.plot([100 * stat["low"], 100 * stat["high"]], [y, y], color=color, linewidth=2,
                    solid_capstyle="round")
            ax.plot(100 * stat["rate"], y, "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label if i == 0 else None)
            ax.annotate(f"{stat['k']}/{stat['n']}", (100 * stat["high"], y), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_yticks(range(len(archs)), [ARCH_NAMES[a] for a in archs])
    ax.invert_yaxis()
    ax.set_xlabel("Fresh healthy runs flagged (%), with 95% interval")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1, 1.02))
    save(fig, "false_alarms")


def diagnosis_matrix(summary: dict) -> None:
    """Planted fault against the cause LHD named, pooled over architectures."""
    pooled = defaultdict(lambda: defaultdict(int))
    for arch in summary["archs"].values():
        for fam, counts in arch["confusion"].items():
            for cause, n in counts.items():
                pooled[fam][cause] += n
    families = [f for f in FAMILY_NAMES if f in pooled]
    causes = [c for c in CAUSES if any(pooled[f].get(c) for f in families)]
    counts = np.array([[pooled[f].get(c, 0) for c in causes] for f in families], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.3 * len(causes) + 3, 0.5 * len(families) + 1.6))
    ax.imshow(shares, cmap=RAMP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(families)):
        for j in range(len(causes)):
            if counts[i, j]:
                ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if shares[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(causes)), [c.replace("_", " ") for c in causes], rotation=25, ha="right")
    ax.set_yticks(range(len(families)), [FAMILY_NAMES[f] for f in families])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Cause LHD named, by planted fault (run counts)")
    save(fig, "diagnosis_matrix")


def per_layer_grid() -> None:
    """Share of broken runs where LHD named both the right layer and the right cause."""
    table = [r for r in read_csv("per_layer.csv") if r["family"] in ("frozen", "dead10", "dead25", "dead50")]
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in table)]
    families = ["frozen", "dead10", "dead25", "dead50"]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.2 * len(archs), 4.2), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        rows = [r for r in table if r["arch"] == arch]
        layers = list(dict.fromkeys(r["layer"] for r in rows if r["family"] == "frozen"))
        grid = np.full((len(layers), len(families)), np.nan)
        for r in rows:
            both = min(r["layer_ok"], r["cause_ok"])
            grid[layers.index(r["layer"]), families.index(r["family"])] = both / r["runs"]
        ax.imshow(np.nan_to_num(grid, nan=0), cmap=RAMP, vmin=0, vmax=1, aspect="auto")
        for i in range(len(layers)):
            for j in range(len(families)):
                text = "" if np.isnan(grid[i, j]) else f"{100 * grid[i, j]:.0f}%"
                ax.text(j, i, text or "n/a", ha="center", va="center", fontsize=8,
                        color="#ffffff" if grid[i, j] > 0.55 else (INK if text else MUTED))
        ax.set_xticks(range(len(families)), ["Frozen", "Dead 10%", "Dead 25%", "Dead 50%"])
        ax.set_yticks(range(len(layers)), layers)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(ARCH_NAMES[arch], fontsize=10)
    fig.suptitle("Right layer and right cause, by planted layer", x=0.01, ha="left",
                 color=INK, fontweight="demibold")
    save(fig, "per_layer")


def sensitivity() -> None:
    """Detection and false alarms across margin and patience, at the chosen floor."""
    rows = [r for r in read_csv("sweep_settings.csv") if r["floor"] == SETTINGS["floor"]]
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    margins = sorted({r["margin"] for r in rows})
    patiences = sorted({r["patience"] for r in rows})
    metrics = [("detection", "Broken runs caught"), ("false_alarm", "Healthy runs flagged"),
               ("both_ok", "Right layer and cause")]
    fig, axes = plt.subplots(len(metrics), len(archs), figsize=(4 * len(archs), 3.1 * len(metrics)),
                             squeeze=False)
    for col, arch in enumerate(archs):
        for row, (key, title) in enumerate(metrics):
            ax = axes[row][col]
            grid = np.zeros((len(margins), len(patiences)))
            for r in rows:
                if r["arch"] == arch:
                    grid[margins.index(r["margin"]), patiences.index(r["patience"])] = r[key]
            ax.imshow(grid, cmap=RAMP, vmin=0, vmax=1, aspect="auto", origin="lower")
            for i in range(len(margins)):
                for j in range(len(patiences)):
                    chosen = margins[i] == SETTINGS["margin"] and patiences[j] == SETTINGS["patience"]
                    ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            fontweight="bold" if chosen else "normal",
                            color="#ffffff" if grid[i, j] > 0.55 else INK)
            ax.set_xticks(range(len(patiences)), patiences)
            ax.set_yticks(range(len(margins)), margins)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{title} (%)" if col == 0 else "", fontsize=10)
            if row == 0:
                ax.set_title(f"{ARCH_NAMES[arch]}\n{title} (%)", fontsize=10)
            if row == len(metrics) - 1:
                ax.set_xlabel("Patience (epochs in a row)")
            if col == 0:
                ax.set_ylabel("Margin")
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    save(fig, "sensitivity")


def calibration_size() -> None:
    """False alarms and detection as the number of healthy calibration runs grows."""
    rows = read_csv("sweep_calibration.csv")
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    colors = dict(zip(archs, (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, ax, title in (("false_alarm", axes[0], "Healthy runs flagged (%)"),
                           ("detection", axes[1], "Broken runs caught (%)")):
        for arch in archs:
            by_k = defaultdict(list)
            for r in rows:
                if r["arch"] == arch:
                    by_k[r["k"]].append(100 * r[key])
            ks = sorted(by_k)
            mean = [np.mean(by_k[k]) for k in ks]
            low = [np.percentile(by_k[k], 10) for k in ks]
            high = [np.percentile(by_k[k], 90) for k in ks]
            ax.fill_between(ks, low, high, color=colors[arch], alpha=0.12, linewidth=0)
            ax.plot(ks, mean, color=colors[arch], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARCH_NAMES[arch])
        ax.set_xlabel("Healthy calibration runs")
        ax.set_title(title)
        ax.set_ylim(-2, 102)
        ax.set_xscale("log")
        ax.set_xticks(sorted({r["k"] for r in rows}), sorted({r["k"] for r in rows}))
        ax.minorticks_off()
    axes[0].legend(loc="upper right", fontsize=8)
    save(fig, "calibration_size")


def ablation() -> None:
    """What each signal contributes: all four, each removed, each alone."""
    rows = read_csv("ablation.csv")
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    colors = dict(zip(archs, (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, (("detection", "Broken runs caught (%)"),
                                       ("both_ok", "Right layer and cause (%)"),
                                       ("false_alarm", "Healthy runs flagged (%)"))):
        for k, arch in enumerate(archs):
            values = {r["variant"]: r[key] for r in rows if r["arch"] == arch}
            ys = np.arange(len(variants)) + (k - 1) * 0.2
            ax.plot([100 * values[v] for v in variants], ys, "o", color=colors[arch], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARCH_NAMES[arch])
        ax.set_title(title)
        ax.set_xlim(-3, 103)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(variants)), [v.replace("_", " ") for v in variants])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, 1.08), ncol=3)
    save(fig, "ablation")


def silent_damage(rows: list[dict]) -> None:
    """Accuracy lost to each fault against when each detector noticed."""
    faults = [r for r in rows if r["role"] == "fault" and r["acc_drop_vs_twin"] is not None
              and r["onset"] == 1 and r["fault_type"] in ("frozen", "dead")]
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in faults)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 3.6), squeeze=False, sharey=True)
    for ax, arch in zip(axes[0], archs):
        group = [r for r in faults if r["arch"] == arch]
        drop = np.array([100 * r["acc_drop_vs_twin"] for r in group])
        lhd = np.array([r["lhd_censored"] for r in group])
        acc = np.array([r["acc_censored"] for r in group])
        ax.scatter(drop, acc, s=22, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="Accuracy curve")
        ax.scatter(drop, lhd, s=22, color=BLUE, edgecolor=SURFACE, linewidth=1, label="LHD")
        never = group[0]["epochs"] + 1
        ax.axhline(never, color=AXIS, linewidth=1)
        ax.annotate("never noticed", (ax.get_xlim()[0], never), xytext=(2, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
        ax.set_title(ARCH_NAMES[arch], fontsize=10)
        ax.set_xlabel("Final accuracy lost vs healthy twin (points)")
    axes[0][0].set_ylabel("Epoch the run was flagged")
    axes[0][0].legend(loc="center right", fontsize=8)
    save(fig, "silent_damage")


def band_build(arch: str, rows: list[dict]) -> None:
    """How the healthy range is built for one ReLU layer: the calibration
    runs that set it, the widened range, and fresh healthy runs that never
    touched it."""
    groups = split_runs(_scored(arch, rows))
    calibration, fresh = groups["calibration"], groups["fresh"]
    if not calibration:
        print(f"skipped healthy_band_{arch}: no calibration runs")
        return
    layer = BAND_LAYERS[arch]
    ref = reference(groups)
    epochs = np.arange(1, len(calibration[0]["history"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for ax, signal in zip(axes, SIGNALS):
        low, high = _widened(ref["band"][layer][signal])
        ax.fill_between(epochs, low, high, color=BAND, linewidth=0)
        for run in calibration:
            ax.plot(epochs, [e["layers"][layer][signal] for e in run["history"]],
                    color=MUTED, linewidth=0.8, alpha=0.8)
        for run in fresh:
            ax.plot(epochs, [e["layers"][layer][signal] for e in run["history"]],
                    color=BLUE, linewidth=0.8, alpha=0.45)
        ax.set_title(SIGNAL_NAMES[signal])
        _epoch_axis(ax, epochs[-1])
    handles = [Patch(facecolor=BAND, label="Healthy range, widened"),
               Line2D([], [], color=MUTED, linewidth=1.2,
                      label=f"{len(calibration)} calibration runs, which set the range"),
               Line2D([], [], color=BLUE, linewidth=1.2, alpha=0.6,
                      label=f"{len(fresh)} fresh healthy runs, never used to set it")]
    fig.suptitle(f"Healthy range for layer {layer}, {ARCH_NAMES[arch]}", x=0.01, y=1.16, ha="left",
                 color=INK, fontweight="demibold")
    fig.text(0.01, 1.06, f"Each epoch the range runs from {1 - SETTINGS['margin']:g} times the lowest "
             f"calibration value to {1 + SETTINGS['margin']:g} times the highest plus {SETTINGS['floor']:g}",
             ha="left", fontsize=9, color=INK_2)
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.13), fontsize=9)
    fig.subplots_adjust(wspace=0.28)
    save(fig, f"healthy_band_{arch}")


def accuracy_curves(rows: list[dict]) -> None:
    """Validation accuracy of healthy runs and of broken runs, averaged by planted fault."""
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.4 * len(archs), 3.8), squeeze=False)
    handles = {}
    for ax, arch in zip(axes[0], archs):
        groups = split_runs(_scored(arch, rows))
        if not groups["calibration"]:
            continue
        ref = reference(groups)
        epochs = np.arange(1, len(groups["calibration"][0]["history"]) + 1)
        ax.fill_between(epochs, ref["accuracy"]["low"] * 100, ref["accuracy"]["high"] * 100, color=BAND,
                        linewidth=0, label="Calibration runs, lowest to highest")
        broken = [r for r in groups["fault"] if r["metadata"]["fault"].get("onset", 1) == 1]
        series = [("Healthy", "Healthy runs, mean", INK, groups["fresh"])]
        series += [(FAMILY_NAMES[f].replace("units ", ""), FAMILY_NAMES[f], DEAD_STEPS.get(f, ORANGE),
                    [r for r in broken if _family(r["metadata"]["fault"]) == f]) for f in PROPOSAL]
        ends = []
        for short, label, color, group in series:
            if not group:
                continue
            mean = 100 * np.mean([[e["val_accuracy"] for e in r["history"]] for r in group], axis=0)
            ax.plot(epochs, mean, color=color, label=label, zorder=3 if color == INK else 2)
            ends.append((mean[-1], short, color))
        _epoch_axis(ax, epochs[-1])
        ax.set_title(f"{ARCH_NAMES[arch]} ({len(broken)} broken runs)", fontsize=10)
        _end_labels(ax, ends, epochs[-1], gap=0.07)
        handles.update(zip(*reversed(ax.get_legend_handles_labels())))
    axes[0][0].set_ylabel("Validation accuracy (%)")
    fig.suptitle("Validation accuracy of healthy and broken runs", x=0.01, y=1.1, ha="left", color=INK,
                 fontweight="demibold")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper right", ncol=3,
               bbox_to_anchor=(1, 1.13), fontsize=9)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "accuracy_curves")


def dose_response() -> None:
    """Share of broken runs caught as more of a layer's units are killed."""
    table = [r for r in read_csv("detectors.csv") if r["family"] in DEAD and r["arch"] in ARCH_COLORS]
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in table)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, detector in zip(axes, ("LHD", ACC3)):
        ends = []
        for k, arch in enumerate(archs):
            points = {r["family"]: r for r in table if r["arch"] == arch and r["detector"] == detector}
            fams = [f for f in DEAD if f in points]
            if not fams:
                continue
            xs = np.array([DEAD_PCT[f] for f in fams]) + (k - (len(archs) - 1) / 2) * 1.1
            ys = [100 * points[f]["caught_rate"] for f in fams]
            ax.vlines(xs, [100 * points[f]["caught_low"] for f in fams],
                      [100 * points[f]["caught_high"] for f in fams], color=ARCH_COLORS[arch], linewidth=1.2)
            ax.plot(xs, ys, color=ARCH_COLORS[arch], label=ARCH_NAMES[arch])
            _dot(ax, xs, ys, ARCH_COLORS[arch])
            ends.append((ys[-1], ARCH_SHORT[arch], ARCH_COLORS[arch]))
        ax.set_title(detector)
        ax.set_xticks([10, 25, 50], ["10%", "25%", "50%"])
        ax.set_xlim(5, 55)
        ax.set_ylim(-3, 104)
        ax.set_xlabel("Units killed in the layer")
        _end_labels(ax, ends, 55, gap=0.07)
    axes[0].set_ylabel("Broken runs caught (%), with 95% interval")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3,
               bbox_to_anchor=(1, 1.08), fontsize=9)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "dose_response")


def detector_comparison() -> None:
    """Every detector on frozen layers, dead units, and healthy runs, pooled over architectures."""
    table = [r for r in read_csv("detectors.csv") if r["arch"] == "pooled"]
    detectors = [d for d in DETECTORS if any(r["detector"] == d for r in table)]

    def counts(detector: str, families: tuple, k: str, n: str) -> tuple[int, int]:
        picked = [r for r in table if r["detector"] == detector and r["family"] in families]
        return sum(r[k] for r in picked), sum(r[n] for r in picked)

    panels = [("Frozen layers caught (%)", ("frozen",), "caught", "broken_runs", "broken runs"),
              ("Dead units caught (%)", DEAD, "caught", "broken_runs", "broken runs"),
              ("Healthy runs flagged (%)", ("proposal",), "false_alarms", "healthy_runs", "fresh healthy runs")]
    colors = {"LHD": BLUE, ACC3: ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (title, families, k_key, n_key, unit) in zip(axes, panels):
        total = 0
        for i, detector in enumerate(detectors):
            k, n = counts(detector, families, k_key, n_key)
            if not n:
                continue
            total = n
            low, high = wilson(k, n)
            color = colors.get(detector, MUTED)
            ax.plot([100 * low, 100 * high], [i, i], color=color, linewidth=2, solid_capstyle="round")
            _dot(ax, 100 * k / n, i, color)
            ax.annotate(f"{100 * k / n:.0f}", (100 * high, i), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
        ax.set_title(title)
        ax.set_xlim(-3, 108)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel(f"{total} {unit}, with 95% interval")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(detectors)), detectors)
    axes[0].set_ylim(len(detectors) - 0.5, -0.5)
    for label in axes[0].get_yticklabels():
        if label.get_text() == "LHD":
            label.set_color(INK)
            label.set_fontweight("demibold")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "detector_comparison")


def harm_chart() -> None:
    """Share of broken runs caught, grouped by how much final accuracy the fault cost."""
    table = {r["harm"]: r for r in read_csv("harm.csv") if r["scope"] == "proposal" and r["arch"] == "pooled"}
    groups = [g for g in HARM_ORDER if g in table]
    if not groups:
        print("skipped harm: no pooled proposal rows")
        return
    series = [("LHD", "lhd", BLUE, "-"), ("Accuracy curve, 3 epochs below healthy", "accuracy", ORANGE, "-"),
              (f"Simple fixed checks ({SIMPLE.lower()})", "simple", MUTED, "--")]
    shorts = {"lhd": "LHD", "accuracy": "Accuracy curve", "simple": "Simple fixed checks"}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(groups))
    ends = []
    for k, (label, key, color, line) in enumerate(series):
        x = xs + (k - 1) * 0.07
        ys = [100 * table[g][f"{key}_rate"] for g in groups]
        ax.vlines(x, [100 * table[g][f"{key}_low"] for g in groups],
                  [100 * table[g][f"{key}_high"] for g in groups], color=color, linewidth=1.2)
        ax.plot(x, ys, color=color, linestyle=line, label=label)
        _dot(ax, x, ys, color)
        ends.append((ys[-1], f"{shorts[key]} {ys[-1]:.0f}%", color))
    ax.set_xticks(xs, [f"{HARM_ORDER[g]}\n{table[g]['broken_runs']} runs" for g in groups])
    ax.set_xlim(-0.3, len(groups) - 1 + 0.3)
    ax.set_ylim(-3, 104)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Final accuracy lost against the healthy twin")
    ax.set_ylabel("Broken runs caught (%), with 95% interval")
    ax.set_title("Broken runs caught, by how much accuracy the fault cost")
    ax.legend(loc="lower right", fontsize=8)
    _end_labels(ax, ends, len(groups) - 1 + 0.1, gap=0.07)
    save(fig, "harm")


def tradeoff() -> None:
    """Broken runs caught against healthy runs flagged as the margin changes."""
    rows = [r for r in read_csv("sweep_settings.csv") if r["floor"] == SETTINGS["floor"]]
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 3.8), sharey=True, squeeze=False)
    metrics = (("detection", BLUE, "Broken runs caught", "caught"),
               ("false_alarm", ORANGE, "Healthy runs flagged", "flagged"))
    for ax, arch in zip(axes[0], archs):
        ends, sizes = [], ""
        for patience, width, line in ((SETTINGS["patience"], 2, "-"), (1, 1.2, "--")):
            points = sorted((r for r in rows if r["arch"] == arch and r["patience"] == patience),
                            key=lambda r: r["margin"])
            if not points:
                continue
            margins = [r["margin"] for r in points]
            for key, color, label, short in metrics:
                ys = [100 * r[key] for r in points]
                ax.plot(margins, ys, color=color, linewidth=width, linestyle=line,
                        label=f"{label}, patience {patience}")
                if patience == SETTINGS["patience"]:
                    _dot(ax, margins, ys, color)
                    ends.append((ys[-1], short, color))
                    sizes = f"{points[0]['faults']} broken, {points[0]['fresh']} healthy runs"
        ax.set_ylim(-3, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(-0.02, max(r["margin"] for r in rows) + 0.02)
        ax.set_xticks(sorted({r["margin"] for r in rows}))
        ax.axvline(SETTINGS["margin"], color=INK_2, linewidth=1)
        ax.annotate("fixed before testing", (SETTINGS["margin"], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK_2)
        ax.set_title(f"{ARCH_NAMES[arch]}\n{sizes}", fontsize=10)
        ax.set_xlabel("Margin")
        _end_labels(ax, ends, ax.get_xlim()[1], gap=0.07)
    axes[0][0].set_ylabel("Runs (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=4,
               bbox_to_anchor=(1, 1.1), fontsize=9)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "tradeoff")


def _attempt(name: str, draw, *args) -> None:
    """Draw one figure, or skip it with a note when its data is missing."""
    try:
        draw(*args)
    except (FileNotFoundError, KeyError, ValueError, IndexError, StopIteration, ZeroDivisionError) as error:
        plt.close("all")
        print(f"skipped {name}: {type(error).__name__} {error}")


def pick_example(rows: list[dict], arch: str, family_name: str) -> str | None:
    """A broken run LHD caught correctly while the accuracy curve stayed quiet longest."""
    group = [r for r in rows if r["arch"] == arch and r["family"] == family_name
             and r["detected"] and r["layer_ok"] and r["cause_ok"]]
    group.sort(key=lambda r: (-r["gap"], r["run_id"]))
    return group[0]["run_id"] if group else None


def main() -> None:
    style()
    with open(RESULTS / "summary.json") as f:
        summary = json.load(f)
    rows = read_csv("runs.csv")
    cumulative_detection(rows)
    false_alarms(summary)
    diagnosis_matrix(summary)
    per_layer_grid()
    silent_damage(rows)
    for arch in summary["archs"]:
        for fam in ("frozen", "dead25"):
            run_id = pick_example(rows, arch, fam)
            if run_id:
                anatomy(arch, run_id, f"anatomy_{arch}_{fam}")
    if (TABLES / "sweep_settings.csv").exists():
        sensitivity()
        calibration_size()
        ablation()
    for arch in ARCH_ORDER:
        _attempt(f"healthy_band_{arch}", band_build, arch, rows)
    _attempt("accuracy_curves", accuracy_curves, rows)
    _attempt("dose_response", dose_response)
    _attempt("detector_comparison", detector_comparison)
    _attempt("harm", harm_chart)
    _attempt("tradeoff", tradeoff)
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
