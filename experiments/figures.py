"""Draws the study's figures from saved runs and the analysis tables.

Usage:
    python -m experiments.figures

Run experiments.analyze first. Writes PNGs to results/figures/.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from experiments.evaluate import RESULTS, SETTINGS, load_runs, reference, split_runs  # noqa: E402
from experiments.grid import ARCH_ORDER  # noqa: E402
from halt.baseline import SIGNALS  # noqa: E402
from halt.replay import replay  # noqa: E402

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
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    """A thin vertical rule at the epoch HALT stops the run."""
    if report:
        ax.axvline(report["epoch"], color=INK_2, linewidth=1)
        ax.annotate("HALT stops", (report["epoch"], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK_2)


def cumulative_detection(rows: list[dict]) -> None:
    """Share of broken runs flagged by each epoch: HALT, accuracy curve, loss curve."""
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(archs), figsize=(5.2 * len(archs), 3.6), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        faults = [r for r in rows if r["arch"] == arch and r["role"] == "fault"
                  and r["family"] in ("frozen", "dead10", "dead25", "dead50")]
        if not faults:
            continue
        epochs = np.arange(0, faults[0]["epochs"] + 1)
        series = [("HALT", "halt_epoch", BLUE), ("Accuracy curve", "acc_epoch", ORANGE),
                  ("Loss curve", "loss_epoch", AQUA)]
        for label, key, color in series:
            share = [100 * np.mean([r[key] is not None and r[key] <= e for r in faults]) for e in epochs]
            ax.step(epochs, share, where="post", color=color, label=label)
            ax.annotate(f"{share[-1]:.0f}%", (epochs[-1], share[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ax.set_ylim(-2, 104)
        ax.set_xlim(0, epochs[-1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"{ARCH_NAMES[arch]} ({len(faults)} broken runs)")
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("Broken runs flagged (%)")
    axes[0][0].legend(loc="center right")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "cumulative_detection")


def false_alarms(summary: dict) -> None:
    """False alarm rate on fresh healthy runs, with 95% Wilson intervals."""
    archs = [a for a in ARCH_ORDER if a in summary["archs"]]
    detectors = [("HALT", "halt_false_alarm", BLUE), ("Accuracy curve", "acc_false_alarm", ORANGE),
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
    """Planted fault against the cause HALT named, pooled over architectures."""
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
    ax.set_title("Cause HALT named, by planted fault (run counts)")
    save(fig, "diagnosis_matrix")


def per_layer_grid() -> None:
    """Share of broken runs where HALT named both the right layer and the right cause."""
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
        halt = np.array([r["halt_censored"] for r in group])
        acc = np.array([r["acc_censored"] for r in group])
        ax.scatter(drop, acc, s=22, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="Accuracy curve")
        ax.scatter(drop, halt, s=22, color=BLUE, edgecolor=SURFACE, linewidth=1, label="HALT")
        never = group[0]["epochs"] + 1
        ax.axhline(never, color=AXIS, linewidth=1)
        ax.annotate("never noticed", (ax.get_xlim()[0], never), xytext=(2, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
        ax.set_title(ARCH_NAMES[arch], fontsize=10)
        ax.set_xlabel("Final accuracy lost vs healthy twin (points)")
    axes[0][0].set_ylabel("Epoch the run was flagged")
    axes[0][0].legend(loc="center right", fontsize=8)
    save(fig, "silent_damage")


def pick_example(rows: list[dict], arch: str, family_name: str) -> str | None:
    """A broken run HALT caught correctly while the accuracy curve stayed quiet longest."""
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
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
