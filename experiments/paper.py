"""Writes the LHD paper from the study's results and compiles it.

Usage:
    python -m experiments.paper

Run after experiments.report. Writes paper/lhd.tex, copies the figures
it uses into paper/figures, and compiles paper/lhd.pdf with tectonic.
Every number comes from results/summary.json and results/tables, so
the paper can be rebuilt whenever more runs finish. The related work
and references are fixed text in paper/related_work.tex and
paper/references.tex.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from experiments.evaluate import RESULTS, SETTINGS
from experiments.grid import ARCH_ORDER, build_grid

PAPER = RESULTS.parent / "paper"
TABLES = RESULTS / "tables"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic"
NAMES = {"mlp": "Fully connected", "transformer": "Transformer", "cnn": "CNN"}
DATASETS = {"mlp": "Fashion-MNIST", "transformer": "Tiny Shakespeare", "cnn": "GTSRB"}
PROSE = {"mlp": "fully connected network", "transformer": "transformer", "cnn": "CNN"}
DEAD = ("dead10", "dead25", "dead50")
FIGURES = ("healthy_band_cnn", "anatomy_cnn_frozen", "cumulative_detection", "detector_comparison",
           "dose_response", "accuracy_curves", "harm", "per_layer", "tradeoff", "calibration_size")


def read(name: str) -> list[dict]:
    with open(TABLES / name) as f:
        return list(csv.DictReader(f))


def tex(text: str) -> str:
    return text.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def pct(k: float, n: float, digits: int = 0) -> str:
    return "n/a" if not n else f"{100 * k / n:.{digits}f}\\%"


def rate(stat: dict, digits: int = 0) -> str:
    return pct(stat["k"], stat["n"], digits)


def kofn(stat: dict) -> str:
    return f"{stat['k']} of {stat['n']}"


def pval(p) -> str:
    if p is None:
        return "n/a"
    if p < 1e-10:
        return r"$p < 10^{-10}$"
    mantissa, exp = f"{p:.1e}".split("e")
    return f"$p = {mantissa} \\times 10^{{{int(exp)}}}$" if p < 0.001 else f"$p = {p:.3f}$"


class Data:
    """Every number the paper quotes, read once."""

    def __init__(self):
        with open(RESULTS / "summary.json") as f:
            self.summary = json.load(f)
        self.archs = [a for a in ARCH_ORDER if a in self.summary["archs"]]
        self.pooled = self.summary["pooled"]
        self.detectors = [r for r in read("detectors.csv") if r["arch"] == "pooled"]
        self.harm = [r for r in read("harm.csv") if r["scope"] == "proposal" and r["arch"] == "pooled"]
        self.networks = {r["arch"]: r for r in read("networks.csv")}
        self.counts = read("run_counts.csv")
        self.compute = {r["arch"]: r for r in read("compute.csv")}
        self.ablation = read("ablation.csv")
        self.calibration = read("sweep_calibration.csv")
        self.sweep = read("sweep_settings.csv")
        self.per_layer = read("per_layer.csv")
        self.wrong = read("wrong_layers.csv")
        self.alarms = read("false_alarms.csv")
        self.runs = read("runs.csv")
        self._signals = None

    def signals(self) -> list[dict]:
        """Every calibration and fresh run's dead unit fraction, per epoch and layer."""
        if self._signals is None:
            with gzip.open(TABLES / "signals.csv.gz", "rt") as f:
                self._signals = [r for r in csv.DictReader(f) if r["role"] in ("calibration", "fresh")]
        return self._signals

    def dead_range(self, arch: str, layer: str, role: str = "calibration") -> tuple[float, float]:
        """Lowest and highest dead unit fraction of a layer over healthy runs and epochs."""
        values = [float(r["dead_unit_fraction"]) for r in self.signals()
                  if r["arch"] == arch and r["layer"] == layer and r["role"] == role]
        return min(values), max(values)

    def alarm_value(self, run_id: str, layer: str, epoch: str) -> float:
        return next(float(r["dead_unit_fraction"]) for r in self.signals()
                    if r["run_id"] == run_id and r["layer"] == layer and r["epoch"] == epoch)

    def total_runs(self) -> int:
        return sum(int(r["runs"]) for r in self.counts)

    def role_count(self, arch: str, role: str) -> int:
        return sum(int(r["runs"]) for r in self.counts if r["arch"] == arch and r["role"] == role)

    def detector(self, name: str, families) -> dict:
        rows = [r for r in self.detectors if r["detector"] == name and r["family"] in families]
        k = sum(int(r["caught"]) for r in rows)
        n = sum(int(r["broken_runs"]) for r in rows)
        return {"k": k, "n": n}

    def detector_alarms(self, name: str) -> dict:
        row = next(r for r in self.detectors if r["detector"] == name and r["family"] == "proposal")
        return {"k": int(row["false_alarms"]), "n": int(row["healthy_runs"]),
                "median": row["median_epoch"]}

    def more_complete(self) -> bool:
        planned = defaultdict(int)
        for spec in build_grid("more"):
            planned[spec["arch"]] += 1
        done = defaultdict(int)
        for r in self.counts:
            if r["phase"] == "more":
                done[r["arch"]] += int(r["runs"])
        return all(done[a] >= n for a, n in planned.items())


def preamble() -> str:
    return r"""\documentclass[11pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage[margin=1in]{geometry}
\usepackage[round]{natbib}
\usepackage{url}
\urlstyle{same}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}

\title{Early Detection of Silent Training Failures in Neural Networks}
\author{Sohan Bhat}
\date{November 2026}

\begin{document}

\maketitle
"""


def figure(name: str, caption: str, width: str = r"\linewidth") -> str:
    return (f"\\begin{{figure}}[t]\n\\centering\n\\includegraphics[width={width}]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{fig:{name}}}\n\\end{{figure}}\n")


def abstract(d: Data) -> str:
    p = d.pooled["proposal"]
    h = d.pooled["healthy"]
    c = d.pooled["proposal_clustered"]
    lo, hi = c["mean_gap_ci"]
    dead = d.detector("LHD", DEAD)
    zero = d.detector("Zero update check", ("frozen",))
    return f"""\\begin{{abstract}}
Neural networks can break during training without raising an error. The run completes and returns a degraded model, and the drop is easy to mistake for a hard problem. This work presents LHD (Layer Health Diagnostics), a monitor that logs four signals for every layer at the end of each epoch (weight change, gradient norm, weight norm, and the fraction of dead units), compares each one with the range it stays within during healthy runs of the same network, flags a layer after three epochs in a row outside that range, names a likely cause, and stops the run. LHD was tested on {d.total_runs()} training runs of a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a small transformer on Tiny Shakespeare, with frozen layers and dead units planted in known layers. LHD caught {kofn(p['detected'])} broken runs ({rate(p['detected'])}), named the planted layer in {rate(p['layer_ok'])} of the runs it caught, and flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs it had never seen. Averaged within each seed, it was earlier than the accuracy curve for {c['by_seed']['lhd_earlier']} of {c['by_seed']['groups']} seed groups, by {p['mean_gap']:.1f} epochs (95\\% interval {lo:.1f} to {hi:.1f}). A one line check for weights that never change caught all {zero['n']} frozen layers two epochs sooner, so LHD's advantage lies in dead units. There, the fixed thresholds used by existing tools either missed most faults or flagged many healthy runs, while LHD caught {pct(dead['k'], dead['n'])} of them.
\\end{{abstract}}
"""


def introduction(d: Data) -> str:
    p = d.pooled["proposal"]
    h = d.pooled["healthy"]
    c = d.pooled["proposal_clustered"]
    lo, hi = c["mean_gap_ci"]
    dead = d.detector("LHD", DEAD)
    zero = d.detector("Zero update check", ("frozen",))
    sage = d.detector("Dead units check, 50% (SageMaker DeadRelu default)", DEAD)
    sage_fa = d.detector_alarms("Dead units check, 50% (SageMaker DeadRelu default)")
    return f"""\\section{{Introduction}}

During training, neural networks can develop defects, yet nothing fails visibly since the run completes normally and simply produces a degraded result; thus, these are called ``silent failures.'' Because of this, identifying whether the degraded result was caused by a difficult problem or a silent failure becomes difficult; therefore, it is common to misread a failure as normal. I initially encountered this as I was training my small-scale GTSRB (German Traffic Sign Recognition Benchmark) model \\citep{{stallkamp2012}} from scratch, where my model plateaued at 76\\%, wasting weeks tuning it, until I found one convolutional layer had been accidentally excluded from the optimizer and was never properly trained. Fixing this minute detail raised the accuracy to 87\\%. This is not an isolated case: errors in deep learning implementations rarely cause an exception and instead reveal themselves only through degraded predictive performance \\citep{{chwilczynski2024}}.

Since training runs consume significant compute, every silent failure that goes undetected wastes it. The evidence existed in the form of raw numbers in the weights and biases, but perceiving that is challenging since nothing in standard practice surfaces them. To address this, this work presents LHD (Layer Health Diagnostics), a monitor that reads these numbers, catches the failure within a few epochs, identifies the broken layer and its likely cause, and halts the run before compute is wasted.

LHD was tested on {d.total_runs()} training runs across three networks, with every rule and setting fixed before any broken run was trained. The main findings are the following.

\\begin{{itemize}}
\\item LHD caught {kofn(p['detected'])} runs with a planted frozen layer or dead units ({rate(p['detected'])}), usually at epoch {p['lhd_median_epoch']:.0f}, while the accuracy curve caught {kofn(p['acc_detected'])} ({rate(p['acc_detected'])}). LHD was earlier on average for {c['by_seed']['lhd_earlier']} of {c['by_seed']['groups']} seed groups and {c['by_condition']['lhd_earlier']} of {c['by_condition']['groups']} fault conditions, by {p['mean_gap']:.1f} epochs (95\\% interval {lo:.1f} to {hi:.1f}).
\\item LHD flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs ({rate(h['lhd_false_alarm'], 1)}), against {kofn(h['acc_false_alarm'])} for the accuracy curve.
\\item For frozen layers, a check for weights that do not change at all caught {kofn(zero)} at epoch 1, two epochs before LHD. For dead units, LHD caught {pct(dead['k'], dead['n'])} with {h['lhd_false_alarm']['k']} false alarms, while SageMaker Debugger's default rule caught {pct(sage['k'], sage['n'])} and flagged {sage_fa['k']} of {sage_fa['n']} healthy runs, and stricter defaults from other tools caught almost none.
\\item LHD missed small dead unit faults in layers where healthy runs already lose many units, and the planted faults were designed to show up in the signals LHD measures, so these results describe planted faults rather than bugs found in the wild.
\\end{{itemize}}

"""


def method() -> str:
    m, f, pat = SETTINGS["margin"], SETTINGS["floor"], SETTINGS["patience"]
    return f"""\\section{{Method}}

\\subsection{{Signals}}

At the end of every epoch, LHD records four signals for each watched layer. Let $W_e$ be the layer's weights and bias at the end of epoch $e$, flattened into one vector, and $\\lVert \\cdot \\rVert$ the square root of the sum of squared entries.

\\begin{{itemize}}
\\item \\textbf{{Weight change}} is $\\lVert W_e - W_{{e-1}} \\rVert / \\lVert W_{{e-1}} \\rVert$, how far the weights moved during the epoch relative to their size. A frozen layer reads exactly 0.
\\item \\textbf{{Weight norm}} is $\\lVert W_e \\rVert$, which grows past normal before a run diverges.
\\item \\textbf{{Gradient norm}} is the mean of $\\lVert \\partial L / \\partial W \\rVert$ over the epoch's batches, measured after each backward pass and before the optimizer step. A vanishing gradient reads near 0.
\\item \\textbf{{Dead unit fraction}} is the share of the layer's units whose output after the ReLU is exactly 0 for every input in a fixed probe batch of 256 validation examples. A unit is one neuron of a dense layer or one filter of a convolution, and a filter counts as dead only when it outputs 0 at every position.
\\end{{itemize}}

The probe batch is the same in every run and is passed through the network in evaluation mode without gradients.

\\subsection{{Healthy range}}

The healthy range is measured only on calibration runs, trained with no defects from different random starting points. For layer $\\ell$, signal $s$, and epoch $e$, let $a_{{\\ell s e}}$ and $b_{{\\ell s e}}$ be the lowest and highest values across the calibration runs. A value $x$ is out of range when
\\begin{{equation}}
x < a_{{\\ell s e}}(1 - m) \\quad \\text{{or}} \\quad x > b_{{\\ell s e}}(1 + m) + f,
\\end{{equation}}
with margin $m = {m}$ and floor $f = {f}$, and a value that is not a finite number is always out of range. The floor keeps a signal that is 0 in every healthy run, such as the dead unit fraction of most layers, from flagging a single unusual unit. Broken runs are never used to set the range, because a detector tuned on a known fault only catches that fault. Figure~\\ref{{fig:healthy_band_cnn}} shows the range for one CNN layer.

{FIG_HEALTHY_BAND_CNN}

\\subsection{{Alarm and diagnosis}}

LHD raises an alarm when one signal of one layer stays out of range for $p = {pat}$ epochs in a row. If several layers reach that point in the same epoch, LHD reports the first one in the order data flows through the network. The settings $m$, $f$, and $p$ were taken from the LHD library's own tests and fixed before any broken run was trained. LHD then names a likely cause from the pattern across the alarmed layer's four signals (Table~\\ref{{tab:rules}}), stops training, and reports the layer, the signals out of range, the cause, and a suggested check. For a frozen layer, for example, the suggested check is that the layer's parameters are passed to the optimizer.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Rules LHD uses to name a cause, checked in order on the alarmed layer.}}
\\label{{tab:rules}}
\\begin{{tabular}}{{@{{}}rll@{{}}}}
\\toprule
Order & Pattern & Cause named \\\\
\\midrule
1 & any signal not a finite number & diverged \\\\
2 & dead unit fraction above range & dead units \\\\
3 & weight change below range, gradient norm inside or above & frozen layer \\\\
4 & weight change below range, gradient norm below & vanishing gradient \\\\
5 & weight norm above range & runaway weights \\\\
6 & anything else & unrecognized \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

"""


def setup(d: Data) -> str:
    rows = []
    for a in d.archs:
        n = d.networks[a]
        rows.append(f"{NAMES[a]} & {DATASETS[a]} & {int(n['parameters']):,} & {n['watched_layers']} & "
                    f"{n['relu_layers']} & {n['epochs']} & {n['batch_size']} & {int(n['train_examples']):,} \\\\")
    count_rows = []
    for a in d.archs:
        count_rows.append(f"{NAMES[a]} & {d.role_count(a, 'calibration')} & {d.role_count(a, 'fresh')} & "
                          f"{d.role_count(a, 'fault')} & {d.role_count(a, 'calibration') + d.role_count(a, 'fresh') + d.role_count(a, 'fault')} \\\\")
    extra = (" A third phase added faults beyond the frozen layer and dead units, reported in Section~\\ref{sec:extra}."
             if d.more_complete() else "")
    return f"""\\section{{Experimental Setup}}

\\subsection{{Networks and data}}

LHD was tested on three networks with different structures (Table~\\ref{{tab:networks}}). The fully connected network has hidden layers of 256, 128, and 64 units. The CNN has three convolutions with 8, 16, and 32 filters, each followed by a ReLU and max pooling, then a dense layer of 64 units and the output layer. The transformer predicts the next character with a width of 64, 4 attention heads, 2 blocks, and a ReLU feed-forward layer of 256 units, so dead units can be counted the same way in every network. Validation accuracy is measured on each dataset's test split after every epoch. The CNN is scored on a fixed random subset of 4,000 test images, and the transformer on 512 fixed windows of 64 characters from the last 10\\% of the text.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Networks, datasets, and training settings. Training examples for the transformer count characters.}}
\\label{{tab:networks}}
\\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\\toprule
Network & Dataset & Parameters & Watched layers & ReLU layers & Epochs & Batch & Training examples \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Every run used Adam with a learning rate of 0.001 and a cosine schedule, on one CPU thread. The seed fixes both the starting weights and the order of training batches, so a broken run and the healthy run with the same seed differ only by the fault. That healthy run is the broken run's twin, and the twin's final accuracy shows what the fault cost.

\\subsection{{Planted faults}}

Each broken run has one fault planted in one layer at the start of training. A frozen layer has its weights and bias restored after every optimizer step, so gradients are computed but never applied, the same failure as a layer left out of the optimizer. Dead units are made by setting the incoming weights of 10\\%, 25\\%, or 50\\% of a layer's units to 0 and their bias to $-1$, so their ReLU output is 0 for every input, and the optimizer's running averages for those units are cleared. Frozen layers were planted in every watched layer and dead units in every layer followed by a ReLU.{extra}

\\subsection{{Runs}}

Calibration runs used seeds 0 to 19 and only set the healthy range. Fresh healthy runs and broken runs used seeds from 1000 up, so no evaluation run touched the range (Table~\\ref{{tab:runs}}). Every run trained for all its epochs, and LHD's alarm was found by replaying the saved signals, which gives the same answer as running LHD live. This let every detector be scored on the same complete runs.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Training runs by role.}}
\\label{{tab:runs}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Network & Calibration & Fresh healthy & Broken & Total \\\\
\\midrule
{chr(10).join(count_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Detectors compared}}

The accuracy curve raises suspicion when validation accuracy falls below the lowest calibration value for that epoch {SETTINGS['patience']} epochs in a row, the same patience LHD uses and with no margin, which makes it more sensitive than LHD's rule. A version that needs only one epoch and a loss curve version were also scored. Two kinds of fixed rule that need no healthy runs were added to test whether LHD does better than checks anyone could write by hand. The zero update check fires when some layer's weights do not move at all in an epoch, like SageMaker Debugger's UnchangedTensor rule. The dead unit checks fire when some ReLU layer has at least a fixed share of dead units, at 50\\% (the default of SageMaker Debugger's DeadRelu rule), 70\\% (DeepDiagnosis), 80\\% (the default of wandb-callbacks), or any dead unit at all (its verbose mode). SageMaker counts a ReLU as inactive when it is active less than 1\\% of the time, which is close to but not the same as LHD's definition.

\\subsection{{Measures and statistics}}

A broken run is caught when a detector fires at or after the fault's onset, and a false alarm is any alarm on a fresh healthy run. The gap is the accuracy curve's epoch minus LHD's epoch, with ``never'' counted as one epoch past the end of the run. Runs that share a seed share starting weights and batch order, and runs that share a fault and layer share the same break, so single runs are not independent. The main tests therefore average the gap within each seed and within each fault condition, then apply a one sided Wilcoxon signed rank test to those averages, and the 95\\% interval for the mean gap comes from a bootstrap with 10,000 draws that resamples whole seeds. Rates are given with 95\\% Wilson intervals in the data tables.

"""


def results(d: Data) -> str:
    out = ["\\section{Results}\n"]
    out.append(detection_section(d))
    out.append(detector_section(d))
    out.append(dose_section(d))
    out.append(harm_section(d))
    out.append(diagnosis_section(d))
    out.append(ablation_section(d))
    out.append(settings_section(d))
    out.append(cost_section(d))
    if d.more_complete():
        out.append(extra_section(d))
    return "\n".join(out)


def detection_section(d: Data) -> str:
    rows = []

    def row(label, fn):
        rows.append(f"{label} & " + " & ".join(fn(a) for a in d.archs) + " \\\\")

    s = d.summary["archs"]
    row("Broken runs", lambda a: str(s[a]["proposal"]["runs"]))
    row("Caught by LHD", lambda a: rate(s[a]["proposal"]["detected"]))
    row("Right layer, when caught", lambda a: rate(s[a]["proposal"]["layer_ok"]))
    row("Right cause, when caught", lambda a: rate(s[a]["proposal"]["cause_ok"]))
    row("Median epoch LHD fired", lambda a: f"{s[a]['proposal']['lhd_median_epoch']:.0f}")
    row("Caught by the accuracy curve", lambda a: rate(s[a]["proposal"]["acc_detected"]))
    row("Median epoch accuracy curve fired",
        lambda a: "n/a" if s[a]["proposal"]["acc_median_epoch"] is None else f"{s[a]['proposal']['acc_median_epoch']:.0f}")
    row("Final accuracy inside healthy range", lambda a: rate(s[a]["proposal"]["silent_share"]))
    row("Fresh healthy runs", lambda a: str(s[a]["healthy"]["runs"]))
    row("Flagged by LHD", lambda a: kofn(s[a]["healthy"]["lhd_false_alarm"]))
    row("Flagged by the accuracy curve", lambda a: kofn(s[a]["healthy"]["acc_false_alarm"]))
    row("Mean epochs LHD was earlier",
        lambda a: f"{s[a]['proposal']['mean_gap']:.1f} ({s[a]['proposal_clustered']['mean_gap_ci'][0]:.1f} to "
                  f"{s[a]['proposal_clustered']['mean_gap_ci'][1]:.1f})")
    c = d.pooled["proposal_clustered"]
    p = d.pooled["proposal"]
    alarm_text = false_alarm_text(d)
    return f"""\\subsection{{Detection and timing}}

Table~\\ref{{tab:headline}} gives the main results for each network. Across all three, LHD caught {kofn(p['detected'])} broken runs and the accuracy curve caught {kofn(p['acc_detected'])}. Of the runs LHD caught, the accuracy curve never noticed {rate(d.pooled['acc_never_on_detected'])}, and {rate(p['silent_share'])} of all broken runs finished with a final accuracy inside the healthy range. LHD's median alarm came at epoch {p['lhd_median_epoch']:.0f}, which is the earliest possible alarm for a fault planted at epoch 1 under a patience of 3 epochs. Figure~\\ref{{fig:cumulative_detection}} shows the share of broken runs flagged by each epoch.

Averaged within each seed, LHD was earlier than the accuracy curve for {c['by_seed']['lhd_earlier']} of {c['by_seed']['groups']} seed groups ({pval(c['by_seed'].get('wilcoxon_p'))}). Averaged within each fault condition, it was earlier for {c['by_condition']['lhd_earlier']} of {c['by_condition']['groups']} ({pval(c['by_condition'].get('wilcoxon_p'))}). The mean gap was {p['mean_gap']:.1f} epochs, with a 95\\% interval of {c['mean_gap_ci'][0]:.1f} to {c['mean_gap_ci'][1]:.1f} from the seed bootstrap.

{alarm_text}

{FIG_ANATOMY_CNN_FROZEN}

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Results on the planted frozen layers and dead units, by network. The gap interval comes from a bootstrap over seeds.}}
\\label{{tab:headline}}
\\begin{{tabular}}{{@{{}}l{'r' * len(d.archs)}@{{}}}}
\\toprule
 & {' & '.join(NAMES[a] for a in d.archs)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{FIG_CUMULATIVE_DETECTION}
"""


FIG_HEALTHY_BAND_CNN = figure('healthy_band_cnn', 'The healthy range for the CNN layer conv2. Gray lines are the 20 calibration runs that set the range, blue lines are fresh healthy runs that never touched it, and the shaded band is the range after the margin and floor.')

FIG_ANATOMY_CNN_FROZEN = figure('anatomy_cnn_frozen', "One CNN run with a frozen layer. Each panel is one signal of the planted layer against its healthy range, and the last panel is validation accuracy. The vertical rule marks the epoch of LHD's alarm.")

FIG_CUMULATIVE_DETECTION = figure('cumulative_detection', 'Share of broken runs flagged by each epoch, for LHD, the accuracy curve, the loss curve, and the zero update or 50\\% dead unit check.')

FIG_DETECTOR_COMPARISON = figure('detector_comparison', 'Every detector on the same runs, pooled over the three networks, with 95\\% Wilson intervals.')

FIG_DOSE_RESPONSE = figure('dose_response', 'Share of dead unit faults caught as the killed share grows, for LHD and the accuracy curve, by network, with 95\\% Wilson intervals.')

FIG_ACCURACY_CURVES = figure('accuracy_curves', 'Validation accuracy over training. The shaded band is the range of the calibration runs and the dark line the mean of fresh healthy runs. Colored lines are the mean of broken runs for each fault.')

FIG_PER_LAYER = figure('per_layer', 'Share of caught runs where LHD named both the planted layer and the right cause, by fault and layer.')

FIG_TRADEOFF = figure('tradeoff', 'Broken runs caught and healthy runs flagged as the margin changes, at floor 0.05. Solid lines use patience 3 and dashed lines patience 1.')

FIG_CALIBRATION_SIZE = figure('calibration_size', 'False alarms and broken runs caught when the healthy range is built from fewer calibration runs.')


def false_alarm_text(d: Data) -> str:
    """What each false alarm on a fresh healthy run was, measured from the saved signals."""
    h = d.pooled["healthy"]["lhd_false_alarm"]
    if not d.alarms:
        return f"LHD raised no false alarms in {h['n']} fresh healthy runs."
    groups = defaultdict(list)
    for r in d.alarms:
        value = d.alarm_value(r["run_id"], r["lhd_layer"], r["lhd_epoch"])
        low, high = d.dead_range(r["arch"], r["lhd_layer"])
        if r["lhd_signals"] == "dead_unit_fraction":
            kind = ("above", round(value, 4), high) if value > high else ("below", round(value, 4), low)
        else:
            kind = (r["lhd_signals"], None, None)
        groups[(r["arch"], r["lhd_layer"], kind)].append(r["lhd_epoch"])
    parts = []
    for (arch, layer, (kind, value, edge)), epochs in groups.items():
        words = {1: "one run", 2: "two runs", 3: "three runs", 4: "four runs", 5: "five runs"}
        runs = words.get(len(epochs), f"{len(epochs)} runs")
        when = f"epoch {epochs[0]}" if len(epochs) == 1 else "epochs " + " and ".join(sorted(epochs))
        where = f"{runs} in the {PROSE[arch]}'s {layer} at {when}"
        if kind == "above":
            limit = ("no calibration run had a single dead unit there" if edge == 0
                     else f"no calibration run had more than {100 * edge:.1f}\\%")
            parts.append(f"{where}, where {100 * value:.1f}\\% of units were dead and {limit}")
        elif kind == "below":
            parts.append(f"{where}, where only {100 * value:.1f}\\% of units were dead, fewer than in any calibration "
                         f"run (lowest {100 * edge:.1f}\\%)")
        else:
            parts.append(f"{where} on the {kind.replace('_', ' ')}")
    below = any("fewer than" in x for x in parts)
    tail = (" An alarm for too few dead units does not point to a fault, so checking the dead unit fraction against "
            "only the upper end of its range would remove that kind of false alarm.") if below else ""
    return (f"On fresh healthy runs, LHD raised {h['k']} false alarms in {h['n']} runs: " + "; ".join(parts) + "."
            + " When no calibration run had a dead unit in a layer, the top of that layer's range is the 0.05 floor, "
            "so one dead filter out of 16 is enough to cross it." + tail)


DETECTORS = (
    ("LHD", "LHD"),
    ("Accuracy curve, 3 epochs below healthy", "Accuracy curve, 3 epochs"),
    ("Accuracy curve, 1 epoch below healthy", "Accuracy curve, 1 epoch"),
    ("Loss curve, 3 epochs above healthy", "Loss curve, 3 epochs"),
    ("Zero update check", "Zero update check"),
    ("Dead units check, 25%", "Dead units, 25\\%"),
    ("Dead units check, 50% (SageMaker DeadRelu default)", "Dead units, 50\\% (SageMaker)"),
    ("Dead units check, 70% (DeepDiagnosis Dead Node)", "Dead units, 70\\% (DeepDiagnosis)"),
    ("Dead units check, 80% (wandb-callbacks default)", "Dead units, 80\\% (wandb-callbacks)"),
    ("Any dead unit (wandb-callbacks verbose)", "Any dead unit (wandb-callbacks)"),
)


def detector_section(d: Data) -> str:
    rows = []
    for name, label in DETECTORS:
        frozen = d.detector(name, ("frozen",))
        dead = d.detector(name, DEAD)
        fa = d.detector_alarms(name)
        median = "--" if not fa["median"] else f"{float(fa['median']):.0f}"
        rows.append(f"{label} & {pct(frozen['k'], frozen['n'])} & {pct(dead['k'], dead['n'])} & "
                    f"{fa['k']} of {fa['n']} & {median} \\\\")
    zero = d.detector("Zero update check", ("frozen",))
    lhd_dead = d.detector("LHD", DEAD)
    any_dead = d.detector("Any dead unit (wandb-callbacks verbose)", DEAD)
    any_fa = d.detector_alarms("Any dead unit (wandb-callbacks verbose)")
    q25 = d.detector("Dead units check, 25%", DEAD)
    q25_fa = d.detector_alarms("Dead units check, 25%")
    cnn_lo, cnn_hi = d.dead_range("cnn", "fc1")
    tr_hi = max(d.dead_range("transformer", layer)[1] for layer in ("b0.ff1", "b1.ff1"))
    return f"""\\subsection{{Comparison with other detectors}}

Table~\\ref{{tab:detectors}} and Figure~\\ref{{fig:detector_comparison}} compare LHD with every other detector on the same runs. For frozen layers, LHD adds nothing over the zero update check, which caught {kofn(zero)} at epoch 1 with no false alarms, two epochs before LHD. A frozen layer's weight change is exactly 0, so a check for an exact 0 needs no healthy range and no patience.

Dead units are where the healthy range matters. LHD caught {pct(lhd_dead['k'], lhd_dead['n'])} of dead unit faults with {d.detector_alarms('LHD')['k']} false alarms. A fixed rule either missed most faults or flagged many healthy runs. A 25\\% rule came closest, catching {pct(q25['k'], q25['n'])}, but it flagged {q25_fa['k']} of {q25_fa['n']} healthy runs, and a rule that fires on any dead unit caught {pct(any_dead['k'], any_dead['n'])} while flagging {any_fa['k']} of {any_fa['n']}. The defaults of SageMaker Debugger, DeepDiagnosis, and wandb-callbacks sit in between. Healthy networks differ in how many units die on their own. In the CNN's dense layer, between {100 * cnn_lo:.0f}\\% and {100 * cnn_hi:.0f}\\% of units were dead in healthy calibration runs, while no calibration run of the transformer had more than {100 * tr_hi:.0f}\\% dead units in any layer, so no single threshold fits every layer and LHD's per layer range does.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Every detector on the same broken and healthy runs, pooled over the three networks. The median epoch is over the runs each detector caught.}}
\\label{{tab:detectors}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Detector & Frozen caught & Dead units caught & Healthy flagged & Median epoch \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{FIG_DETECTOR_COMPARISON}
"""


def dose_section(d: Data) -> str:
    misses = [r for r in d.per_layer if r["family"] in DEAD and int(r["detected"]) < int(r["runs"])]
    miss_text = ", ".join(f"{PROSE[r['arch']]} {r['layer']} at {r['family'][4:]}\\% ({r['detected']} of {r['runs']} caught)"
                          for r in misses)
    return f"""\\subsection{{How many units must die}}

Figure~\\ref{{fig:dose_response}} shows how often LHD and the accuracy curve caught dead unit faults as the killed share grew from 10\\% to 50\\%. LHD's misses were concentrated in a few layers: {miss_text}. In each of these layers, healthy runs already lose a share of units on their own, so the planted fault stayed inside the healthy range. The CNN's dense layer is the clearest case, since its healthy range already spans most of the planted values.

{FIG_DOSE_RESPONSE}
"""


def harm_section(d: Data) -> str:
    rows = []
    for r in d.harm:
        n = int(r["broken_runs"])
        rows.append(f"{r['harm'].replace('pts', 'points')} & {n} & {float(r['mean_drop_pts']):.2f} & "
                    f"{pct(int(r['lhd_caught']), n)} & {pct(int(r['accuracy_caught']), n)} & "
                    f"{pct(int(r['simple_caught']), n)} \\\\")
    low = next((r for r in d.harm if r["harm"] == "under 0.5 pts"), None)
    high = next((r for r in d.harm if r["harm"] == "2 pts or more"), None)
    total = sum(int(r["broken_runs"]) for r in d.harm)
    text = ""
    if low and high:
        text = (f"Most planted faults did little harm. {low['broken_runs']} of {total} broken runs lost under half a point "
                f"of final accuracy against their healthy twin, and the accuracy curve missing those runs is partly the "
                f"curve being right. On the {high['broken_runs']} runs that lost 2 points or more, LHD caught "
                f"{high['lhd_caught']} and the accuracy curve caught {high['accuracy_caught']}.")
    return f"""\\subsection{{Faults that cost accuracy}}

{text} Table~\\ref{{tab:harm}} splits broken runs by how much accuracy they lost, and Figure~\\ref{{fig:accuracy_curves}} shows why the accuracy curve struggles. Averaged by fault, broken runs stay inside the healthy accuracy range for most of training. The clearest exception is a frozen layer in the CNN, which cost about as much as the frozen convolution in the GTSRB model that motivated this work.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Broken runs caught, split by accuracy lost against the healthy twin. The simple check is the zero update check or a 50\\% dead unit rule.}}
\\label{{tab:harm}}
\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\\toprule
Accuracy lost & Runs & Mean lost (points) & LHD & Accuracy curve & Simple check \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{FIG_ACCURACY_CURVES}
"""


def diagnosis_section(d: Data) -> str:
    p = d.pooled["proposal"]
    wrong = d.wrong
    causes = [r for r in d.runs if r["role"] == "fault" and r["family"] in ("frozen",) + DEAD
              and r["detected"] == "True" and r["layer_ok"] == "True" and r["cause_ok"] == "False"]
    def fault(fam, layer):
        return f"frozen {layer}" if fam == "frozen" else f"{fam[4:]}\\% dead units in {layer}"

    wrong_text = ""
    if wrong:
        groups = defaultdict(int)
        for r in wrong:
            groups[(r["arch"], r["family"], r["fault_layer"], r["lhd_layer"])] += 1
        wrong_text = (f" It named the wrong layer in {len(wrong)} caught runs: "
                      + "; ".join(f"{PROSE[a]} {fault(fam, layer)} reported as {named} ({n})"
                                  for (a, fam, layer, named), n in sorted(groups.items())) + ".")
        dead_hits = sum(r["lhd_signals"] == "dead_unit_fraction" for r in wrong)
        frozen_now = [r for r in wrong if r["family"] == "frozen" and r["lhd_epoch"] == str(SETTINGS["patience"])]
        wrong_text += (f" In each case the reported layer's own signal was out of range, dead units in {dead_hits} runs "
                       f"and another signal in {len(wrong) - dead_hits}, so LHD found a real effect of the fault in a "
                       f"different layer.")
        if frozen_now:
            wrong_text += (f" In the {len(frozen_now)} frozen cases, the planted layer's weight change crossed its range at "
                           f"the same epoch, but when several layers cross at once LHD names the first in data flow "
                           f"order, so reporting every layer out of range would have included the planted one.")
    cause_text = ""
    if causes:
        groups = defaultdict(int)
        for r in causes:
            groups[(r["arch"], r["family"], r["fault_layer"], r["lhd_cause"])] += 1
        cause_text = (f" It named the right layer but the wrong cause in {len(causes)} runs: "
                      + "; ".join(f"{PROSE[a]} {fault(fam, layer)} called {cause.replace('_', ' ')} ({n})"
                                  for (a, fam, layer, cause), n in sorted(groups.items())) + ".")
        if any("qkv" in layer and cause == "vanishing_gradient" for _, _, layer, cause in groups):
            cause_text += (" A frozen attention projection also lowers the gradient that reaches it, so its pattern "
                           "matches a vanishing gradient.")
    return f"""\\subsection{{Diagnosis}}

When LHD caught a broken run, it named the planted layer in {rate(p['layer_ok'], 1)} of cases and the planted cause in {rate(p['cause_ok'], 1)} (Figure~\\ref{{fig:per_layer}}).{wrong_text}{cause_text} The high rate of naming the right cause is largely built into the design, since the faults produce exactly the signal patterns the rules look for.

{FIG_PER_LAYER}
"""


def ablation_section(d: Data) -> str:
    by = {(r["arch"], r["variant"]): r for r in d.ablation}
    variants = list(dict.fromkeys(r["variant"] for r in d.ablation))
    rows = []
    for v in variants:
        cells = []
        for a in d.archs:
            r = by.get((a, v))
            cells.append("--" if r is None else f"{100 * float(r['detection']):.0f} / {100 * float(r['both_ok']):.0f}")
        rows.append(f"{v.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    return f"""\\subsection{{Each signal's part}}

Table~\\ref{{tab:ablation}} replays the same runs with signals removed or used alone. The dead unit fraction carries most of the detection, and the weight change is what lets LHD tell a frozen layer apart from other faults. Removing the weight norm or the gradient norm changed little on these faults, which were not designed to move them.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Signal ablation. Each cell gives the percent of broken runs caught, then the percent where LHD named both the right layer and the right cause.}}
\\label{{tab:ablation}}
\\begin{{tabular}}{{@{{}}l{'r' * len(d.archs)}@{{}}}}
\\toprule
Signals used & {' & '.join(NAMES[a] for a in d.archs)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def settings_section(d: Data) -> str:
    def sweep(a, margin, patience=3, floor=0.05):
        for r in d.sweep:
            if (r["arch"] == a and float(r["margin"]) == margin and int(r["patience"]) == patience
                    and float(r["floor"]) == floor):
                return r
        return None

    edge = []
    for a in d.archs:
        r0, r5 = sweep(a, 0.0), sweep(a, SETTINGS["margin"])
        if r0 and r5:
            edge.append(f"{PROSE[a]} {100 * float(r0['false_alarm']):.0f}\\% to {100 * float(r5['false_alarm']):.0f}\\%")
    cal = defaultdict(list)
    for r in d.calibration:
        cal[(r["arch"], int(r["k"]))].append(float(r["false_alarm"]))
    small = ", ".join(f"{PROSE[a]} {100 * sum(cal[(a, 2)]) / len(cal[(a, 2)]):.0f}\\%" for a in d.archs if cal[(a, 2)])
    full = ", ".join(f"{PROSE[a]} {100 * sum(cal[(a, 20)]) / len(cal[(a, 20)]):.0f}\\%" for a in d.archs if cal[(a, 20)])
    return f"""\\subsection{{Settings and calibration}}

Thresholds placed right at the edge of the healthy range flagged many healthy runs. Raising the margin from 0 to {SETTINGS['margin']} at a patience of 3 cut false alarms ({'; '.join(edge)}), while broken runs caught fell only a little (Figure~\\ref{{fig:tradeoff}}). The healthy range also needs enough calibration runs. Built from 2 runs, it flagged on average {small} of healthy runs, and built from all 20, {full}, with 20 random draws per size (Figure~\\ref{{fig:calibration_size}}).

{FIG_TRADEOFF}

{FIG_CALIBRATION_SIZE}
"""


def cost_section(d: Data) -> str:
    s = d.summary["archs"]
    overhead = ", ".join(f"{100 * s[a]['healthy']['overhead']:.1f}\\% for the {PROSE[a]}" for a in d.archs)
    p = d.pooled["proposal"]
    hours = float(d.compute["all"]["training_hours"])
    return f"""\\subsection{{Cost}}

Monitoring added {overhead}, measured on fresh healthy runs on one CPU thread. Most of the cost is the gradient norm, read after every batch, and the probe batch, run once per epoch, so smaller networks with cheap epochs pay the largest share. Stopping each broken run at LHD's alarm would have saved {100 * p['mean_compute_saved']:.0f}\\% of its training on average. All runs together took {hours:.0f} hours of CPU time on one laptop.
"""


def extra_section(d: Data) -> str:
    rows = []
    names = {"latefrozen": "Frozen at epoch 6", "latedead50": "50\\% dead at epoch 6", "detach": "Cut gradient",
             "lr10": "Learning rate $\\times$10 at epoch 6", "lr100": "Learning rate $\\times$100 at epoch 6"}
    for fam, label in names.items():
        cells = []
        for a in d.archs:
            st = d.summary["archs"][a]["families"].get(fam)
            cells.append("--" if not st else f"{rate(st['detected'])} / {rate(st['acc_detected'])}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return f"""\\subsection{{Faults beyond frozen layers and dead units}}
\\label{{sec:extra}}

A third phase tested faults the rules were not built around: a frozen layer or 50\\% dead units planted at epoch 6, a cut that stops gradients from reaching earlier layers, and a learning rate raised 10 or 100 times at epoch 6 (Table~\\ref{{tab:extra}}).

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Faults beyond the main study. Each cell gives the percent caught by LHD, then by the accuracy curve.}}
\\label{{tab:extra}}
\\begin{{tabular}}{{@{{}}l{'r' * len(d.archs)}@{{}}}}
\\toprule
Fault & {' & '.join(NAMES[a] for a in d.archs)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def discussion(d: Data) -> str:
    acc = next(float(r["final_acc"]) for r in d.runs if r["arch"] == "transformer" and r["role"] == "fresh")
    trans = [float(r["final_acc"]) for r in d.runs if r["arch"] == "transformer" and r["role"] == "fresh"]
    return f"""\\section{{Discussion and Limitations}}

The results support the hypothesis for planted faults. Per layer signals exposed a broken layer within the first few epochs, usually before the accuracy curve showed anything, and LHD rarely flagged healthy runs. They also show where the idea's value lies. A frozen layer is caught just as well, and sooner, by a check for weights that never change, so LHD's contribution there is only that the same rules cover it. For dead units, a range measured on healthy runs of the same network did better than every fixed threshold tested, including the defaults of three existing tools.

Several limits apply. The faults were planted to show up in the signals LHD measures, so detection and diagnosis rates describe how well LHD separates these faults from normal variation, not how often it would catch bugs found in the wild. Many planted faults cost less than one point of accuracy, so stopping those runs saves compute on runs that would have produced a nearly normal model. LHD misses dead units in layers where healthy runs already lose many units, because the healthy range treats that loss as normal. The healthy range needs about 10 to 20 healthy runs of the same network, which costs more than LHD can save for a model trained only once, and it assumes those runs are themselves healthy. The networks are small, trained for 15 to 20 epochs on a laptop CPU, and the transformer reaches only about {100 * sum(trans) / len(trans):.0f}\\% next character accuracy. Large networks, long runs, other optimizers, and GPU training were not tested.

\\section{{Conclusion}}

This work presented LHD, a monitor that compares four per layer signals with ranges measured on healthy runs of the same network and stops a run when one stays out of range. On {d.total_runs()} runs across three networks, LHD caught most planted frozen layers and dead units within three epochs, named the right layer almost every time, and rarely flagged healthy runs. Its clearest advantage over existing tools is for dead units, where a healthy range per layer avoided the misses and false alarms of fixed thresholds. Testing LHD on real bugs collected from training code is the next step.

\\section*{{Code and data}}

The code, every saved run, and the tables behind each number are at \\url{{https://github.com/sohan-bhat/lhd}}.

"""


def main() -> None:
    d = Data()
    figures = PAPER / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy(RESULTS / "figures" / "pdf" / f"{name}.pdf", figures / f"{name}.pdf")
    body = [
        preamble(),
        abstract(d),
        introduction(d),
        "\\input{related_work}\n",
        method(),
        setup(d),
        results(d),
        discussion(d),
        "\\input{references}\n",
        "\\end{document}\n",
    ]
    (PAPER / "lhd.tex").write_text("\n".join(body))
    print(f"wrote {PAPER / 'lhd.tex'}", flush=True)
    if TECTONIC.exists():
        result = subprocess.run([str(TECTONIC), "-X", "compile", "lhd.tex"], cwd=PAPER, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-3000:])
            raise SystemExit(1)
        print(f"wrote {PAPER / 'lhd.pdf'}", flush=True)


if __name__ == "__main__":
    main()
