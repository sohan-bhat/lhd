"""Writes the LHD paper from the study's results and compiles it.

Usage:
    python -m experiments.paper

Run after experiments.report. Draws the paper's figures with
experiments.paper_figures, writes paper/lhd.tex, and compiles
paper/lhd.pdf with tectonic. Every number comes from
results/summary.json and results/tables, so the paper can be rebuilt
whenever more runs finish. The related work and references are fixed
text in paper/related_work.tex and paper/references.tex, copied into
lhd.tex so the .tex compiles on its own next to the figures folder.
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from experiments.evaluate import RESULTS, SETTINGS
from experiments.grid import ARCH_ORDER, build_grid

PAPER = RESULTS.parent / "paper"
TABLES = RESULTS / "tables"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic"
NAMES = {"mlp": "Fully connected", "transformer": "Transformer", "cnn": "CNN"}
PROSE = {"mlp": "fully connected network", "transformer": "transformer", "cnn": "CNN"}
DATASETS = {"mlp": "Fashion-MNIST", "transformer": "Tiny Shakespeare", "cnn": "GTSRB"}
DEAD = ("dead10", "dead25", "dead50")
FIGURES = ("how_lhd_works", "detection_over_time", "detectors_tradeoff", "per_layer")
SAGEMAKER = "Dead units check, 50% (SageMaker DeadRelu default)"
DEEPDIAGNOSIS = "Dead units check, 70% (DeepDiagnosis Dead Node)"
WANDB = "Dead units check, 80% (wandb-callbacks default)"
ANY_DEAD = "Any dead unit (wandb-callbacks verbose)"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def read(name: str) -> list[dict]:
    with open(TABLES / name) as f:
        return list(csv.DictReader(f))


def pct(k: float, n: float, digits: int = 0) -> str:
    return "n/a" if not n else f"{100 * k / n:.{digits}f}\\%"


def rate(stat: dict, digits: int = 0) -> str:
    return pct(stat["k"], stat["n"], digits)


def kofn(stat: dict) -> str:
    return f"{stat['k']} of {stat['n']}"


def count(n: int, noun: str) -> str:
    word = WORDS.get(n, str(n))
    return f"{word} {noun}" if n == 1 else f"{word} {noun}s"


def listing(items: list[str]) -> str:
    """Items joined as English, with "and" before the last."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


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
        """Every calibration and fresh run's signals, per epoch and layer."""
        if self._signals is None:
            with gzip.open(TABLES / "signals.csv.gz", "rt") as f:
                self._signals = [r for r in csv.DictReader(f) if r["role"] in ("calibration", "fresh")]
        return self._signals

    def dead_range(self, arch: str, layer: str) -> tuple[float, float]:
        """Lowest and highest dead unit fraction of a layer over calibration runs and epochs."""
        values = [float(r["dead_unit_fraction"]) for r in self.signals()
                  if r["arch"] == arch and r["layer"] == layer and r["role"] == "calibration"]
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
        return {"k": sum(int(r["caught"]) for r in rows), "n": sum(int(r["broken_runs"]) for r in rows)}

    def detector_alarms(self, name: str) -> dict:
        row = next(r for r in self.detectors if r["detector"] == name and r["family"] == "proposal")
        return {"k": int(row["false_alarms"]), "n": int(row["healthy_runs"])}

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
\usepackage{array}
\usepackage{amsmath}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}

\title{Early Detection of Silent Training Failures in Neural Networks}
\author{Sohan Bhat}
\date{November 2026}

\begin{document}

\maketitle
"""


def figure(name: str, caption: str) -> str:
    return (f"\\begin{{figure}}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{fig:{name}}}\n\\end{{figure}}\n")


FIG = {
    "how_lhd_works": figure("how_lhd_works", "One CNN run with a frozen layer. (a) The frozen layer's weight change "
                            "against the range of the 20 calibration runs. It stays at 0, outside the range, and LHD "
                            "stops the run after three epochs. (b) Validation accuracy of the same run, which is still "
                            "inside the healthy range when LHD stops it."),
    "detection_over_time": figure("detection_over_time", "Share of broken runs flagged by each epoch, for LHD and the "
                                  "accuracy curve, in each network."),
    "detectors_tradeoff": figure("detectors_tradeoff", "Dead unit faults caught against healthy runs flagged, for "
                                 "every detector on the same runs, pooled over the three networks. Detectors toward the "
                                 "upper left catch more faults while flagging fewer healthy runs."),
    "per_layer": figure("per_layer", "Share of broken runs in which LHD caught the fault and named both the planted "
                        "layer and the right cause, for every layer and fault. Each cell is 10 broken runs. Blank cells "
                        "are faults that were not planted, since dead units only go into layers followed by a ReLU."),
}


def abstract(d: Data) -> str:
    p, h, c = d.pooled["proposal"], d.pooled["healthy"], d.pooled["proposal_clustered"]
    lo, hi = c["mean_gap_ci"]
    dead = d.detector("LHD", DEAD)
    return f"""\\begin{{abstract}}
During training, neural networks can develop defects that raise no error, so the run completes and returns a degraded model that is easy to mistake for a hard problem. This work presents LHD (Layer Health Diagnostics), a monitor that logs four signals for every layer at the end of each epoch (weight change, gradient norm, weight norm, and the fraction of dead units) and compares each one with the range it stays within during healthy runs of the same network. When a signal stays outside that range for three epochs in a row, LHD names the layer and a likely cause and stops the run. Across {d.total_runs()} training runs of a fully connected network, a CNN, and a small transformer, with frozen layers and dead units planted in known layers, LHD caught {kofn(p['detected'])} broken runs, named the planted layer in {rate(p['layer_ok'])} of them, and flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs. Averaged within each seed, it fired {p['mean_gap']:.1f} epochs before the accuracy curve (95\\% interval {lo:.1f} to {hi:.1f}). A check for weights that never change caught every frozen layer two epochs sooner, so LHD helps most with dead units, where it caught {pct(dead['k'], dead['n'])} of faults while the fixed thresholds of existing tools either missed most of them or flagged many healthy runs.
\\end{{abstract}}
"""


def introduction(d: Data) -> str:
    p, h, c = d.pooled["proposal"], d.pooled["healthy"], d.pooled["proposal_clustered"]
    dead = d.detector("LHD", DEAD)
    sage = d.detector(SAGEMAKER, DEAD)
    sage_fa = d.detector_alarms(SAGEMAKER)
    return f"""\\section{{Introduction}}

During training, neural networks can develop defects, yet nothing fails visibly since the run completes normally and simply produces a degraded result; thus, these are called ``silent failures.'' Because of this, identifying whether the degraded result was caused by a difficult problem or a silent failure becomes difficult; therefore, it is common to misread a failure as normal. I initially encountered this as I was training my small-scale GTSRB (German Traffic Sign Recognition Benchmark) model \\citep{{stallkamp2012}} from scratch, where my model plateaued at 76\\%, wasting weeks tuning it, until I found one convolutional layer had been accidentally excluded from the optimizer and was never properly trained. Fixing this minute detail raised the accuracy to 87\\%. This is not an isolated case: errors in deep learning implementations rarely cause an exception and instead reveal themselves only through degraded predictive performance \\citep{{chwilczynski2024}}.

Since training runs consume significant compute, every silent failure that goes undetected wastes it. The evidence existed in the form of raw numbers in the weights and biases, but perceiving that is challenging since nothing in standard practice surfaces them. To address this, this work presents LHD (Layer Health Diagnostics), a monitor that reads these numbers, catches the failure within a few epochs, identifies the broken layer and its likely cause, and halts the run before compute is wasted.

To test LHD, {d.total_runs()} networks were trained, and every rule and setting was fixed before the first broken run. LHD caught {kofn(p['detected'])} broken runs, usually at epoch {p['lhd_median_epoch']:.0f}, while the accuracy curve caught {kofn(p['acc_detected'])}, and it was earlier on average for {c['by_seed']['lhd_earlier']} of {c['by_seed']['groups']} seed groups. It also flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs, against {kofn(h['acc_false_alarm'])} for the accuracy curve. However, a simple check for weights that do not change caught every frozen layer at epoch 1, two epochs before LHD, so the advantage of LHD comes from dead units. For those, it caught {pct(dead['k'], dead['n'])} with {h['lhd_false_alarm']['k']} false alarms, while the default rule of SageMaker Debugger caught {pct(sage['k'], sage['n'])} and flagged {sage_fa['k']} of {sage_fa['n']} healthy runs. Because the faults were planted to show up in the signals LHD measures, these results describe planted faults rather than bugs found in real projects.

"""


def method() -> str:
    m, f, pat = SETTINGS["margin"], SETTINGS["floor"], SETTINGS["patience"]
    return f"""\\section{{Method}}

\\subsection{{Signals}}

At the end of every epoch, LHD records four signals for each layer it watches. Let $W_e$ be the layer's weights and bias at the end of epoch $e$, flattened into one vector, and let $\\lVert \\cdot \\rVert$ be the square root of the sum of squared entries. The weight change, $\\lVert W_e - W_{{e-1}} \\rVert / \\lVert W_{{e-1}} \\rVert$, measures how far the weights moved during the epoch relative to their size, so a frozen layer reads exactly 0. The weight norm, $\\lVert W_e \\rVert$, grows past normal before a run diverges. The gradient norm is the mean of $\\lVert \\partial L / \\partial W \\rVert$ over the epoch's batches, taken after each backward pass and before the optimizer step, and a vanishing gradient reads near 0. Finally, the dead unit fraction is the share of the layer's units whose output after the ReLU is exactly 0 for every input in a fixed probe batch of 256 validation examples. A unit is one neuron of a dense layer or one filter of a convolution, and a filter counts as dead only when it outputs 0 at every position. The probe batch is the same in every run and passes through the network in evaluation mode without gradients.

\\subsection{{Healthy range}}

The healthy range comes only from calibration runs, which are trained with no defects from different random starting points. For layer $\\ell$, signal $s$, and epoch $e$, let $a_{{\\ell s e}}$ and $b_{{\\ell s e}}$ be the lowest and highest values across the calibration runs. A value $x$ is out of range when
\\begin{{equation}}
x < a_{{\\ell s e}}(1 - m) \\quad \\text{{or}} \\quad x > b_{{\\ell s e}}(1 + m) + f,
\\end{{equation}}
with margin $m = {m}$ and floor $f = {f}$, and a value that is not a finite number is always out of range. The floor keeps a signal that is 0 in every healthy run, such as the dead unit fraction of most layers, from flagging a single unusual unit. Broken runs are never used to set the range, since a detector tuned on a known fault only catches that fault.

\\subsection{{Alarm and diagnosis}}

LHD raises an alarm once one signal of one layer stays out of range for $p = {pat}$ epochs in a row. If several layers reach that point in the same epoch, it reports the first one in the order data flows through the network. The settings $m$, $f$, and $p$ came from the LHD library's own tests and were fixed before any broken run was trained. LHD then names a likely cause from the pattern across the alarmed layer's four signals (Table~\\ref{{tab:rules}}), stops training, and reports the layer, the signals out of range, the cause, and a suggested check. For a frozen layer, for example, it suggests checking that the layer's parameters are passed to the optimizer. Figure~\\ref{{fig:how_lhd_works}} shows this on one CNN run with a frozen layer.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{How LHD names a cause. The rules are checked from top to bottom on the layer that raised the alarm.}}
\\label{{tab:rules}}
\\begin{{tabular}}{{@{{}}p{{3.9in}}l@{{}}}}
\\toprule
When the alarmed layer shows & LHD names \\\\
\\midrule
a signal that is not a finite number & diverged \\\\
more dead units than its range allows & dead units \\\\
weight change below its range while the gradient is inside or above & frozen layer \\\\
weight change and gradient both below their ranges & vanishing gradient \\\\
weight norm above its range & runaway weights \\\\
any other pattern & unrecognized \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{FIG['how_lhd_works']}
"""


def setup(d: Data) -> str:
    rows = []
    for a in d.archs:
        n = d.networks[a]
        rows.append(f"{NAMES[a]} & {DATASETS[a]} & {int(n['parameters']):,} & {n['epochs']} & "
                    f"{d.role_count(a, 'calibration')} & {d.role_count(a, 'fresh')} & {d.role_count(a, 'fault')} \\\\")
    extra = (" A third set of runs added faults beyond these two, reported in Section~\\ref{sec:extra}."
             if d.more_complete() else "")
    return f"""\\section{{Experimental Setup}}

\\subsection{{Networks and data}}

LHD was tested on three networks with different structures (Table~\\ref{{tab:networks}}). The fully connected network has hidden layers of 256, 128, and 64 units. The CNN has three convolutions with 8, 16, and 32 filters, each followed by a ReLU and max pooling, then a dense layer of 64 units and the output layer. The transformer predicts the next character with a width of 64, 4 attention heads, 2 blocks, and a feed-forward layer of 256 units that uses a ReLU, so dead units can be counted the same way in all three. After every epoch, validation accuracy is measured on the dataset's test split. The CNN is scored on a fixed random subset of 4,000 test images and the transformer on 512 fixed windows of 64 characters from the last 10\\% of the text.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{The three networks and the runs trained for each. Calibration runs set the healthy range, and fresh healthy runs measure false alarms.}}
\\label{{tab:networks}}
\\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\\toprule
 & & & & \\multicolumn{{3}}{{c}}{{Runs}} \\\\
\\cmidrule(l){{5-7}}
Network & Dataset & Parameters & Epochs & Calibration & Fresh healthy & Broken \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Every run used Adam with a learning rate of 0.001 and a cosine schedule on one CPU thread. The seed fixes both the starting weights and the order of training batches, so a broken run and the healthy run with the same seed differ only by the fault. That healthy run is the broken run's twin, and the twin's final accuracy shows how much the fault cost.

\\subsection{{Planted faults}}

Each broken run has one fault planted in one layer at the start of training. A frozen layer has its weights and bias put back after every optimizer step, so gradients are computed but never applied, which is the same failure as a layer left out of the optimizer. Dead units are made by setting the incoming weights of 10\\%, 25\\%, or 50\\% of a layer's units to 0 and their bias to $-1$, so their ReLU output is 0 for every input, and the optimizer's running averages for those units are cleared. Frozen layers were planted in every watched layer and dead units in every layer followed by a ReLU.{extra}

\\subsection{{Runs}}

Calibration runs used seeds 0 to 19 and were used only to set the healthy range. Fresh healthy runs and broken runs used seeds from 1000 up, so no scored run ever touched the range. Every run trained for all of its epochs, and LHD's alarm was found by replaying the saved signals. Replaying gives the same answer as running LHD live and lets every detector be scored on the same complete runs.

\\subsection{{Detectors compared}}

The accuracy curve raises suspicion when validation accuracy falls below the lowest calibration value for that epoch {SETTINGS['patience']} epochs in a row. It uses the same patience as LHD and no margin, which makes it more sensitive than LHD's own rule. A version that needs only one epoch and a version based on the loss curve were also scored. Two kinds of fixed rule that need no healthy runs test whether LHD does better than checks anyone could write by hand. The zero update check fires when some layer's weights do not move at all in an epoch, like the UnchangedTensor rule of SageMaker Debugger. The dead unit checks fire when some ReLU layer has at least a fixed share of dead units, at 50\\% (the default of SageMaker Debugger's DeadRelu rule), 70\\% (DeepDiagnosis), 80\\% (the default of wandb-callbacks), or a single dead unit (its verbose mode). SageMaker counts a ReLU as inactive when it fires less than 1\\% of the time, which is close to LHD's definition but not the same.

\\subsection{{Measures and statistics}}

A broken run is caught when a detector fires at or after the fault's onset, and a false alarm is any alarm on a fresh healthy run. The gap is the accuracy curve's epoch minus LHD's epoch, with ``never'' counted as one epoch past the end of the run. Runs that share a seed share their starting weights and batch order, and runs that share a fault and layer share the same break, so single runs are not independent. The main tests therefore average the gap within each seed and within each fault condition and apply a one sided Wilcoxon signed rank test to those averages. The 95\\% interval for the mean gap comes from a bootstrap with 10,000 draws that resamples whole seeds.

"""


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
        when = f"epoch {epochs[0]}" if len(epochs) == 1 else "epochs " + " and ".join(sorted(epochs))
        where = f"{count(len(epochs), 'run')} in the {PROSE[arch]}'s {layer} at {when}"
        if kind == "above":
            limit = ("no calibration run had a single dead unit there" if edge == 0
                     else f"no calibration run had more than {100 * edge:.1f}\\%")
            parts.append(f"{where}, where {100 * value:.1f}\\% of units were dead and {limit}")
        elif kind == "below":
            parts.append(f"{where}, where only {100 * value:.1f}\\% of units were dead, fewer than in any calibration "
                         f"run (lowest {100 * edge:.1f}\\%)")
        else:
            parts.append(f"{where} on the {kind.replace('_', ' ')}")
    text = f"On fresh healthy runs, LHD raised {h['k']} false alarms in {h['n']} runs: " + "; ".join(parts) + "."
    if any(kind == "above" and edge == 0 for (_, _, (kind, _, edge)) in groups):
        text += (" When no calibration run has a dead unit in a layer, the top of its range is the floor of 0.05, so "
                 "one dead filter out of 16 is enough to cross it.")
    if any(kind == "below" for (_, _, (kind, _, _)) in groups):
        text += (" An alarm for too few dead units does not point to any fault, and checking the dead unit fraction "
                 "against only the top of its range would remove it.")
    return text


def results(d: Data) -> str:
    sections = [detection_section(d), detector_section(d), dose_section(d), harm_section(d),
                diagnosis_section(d), ablation_section(d), settings_section(d), cost_section(d)]
    if d.more_complete():
        sections.append(extra_section(d))
    return "\\section{Results}\n\n" + "\n".join(sections)


def detection_section(d: Data) -> str:
    s = d.summary["archs"]
    p, c = d.pooled["proposal"], d.pooled["proposal_clustered"]
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{NAMES[a]}}}" for a in d.archs)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(d.archs)))
    sub = " & ".join("LHD & Accuracy" for _ in d.archs)

    def median(v):
        return "never" if v is None else f"{v:.0f}"

    caught = " & ".join(f"\\textbf{{{rate(s[a]['proposal']['detected'])}}} & {rate(s[a]['proposal']['acc_detected'])}"
                        for a in d.archs)
    epoch = " & ".join(f"\\textbf{{{median(s[a]['proposal']['lhd_median_epoch'])}}} & "
                       f"{median(s[a]['proposal']['acc_median_epoch'])}" for a in d.archs)
    flagged = " & ".join(f"\\textbf{{{kofn(s[a]['healthy']['lhd_false_alarm'])}}} & "
                         f"{kofn(s[a]['healthy']['acc_false_alarm'])}" for a in d.archs)
    return f"""\\subsection{{Detection and timing}}

Across the three networks, LHD caught {kofn(p['detected'])} broken runs and the accuracy curve caught {kofn(p['acc_detected'])} (Table~\\ref{{tab:headline}}). Of the runs LHD caught, the accuracy curve never noticed {rate(d.pooled['acc_never_on_detected'])}, and {rate(p['silent_share'])} of all broken runs finished with a final accuracy inside the healthy range. The median alarm came at epoch {p['lhd_median_epoch']:.0f}, which is the earliest possible alarm for a fault planted at epoch 1 when three epochs out of range are required, and Figure~\\ref{{fig:detection_over_time}} shows the share of broken runs each method had flagged by each epoch. Averaged within each seed, LHD was earlier than the accuracy curve for {c['by_seed']['lhd_earlier']} of {c['by_seed']['groups']} seed groups ({pval(c['by_seed'].get('wilcoxon_p'))}), and averaged within each fault condition it was earlier for {c['by_condition']['lhd_earlier']} of {c['by_condition']['groups']} ({pval(c['by_condition'].get('wilcoxon_p'))}). The mean gap was {p['mean_gap']:.1f} epochs, with a 95\\% interval of {c['mean_gap_ci'][0]:.1f} to {c['mean_gap_ci'][1]:.1f} from the seed bootstrap.

{false_alarm_text(d)}

\\begin{{table}}[t]
\\centering
\\small
\\caption{{LHD (bold) next to the accuracy curve on the planted frozen layers and dead units. The median epoch is over the runs each method caught.}}
\\label{{tab:headline}}
\\begin{{tabular}}{{@{{}}l{'rr' * len(d.archs)}@{{}}}}
\\toprule
 & {header} \\\\
{rules}
 & {sub} \\\\
\\midrule
Broken runs caught & {caught} \\\\
Median epoch of the alarm & {epoch} \\\\
Healthy runs flagged & {flagged} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{FIG['detection_over_time']}
"""


def detector_section(d: Data) -> str:
    zero = d.detector("Zero update check", ("frozen",))
    lhd = d.detector("LHD", DEAD)
    lhd_fa = d.detector_alarms("LHD")
    q25, q25_fa = d.detector("Dead units check, 25%", DEAD), d.detector_alarms("Dead units check, 25%")
    any_dead, any_fa = d.detector(ANY_DEAD, DEAD), d.detector_alarms(ANY_DEAD)
    sage, dd, wb = (d.detector(n, DEAD) for n in (SAGEMAKER, DEEPDIAGNOSIS, WANDB))
    cnn_lo, cnn_hi = d.dead_range("cnn", "fc1")
    s10, s25 = d.detector(SAGEMAKER, ("dead10",)), d.detector(SAGEMAKER, ("dead25",))
    natural = ""
    if s10["k"] >= s25["k"]:
        natural = (f" Some of these catches have nothing to do with the planted fault, because the same rules also fire on "
                   f"units that died naturally. The 50\\% rule caught {kofn(s10)} runs with 10\\% dead units but only "
                   f"{kofn(s25)} with 25\\%.")
    tr_hi = max(d.dead_range("transformer", layer)[1] for layer in ("b0.ff1", "b1.ff1"))
    transformer_dead = ("no calibration run of the transformer had a single dead unit in its ReLU layers" if tr_hi == 0
                        else f"no calibration run of the transformer had more than {100 * tr_hi:.0f}\\% dead units in any layer")
    return f"""\\subsection{{Comparison with other detectors}}

For frozen layers, LHD adds nothing over the zero update check, which caught {kofn(zero)} at epoch 1 with no false alarms, two epochs before LHD. A frozen layer's weight change is exactly 0, so a check for an exact 0 needs neither a healthy range nor patience.

Dead units are different, because healthy networks lose units on their own and the amount varies widely. In healthy calibration runs of the CNN's dense layer, between {100 * cnn_lo:.0f}\\% and {100 * cnn_hi:.0f}\\% of units were dead, while {transformer_dead}. A single fixed threshold therefore misses faults in one network or flags healthy runs in another (Figure~\\ref{{fig:detectors_tradeoff}}). LHD caught {pct(lhd['k'], lhd['n'])} of dead unit faults with {lhd_fa['k']} false alarms in {lhd_fa['n']} healthy runs. A 25\\% rule came closest at {pct(q25['k'], q25['n'])}, but it flagged {q25_fa['k']} healthy runs, and a rule that fires on a single dead unit caught {pct(any_dead['k'], any_dead['n'])} while flagging {any_fa['k']}. The defaults of SageMaker Debugger (50\\%), DeepDiagnosis (70\\%), and wandb-callbacks (80\\%) caught {pct(sage['k'], sage['n'])}, {pct(dd['k'], dd['n'])}, and {pct(wb['k'], wb['n'])}.{natural}

{FIG['detectors_tradeoff']}
"""


def dose_section(d: Data) -> str:
    misses = [r for r in d.per_layer if r["family"] in DEAD and int(r["detected"]) < int(r["runs"])]
    miss_text = ", ".join(f"{PROSE[r['arch']]} {r['layer']} at {r['family'][4:]}\\% ({r['detected']} of {r['runs']} caught)"
                          for r in misses)
    lhd = [d.detector("LHD", (f,)) for f in DEAD]
    acc = [d.detector("Accuracy curve, 3 epochs below healthy", (f,)) for f in DEAD]
    lhd_text = listing([pct(x["k"], x["n"]) for x in lhd])
    acc_text = listing([pct(x["k"], x["n"]) for x in acc])
    return f"""\\subsection{{How many units must die}}

With 10\\%, 25\\%, and 50\\% of a layer's units killed, LHD caught {lhd_text} of runs, against {acc_text} for the accuracy curve. Its misses came from a few layers: {miss_text}. In each of them, healthy runs already lose a share of units on their own, so the planted fault stayed inside the healthy range. The CNN's dense layer is the clearest case, since its healthy range already covers most of the planted values, and it shows up as the pale row of Figure~\\ref{{fig:per_layer}}.

{FIG['per_layer']}
"""


def mean_drop(d: Data, arch: str, family: str) -> float | None:
    drops = [float(r["acc_drop_vs_twin"]) for r in d.runs
             if r["arch"] == arch and r["family"] == family and r["acc_drop_vs_twin"] not in ("", None)]
    return 100 * sum(drops) / len(drops) if drops else None


def harm_section(d: Data) -> str:
    low = next((r for r in d.harm if r["harm"] == "under 0.5 pts"), None)
    high = next((r for r in d.harm if r["harm"] == "2 pts or more"), None)
    total = sum(int(r["broken_runs"]) for r in d.harm)
    text = ""
    if low and high:
        text = (f"Most planted faults did little harm. In {low['broken_runs']} of {total} broken runs, the final accuracy "
                f"was less than half a point below the healthy twin's, and the accuracy curve missing those runs is partly "
                f"the curve being right. Among the {high['broken_runs']} runs that lost 2 points or more, LHD caught "
                f"{high['lhd_caught']} and the accuracy curve caught {high['accuracy_caught']}.")
    cnn_frozen, cnn_dead = mean_drop(d, "cnn", "frozen"), mean_drop(d, "cnn", "dead50")
    others = [mean_drop(d, a, f) for a in d.archs if a != "cnn" for f in ("frozen",) + DEAD]
    others = [v for v in others if v is not None]
    cost = ""
    if cnn_frozen is not None and others:
        cost = (f" Only one fault cost much on average. A frozen CNN layer lowered final accuracy by {cnn_frozen:.1f} points "
                f"against the healthy twin, about as much as the frozen convolution in the GTSRB model that motivated this "
                f"work, while 50\\% dead units in the CNN cost {cnn_dead:.1f} points and no fault in the other two networks "
                f"cost more than {max(others):.1f} points on average.")
    return f"""\\subsection{{Faults that cost accuracy}}

{text}{cost}
"""


def diagnosis_section(d: Data) -> str:
    p = d.pooled["proposal"]
    wrong = d.wrong
    causes = [r for r in d.runs if r["role"] == "fault" and r["family"] in ("frozen",) + DEAD
              and r["detected"] == "True" and r["layer_ok"] == "True" and r["cause_ok"] == "False"]

    def fault(fam, layer):
        return f"a frozen {layer}" if fam == "frozen" else f"{fam[4:]}\\% dead units in {layer}"

    wrong_text = ""
    if wrong:
        groups = defaultdict(int)
        for r in wrong:
            groups[(r["arch"], r["family"], r["fault_layer"], r["lhd_layer"])] += 1
        listing = "; ".join(f"{fault(fam, layer)} in the {PROSE[a]}, reported as {named} ({n})"
                            for (a, fam, layer, named), n in sorted(groups.items()))
        dead_hits = sum(r["lhd_signals"] == "dead_unit_fraction" for r in wrong)
        frozen_now = [r for r in wrong if r["family"] == "frozen" and r["lhd_epoch"] == str(SETTINGS["patience"])]
        wrong_text = (f" It named the wrong layer in {len(wrong)} caught runs ({listing}). In each of these, the reported "
                      f"layer's own signal was out of range, with too many dead units in {dead_hits} and another signal in "
                      f"{len(wrong) - dead_hits}, so LHD found a real effect of the fault in a different layer.")
        if frozen_now:
            n_frozen = len(frozen_now)
            cases = f"{WORDS.get(n_frozen, n_frozen)} frozen case" + ("s" if n_frozen > 1 else "")
            wrong_text += (f" In the {cases}, the planted layer's weight change crossed its range in the same epoch, but LHD "
                           f"names only the first layer in data flow order, so reporting every layer out of range would "
                           f"have included the planted one.")
    cause_text = ""
    if causes:
        groups = defaultdict(int)
        for r in causes:
            groups[(r["arch"], r["family"], r["fault_layer"], r["lhd_cause"])] += 1
        listing = "; ".join(f"{fault(fam, layer)} in the {PROSE[a]}, called a {cause.replace('_', ' ')} ({n})"
                            for (a, fam, layer, cause), n in sorted(groups.items()))
        cause_text = f" It found the right layer but named the wrong cause in {len(causes)} runs ({listing})."
        if any("qkv" in layer and cause == "vanishing_gradient" for _, _, layer, cause in groups):
            cause_text += (" A frozen attention projection also lowers the gradient that reaches it, so its pattern "
                           "matches a vanishing gradient.")
    return f"""\\subsection{{Diagnosis}}

When LHD caught a broken run, it named the planted layer {rate(p['layer_ok'], 1)} of the time and the planted cause {rate(p['cause_ok'], 1)} of the time (Figure~\\ref{{fig:per_layer}}).{wrong_text}{cause_text} The high rate of naming the right cause is largely built into the design, since the planted faults produce exactly the signal patterns the rules look for.
"""


def pooled_ablation(d: Data) -> list[tuple[str, float, float]]:
    """Each signal variant's caught and right-diagnosis rates, pooled over networks by run count."""
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in d.ablation:
        n = int(r["faults"])
        t = totals[r["variant"]]
        t[0] += float(r["detection"]) * n
        t[1] += float(r["both_ok"]) * n
        t[2] += n
    order = list(dict.fromkeys(r["variant"] for r in d.ablation))
    return [(v, totals[v][0] / totals[v][2], totals[v][1] / totals[v][2]) for v in order]


def ablation_section(d: Data) -> str:
    rows = pooled_ablation(d)
    base = next(r for r in rows if r[0] == "all four")

    def delta(value, ref):
        diff = 100 * (value - ref)
        if abs(diff) < 0.5:
            return "same"
        sign = "+" if diff > 0 else "$-$"
        return f"{sign}{abs(diff):.0f}"

    lines = [f"All four & \\textbf{{{100 * base[1]:.0f}\\%}} & & \\textbf{{{100 * base[2]:.0f}\\%}} & \\\\", "\\midrule"]
    first_only = True
    for name, caught, both in rows:
        if name == "all four":
            continue
        if name.startswith("only") and first_only:
            lines.append("\\midrule")
            first_only = False
        label = name.replace("without ", "Without ").replace("only ", "Only ").replace("_", " ")
        lines.append(f"{label} & {100 * caught:.0f}\\% & {delta(caught, base[1])} & {100 * both:.0f}\\% & "
                     f"{delta(both, base[2])} \\\\")
    without_dead = next(r for r in rows if r[0] == "without dead_unit_fraction" or r[0] == "without dead unit fraction")
    without_change = next(r for r in rows if r[0] in ("without weight_change", "without weight change"))
    return f"""\\subsection{{What each signal adds}}

Table~\\ref{{tab:ablation}} shows the same runs replayed with each signal removed, or with one signal used alone. Without the dead unit fraction, LHD caught {100 * without_dead[1]:.0f}\\% of broken runs instead of {100 * base[1]:.0f}\\%, so that signal carries most of the detection. Without the weight change, it named the right layer and cause in {100 * without_change[2]:.0f}\\% instead of {100 * base[2]:.0f}\\%, since the weight change is what separates a frozen layer from other faults. Removing the weight norm or the gradient norm changed little, because none of the planted faults was designed to move them.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{What happens when signals are removed or used alone, pooled over the three networks. Changes are in percentage points against all four signals.}}
\\label{{tab:ablation}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Signals used & \\multicolumn{{2}}{{c}}{{Broken runs caught}} & \\multicolumn{{2}}{{c}}{{Right layer and cause}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(l){{4-5}}
 & Rate & Change & Rate & Change \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def settings_section(d: Data) -> str:
    def pooled_sweep(margin, patience=3, floor=0.05):
        rows = [r for r in d.sweep if float(r["margin"]) == margin and int(r["patience"]) == patience
                and float(r["floor"]) == floor]
        fresh = sum(int(r["fresh"]) for r in rows)
        faults = sum(int(r["faults"]) for r in rows)
        alarms = sum(float(r["false_alarm"]) * int(r["fresh"]) for r in rows) / fresh if fresh else 0
        caught = sum(float(r["detection"]) * int(r["faults"]) for r in rows) / faults if faults else 0
        return alarms, caught

    edge_fa, edge_caught = pooled_sweep(0.0)
    used_fa, used_caught = pooled_sweep(SETTINGS["margin"])
    cal = defaultdict(list)
    for r in d.calibration:
        cal[(r["arch"], int(r["k"]))].append(float(r["false_alarm"]))

    def worst(k):
        values = {a: sum(cal[(a, k)]) / len(cal[(a, k)]) for a in d.archs if cal[(a, k)]}
        a = max(values, key=values.get)
        return a, values[a]

    small_arch, small = worst(2)
    full_arch, full = worst(20)
    return f"""\\subsection{{Settings and calibration}}

Thresholds placed right at the edge of the healthy range flagged {100 * edge_fa:.0f}\\% of healthy runs. Widening them by the margin of {SETTINGS['margin']} used here cut that to {100 * used_fa:.0f}\\%, while the share of broken runs caught fell only from {100 * edge_caught:.0f}\\% to {100 * used_caught:.0f}\\%. The range also needs enough calibration runs. Built from 2 runs, it flagged up to {100 * small:.0f}\\% of healthy runs (in the {PROSE[small_arch]}), and built from all 20, at most {100 * full:.0f}\\%.
"""


def cost_section(d: Data) -> str:
    s = d.summary["archs"]
    overhead = listing([f"{100 * s[a]['healthy']['overhead']:.1f}\\% for the {PROSE[a]}" for a in d.archs])
    p = d.pooled["proposal"]
    hours = float(d.compute["all"]["training_hours"])
    return f"""\\subsection{{Cost}}

Monitoring added {overhead} to the training time, measured on fresh healthy runs on one CPU thread. Most of that time goes to the gradient norm, read after every batch, and to the probe batch, run once per epoch, so the overhead is largest for the network with the cheapest epochs. Stopping each broken run at LHD's alarm would have saved {100 * p['mean_compute_saved']:.0f}\\% of its training on average, and all runs together took {hours:.0f} hours of CPU time on one laptop.
"""


def extra_section(d: Data) -> str:
    names = {"latefrozen": "Frozen layer at epoch 6", "latedead50": "50\\% dead units at epoch 6",
             "detach": "Gradient cut", "lr10": "Learning rate $\\times 10$ at epoch 6",
             "lr100": "Learning rate $\\times 100$ at epoch 6"}
    rows = []
    for fam, label in names.items():
        k = n = acc = 0
        for a in d.archs:
            st = d.summary["archs"][a]["families"].get(fam)
            if st:
                k += st["detected"]["k"]
                n += st["detected"]["n"]
                acc += st["acc_detected"]["k"]
        if n:
            rows.append(f"{label} & {n} & \\textbf{{{pct(k, n)}}} & {pct(acc, n)} \\\\")
    return f"""\\subsection{{Faults beyond frozen layers and dead units}}
\\label{{sec:extra}}

A third set of runs tested faults the rules were not built around: a frozen layer or 50\\% dead units planted at epoch 6, a cut that stops gradients from reaching earlier layers, and a learning rate raised 10 or 100 times at epoch 6 (Table~\\ref{{tab:extra}}).

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Faults beyond the main study, pooled over the three networks.}}
\\label{{tab:extra}}
\\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\\toprule
Fault & Runs & LHD & Accuracy curve \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def discussion(d: Data) -> str:
    trans = [float(r["final_acc"]) for r in d.runs if r["arch"] == "transformer" and r["role"] == "fresh"]
    p, h = d.pooled["proposal"], d.pooled["healthy"]
    return f"""\\section{{Discussion and Limitations}}

The results support the hypothesis for planted faults. Per layer signals exposed a broken layer within the first few epochs, usually before the accuracy curve showed anything, and LHD rarely flagged healthy runs. A frozen layer, however, is caught just as well and sooner by a check for weights that never change, so for that fault the only gain is that the same rules also cover it. For dead units, a range measured on healthy runs of the same network did better than every fixed threshold tested, including the defaults of three existing tools.

The faults were planted to show up in the signals LHD measures, so the detection and diagnosis rates describe how well LHD separates these faults from normal variation, not how often it would catch bugs in real projects. Many planted faults also cost less than one point of accuracy, which means stopping those runs saves compute on runs that would have produced a nearly normal model. LHD misses dead units in layers where healthy runs already lose many units, because the healthy range treats that loss as normal. Building the range takes about 10 to 20 healthy runs of the same network, which costs more than LHD can save for a model trained only once, and it assumes those runs are themselves healthy. Finally, the networks are small and were trained for 15 to 20 epochs on a laptop CPU, the transformer reaches only about {100 * sum(trans) / len(trans):.0f}\\% next character accuracy, and large networks, long runs, other optimizers, and GPU training were not tested.

\\section{{Conclusion}}

This work presented LHD, a monitor that compares four signals of every layer with ranges measured on healthy runs of the same network and stops a run when one stays out of range. On {d.total_runs()} runs across three networks, it caught {rate(p['detected'])} of planted frozen layers and dead units, usually within three epochs, named the right layer in {rate(p['layer_ok'])} of those runs, and flagged {kofn(h['lhd_false_alarm'])} healthy runs. Its advantage over existing tools is clearest for dead units, where a healthy range for each layer avoided both the misses and the false alarms of fixed thresholds. The next step is to test LHD on real bugs collected from training code.

\\section*{{Code and data}}

The code, every saved run, and the tables behind each number are at \\url{{https://github.com/sohan-bhat/lhd}}.

"""


def main() -> None:
    from experiments import paper_figures

    paper_figures.main()
    d = Data()
    missing = [n for n in FIGURES if not (PAPER / "figures" / f"{n}.pdf").exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    related = (PAPER / "related_work.tex").read_text()
    references = (PAPER / "references.tex").read_text()
    body = [preamble(), abstract(d), introduction(d), related, method(), setup(d),
            results(d), discussion(d), references, "\\end{document}\n"]
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
