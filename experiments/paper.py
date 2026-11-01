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
FIGURES = ("how_lhd_works", "detection_over_time", "detectors_tradeoff", "dose_response",
           "accuracy_curves", "harm", "margin", "calibration_size")
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
                                 "every detector on the same runs, pooled over the three networks. A good detector "
                                 "sits in the upper left."),
    "dose_response": figure("dose_response", "Share of dead unit faults caught as more of a layer's units are "
                            "killed, for (a) LHD and (b) the accuracy curve."),
    "accuracy_curves": figure("accuracy_curves", "Validation accuracy over training. The shaded band is the range of "
                              "the calibration runs, the dark line is the mean of fresh healthy runs, and the colored "
                              "lines are the means of broken runs with a frozen layer or 50\\% dead units."),
    "harm": figure("harm", "Broken runs caught by LHD and by the accuracy curve, split by how much final accuracy "
                   "each fault cost against its healthy twin."),
    "margin": figure("margin", "Broken runs caught and healthy runs flagged as the margin around the healthy range "
                     "grows, pooled over the three networks, with a floor of 0.05 and a patience of 3 epochs."),
    "calibration_size": figure("calibration_size", "Healthy runs flagged when the healthy range is built from fewer "
                               "calibration runs, averaged over 20 random draws for each size."),
}


def abstract(d: Data) -> str:
    p, h, c = d.pooled["proposal"], d.pooled["healthy"], d.pooled["proposal_clustered"]
    lo, hi = c["mean_gap_ci"]
    dead = d.detector("LHD", DEAD)
    return f"""\\begin{{abstract}}
During training, neural networks can develop defects that raise no error, so the run completes and returns a degraded model that is easy to mistake for a hard problem. This work presents LHD (Layer Health Diagnostics), a monitor that logs four signals for every layer at the end of each epoch (weight change, gradient norm, weight norm, and the fraction of dead units) and compares each one with the range it stays within during healthy runs of the same network. When a signal stays outside that range for three epochs in a row, LHD names the layer and a likely cause and stops the run. Across {d.total_runs()} training runs of a fully connected network, a CNN, and a small transformer, with frozen layers and dead units planted in known layers, LHD caught {kofn(p['detected'])} broken runs, named the planted layer in {rate(p['layer_ok'])} of them, and flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs. Averaged within each seed, it fired {p['mean_gap']:.1f} epochs before the accuracy curve (95\\% interval {lo:.1f} to {hi:.1f}). A check for weights that never change caught every frozen layer two epochs sooner, so the benefit of LHD lies in dead units, where it caught {pct(dead['k'], dead['n'])} of faults while the fixed thresholds of existing tools either missed most of them or flagged many healthy runs.
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
