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


def fault_name(fam: str, layer: str) -> str:
    return f"a frozen {layer}" if fam == "frozen" else f"{fam[4:]}\\% dead units in {layer}"


def count_word(n: int, noun: str) -> str:
    word = WORDS.get(n, str(n))
    return f"{word} {noun}" if n == 1 else f"{word} {noun}s"


def interval(stat: dict) -> str:
    """The 95% Wilson interval of a rate, as "a to b%"."""
    return f"{100 * stat['low']:.0f} to {100 * stat['high']:.0f}\\%"


def interval1(stat: dict) -> str:
    return f"{100 * stat['low']:.1f} to {100 * stat['high']:.1f}\\%"


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

    def detector_right(self, name: str, families) -> dict:
        """Broken runs a detector caught at the planted layer."""
        rows = [r for r in self.detectors if r["detector"] == name and r["family"] in families]
        k = sum(int(r["caught_right_layer"]) for r in rows if r.get("caught_right_layer") not in ("", None))
        return {"k": k, "n": sum(int(r["broken_runs"]) for r in rows)}

    def proposal_rows(self) -> list[dict]:
        return [r for r in self.runs if r["role"] == "fault" and r["family"] in ("frozen",) + DEAD]

    def both_gap(self) -> dict:
        """The gap on broken runs that both LHD and the accuracy curve caught."""
        both = [r for r in self.proposal_rows() if r["detected"] == "True" and r["acc_detected"] == "True"]
        gaps = sorted(int(r["acc_epoch"]) - int(r["lhd_epoch"]) for r in both)
        if not gaps:
            return {"n": 0}
        mid = len(gaps) // 2
        median = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2
        return {"n": len(gaps), "mean": sum(gaps) / len(gaps), "median": median,
                "acc_earlier": sum(g < 0 for g in gaps), "ties": sum(g == 0 for g in gaps)}

    def conditions(self) -> dict:
        """How each fault condition (fault and layer) came out between LHD and the accuracy curve."""
        groups = defaultdict(list)
        for r in self.proposal_rows():
            groups[(r["arch"], r["family"], r["fault_layer"])].append(r)
        out = {"lhd": 0, "neither": 0, "same": 0, "acc": []}
        for key, rows in groups.items():
            mean = sum(float(r["gap"]) for r in rows) / len(rows)
            if mean > 0:
                out["lhd"] += 1
            elif mean < 0:
                out["acc"].append(key)
            elif not any(r["detected"] == "True" or r["acc_detected"] == "True" for r in rows):
                out["neither"] += 1
            else:
                out["same"] += 1
        out["groups"] = len(groups)
        return out

    def lhd_rate(self, arch: str, family: str, layer: str) -> float:
        rows = [r for r in self.proposal_rows() if r["arch"] == arch and r["family"] == family and r["fault_layer"] == layer]
        return sum(r["detected"] == "True" for r in rows) / len(rows) if rows else 0.0

    def composition(self) -> dict:
        calibration = sum(int(r["runs"]) for r in self.counts if r["role"] == "calibration")
        fresh = sum(int(r["runs"]) for r in self.counts if r["role"] == "fresh")
        main = sum(int(r["runs"]) for r in self.counts if r["role"] == "fault" and r["phase"] != "more")
        other = sum(int(r["runs"]) for r in self.counts if r["role"] == "fault" and r["phase"] == "more")
        return {"calibration": calibration, "fresh": fresh, "main": main, "other": other,
                "total": calibration + fresh + main + other}

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
\author{Sohan Bhat\\\\ \\small Heritage High School, Frisco, Texas\\\\ \\small \\texttt{sohanrambhat@gmail.com}}
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
    "detectors_tradeoff": figure("detectors_tradeoff", "Dead unit faults caught at the planted layer against healthy "
                                 "runs flagged, for LHD and the fixed rules held to three epochs, pooled over the three "
                                 "networks. Detectors toward the "
                                 "upper left catch more faults while flagging fewer healthy runs."),
    "per_layer": figure("per_layer", "Share of broken runs in which LHD caught the fault and named both the planted "
                        "layer and the right cause, for every layer and fault. Each cell is 10 broken runs. Blank cells "
                        "are faults that were not planted, since dead units only go into layers followed by a ReLU."),
}


def abstract(d: Data) -> str:
    p, h = d.pooled["proposal"], d.pooled["healthy"]
    harm = next((r for r in d.harm if r["harm"] == "2 pts or more"), None)
    both = d.both_gap()
    comp = d.composition()
    dead = d.detector("LHD", DEAD)
    harm_text = (f" On the {harm['broken_runs']} broken runs whose fault cost two or more points of final accuracy, LHD "
                 f"caught {harm['lhd_caught']} and the accuracy curve caught {harm['accuracy_caught']}." if harm else "")
    return f"""\\begin{{abstract}}
During training, neural networks can develop defects that raise no error, so the run completes and returns a degraded model that is easy to mistake for a hard problem. This work presents LHD (Layer Health Diagnostics), a monitor that logs four signals for every layer at the end of each epoch (weight change, gradient norm, weight norm, and the fraction of dead units) and compares each one with the range it stays within during healthy runs of the same network. When a signal stays outside that range for three epochs in a row, LHD names the layer and a likely cause and stops the run. LHD was tested on {comp['total']} training runs of a fully connected network, a CNN, and a small transformer, in which frozen layers and dead units were planted in known layers.{harm_text} Across all {p['runs']} broken runs with these faults, LHD caught {rate(p['detected'])} (95\\% interval {interval(p['detected'])}) and named the planted layer in {rate(p['layer_ok'])} of the runs it caught, while it flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs ({interval1(h['lhd_false_alarm'])}). On the {both['n']} runs that both LHD and the accuracy curve caught, LHD fired {both['mean']:.1f} epochs earlier on average. A check for weights that never change caught every frozen layer two epochs sooner, so LHD helps most with dead units, where it caught {pct(dead['k'], dead['n'])} of faults while the fixed thresholds used by existing tools either missed most of them or flagged many healthy runs.
\\end{{abstract}}
"""


def introduction(d: Data) -> str:
    p, h, c = d.pooled["proposal"], d.pooled["healthy"], d.pooled["proposal_clustered"]
    harm = next((r for r in d.harm if r["harm"] == "2 pts or more"), None)
    low = next((r for r in d.harm if r["harm"] == "under 0.5 pts"), None)
    both = d.both_gap()
    dead = d.detector("LHD", DEAD)
    sage = d.detector_right(SAGEMAKER + ", 3 epochs", DEAD)
    sage_fa = d.detector_alarms(SAGEMAKER + ", 3 epochs")
    seeds = c["by_seed"]
    seed_text = (f"all {seeds['groups']} seed groups favored LHD" if seeds["lhd_earlier"] == seeds["groups"]
                 else f"{seeds['lhd_earlier']} of {seeds['groups']} seed groups favored LHD")
    harm_text = ""
    if harm and low:
        harm_text = (f"The clearest result is on the faults that did real damage. Of the {harm['broken_runs']} broken runs "
                     f"whose fault cost two or more points of final accuracy, LHD caught {harm['lhd_caught']} and the "
                     f"accuracy curve caught {harm['accuracy_caught']}. Those runs are a minority, since {low['broken_runs']} "
                     f"of the {p['runs']} planted faults cost less than half a point. ")
    return f"""\\section{{Introduction}}

During training, neural networks can develop defects, yet nothing fails visibly since the run completes normally and simply produces a degraded result; thus, these are called ``silent failures.'' Because of this, identifying whether the degraded result was caused by a difficult problem or a silent failure becomes difficult; therefore, it is common to misread a failure as normal. I initially encountered this as I was training my small-scale GTSRB (German Traffic Sign Recognition Benchmark) model \\citep{{stallkamp2012}} from scratch, where my model plateaued at 76\\%, wasting weeks tuning it, until I found one convolutional layer had been accidentally excluded from the optimizer and was never properly trained. Fixing this minute detail raised the accuracy to 87\\%. This is not an isolated case: errors in deep learning implementations rarely cause an exception and instead reveal themselves only through degraded predictive performance \\citep{{chwilczynski2024}}.

Since training runs consume significant compute, every silent failure that goes undetected wastes it. The evidence existed in the form of raw numbers in the weights and biases, but perceiving that is challenging since nothing in standard practice surfaces them. To address this, this work presents LHD (Layer Health Diagnostics), a monitor that reads these numbers, catches the failure within a few epochs, identifies the broken layer and its likely cause, and halts the run before compute is wasted.

{harm_text}Across all {p['runs']} broken runs with a frozen layer or dead units, LHD caught {kofn(p['detected'])}, usually at epoch {p['lhd_median_epoch']:.0f}, while the accuracy curve caught {kofn(p['acc_detected'])}. On the {both['n']} runs both caught, LHD was earlier by {both['mean']:.1f} epochs on average, and averaged within each seed, {seed_text}. LHD also flagged {kofn(h['lhd_false_alarm'])} fresh healthy runs, against {kofn(h['acc_false_alarm'])} for the accuracy curve. However, a simple check for weights that do not change caught every frozen layer two epochs before LHD, so the advantage of LHD comes from dead units. There it caught {pct(dead['k'], dead['n'])} with {h['lhd_false_alarm']['k']} false alarms, while the default rule of SageMaker Debugger, held to the same three epochs and counted only when it named the planted layer, caught {pct(sage['k'], sage['n'])} and flagged {sage_fa['k']} of {sage_fa['n']} healthy runs. Only frozen layers and dead units were planted in the main study, and they were planted to show up in the signals LHD measures, so these results describe planted faults rather than bugs found in real projects.

"""


def method(d: Data) -> str:
    m, f, pat = SETTINGS["margin"], SETTINGS["floor"], SETTINGS["patience"]
    untested = (" The rules for divergence, vanishing gradients, and runaway weights were exercised only by the extra "
                "faults in Section~\\ref{sec:extra}." if d.more_complete()
                else " The rules for divergence, vanishing gradients, and runaway weights were not tested.")
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

LHD raises an alarm once one signal of one layer stays out of range for $p = {pat}$ epochs in a row. If several layers reach that point in the same epoch, it reports the first one in the order data flows through the network. The settings $m$, $f$, and $p$ come from the tests of the LHD library, written on 12 October 2026, and were committed to the experiment code on 31 October 2026 (commit \\texttt{{f55a811}} in the code repository) before any of the study's training runs started. An earlier draft of this work defined the threshold as a multiple of the largest healthy value, and the rule above replaced it, adding the floor and a margin below the lowest healthy value, before any run was trained. LHD then names a likely cause from the pattern across the alarmed layer's four signals (Table~\\ref{{tab:rules}}), stops training, and reports the layer, the signals out of range, the cause, and a suggested check. For a frozen layer, for example, it suggests checking that the layer's parameters are passed to the optimizer. The main study planted only frozen layers and dead units, so it tested only the frozen layer and dead unit rules.{untested} Figure~\\ref{{fig:how_lhd_works}} shows this on one CNN run with a frozen layer.

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
        main = sum(int(r["runs"]) for r in d.counts if r["arch"] == a and r["role"] == "fault" and r["phase"] != "more")
        other = sum(int(r["runs"]) for r in d.counts if r["arch"] == a and r["role"] == "fault" and r["phase"] == "more")
        rows.append(f"{NAMES[a]} & {DATASETS[a]} & {int(n['parameters']):,} & {n['epochs']} & "
                    f"{d.role_count(a, 'calibration')} & {d.role_count(a, 'fresh')} & {main} & {other} \\\\")
    extra = (" A third set of runs added faults beyond these two, reported in Section~\\ref{sec:extra}."
             if d.more_complete() else "")
    return f"""\\section{{Experimental Setup}}

\\subsection{{Networks and data}}

LHD was tested on three networks with different structures (Table~\\ref{{tab:networks}}). The fully connected network has hidden layers of 256, 128, and 64 units. The CNN has three convolutions with 8, 16, and 32 filters, each followed by a ReLU and max pooling, then a dense layer of 64 units and the output layer. The transformer predicts the next character with a width of 64, 4 attention heads, 2 blocks, and a feed-forward layer of 256 units that uses a ReLU, so dead units can be counted the same way in all three. After every epoch, validation accuracy is measured on the dataset's test split. The CNN is scored on a fixed random subset of 4,000 test images and the transformer on 512 fixed windows of 64 characters from the last 10\\% of the text. LHD watches every layer with weights: fc1 to fc4 in the fully connected network; conv1, conv2, conv3, fc1, and fc2 in the CNN; and 11 layers in the transformer, namely the token and position embeddings (tok, pos), the attention input and output projections of each block (b0.qkv, b0.proj, b1.qkv, b1.proj), the two feed-forward layers of each block (b0.ff1, b0.ff2, b1.ff1, b1.ff2), and the output layer (head). The layers followed by a ReLU, where units can die, are fc1 to fc3 in the fully connected network, conv1 to conv3 and fc1 in the CNN, and b0.ff1 and b1.ff1 in the transformer.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{The three networks and the runs trained for each. Calibration runs set the healthy range, fresh healthy runs measure false alarms, and the results in Section~5 use the runs with a frozen layer or dead units. Runs with other faults appear only in Section~\\ref{{sec:extra}}.}}
\\label{{tab:networks}}
\\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\\toprule
 & & & & \\multicolumn{{4}}{{c}}{{Runs}} \\\\
\\cmidrule(l){{5-8}}
 & & & & & Fresh & Frozen or & Other \\\\
Network & Dataset & Parameters & Epochs & Calibration & healthy & dead units & faults \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Every run used Adam with a learning rate of 0.001 and a cosine schedule on one CPU thread. The seed fixes both the starting weights and the order of training batches, so a broken run and the healthy run with the same seed differ only by the fault. That healthy run is the broken run's twin, and the twin's final accuracy shows how much the fault cost.

\\subsection{{Planted faults}}

Each broken run has one fault planted in one layer at the start of training. A frozen layer has its weights and bias put back after every optimizer step, so gradients are computed but never applied, which is the same failure as a layer left out of the optimizer. Dead units are made by setting the incoming weights of 10\\%, 25\\%, or 50\\% of a layer's units to 0 and their bias to $-1$, so their ReLU output is 0 for every input. The units are drawn at random without replacement by a generator seeded with the run's seed plus 7919, and their number is the fraction times the layer's width, rounded, with at least one unit. The optimizer's running averages for those units are cleared. Frozen layers were planted in every watched layer and dead units in every layer followed by a ReLU.{extra}

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
    both = d.both_gap()
    cond = d.conditions()
    seeds = c["by_seed"]
    seed_text = (f"all {seeds['groups']} seed groups favored LHD, which is the most extreme result a one sided "
                 f"Wilcoxon test can give with {seeds['groups']} groups" if seeds["lhd_earlier"] == seeds["groups"]
                 else f"{seeds['lhd_earlier']} of {seeds['groups']} seed groups favored LHD ({pval(seeds.get('wilcoxon_p'))})")
    parts = []
    if cond["neither"]:
        parts.append(f"neither detector fired in {count_word(cond['neither'], 'condition')}")
    if cond["same"]:
        parts.append(f"both fired at the same epoch in {count_word(cond['same'], 'condition')}")
    if cond["acc"]:
        missed = all(fam in DEAD and d.lhd_rate(a, fam, layer) < 0.5 for a, fam, layer in cond["acc"])
        where = (", all in layers where LHD missed most of the planted dead units" if missed
                 else " (" + listing([f"{fault_name(fam, layer)} in the {PROSE[a]}" for a, fam, layer in cond["acc"]]) + ")")
        parts.append(f"the accuracy curve was earlier in {count_word(len(cond['acc']), 'condition')}{where}")
    cond_text = (". Of the other " + str(cond["groups"] - cond["lhd"]) + ", " + listing(parts)) if parts else ""
    return f"""\\subsection{{Detection and timing}}

Across the three networks, LHD caught {kofn(p['detected'])} broken runs ({rate(p['detected'])}, 95\\% interval {interval(p['detected'])}) and the accuracy curve caught {kofn(p['acc_detected'])} (Table~\\ref{{tab:headline}}). Of the runs LHD caught, the accuracy curve never noticed {rate(d.pooled['acc_never_on_detected'])}, and {rate(p['silent_share'])} of all broken runs finished with a final accuracy inside the healthy range. The median alarm came at epoch {p['lhd_median_epoch']:.0f}, which is the earliest possible alarm for a fault planted at epoch 1 when three epochs out of range are required, and Figure~\\ref{{fig:detection_over_time}} shows the share of broken runs each method had flagged by each epoch.

On the {both['n']} runs that both LHD and the accuracy curve caught, LHD fired {both['mean']:.1f} epochs earlier on average (median {both['median']:.0f}), the accuracy curve was earlier in {count_word(both['acc_earlier'], 'run')}, and {both['ties']} were ties. Averaged within each seed, {seed_text}. Averaged within each fault condition, {cond['lhd']} of {cond['groups']} conditions favored LHD{cond_text}. Counting a run the accuracy curve never caught as one epoch past its end, the mean gap over all broken runs was {p['mean_gap']:.1f} epochs (95\\% interval {c['mean_gap_ci'][0]:.1f} to {c['mean_gap_ci'][1]:.1f} from a bootstrap over seeds), but that figure leans on the runs the accuracy curve missed, so the gap on runs both caught is the fairer measure.

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


FIXED = (
    ("Zero update check", "Zero update check"),
    ("Dead units check, 25%", "25\\% dead rule"),
    (SAGEMAKER, "SageMaker Debugger (50\\%)"),
    (DEEPDIAGNOSIS, "DeepDiagnosis (70\\%)"),
    (WANDB, "wandb-callbacks (80\\%)"),
    (ANY_DEAD, "Any dead unit (wandb-callbacks verbose)"),
)


def detector_section(d: Data) -> str:
    p3 = ", 3 epochs"
    lhd_frozen, lhd_dead = d.detector_right("LHD", ("frozen",)), d.detector_right("LHD", DEAD)
    lhd_fa = d.detector_alarms("LHD")
    rows = [f"\\textbf{{LHD}} & \\textbf{{{pct(lhd_frozen['k'], lhd_frozen['n'])}}} & "
            f"\\textbf{{{pct(lhd_dead['k'], lhd_dead['n'])}}} & \\textbf{{{lhd_fa['k']} of {lhd_fa['n']}}} \\\\"]
    for name, label in FIXED:
        fr, de, fa = d.detector_right(name + p3, ("frozen",)), d.detector_right(name + p3, DEAD), d.detector_alarms(name + p3)
        rows.append(f"{label} & {pct(fr['k'], fr['n'])} & {pct(de['k'], de['n'])} & {fa['k']} of {fa['n']} \\\\")
    zero = d.detector_right("Zero update check" + p3, ("frozen",))
    zero1 = d.detector("Zero update check", ("frozen",))
    q25, q25_fa = d.detector_right("Dead units check, 25%" + p3, DEAD), d.detector_alarms("Dead units check, 25%" + p3)
    any_dead, any_fa = d.detector_right(ANY_DEAD + p3, DEAD), d.detector_alarms(ANY_DEAD + p3)
    sage, sage_fa = d.detector_right(SAGEMAKER + p3, DEAD), d.detector_alarms(SAGEMAKER + p3)
    s10, s10r = d.detector(SAGEMAKER + p3, ("dead10",)), d.detector_right(SAGEMAKER + p3, ("dead10",))
    s25, s25r = d.detector(SAGEMAKER + p3, ("dead25",)), d.detector_right(SAGEMAKER + p3, ("dead25",))
    p1_fa = {n: d.detector_alarms(n)["k"] for n, _ in FIXED}
    p3_fa = {n: d.detector_alarms(n + p3)["k"] for n, _ in FIXED}
    p1_dead = {n: d.detector(n, DEAD)["k"] for n, _ in FIXED}
    p3_dead = {n: d.detector(n + p3, DEAD)["k"] for n, _ in FIXED}
    fa_change = max(abs(p1_fa[n] - p3_fa[n]) for n in p1_fa)
    dead_change = max(p1_dead[n] - p3_dead[n] for n in p1_dead)
    p_change = (f", so false alarms did not change and dead unit catches fell by at most {dead_change}" if fa_change == 0
                else f", so false alarms changed by at most {fa_change} and dead unit catches by at most {dead_change}")
    cnn_lo, cnn_hi = d.dead_range("cnn", "fc1")
    tr_hi = max(d.dead_range("transformer", layer)[1] for layer in ("b0.ff1", "b1.ff1"))
    transformer_dead = ("no calibration run of the transformer had a single dead unit in its ReLU layers" if tr_hi == 0
                        else f"no calibration run of the transformer had more than {100 * tr_hi:.0f}\\% dead units in any layer")
    return f"""\\subsection{{Comparison with other detectors}}

The fixed rules were scored the same way as LHD. Each one had to hold for three epochs in a row, and a broken run counts as caught only when the rule named the planted layer (Table~\\ref{{tab:detectors}}). Requiring three epochs instead of one changed little, since a layer's dead unit fraction rarely drops back once it crosses a threshold{p_change}. Counting only the planted layer mattered more. Counted at any layer, the 50\\% rule caught more runs with 10\\% dead units ({s10['k']}) than with 25\\% ({s25['k']}), because it fired on units that died on their own in other layers. Counted only at the planted layer, it caught {s10r['k']} and {s25r['k']}.

For frozen layers, LHD adds nothing over the zero update check, which caught {kofn(zero)} at the planted layer with no false alarms, and which fires at epoch 1, two epochs before LHD, when a single epoch is enough. A frozen layer's weight change is exactly 0, so a check for an exact 0 needs neither a healthy range nor patience.

Dead units are different, because healthy networks lose units on their own and the amount varies widely. In healthy calibration runs of the CNN's dense layer, between {100 * cnn_lo:.0f}\\% and {100 * cnn_hi:.0f}\\% of units were dead, while {transformer_dead}. A single fixed threshold therefore misses faults in one network or flags healthy runs in another (Figure~\\ref{{fig:detectors_tradeoff}}). LHD caught {pct(lhd_dead['k'], lhd_dead['n'])} of dead unit faults at the planted layer with {lhd_fa['k']} false alarms in {lhd_fa['n']} healthy runs. The 25\\% rule came closest at {pct(q25['k'], q25['n'])} but flagged {q25_fa['k']} healthy runs, a rule that fires on a single dead unit caught {pct(any_dead['k'], any_dead['n'])} while flagging {any_fa['k']}, and the default rule of SageMaker Debugger caught {pct(sage['k'], sage['n'])} while flagging {sage_fa['k']}.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Each detector held to three epochs in a row, with a broken run counted only when the detector named the planted layer. The accuracy and loss curves name no layer, so they are compared in Table~\\ref{{tab:headline}} instead.}}
\\label{{tab:detectors}}
\\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Caught at the planted layer}} & \\\\
\\cmidrule(lr){{2-3}}
Detector & Frozen layers & Dead units & Healthy runs flagged \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

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

These checks were run after the main results, by replaying the same saved runs, and the settings were fixed before them, not picked from them. Thresholds placed right at the edge of the healthy range flagged {100 * edge_fa:.0f}\\% of healthy runs. Widening them by the margin of {SETTINGS['margin']} used here cut that to {100 * used_fa:.0f}\\%, while the share of broken runs caught fell only from {100 * edge_caught:.0f}\\% to {100 * used_caught:.0f}\\%. The range also needs enough calibration runs. Built from 2 runs, it flagged up to {100 * small:.0f}\\% of healthy runs (in the {PROSE[small_arch]}), and built from all 20, at most {100 * full:.0f}\\%.
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

The faults were planted to show up in the signals LHD measures, so the detection and diagnosis rates describe how well LHD separates these faults from normal variation, not how often it would catch bugs in real projects. Many planted faults also cost less than one point of accuracy, which means stopping those runs saves compute on runs that would have produced a nearly normal model. LHD misses dead units in layers where healthy runs already lose many units, because the healthy range treats that loss as normal. Building the range takes about 10 to 20 healthy runs of the same network, which costs more than LHD can save for a model trained only once, and it assumes those runs are themselves healthy. The floor of 0.05 is also too low for small layers. One dead filter out of 16 is 6.25\\%, already above the floor, so a later version should scale the floor to the layer's width. Finally, the networks are small and were trained for 15 to 20 epochs on a laptop CPU, the transformer reaches only about {100 * sum(trans) / len(trans):.0f}\\% next character accuracy, and large networks, long runs, other optimizers, and GPU training were not tested.

\\section{{Conclusion}}

This work presented LHD, a monitor that compares four signals of every layer with ranges measured on healthy runs of the same network and stops a run when one stays out of range. On {d.total_runs()} runs across three networks, it caught {rate(p['detected'])} of planted frozen layers and dead units, usually within three epochs, named the right layer in {rate(p['layer_ok'])} of those runs, and flagged {kofn(h['lhd_false_alarm'])} healthy runs. Its advantage over existing tools is clearest for dead units, where a healthy range for each layer avoided both the misses and the false alarms of fixed thresholds. The next step is to test LHD on real bugs collected from training code.

\\section*{{Use of generative AI}}

Claude, an AI assistant made by Anthropic, wrote most of the experiment and analysis code, ran the training runs, and drafted parts of this text under the author's direction. The author designed the study, chose its methods, and is responsible for all of its content.

\\section*{{Code and data}}

The code, every saved run, and the tables behind each number are at \\url{{https://github.com/sohan-bhat/lhd}}.

"""


def tables(text: str) -> dict:
    """Every table block in the paper text, keyed by its label."""
    out = {}
    for block in text.split("\\begin{table}")[1:]:
        block = "\\begin{table}" + block.split("\\end{table}")[0] + "\\end{table}"
        if "\\label{" in block:
            out[block.split("\\label{")[1].split("}")[0]] = block
    return out


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", help="print one table of the paper by label, such as tab:headline, and stop")
    args = parser.parse_args(argv)
    if args.table:
        d = Data()
        text = "\n".join([method(d), setup(d), results(d)])
        found = tables(text)
        if args.table not in found:
            raise SystemExit(f"no table {args.table}; tables are {', '.join(sorted(found))}")
        print(found[args.table])
        return
    from experiments import paper_figures

    paper_figures.main()
    d = Data()
    missing = [n for n in FIGURES if not (PAPER / "figures" / f"{n}.pdf").exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    related = (PAPER / "related_work.tex").read_text()
    references = (PAPER / "references.tex").read_text()
    body = [preamble(), abstract(d), introduction(d), related, method(d), setup(d),
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
