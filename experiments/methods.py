"""Writes results/METHODS.md, a factual record of how the data was collected.

Usage:
    python -m experiments.methods

Run after experiments.export. Every number is read from the code or
the exported tables, so the record always matches the saved runs.
It is a record to write a paper from, not paper prose.
"""

from __future__ import annotations

import csv
import platform
import subprocess
from collections import defaultdict

from experiments.evaluate import DEAD_CHECKS, HARM_BINS, RESULTS, SETTINGS
from experiments.grid import (
    CALIBRATION_SEEDS,
    DEAD_FRACTIONS,
    DETACH_CUTS,
    EXTRA_SEEDS,
    FAULT_SEEDS,
    FRESH_SEEDS,
    LATE_ONSET,
    build_grid,
)
from experiments.train import DEAD_BIAS

TABLES = RESULTS / "tables"
NAMES = {"mlp": "Fully connected", "cnn": "CNN", "transformer": "Transformer"}


def read(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def span(seeds: list[int]) -> str:
    return f"{seeds[0]} to {seeds[-1]}" if seeds else "none"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return lines


def machine() -> str:
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    return cpu or platform.machine()


def versions() -> str:
    import matplotlib
    import numpy
    import scipy
    import torch

    return (f"Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {numpy.__version__}, "
            f"SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}")


def networks_section() -> list[str]:
    nets = read("networks.csv")
    layers = read("layers.csv")
    out = table(
        ["Network", "Dataset", "Task", "Classes or vocabulary", "Training examples", "Validation examples",
         "Parameters", "Watched layers", "ReLU layers", "Epochs", "Batch size", "Steps per epoch"],
        [[NAMES[n["arch"]], n["dataset"], n["task"], n["classes_or_vocab"], f"{int(n['train_examples']):,}",
          f"{int(n['val_examples']):,}", f"{int(n['parameters']):,}", n["watched_layers"], n["relu_layers"],
          n["epochs"], n["batch_size"], n["steps_per_epoch"]] for n in nets])
    out += ["", "For the transformer, training examples count characters in the training text and "
            "validation examples count fixed windows of 64 characters, so 512 windows give 32,768 "
            "scored next character predictions.", ""]
    by_arch = defaultdict(list)
    for row in layers:
        by_arch[row["arch"]].append(row)
    out += table(["Network", "Layer", "Module", "Weight shape", "Parameters", "ReLU after it"],
                 [[NAMES[r["arch"]], r["layer"], r["module"], r["weight_shape"], f"{int(r['parameters']):,}",
                   "yes" if r["relu_after"] == "True" else "no"] for a in by_arch for r in by_arch[a]])
    return out


def design_section() -> list[str]:
    planned = defaultdict(int)
    for phase in ("core", "seeds", "more"):
        for spec in build_grid(phase):
            planned[(spec["arch"], phase, spec["role"])] += 1
    done = {(r["arch"], r["phase"], r["role"]): int(r["runs"]) for r in read("run_counts.csv")}
    rows = []
    for (arch, phase, role), n in sorted(planned.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
        rows.append([NAMES[arch], phase, role, n, done.get((arch, phase, role), 0)])
    total_planned = sum(planned.values())
    total_done = sum(done.values())
    return [
        *table(["Network", "Phase", "Role", "Planned runs", "Finished runs"], rows),
        "",
        f"Planned runs in total {total_planned:,}. Finished when this file was written {total_done:,}.",
    ]


def compute_section() -> list[str]:
    rows = read("compute.csv")
    return table(["Network", "Runs", "Training hours", "Hours spent inside LHD", "Mean minutes per run"],
                 [[NAMES.get(r["arch"], "All networks"), r["runs"], f"{float(r['training_hours']):.1f}",
                   f"{float(r['lhd_hours']):.1f}",
                   "" if not r["mean_minutes_per_run"] else f"{float(r['mean_minutes_per_run']):.1f}"]
                  for r in rows])


def main() -> None:
    m, f, p = SETTINGS["margin"], SETTINGS["floor"], SETTINGS["patience"]
    fractions = ", ".join(f"{int(x * 100)}%" for x in DEAD_FRACTIONS)
    cuts = "; ".join(f"{NAMES[a]} {' and '.join(c)}" for a, c in DETACH_CUTS.items()).replace(";", ",")
    harm = ", ".join(label for _, _, label in HARM_BINS)
    dead_checks = ", ".join(f"{int(t * 100)}%" for t in DEAD_CHECKS)
    out = [
        "# LHD study, methods record",
        "",
        "This file records how every number in the study was produced. It is written by "
        "`python -m experiments.methods` from the code and the saved runs, so it always matches the "
        "data. It is a record of facts to write a paper from, not paper text.",
        "",
        "## 1. What the study tests",
        "",
        "The proposal (Bhat, October 2026) states this hypothesis.",
        "",
        "> This work hypothesizes that per-layer internal signals (weight change, gradient norm, weight "
        "norm, and fraction of zero outputs) will expose a broken layer within the first few epochs, before "
        "the accuracy curve shows anything wrong, without flagging healthy runs.",
        "",
        "The study turns that into four measurable questions.",
        "",
        "1. Does LHD flag broken runs, and does it flag them earlier than the accuracy curve?",
        "2. When LHD flags a run, does it name the layer where the fault was planted and the right cause?",
        "3. How often does LHD flag healthy runs it has never seen?",
        "4. Do the same rules work on three different networks when only the healthy baselines are rebuilt?",
        "",
        "Every broken run has a fault planted on purpose, so the true answer for each run is known.",
        "",
        "## 2. Networks and data",
        "",
        *networks_section(),
        "",
        "Images are scaled to the range 0 to 1 and standardized with the mean and standard deviation of "
        "the training set. Fashion-MNIST uses one mean for all pixels and GTSRB uses one per color "
        "channel. GTSRB images are resized to 32 by 32 pixels with bilinear resampling. The GTSRB files "
        "come from the Hugging Face copy of the torchvision splits because the original host refused "
        "connections. Tiny Shakespeare is split into the first 90% of characters for training and the "
        "last 10% for validation, one token per character.",
        "",
        "The validation set is each dataset's official test split. Nothing is tuned on it. It only feeds "
        "the accuracy and loss curves that LHD is compared against. The CNN is scored on a fixed random "
        "4,000 image subset of the GTSRB test split, chosen once with seed 2, to save time.",
        "",
        "## 3. How one training run works",
        "",
        "1. Set the random seed, then build the network, so the seed fixes the starting weights.",
        "2. Seed a separate generator with the same seed. It fixes the order of training batches.",
        "3. Train with Adam at learning rate 0.001 and a cosine schedule that falls toward 0 over the run.",
        "4. At the start of each epoch, switch on the planted fault if this is its onset epoch.",
        "5. After every backward pass and before the optimizer step, LHD records each watched layer's "
        "gradient norm.",
        "6. After every optimizer step, a frozen layer gets its old weights put back.",
        "7. At the end of each epoch, score the validation set, then LHD measures every watched layer.",
        "8. Every run trains for all its epochs even when LHD would have stopped it, so each saved run "
        "can be replayed through any detector afterward.",
        "9. Save the run as JSON with every epoch's signals, accuracy, loss, and timing.",
        "",
        "Each run uses 1 CPU thread and 5 runs train at once. Replaying a saved run through LHD gives "
        "the same result as running LHD live, which a test in `tests/test_replay.py` checks.",
        "",
        "## 4. What LHD measures",
        "",
        "For each watched layer, each epoch. W is the layer's weights and bias flattened into one vector.",
        "",
        *table(["Signal", "Formula", "When it is measured"], [
            ["Weight change", "‖W_end − W_start‖ / ‖W_start‖", "Start and end of the epoch"],
            ["Weight norm", "‖W_end‖", "End of the epoch"],
            ["Gradient norm", "mean over batches of ‖∂loss/∂W‖", "Every batch, after the backward pass"],
            ["Zero activation fraction", "share of units whose output is exactly 0 for every probe input",
             "End of the epoch, on the probe batch"],
        ]),
        "",
        "‖·‖ is the square root of the sum of squared entries. The probe batch is the same 256 "
        "validation examples in every run, chosen once with seed 0. It is run in evaluation mode with no "
        "gradients. A unit is one neuron of a dense layer or one filter of a convolution. A filter counts "
        "as dead only when it outputs 0 at every position for every probe input.",
        "",
        "## 5. The healthy range and the alarm rule",
        "",
        f"The healthy range is built only from calibration runs, seeds {span(CALIBRATION_SEEDS)}, "
        f"{len(CALIBRATION_SEEDS)} runs per network with no fault. For every layer, signal, and epoch it "
        "keeps the lowest and highest value across those runs. Broken runs never touch it.",
        "",
        "A value counts as out of range when it is below low × (1 − m) or above high × (1 + m) + f, "
        f"with margin m = {m} and floor f = {f}. A value that is not a finite number is always out of range.",
        "",
        f"LHD raises an alarm when one signal of one layer stays out of range for p = {p} epochs in a "
        "row. If several layers reach that point in the same epoch, LHD reports the first one in the "
        "order data flows through the network.",
        "",
        f"The settings m = {m}, f = {f}, and p = {p} were taken from the LHD library's own tests and "
        "fixed before any broken run was trained. They were never tuned on broken runs. Section 11 shows "
        "how results change across other settings.",
        "",
        "## 6. How LHD names a cause",
        "",
        "The rules run in this order on the alarmed layer's signals at the alarm epoch.",
        "",
        *table(["Order", "Pattern", "Cause named"], [
            [1, "any signal not a finite number", "diverged"],
            [2, "zero activation fraction above range", "dead units"],
            [3, "weight change below range and gradient norm inside or above range", "frozen layer"],
            [4, "weight change below range and gradient norm below range", "vanishing gradient"],
            [5, "weight norm above range", "runaway weights"],
            [6, "anything else", "unrecognized"],
        ]),
        "",
        "## 7. Planted faults",
        "",
        "Every fault goes into one layer of one run.",
        "",
        *table(["Fault", "How it is planted", "Onset epoch", "Where"], [
            ["Frozen layer", "after every optimizer step the layer's weights and bias are put back, so "
             "gradients are computed but never applied", 1, "every watched layer"],
            [f"Dead units, {fractions}", "that share of the layer's units, picked at random with seed + 7919, "
             f"get incoming weights of 0 and a bias of {DEAD_BIAS}, so their ReLU output is 0 for every input. "
             "Adam's running averages for those units are cleared", 1, "every layer followed by a ReLU"],
            ["Late frozen layer", "as above", LATE_ONSET, "every watched layer"],
            ["Late dead units, 50%", "as above", LATE_ONSET, "every layer followed by a ReLU"],
            ["Cut gradient", "the input to one module is detached, so every layer before it gets no "
             "gradient", 1, cuts],
            ["Learning rate jump", "the learning rate is multiplied by 10 or by 100", LATE_ONSET, "whole network"],
        ]),
        "",
        "The proposal's faults are the frozen layer and dead units planted at epoch 1. The others are "
        "extra tests in the more phase.",
        "",
        "## 8. Runs and seeds",
        "",
        *table(["Role", "Seeds", "Purpose"], [
            ["Calibration", span(CALIBRATION_SEEDS), "build the healthy range, nothing else"],
            ["Fresh healthy", f"{span(FRESH_SEEDS['core'])}, then {span(FRESH_SEEDS['more'])}",
             "measure false alarms"],
            ["Broken, core phase", span(FAULT_SEEDS["core"]), "the proposal's faults"],
            ["Broken, seeds phase", span(FAULT_SEEDS["seeds"]), "the proposal's faults on 5 more seeds"],
            ["Broken, more phase", span(EXTRA_SEEDS), "faults beyond the proposal"],
        ]),
        "",
        "Calibration seeds and evaluation seeds never overlap. Every broken run has a healthy twin, the "
        "fresh run with the same seed, which starts from the same weights and sees batches in the same "
        "order. The twin gives the accuracy each broken run would have reached without its fault.",
        "",
        *design_section(),
        "",
        "## 9. Detectors compared",
        "",
        "Every detector sees the same saved runs and, where it needs one, the same calibration runs.",
        "",
        *table(["Detector", "Rule", "Needs healthy runs", "Defined"], [
            ["LHD", f"section 5, patience {p}", "yes", "before testing"],
            ["Accuracy curve, 3 epochs", f"validation accuracy below the lowest calibration value for that "
             f"epoch, {p} epochs in a row", "yes", "before testing"],
            ["Accuracy curve, 1 epoch", "the same with 1 epoch", "yes", "before testing"],
            ["Loss curve, 3 epochs", f"validation loss above the highest calibration value for that epoch, "
             f"{p} epochs in a row", "yes", "before testing"],
            ["Zero update check", "some watched layer's weight change is exactly 0 in an epoch", "no",
             "after the core results"],
            [f"Dead units check, {dead_checks}", "some layer has at least that share of dead units", "no",
             "after the core results"],
            ["Zero update or dead units 50%", "either of the two checks above", "no", "after the core results"],
            ["SageMaker DeadRelu default", "the 50% dead units check above. SageMaker counts a ReLU as "
             "inactive when it is active less than 1% of the time, close to LHD's never active", "no",
             "after the core results"],
            ["DeepDiagnosis Dead Node", "some ReLU layer has more than 70% inactive units, the check in "
             "Wardat et al. (2022)", "no", "after the core results"],
            ["wandb-callbacks default", "some ReLU layer has at least 80% dead units, the default of the "
             "package's DeadReluCallback", "no", "after the core results"],
            ["wandb-callbacks verbose", "some ReLU layer has at least one dead unit, the package's verbose "
             "mode", "no", "after the core results"],
        ]),
        "",
        "The accuracy curve rule uses no margin, which makes it more sensitive than LHD's rule. The "
        "simple checks were added after the core results to test whether LHD does better than rules "
        "anyone could write by hand. They can only make LHD look the same or worse. Every dead unit "
        "check looks only at layers followed by a ReLU. The wandb-callbacks rules copy the dead ReLU "
        "warning of the package cited in the proposal, which counts a unit as dead when it never fires "
        "on the inputs it is given, the same definition LHD uses.",
        "",
        "## 10. Outcome measures",
        "",
        *table(["Measure", "Definition"], [
            ["Caught", "the detector fired at or after the fault's onset epoch"],
            ["Alarm epoch", "the epoch the detector fired"],
            ["Right layer", "LHD named the planted layer. For a cut gradient, any layer before the cut"],
            ["Right cause", "frozen layer for frozen, dead units for dead, vanishing gradient for a cut"],
            ["False alarm", "the detector fired at any epoch of a fresh healthy run"],
            ["Gap", "accuracy curve epoch minus LHD epoch, with never counted as one epoch past the end"],
            ["Compute saved", "share of the run's epochs left after LHD's alarm, 0 when not caught"],
            ["Silent", "final validation accuracy at or above the lowest calibration final accuracy"],
            ["Accuracy cost", "the healthy twin's final accuracy minus the broken run's final accuracy"],
            ["Harm group", harm],
            ["Overhead", "time spent inside LHD divided by the whole run's time, on fresh healthy runs"],
        ]),
        "",
        "## 11. Statistics",
        "",
        "Rates come with 95% Wilson score intervals. Runs that share a seed share starting weights and "
        "batch order, and runs that share a fault and layer share the same break, so single runs are "
        "not independent. The main tests average the gap within each seed and within each fault "
        "condition first, then run a one sided Wilcoxon signed rank test on those averages. The 95% "
        "interval for the mean gap comes from a bootstrap with 10,000 draws that resamples whole seeds "
        "within each network. Run by run tests are reported for reference only.",
        "",
        "Three sweeps replay the same saved runs. The settings sweep tries margins 0, 0.1, 0.25, 0.5, and "
        "0.75, floors 0, 0.02, 0.05, and 0.1, and patience 1 to 5. The calibration sweep builds the "
        "healthy range from 2, 3, 5, 10, 15, and 20 calibration runs, 20 random draws each. The signal "
        "ablation turns each signal off, or uses one signal alone.",
        "",
        "## 12. Software, hardware, and compute",
        "",
        f"{versions()}. All runs trained on one laptop CPU, {machine()}, with 1 thread per run. The "
        "laptop was often thermally throttled and shared with other work, so timing numbers are rough.",
        "",
        *compute_section(),
        "",
        "## 13. Choices and changes made during the study",
        "",
        *table(["What", "When", "Effect on LHD's results"], [
            ["Datasets, networks, epochs, and batch sizes were chosen for this study", "before training",
             "none intended"],
            ["The accuracy curve's alarm was defined as below every healthy run for 3 epochs in a row",
             "before testing", "this is how the proposal's comparison was made measurable"],
            [f"Dead units get a bias of {DEAD_BIAS} instead of the library default of -10",
             "after a pilot run, before the study runs",
             "-10 inflated the weight norm far past normal, which would have handed LHD an easy signal. "
             "-1 makes dead units harder for LHD to catch"],
            ["The CNN was made smaller, the CNN and transformer were cut to 15 epochs, the transformer to "
             "100 steps per epoch, and the CNN scored on 4,000 test images",
             "before any CNN or transformer run", "none intended, done to fit the laptop's compute"],
            ["Adam's running averages are cleared for killed units",
             "after the core fully connected runs, before any late fault ran",
             "affects only late dead unit faults, which would otherwise come back to life"],
            ["Seed and fault condition level statistics became the main tests",
             "after the fully connected core results", "more conservative than run by run tests"],
            ["Simple fixed checks and the harm split were added", "after the core results",
             "can only make LHD look the same or worse"],
        ]),
        "",
        "## 14. Where the code differs from the proposal's wording",
        "",
        *table(["Proposal says", "The study does", "Why"], [
            ["the zero activation fraction is how much of a layer's outputs are exactly zero",
             "the share of a layer's units that output exactly zero for every probe input, so a unit "
             "counts only when it is dead for all inputs",
             "this is how the LHD library's signal was written. The proposal's sentence that dead units "
             "read near one fits both versions. The paper should describe the one in the code"],
            ["LHD stops the run once a fault is confirmed",
             "every run trains to the end and LHD's alarm is found by replaying the saved run",
             "so every detector can be scored on the same full runs. Replay matches live LHD, which "
             "`tests/test_replay.py` checks, and `experiments/demo.py` shows a live run being stopped"],
            ["thresholds are set just outside the healthy range",
             f"the range is widened by a margin of {m} and a floor of {f} before a value counts as out",
             "thresholds right at the range's edge flag many healthy runs, which section 11's sweep shows"],
            ["a layer is flagged after numerous consecutive epochs", f"{p} consecutive epochs",
             "taken from the LHD library's tests"],
            ["the CNN is trained first, then the fully connected network and the transformer",
             "the fully connected network ran first, then the transformer, then the CNN",
             "to fit the laptop's schedule. The rules were fixed before any of them ran"],
            ["the frozen layer is tested first, followed by dead neurons",
             "both faults ran in the same phase", "order does not change a replayed result"],
            ["frozen layers and dead neurons",
             "also late faults, cut gradients, learning rate jumps, 5 more seeds, and extra baselines",
             "extra tests beyond the proposal, reported apart from its main results"],
        ]),
        "",
        "## 15. Where the data is",
        "",
        *table(["File", "Contents"], [
            ["results/runs/<network>.jsonl.gz", "every saved run, one JSON object per line"],
            ["results/tables/runs.csv", "one scored row per fresh and broken run"],
            ["results/tables/epochs.csv", "accuracy and loss for every run and epoch"],
            ["results/tables/signals.csv.gz", "every signal for every run, epoch, and layer"],
            ["results/tables/detectors.csv", "every detector on the same runs"],
            ["results/LHD_data.xlsx", "every table in one workbook"],
            ["results/REPORT.md", "the results with figures"],
        ]),
        "",
    ]
    (RESULTS / "METHODS.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'METHODS.md'}")


if __name__ == "__main__":
    main()
