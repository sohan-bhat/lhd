"""Builds the short LHD paper's numbers, tables, and figures, and compiles it.

Usage:
    python -m experiments.paper_short             numbers, tables, figures, and the PDF
    python -m experiments.paper_short --facts     print every number with its source and stop
    python -m experiments.paper_short --outline   also write the outline, only if lhd_short.tex is missing

Every number comes from the saved runs through the Data reader that
experiments.paper uses, so the short paper and the long one read the same
tables. paper/short/numbers.tex defines one macro per number, such as
\\lhdRuns for the total run count, and the outline lists each one with the
file it comes from.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments import data as data_code
from experiments import models
from experiments import train as train_code
from experiments import paper as long
from experiments.data import PROBE_SIZE, TextTask
from experiments.evaluate import DEAD_CHECKS, SETTINGS
from experiments.grid import CALIBRATION_SEEDS, DEAD_FRACTIONS, FAULT_SEEDS, FRESH_SEEDS, LATE_ONSET
from experiments.train import ARCHS, DEAD_BIAS, train_run

SHORT = long.PAPER / "short"
SUMMARY = "results/summary.json"
CODE = {"grid": "experiments/grid.py", "train": "experiments/train.py", "models": "experiments/models.py",
        "data": "experiments/data.py", "evaluate": "experiments/evaluate.py", "paper": "experiments/paper.py"}
LETTERS = "ABCDEFGHIJ"
FIXED_KEYS = {"Zero update check": "Zero", "Dead units check, 25%": "Quarter", long.SAGEMAKER: "Sage",
              long.DEEPDIAGNOSIS: "Deep", long.WANDB: "Wandb", long.ANY_DEAD: "AnyDead"}
EXTRA_KEYS = {"latefrozen": "LateFrozen", "latedead50": "LateDead", "detach": "Cut", "lr10": "LrTen",
              "lr100": "LrHundred"}


def table(name: str) -> str:
    return f"results/tables/{name}"


@dataclass(frozen=True)
class Fact:
    macro: str
    value: str
    what: str
    source: str

    def shown(self) -> str:
        """The value as it reads in the PDF, for the outline's comments."""
        text = self.value.replace("\\%", "%").replace("$-$", "-").replace("\\times", "x")
        return re.sub(r"[${}]", "", text).replace("p < 10^-10", "p < 1e-10")


class Sheet:
    """Every number of the short paper, keyed by macro name."""

    def __init__(self):
        self.facts: dict[str, Fact] = {}

    def add(self, macro: str, value, what: str, source: str) -> None:
        if not re.fullmatch(r"[A-Za-z]+", macro):
            raise ValueError(f"macro names take letters only: {macro}")
        if macro in self.facts:
            raise ValueError(f"macro {macro} added twice")
        self.facts[macro] = Fact(macro, str(value), what, source)

    def __getitem__(self, macro: str) -> Fact:
        return self.facts[macro]

    def tex(self) -> str:
        lines = ["% Written by experiments/paper_short.py from the saved runs. Rebuild it, never edit it."]
        lines += [f"\\newcommand{{\\lhd{f.macro}}}{{{f.value}}}" for f in self.facts.values()]
        return "\n".join(lines) + "\n"


def git_date(sha: str) -> str:
    out = subprocess.run(["git", "show", "-s", "--format=%as", sha], cwd=long.RESULTS.parent,
                         capture_output=True, text=True, check=True).stdout.strip()
    day = datetime.strptime(out, "%Y-%m-%d")
    return f"{day.day} {day:%B %Y}"


def build(arch: str):
    if arch == "mlp":
        return models.MLP(784, 10)
    if arch == "cnn":
        return models.CNN(43)
    return models.TinyTransformer(65, ARCHS["transformer"]["context"])


def width(arch: str, layer: str) -> int:
    module = build(arch).lhd_layers()[layer][0]
    for name in ("out_features", "out_channels", "embedding_dim"):
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"no width for {arch} {layer}")


def source_number(module, pattern: str) -> str:
    """A constant written inline in a function, read from the code itself."""
    found = re.search(pattern, inspect.getsource(module))
    if not found:
        raise ValueError(f"pattern {pattern} not found in {module.__name__}")
    return found.group(1)


def long_tex() -> str:
    return (long.PAPER / "lhd.tex").read_text()


def headline(s: Sheet, d: long.Data) -> None:
    p, h, c = d.pooled["proposal"], d.pooled["healthy"], d.pooled["proposal_clustered"]
    comp, both, cond = d.composition(), d.both_gap(), d.conditions()
    s.add("Runs", d.total_runs(), "runs in total", table("run_counts.csv"))
    s.add("CalibrationRuns", comp["calibration"], "calibration runs, all networks", SUMMARY)
    s.add("CalibrationPerNet", len(CALIBRATION_SEEDS), "calibration runs per network", CODE["grid"])
    s.add("FreshRuns", comp["fresh"], "fresh healthy runs, all networks", SUMMARY)
    s.add("BrokenRuns", comp["main"], "broken runs with a frozen layer or dead units", SUMMARY)
    s.add("OtherRuns", comp["other"], "runs with the extra faults (third set)", SUMMARY)
    s.add("Caught", p["detected"]["k"], "broken runs LHD caught", SUMMARY)
    s.add("CaughtRate", long.rate(p["detected"]), "share of broken runs LHD caught", SUMMARY)
    s.add("CaughtInterval", long.interval(p["detected"]), "95% Wilson interval of that share", SUMMARY)
    s.add("AccCaught", p["acc_detected"]["k"], "broken runs the accuracy curve caught", SUMMARY)
    s.add("AccCaughtRate", long.rate(p["acc_detected"]), "share the accuracy curve caught", SUMMARY)
    s.add("MedianEpoch", f"{p['lhd_median_epoch']:.0f}", "median epoch of LHD's alarm", SUMMARY)
    s.add("AccMedianEpoch", f"{p['acc_median_epoch']:.0f}", "median epoch of the accuracy curve's alarm", SUMMARY)
    s.add("AccNever", long.rate(d.pooled["acc_never_on_detected"]),
          "runs LHD caught that the accuracy curve never caught", SUMMARY)
    s.add("SilentShare", long.rate(p["silent_share"]),
          "broken runs whose final accuracy ended inside the healthy range", SUMMARY)
    s.add("BothCaught", both["n"], "broken runs both LHD and the accuracy curve caught", table("runs.csv"))
    s.add("BothGap", f"{both['mean']:.1f}", "mean epochs LHD was earlier on those runs", table("runs.csv"))
    s.add("BothMedian", f"{both['median']:.0f}", "median epochs LHD was earlier on those runs", table("runs.csv"))
    s.add("BothAccEarlier", both["acc_earlier"], "of those runs, accuracy curve earlier", table("runs.csv"))
    s.add("BothTies", both["ties"], "of those runs, same epoch", table("runs.csv"))
    s.add("LayerRight", long.rate(p["layer_ok"], 1), "caught runs where LHD named the planted layer", SUMMARY)
    s.add("LayerRightRuns", long.kofn(p["layer_ok"]), "the same as runs", SUMMARY)
    s.add("CauseRight", long.rate(p["cause_ok"], 1), "caught runs where LHD named the planted cause", SUMMARY)
    s.add("CauseRightRuns", long.kofn(p["cause_ok"]), "the same as runs", SUMMARY)
    s.add("FalseAlarms", h["lhd_false_alarm"]["k"], "fresh healthy runs LHD flagged", SUMMARY)
    s.add("HealthyRuns", h["lhd_false_alarm"]["n"], "fresh healthy runs scored", SUMMARY)
    s.add("FalseAlarmRate", long.rate(h["lhd_false_alarm"], 1), "LHD false alarm rate", SUMMARY)
    s.add("FalseAlarmInterval", long.interval1(h["lhd_false_alarm"]), "95% Wilson interval of that rate", SUMMARY)
    s.add("AccFalseAlarms", h["acc_false_alarm"]["k"], "fresh healthy runs the accuracy curve flagged", SUMMARY)
    s.add("AccFalseAlarmInterval", long.interval1(h["acc_false_alarm"]), "95% Wilson interval of that rate",
          SUMMARY)
    seeds = c["by_seed"]
    s.add("SeedGroups", seeds["groups"], "seed groups (network and seed)", SUMMARY)
    s.add("SeedGroupsLHD", seeds["lhd_earlier"], "seed groups whose mean gap favored LHD", SUMMARY)
    s.add("SeedP", long.pval(seeds.get("wilcoxon_p")), "one sided Wilcoxon over seed groups", SUMMARY)
    s.add("Conditions", cond["groups"], "fault conditions (network, fault, layer)", table("runs.csv"))
    s.add("ConditionsLHD", cond["lhd"], "conditions whose mean gap favored LHD", table("runs.csv"))
    s.add("ConditionsAcc", len(cond["acc"]), "conditions where the accuracy curve was earlier", table("runs.csv"))
    s.add("ConditionsSame", cond["same"], "conditions where both fired at the same epoch", table("runs.csv"))
    s.add("ConditionsNeither", cond["neither"], "conditions where neither fired", table("runs.csv"))
    s.add("ConditionP", long.pval(c["by_condition"].get("wilcoxon_p")), "one sided Wilcoxon over conditions",
          SUMMARY)
    s.add("MeanGapAll", f"{p['mean_gap']:.1f}", "mean gap over all broken runs, never as one past the end", SUMMARY)
    s.add("MeanGapInterval", f"{c['mean_gap_ci'][0]:.1f} to {c['mean_gap_ci'][1]:.1f}",
          "95% bootstrap interval over seeds", SUMMARY)


def harm(s: Sheet, d: long.Data) -> None:
    rows = {r["harm"]: r for r in d.harm}
    for key, label in (("under 0.5 pts", "Small"), ("0.5 to 2 pts", "Mid"), ("2 pts or more", "Big")):
        r = rows[key]
        s.add(f"Harm{label}", r["broken_runs"], f"broken runs whose fault cost {key}", table("harm.csv"))
        s.add(f"Harm{label}LHD", r["lhd_caught"], "of those, LHD caught", table("harm.csv"))
        s.add(f"Harm{label}Acc", r["accuracy_caught"], "of those, the accuracy curve caught", table("harm.csv"))
    others = [long.mean_drop(d, a, f) for a in d.archs if a != "cnn" for f in ("frozen",) + long.DEAD]
    s.add("CnnFrozenDrop", f"{long.mean_drop(d, 'cnn', 'frozen'):.1f}",
          "mean accuracy points a frozen CNN layer cost against the twin", table("runs.csv"))
    s.add("CnnHalfDeadDrop", f"{long.mean_drop(d, 'cnn', 'dead50'):.1f}",
          "mean points half a CNN layer's units dead cost", table("runs.csv"))
    s.add("OtherMaxDrop", f"{max(v for v in others if v is not None):.1f}",
          "largest mean cost of any fault in the other two networks", table("runs.csv"))


def detectors(s: Sheet, d: long.Data) -> None:
    p3, frozen = ", 3 epochs", ("frozen",)
    lhd_frozen, lhd_dead = d.detector_right("LHD", frozen), d.detector_right("LHD", long.DEAD)
    s.add("LhdFrozenRight", long.pct(lhd_frozen["k"], lhd_frozen["n"]), "frozen layers LHD caught at the planted layer",
          table("detectors.csv"))
    s.add("LhdDeadRight", long.pct(lhd_dead["k"], lhd_dead["n"]), "dead unit faults LHD caught at the planted layer",
          table("detectors.csv"))
    any_dead = d.detector("LHD", long.DEAD)
    s.add("LhdDeadAny", long.pct(any_dead["k"], any_dead["n"]), "dead unit faults LHD caught at any layer",
          table("detectors.csv"))
    for name, key in FIXED_KEYS.items():
        fr, de, fa = d.detector_right(name + p3, frozen), d.detector_right(name + p3, long.DEAD), \
            d.detector_alarms(name + p3)
        label = dict(long.FIXED)[name].replace("\\%", "%")
        s.add(f"{key}Frozen", long.pct(fr["k"], fr["n"]), f"{label}, frozen layers caught at the planted layer",
              table("detectors.csv"))
        s.add(f"{key}Dead", long.pct(de["k"], de["n"]), f"{label}, dead unit faults caught at the planted layer",
              table("detectors.csv"))
        s.add(f"{key}Flagged", fa["k"], f"{label}, healthy runs flagged (of {fa['n']})", table("detectors.csv"))
    zero3 = d.detector_right("Zero update check" + p3, frozen)
    zero1 = d.detector("Zero update check", frozen)
    zero_epoch = float(d.family_row("Zero update check", "frozen")["median_epoch"])
    lhd_epoch = float(d.family_row("LHD", "frozen")["median_epoch"])
    s.add("ZeroFrozenRuns", long.kofn(zero3), "frozen layers the zero update check caught at the planted layer",
          table("detectors.csv"))
    s.add("ZeroFrozenOneEpoch", long.kofn(zero1), "the same check when one epoch is enough", table("detectors.csv"))
    s.add("ZeroEpoch", f"{zero_epoch:.0f}", "median epoch of that one epoch check", table("detectors.csv"))
    s.add("LhdFrozenEpoch", f"{lhd_epoch:.0f}", "median epoch of LHD's alarm on frozen layers", table("detectors.csv"))
    s.add("ZeroLead", f"{lhd_epoch - zero_epoch:.0f}", "epochs the zero update check fires before LHD",
          table("detectors.csv"))
    p1_fa = {n: d.detector_alarms(n)["k"] for n in FIXED_KEYS}
    p3_fa = {n: d.detector_alarms(n + p3)["k"] for n in FIXED_KEYS}
    p1_dead = {n: d.detector(n, long.DEAD)["k"] for n in FIXED_KEYS}
    p3_dead = {n: d.detector(n + p3, long.DEAD)["k"] for n in FIXED_KEYS}
    s.add("PatienceAlarmChange", max(abs(p1_fa[n] - p3_fa[n]) for n in p1_fa),
          "largest change in false alarms between one and three epochs, fixed rules", table("detectors.csv"))
    s.add("PatienceDeadChange", max(p1_dead[n] - p3_dead[n] for n in p1_dead),
          "largest drop in dead unit catches between one and three epochs", table("detectors.csv"))
    for fam, key in (("dead10", "Ten"), ("dead25", "TwentyFive")):
        anywhere = d.detector(long.SAGEMAKER + p3, (fam,))
        right = d.detector_right(long.SAGEMAKER + p3, (fam,))
        s.add(f"SageAny{key}", anywhere["k"], f"50% rule catches of {fam[4:]}% dead units at any layer",
              table("detectors.csv"))
        s.add(f"SageRight{key}", right["k"], f"50% rule catches of {fam[4:]}% dead units at the planted layer",
              table("detectors.csv"))
    low, high = d.dead_range("cnn", "fc1")
    s.add("CnnDenseLow", f"{100 * low:.0f}\\%", "fewest dead units in the CNN's fc1, calibration runs",
          table("signals.csv.gz"))
    s.add("CnnDenseHigh", f"{100 * high:.0f}\\%", "most dead units in the CNN's fc1, calibration runs",
          table("signals.csv.gz"))
    top = max(d.dead_range("transformer", layer)[1] for layer in ("b0.ff1", "b1.ff1"))
    s.add("TransformerDeadMax", f"{100 * top:.0f}\\%", "most dead units in any transformer ReLU layer, calibration",
          table("signals.csv.gz"))


def false_alarms(s: Sheet, d: long.Data) -> None:
    for letter, r in zip(LETTERS, d.alarms):
        value = d.alarm_value(r["run_id"], r["lhd_layer"], r["lhd_epoch"])
        low, high = d.dead_range(r["arch"], r["lhd_layer"])
        kind = "above" if value > high else "below"
        s.add(f"Alarm{letter}Run", r["run_id"], "false alarm run", table("false_alarms.csv"))
        s.add(f"Alarm{letter}Layer", r["lhd_layer"], "layer it named", table("false_alarms.csv"))
        s.add(f"Alarm{letter}Epoch", r["lhd_epoch"], "epoch it fired", table("false_alarms.csv"))
        s.add(f"Alarm{letter}Signal", r["lhd_signals"].replace("_", " "), "signal out of range",
              table("false_alarms.csv"))
        s.add(f"Alarm{letter}Value", f"{100 * value:.1f}\\%", "dead units in that layer at the alarm",
              table("signals.csv.gz"))
        s.add(f"Alarm{letter}Edge", f"{100 * (high if kind == 'above' else low):.1f}\\%",
              f"calibration {'highest' if kind == 'above' else 'lowest'} for that layer, so the alarm is {kind}",
              table("signals.csv.gz"))
    s.add("Floor", SETTINGS["floor"], "floor f", CODE["evaluate"])
    zero_edge = [r for r in d.alarms if d.dead_range(r["arch"], r["lhd_layer"])[1] == 0]
    if zero_edge:
        r = zero_edge[0]
        units = width(r["arch"], r["lhd_layer"])
        s.add("SmallLayerUnits", units, f"units in the {long.PROSE[r['arch']]}'s {r['lhd_layer']}", CODE["models"])
        s.add("OneUnitShare", f"{100 / units:.2f}\\%", "one dead unit in that layer", CODE["models"])


def method_facts(s: Sheet, d: long.Data) -> None:
    s.add("Margin", SETTINGS["margin"], "margin m", CODE["evaluate"])
    s.add("Patience", SETTINGS["patience"], "patience p, epochs in a row", CODE["evaluate"])
    s.add("ProbeSize", PROBE_SIZE, "validation examples in the probe batch", CODE["data"])
    settings_commit = re.search(r"commit/([0-9a-f]{7})", long.method(d)).group(1)
    s.add("SettingsCommit", settings_commit, "commit that fixed m, f, and p", CODE["paper"])
    s.add("SettingsDate", git_date(settings_commit), "date of that commit", "git log")
    named = Counter()
    for fam in EXTRA_KEYS:
        named.update(d.extra(fam)["causes"])
    s.add("DivergenceFired", named.get("diverged", 0), "runs where the divergence rule fired, all faults",
          SUMMARY)
    s.add("VanishingFired", named.get("vanishing_gradient", 0), "runs named vanishing gradient, extra faults only",
          SUMMARY)
    s.add("RunawayFired", named.get("runaway_weights", 0), "runs named runaway weights, extra faults only", SUMMARY)


def setup_facts(s: Sheet, d: long.Data) -> None:
    for a in d.archs:
        key = {"mlp": "Mlp", "cnn": "Cnn", "transformer": "Transformer"}[a]
        s.add(f"{key}Params", f"{int(d.networks[a]['parameters']):,}", f"parameters, {long.PROSE[a]}",
              table("networks.csv"))
        s.add(f"{key}Epochs", ARCHS[a]["epochs"], f"epochs, {long.PROSE[a]}", CODE["train"])
        s.add(f"{key}Fresh", d.role_count(a, "fresh"), f"fresh healthy runs, {long.PROSE[a]}", SUMMARY)
        s.add(f"{key}Broken", d.main_count(a), f"broken runs in the main study, {long.PROSE[a]}", SUMMARY)
        s.add(f"{key}Other", d.other_count(a), f"runs with extra faults, {long.PROSE[a]}", SUMMARY)
    mlp, cnn = build("mlp"), build("cnn")
    s.add("MlpWidths", ", ".join(str(mlp.lhd_layers()[n][0].out_features) for n in ("fc1", "fc2", "fc3")),
          "hidden widths of the fully connected network", CODE["models"])
    s.add("CnnFilters", ", ".join(str(cnn.lhd_layers()[n][0].out_channels) for n in ("conv1", "conv2", "conv3")),
          "filters in the CNN's three convolutions", CODE["models"])
    s.add("CnnDense", cnn.fc1.out_features, "units in the CNN's dense layer", CODE["models"])
    defaults = inspect.signature(models.TinyTransformer).parameters
    for name, label in (("width", "Width"), ("heads", "Heads"), ("depth", "Blocks"), ("hidden", "Hidden")):
        s.add(f"Transformer{label}", defaults[name].default, f"transformer {name}", CODE["models"])
    s.add("CnnValImages", f"{ARCHS['cnn']['val_size']:,}", "CNN validation images", CODE["train"])
    s.add("TextWindows", inspect.signature(TextTask).parameters["val_windows"].default,
          "transformer validation windows", CODE["data"])
    s.add("TextContext", ARCHS["transformer"]["context"], "characters per window", CODE["train"])
    split = float(source_number(data_code, r"int\(([\d.]+) \* len\(tokens\)\)"))
    s.add("TextHeldOut", f"{100 * (1 - split):.0f}\\%", "share of the text held out for validation", CODE["data"])
    rates = {cfg["lr"] for cfg in ARCHS.values()}
    s.add("LearningRate", f"{rates.pop():g}" if len(rates) == 1 else "varies", "Adam learning rate", CODE["train"])
    s.add("Threads", inspect.signature(train_run).parameters["threads"].default, "CPU threads per run", CODE["train"])
    s.add("CalibrationSeeds", f"{min(CALIBRATION_SEEDS)} to {max(CALIBRATION_SEEDS)}", "calibration seeds",
          CODE["grid"])
    s.add("ScoredSeedStart", min(FRESH_SEEDS["core"] + FAULT_SEEDS["core"]), "first seed of scored runs",
          CODE["grid"])
    s.add("SeedsPerFault", len(FAULT_SEEDS["core"]) + len(FAULT_SEEDS["seeds"]), "seeds per fault condition",
          CODE["grid"])
    s.add("DeadFractions", long.listing([f"{100 * f:g}\\%" for f in DEAD_FRACTIONS]), "dead unit shares planted",
          CODE["grid"])
    s.add("DeadBias", f"${DEAD_BIAS:g}$", "bias given to killed units", CODE["train"])
    s.add("UnitSeedOffset", source_number(train_code, r"seed \+ (\d+)"),
          "added to the run's seed to pick the killed units", CODE["train"])
    s.add("DeadChecks", long.listing([f"{100 * t:g}\\%" for t in DEAD_CHECKS]), "fixed dead unit thresholds scored",
          CODE["evaluate"])
    fresh = d.fresh_by_phase()
    s.add("FreshFirstSet", fresh["core"], "fresh healthy runs in the first set", table("runs.csv"))
    s.add("FreshThirdSet", fresh["more"], "fresh healthy runs in the third set", table("runs.csv"))
    s.add("PlanCommit", long.PLAN_COMMIT, "commit with the run plan", CODE["paper"])
    s.add("PlanDate", git_date(long.PLAN_COMMIT), "date of that commit", "git log")
    s.add("DraftCommit", long.DRAFT["commit"], "commit of the earlier draft", CODE["paper"])
    s.add("DraftFresh", long.DRAFT["fresh"], "fresh healthy runs done at that draft", CODE["paper"])
    s.add("DraftAlarms", long.DRAFT["alarms"], "false alarms that draft reported", CODE["paper"])
    h = d.pooled["healthy"]["lhd_false_alarm"]
    s.add("LaterFresh", sum(fresh.values()) - long.DRAFT["fresh"], "fresh healthy runs finished after that draft",
          table("runs.csv"))
    s.add("LaterAlarms", h["k"] - long.DRAFT["alarms"], "false alarms those later runs added", SUMMARY)


def limitation_facts(s: Sheet, d: long.Data) -> None:
    epochs = [ARCHS[a]["epochs"] for a in d.archs]
    s.add("EpochRange", f"{min(epochs)} to {max(epochs)}", "epochs per run", CODE["train"])
    trans = [float(r["final_acc"]) for r in d.runs if r["arch"] == "transformer" and r["role"] == "fresh"]
    s.add("TransformerAcc", f"{100 * sum(trans) / len(trans):.0f}\\%",
          "mean final next character accuracy of fresh transformer runs", table("runs.csv"))
    for fam in long.DEAD:
        row = next(r for r in d.per_layer if r["arch"] == "cnn" and r["layer"] == "fc1" and r["family"] == fam)
        s.add(f"CnnDense{LETTERS[long.DEAD.index(fam)]}Caught", f"{row['detected']} of {row['runs']}",
              f"CNN fc1 runs with {fam[4:]}% dead units that LHD caught", table("per_layer.csv"))
    cal = defaultdict(list)
    for r in d.calibration:
        cal[(r["arch"], int(r["k"]))].append(float(r["false_alarm"]))
    for i, k in enumerate(sorted({k for _, k in cal})):
        values = {a: sum(cal[(a, k)]) / len(cal[(a, k)]) for a in d.archs if cal[(a, k)]}
        worst = max(values, key=values.get)
        s.add(f"Cal{LETTERS[i]}Runs", k, "calibration runs used to build the range", table("sweep_calibration.csv"))
        s.add(f"Cal{LETTERS[i]}Worst", f"{100 * values[worst]:.0f}\\%",
              "highest mean false alarm rate over networks at that size", table("sweep_calibration.csv"))
        s.add(f"Cal{LETTERS[i]}Net", long.PROSE[worst], "network with that rate", table("sweep_calibration.csv"))


def appendix_facts(s: Sheet, d: long.Data) -> None:
    fam = ("frozen",) + long.DEAD
    for name, key in (("Accuracy curve, 1 epoch below healthy", "AccOne"), ("Loss curve, 3 epochs above healthy", "Loss")):
        caught, alarms = d.detector(name, fam), d.detector_alarms(name)
        s.add(f"{key}Caught", long.kofn(caught), f"{name}, broken runs caught", table("detectors.csv"))
        s.add(f"{key}Flagged", long.kofn(alarms), f"{name}, healthy runs flagged", table("detectors.csv"))
    for i, f in enumerate(long.DEAD):
        lhd, acc = d.detector("LHD", (f,)), d.detector("Accuracy curve, 3 epochs below healthy", (f,))
        s.add(f"Dose{LETTERS[i]}Lhd", long.pct(lhd["k"], lhd["n"]), f"LHD catches with {f[4:]}% dead units",
              table("detectors.csv"))
        s.add(f"Dose{LETTERS[i]}Acc", long.pct(acc["k"], acc["n"]), f"accuracy curve catches with {f[4:]}% dead units",
              table("detectors.csv"))
    misses = [r for r in d.per_layer if r["family"] in long.DEAD and int(r["detected"]) < int(r["runs"])]
    s.add("DoseMisses", "; ".join(f"{long.PROSE[r['arch']]} {r['layer']} at {r['family'][4:]}\\% "
                                  f"({r['detected']} of {r['runs']})" for r in misses),
          "every layer and share where LHD missed some dead unit runs", table("per_layer.csv"))
    groups = Counter((r["arch"], r["family"], r["fault_layer"], r["lhd_layer"]) for r in d.wrong)
    s.add("WrongLayerRuns", len(d.wrong), "caught runs where LHD named another layer", table("wrong_layers.csv"))
    s.add("WrongLayerList", "; ".join(f"{long.fault_name(f, layer)} in the {long.PROSE[a]} named {named} ({n})"
                                      for (a, f, layer, named), n in sorted(groups.items())),
          "those runs", table("wrong_layers.csv"))
    causes = Counter((r["arch"], r["family"], r["fault_layer"], r["lhd_cause"]) for r in d.runs
                     if r["role"] == "fault" and r["family"] in fam and r["detected"] == "True"
                     and r["layer_ok"] == "True" and r["cause_ok"] == "False")
    s.add("WrongCauseRuns", sum(causes.values()), "right layer but wrong cause", table("runs.csv"))
    s.add("WrongCauseList", "; ".join(f"{long.fault_name(f, layer)} in the {long.PROSE[a]} called "
                                      f"{cause.replace('_', ' ')} ({n})" for (a, f, layer, cause), n in sorted(causes.items())),
          "those runs", table("runs.csv"))
    rows = {name: (caught, both) for name, caught, both in long.pooled_ablation(d)}
    for name, key in (("all four", "AllFour"), ("without dead_unit_fraction", "NoDead"),
                      ("without weight_change", "NoChange")):
        caught, both = rows[name]
        s.add(f"Ablation{key}Caught", f"{100 * caught:.0f}\\%", f"broken runs caught, {name.replace('_', ' ')}",
              table("ablation.csv"))
        s.add(f"Ablation{key}Both", f"{100 * both:.0f}\\%", f"right layer and cause, {name.replace('_', ' ')}",
              table("ablation.csv"))
    sweep = {}
    for margin in (0.0, SETTINGS["margin"]):
        picked = [r for r in d.sweep if float(r["margin"]) == margin and int(r["patience"]) == SETTINGS["patience"]
                  and float(r["floor"]) == SETTINGS["floor"]]
        fresh, faults = sum(int(r["fresh"]) for r in picked), sum(int(r["faults"]) for r in picked)
        sweep[margin] = (sum(float(r["false_alarm"]) * int(r["fresh"]) for r in picked) / fresh,
                         sum(float(r["detection"]) * int(r["faults"]) for r in picked) / faults)
    s.add("EdgeFlagged", f"{100 * sweep[0.0][0]:.0f}\\%", "healthy runs flagged with no margin",
          table("sweep_settings.csv"))
    s.add("EdgeCaught", f"{100 * sweep[0.0][1]:.0f}\\%", "broken runs caught with no margin", table("sweep_settings.csv"))
    s.add("MarginFlagged", f"{100 * sweep[SETTINGS['margin']][0]:.0f}\\%", "healthy runs flagged with the margin used",
          table("sweep_settings.csv"))
    s.add("MarginCaught", f"{100 * sweep[SETTINGS['margin']][1]:.0f}\\%", "broken runs caught with the margin used",
          table("sweep_settings.csv"))
    summary = d.summary["archs"]
    for a in d.archs:
        key = {"mlp": "Mlp", "cnn": "Cnn", "transformer": "Transformer"}[a]
        s.add(f"{key}Overhead", f"{100 * summary[a]['healthy']['overhead']:.1f}\\%",
              f"training time LHD added, {long.PROSE[a]}", SUMMARY)
    s.add("ComputeSaved", f"{100 * d.pooled['proposal']['mean_compute_saved']:.0f}\\%",
          "mean training saved by stopping each broken run at LHD's alarm", SUMMARY)
    s.add("TrainingHours", f"{float(d.compute['all']['training_hours']):.0f}", "CPU hours for all runs",
          table("compute.csv"))
    extra_facts(s, d)


def extra_facts(s: Sheet, d: long.Data) -> None:
    import math

    s.add("LateOnset", LATE_ONSET, "epoch the late faults and learning rate jumps start", CODE["grid"])
    for fam, key in EXTRA_KEYS.items():
        info = d.extra(fam)
        s.add(f"{key}Runs", info["detected"][1], f"runs, {fam}", SUMMARY)
        s.add(f"{key}Caught", info["detected"][0], f"LHD caught, {fam}", SUMMARY)
        s.add(f"{key}Both", info["both"][0], f"LHD named layer and cause, {fam}", table("runs.csv"))
        s.add(f"{key}Acc", info["acc"], f"accuracy curve caught, {fam}", SUMMARY)
        named = [f"{long.CAUSE_PHRASE.get(c, c.replace('_', ' '))} ({n})" for c, n in info["causes"].most_common()
                 if c != "not detected"]
        s.add(f"{key}Causes", "; ".join(named), f"causes LHD named, {fam}", SUMMARY)
    short = min(int(d.networks[a]["epochs"]) for a in d.archs)
    normal = 0.001 * (1 + math.cos(math.pi * (LATE_ONSET - 1) / short)) / 2
    s.add("LrNormal", f"{normal:.4g}", f"learning rate at epoch {LATE_ONSET} in the {short} epoch networks",
          CODE["train"])
    s.add("LrTimesTen", f"{10 * normal:.4g}", "the same after a tenfold jump", CODE["train"])
    s.add("LrTimesHundred", f"{100 * normal:.4g}", "the same after a hundredfold jump", CODE["train"])
    zero1, zero3 = d.family_row("Zero update check", "latefrozen"), d.family_row("Zero update check, 3 epochs",
                                                                                 "latefrozen")
    s.add("LateZeroEpoch", f"{float(zero1['median_epoch']):.0f}", "zero update check epoch on late frozen layers",
          table("detectors.csv"))
    s.add("LateZeroTies", f"{zero3['ties']} of {zero3['broken_runs']}",
          "late frozen runs where the three epoch check fired with LHD", table("detectors.csv"))
    cut = d.family_row("Zero update check, 3 epochs", "detach")
    s.add("CutZero", f"{cut['caught_right_layer']} of {cut['broken_runs']}",
          "gradient cuts the zero update check caught before the cut", table("detectors.csv"))
    sage = d.family_row(long.SAGEMAKER + ", 3 epochs", "latedead50")
    s.add("LateDeadSage", f"{sage['caught_right_layer']} of {sage['broken_runs']}",
          "late half dead runs the 50% rule caught at the planted layer", table("detectors.csv"))


LONG_TABLES = {"tab:rules": "rules", "tab:headline": "headline", "tab:detectors": "detectors",
               "tab:ablation": "ablation", "tab:extra": "extra"}


def tabular(block: str) -> str:
    """The tabular of a long paper table, without its float or caption."""
    start = block.index("\\begin{tabular}")
    end = block.index("\\end{tabular}") + len("\\end{tabular}")
    return block[start:end] + "\n"


def networks_table(d: long.Data) -> str:
    """Table 2 in one header row, with a total that adds up to every run."""
    rows, totals = [], [0, 0, 0, 0]
    for a in d.archs:
        n = d.networks[a]
        runs = [d.role_count(a, "calibration"), d.role_count(a, "fresh"), d.main_count(a), d.other_count(a)]
        totals = [t + r for t, r in zip(totals, runs)]
        rows.append(f"{long.NAMES[a]} & {long.DATASETS[a]} & {int(n['parameters']):,} & {n['epochs']} & "
                    + " & ".join(str(r) for r in runs) + " \\\\")
    if sum(totals) != d.total_runs():
        raise SystemExit(f"Table 2 adds up to {sum(totals)} runs, not {d.total_runs()}")
    rows += ["\\midrule", "Total & & & & " + " & ".join(str(t) for t in totals) + " \\\\"]
    return ("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
            " & & & & & Fresh & Frozen & Other \\\\\n"
            "Network & Dataset & Parameters & Epochs & Calibration & healthy & or dead & faults \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def write_tables(d: long.Data) -> None:
    found = long.tables("\n".join([long.method(d), long.setup(d), long.results(d)]))
    missing = [label for label in LONG_TABLES if label not in found]
    if missing:
        raise SystemExit(f"long paper tables not found: {missing}")
    folder = SHORT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for label, name in LONG_TABLES.items():
        (folder / f"{name}.tex").write_text(tabular(found[label]))
    (folder / "networks.tex").write_text(networks_table(d))
    print(f"wrote {len(LONG_TABLES) + 1} tables to {folder}", flush=True)


def write_figures() -> dict:
    """The long paper's four figures, drawn by the same code into paper/short/figures."""
    from experiments import paper_figures as figs

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    figs.PAPER_FIGURES = SHORT / "figures"
    figs.setup()
    rows = figs.read_csv("runs.csv")
    return {"how_lhd_works": figs.how_lhd_works(rows), "detection_over_time": figs.detection_over_time(rows),
            "detectors_tradeoff": figs.detectors_tradeoff(), "per_layer": figs.per_layer(rows)}


def figure_facts(s: Sheet, drawn: dict) -> None:
    one = drawn["how_lhd_works"]
    source = "experiments/paper_figures.py"
    s.add("FigRun", one["run_id"], "run in Figure 1", source)
    s.add("FigLayer", one["layer"], "frozen layer in Figure 1", source)
    s.add("FigAlarm", one["alarm"], "epoch LHD stops that run", source)
    s.add("FigAccuracy", f"{one['accuracy']:.1f}\\%", "its validation accuracy when LHD stops it", source)
    s.add("FigAccuracyRange", f"{one['range'][0]:.1f} to {one['range'][1]:.1f}\\%",
          "calibration accuracy range at that epoch", source)
    s.add("FigFinal", f"{one['final']:.1f}\\%", "its final accuracy", source)
    s.add("FigFinalRange", f"{one['final_range'][0]:.1f} to {one['final_range'][1]:.1f}\\%",
          "calibration accuracy range at the last epoch", source)
    for a, shares in drawn["detection_over_time"].items():
        key = {"mlp": "Mlp", "cnn": "Cnn", "transformer": "Transformer"}[a]
        s.add(f"Time{key}Lhd", f"{shares['LHD'][-1]:.0f}\\%", f"broken runs LHD flagged by the end, {long.PROSE[a]}",
              source)
        s.add(f"Time{key}Acc", f"{shares['Accuracy curve'][-1]:.0f}\\%",
              f"broken runs the accuracy curve flagged by the end, {long.PROSE[a]}", source)


PLACEHOLDER = "SOHAN WRITES THIS"


def block(s: Sheet, budget: str, macros: list[str], notes: list[str]) -> str:
    """A section's budget, the numbers it needs with their files, other facts, and the placeholder."""
    lines = [f"% BUDGET {budget}"]
    if macros:
        lines.append("% NUMBERS. Each \\lhd macro prints the value next to it, and typing the value works too.")
        pad = min(22, max(len(s[m].shown()) for m in macros))
        name = max(len(m) for m in macros) + 4
        for m in macros:
            tag = "\\lhd" + m
            lines.append(f"%   {s[m].shown():<{pad}}  {tag:<{name}}  {s[m].what}  [{s[m].source}]")
    if notes:
        lines.append("% FACTS")
        lines += [f"%   {n}" for n in notes]
    lines.append(PLACEHOLDER)
    return "\n".join(lines) + "\n"


def figure_float(name: str, shows: str) -> str:
    return (f"% FIGURE shows {shows}\n\\begin{{figure}}[t]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n\\caption{{{PLACEHOLDER}}}\n"
            f"\\label{{fig:{name}}}\n\\end{{figure}}\n")


def table_float(name: str, label: str, shows: str) -> str:
    return (f"% TABLE shows {shows}. The tabular is rebuilt from the data, so edit only the caption.\n"
            f"\\begin{{table}}[t]\n\\centering\n\\small\n\\caption{{{PLACEHOLDER}}}\n\\label{{tab:{label}}}\n"
            f"\\input{{tables/{name}}}\n\\end{{table}}\n")


def story(tex: str) -> tuple[str, str]:
    before = re.search(r"plateaued at (\d+)\\%", tex).group(1)
    after = re.search(r"raised the accuracy to (\d+)\\%", tex).group(1)
    return before, after


def related_notes() -> list[str]:
    text = (long.PAPER / "related_work.tex").read_text()
    notes = []
    for name, body in re.findall(r"\\paragraph\{([^}]*)\}(.*?)(?=\\paragraph|\\medskip)", text, re.S):
        keys = sorted({k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", body) for k in group.split(",")})
        quoted = [(n + w).replace("\\%", "%") for n, w in
                  re.findall(r"(?<![\w-])(\d+(?:\.\d+)?(?:\\%)?(?: of (?:the )?\d+)?)((?: [A-Za-z0-9-]+){0,2})", body)
                  if not re.fullmatch(r"(19|20)\d\d", n)]
        notes.append(f"{name.rstrip('.')}  cites {', '.join(keys)}"
                     + (f"  numbers it quotes: {'; '.join(quoted)}" if quoted else ""))
    refs = re.findall(r"\\bibitem\[([^\]]*)\]\{([^}]*)\}", (long.PAPER / "references.tex").read_text())
    notes.append("All citation keys in paper/references.tex: " + "; ".join(f"{k} {label}" for label, k in refs))
    return [n + "  [paper/related_work.tex]" if not n.startswith("All") else n for n in notes]


def layer_notes() -> list[str]:
    notes = []
    for arch in long.ARCH_ORDER:
        model = build(arch)
        notes.append(f"{long.PROSE[arch]}: watched layers {', '.join(model.lhd_layers())}; ReLU layers where units "
                     f"can die {', '.join(models.relu_layers(model))}  [experiments/models.py]")
    return notes


def ai_statement(tex: str) -> str:
    found = re.search(r"\\section\*\{Use of generative AI\}\n\n(.*?)\n\n\\section\*", tex, re.S)
    return found.group(1).strip() if found else ""

def limitation_items(s: Sheet) -> list[str]:
    """The nine limitations that must stay in, with the numbers behind each."""
    def v(m: str) -> str:
        return s[m].shown()

    return [
        f"1. The zero update check catches frozen layers {v('ZeroLead')} epochs sooner ({v('ZeroFrozenOneEpoch')} at "
        f"epoch {v('ZeroEpoch')}); LHD's advantage is dead units",
        "2. Faults were planted to show up in the signals LHD measures; they are not real bugs",
        "3. Calibration needs 10 to 20 healthy runs of the same network and assumes they are healthy; this costs more "
        f"than LHD saves for a model trained once. The study used {v('CalibrationPerNet')} per network, and the Cal "
        "numbers above show the false alarms at each size",
        "4. LHD misses dead units where healthy runs already lose many units (the CNN's dense layer, "
        f"{v('CnnDenseLow')} to {v('CnnDenseHigh')} dead in healthy runs)",
        f"5. {v('HarmSmall')} of {v('BrokenRuns')} planted faults cost less than half a point of accuracy",
        "6. Only the frozen layer and dead unit rules were tested in the main study; the divergence rule never fired "
        f"({v('DivergenceFired')} runs). In the main study the vanishing gradient rule fired only as a wrong cause, "
        f"{v('WrongCauseList')}",
        f"7. The {v('Floor')} floor is too low for small layers; one dead unit of {v('SmallLayerUnits')} is "
        f"{v('OneUnitShare')}",
        f"8. Small networks, laptop CPU, {v('EpochRange')} epochs; the transformer reaches about {v('TransformerAcc')} "
        "accuracy; no GPU, large models, or other optimizers",
        f"9. The third set of runs ({v('FreshThirdSet')} healthy runs and the extra faults) was trained after the main "
        "results",
    ]


def preamble() -> str:
    return long.preamble().replace("\\begin{document}", "\\usepackage{placeins}\n\\input{numbers}\n\n\\begin{document}")


def outline(s: Sheet, d: long.Data) -> str:
    tex = long_tex()
    before, after = story(tex)
    hdr = ["Runs", "BrokenRuns", "Caught", "CaughtRate", "CaughtInterval", "AccCaught", "LayerRight", "FalseAlarms",
           "HealthyRuns", "FalseAlarmInterval", "BothCaught", "BothGap", "HarmBig", "HarmBigLHD", "HarmBigAcc",
           "ZeroLead", "LhdDeadRight"]
    nets = "fully connected network on Fashion-MNIST, CNN on GTSRB, small transformer on Tiny Shakespeare  [experiments/paper.py]"
    abstract = ("\\begin{abstract}\n"
                + block(s, "120 words at most", hdr,
                        ["LHD stands for Layer Health Diagnostics", nets,
                         "Faults planted in the main study: frozen layers and dead units"])
                + "\\end{abstract}\n")
    intro = ("\\section{Introduction}\n\\label{sec:intro}\n\n"
             + block(s, "0.75 page",
                     ["Runs", "BrokenRuns", "Caught", "AccCaught", "MedianEpoch", "BothCaught", "BothGap",
                      "SeedGroups", "SeedGroupsLHD", "FalseAlarms", "HealthyRuns", "AccFalseAlarms", "HarmBig",
                      "HarmBigLHD", "HarmBigAcc", "HarmSmall", "ZeroLead", "LhdDeadRight", "SageDead", "SageFlagged"],
                     [f"Your GTSRB model plateaued at {before}% with one convolution left out of the optimizer, and "
                      f"fixing it gave {after}%. This is your own account, not from the runs  [paper/lhd.tex Section 1]",
                      "Cite \\citep{stallkamp2012} for GTSRB and \\citep{chwilczynski2024} for errors in deep learning "
                      "code that raise no exception",
                      "When it fires, LHD reports the layer, the signals out of range, a likely cause, and a suggested "
                      "check, then stops the run  [src/lhd/monitor.py]"]))
    related = ("\\section{Related Work}\n\\label{sec:related}\n\n"
               + block(s, "0.5 page", [], related_notes()
                       + ["Thresholds the fixed rules use: SageMaker DeadRelu 50%, DeepDiagnosis 70%, wandb-callbacks "
                          "80%. LHD instead measures a range for every layer, signal, and epoch from healthy runs of "
                          "the same network  [experiments/paper.py, src/lhd/baseline.py]"]))
    method = ("\\section{Method}\n\\label{sec:method}\n\n"
              + block(s, "1 page, including Equation 1, Table 1, and Figure 1",
                      ["ProbeSize", "Margin", "Floor", "Patience", "CalibrationPerNet", "SettingsCommit", "SettingsDate",
                       "DivergenceFired", "VanishingFired", "RunawayFired"],
                      ["Four signals per layer at the end of every epoch  [src/lhd/signals.py]",
                       "  weight change ||W_e - W_(e-1)|| / ||W_(e-1)||, weights and bias flattened, exactly 0 for a frozen layer",
                       "  gradient norm, mean of ||dL/dW|| over the epoch's batches, after backward and before the optimizer step",
                       "  weight norm ||W_e||",
                       "  dead unit fraction, share of units whose ReLU output is exactly 0 for every probe input; a filter "
                       "counts only when 0 at every position; probe batch fixed, evaluation mode, no gradients",
                       "Range from calibration runs only, lowest a and highest b per layer, signal, and epoch; a value "
                       "that is not finite is always out of range; broken runs never set the range  [src/lhd/baseline.py]",
                       "The floor keeps a signal that is 0 in every healthy run from flagging one unusual unit",
                       "Alarm when one signal of one layer is out of range p epochs in a row; ties go to the first layer "
                       "in data flow order  [src/lhd/detector.py]",
                       "m, f, and p come from the library tests written 12 October 2026 and were committed before any "
                       "study run; an earlier draft used a multiple of the largest healthy value  [paper/lhd.tex Section 3.3]",
                       "Cause rules in Table 1, checked top to bottom  [src/lhd/diagnosis.py]; the main study tested only "
                       "the frozen layer and dead unit rules"])
              + "\n\\begin{equation}\nx < a_{\\ell s e}(1 - m) \\quad \\text{or} \\quad x > b_{\\ell s e}(1 + m) + f\n"
                "\\label{eq:range}\n\\end{equation}\n\n"
              + table_float("rules", "rules", "the cause LHD names for each pattern of the alarmed layer's signals")
              + "\n% NUMBERS for Figure 1\n"
              + "\n".join(f"%   {s[m].shown():<22}  \\lhd{m:<20}  {s[m].what}" for m in
                          ("FigRun", "FigLayer", "FigAlarm", "FigAccuracy", "FigAccuracyRange", "FigFinal",
                           "FigFinalRange", "CalibrationPerNet")) + "\n"
              + figure_float("how_lhd_works", "(a) the frozen layer's weight change against the calibration range and "
                                              "(b) the same run's validation accuracy"))
    setup = ("\\section{Experimental Setup}\n\\label{sec:setup}\n\n"
             + block(s, "0.5 page, including Table 2",
                     ["Runs", "CalibrationRuns", "FreshRuns", "BrokenRuns", "OtherRuns", "MlpWidths", "CnnFilters",
                      "CnnDense", "TransformerWidth", "TransformerHeads", "TransformerBlocks", "TransformerHidden",
                      "LearningRate", "Threads", "CalibrationSeeds", "ScoredSeedStart", "SeedsPerFault", "DeadFractions",
                      "DeadBias", "DeadChecks", "Patience", "PlanCommit", "PlanDate"],
                     ["Adam with a cosine schedule, one CPU thread  [experiments/train.py]",
                      "A broken run and the fresh healthy run with the same seed differ only by the fault; that run "
                      "is its twin, and the twin's final accuracy gives the fault's cost",
                      "Frozen layer: weights and bias put back after every optimizer step, the same failure as a layer "
                      "left out of the optimizer; planted in every watched layer  [src/lhd/faults.py]",
                      "Dead units: incoming weights 0 and bias as above, so the ReLU output is 0 for every input; "
                      "planted in every layer followed by a ReLU  [experiments/train.py]",
                      "Every run trained to the end, and LHD's alarm comes from replaying the saved signals  "
                      "[src/lhd/replay.py]",
                      "Accuracy curve: fires when validation accuracy is below the lowest calibration value p epochs "
                      "in a row, with no margin  [experiments/evaluate.py]",
                      "Zero update check: a layer's weights do not move in an epoch, like SageMaker's UnchangedTensor",
                      "Dead unit checks at the thresholds above: SageMaker DeadRelu 50%, DeepDiagnosis 70%, "
                      "wandb-callbacks 80%, plus a 25% rule and any dead unit; all held to p epochs and counted only "
                      "at the planted layer",
                      "Caught means the detector fired at or after the fault's onset; a false alarm is any alarm on a "
                      "fresh healthy run; the gap is the accuracy curve's epoch minus LHD's, never counted as one "
                      "past the end"])
             + table_float("networks", "networks", "each network's dataset, size, epochs, and runs, with a total row"))
    results = ("\\section{Results}\n\\label{sec:results}\n% BUDGET 2 pages for the four parts below, "
               "split as suggested under each heading\n\n") + "\n".join([
        "\\subsection{LHD and the accuracy curve}\n\\label{sec:detection}\n\n"
        + block(s, "about 0.6 page, including Table 3",
                ["Caught", "BrokenRuns", "CaughtRate", "CaughtInterval", "AccCaught", "AccCaughtRate", "MedianEpoch",
                 "AccMedianEpoch", "AccNever", "SilentShare", "BothCaught", "BothGap", "BothMedian", "BothAccEarlier",
                 "BothTies", "SeedGroups", "SeedGroupsLHD", "LayerRight", "LayerRightRuns", "CauseRight",
                 "CauseRightRuns"],
                ["Per network values are in Table 3  [tables/headline.tex]",
                 "Epoch 3 is the earliest alarm possible for a fault planted at epoch 1 when p is 3",
                 "Figure A1 in the appendix shows the share flagged by each epoch"])
        + table_float("headline", "headline", "LHD next to the accuracy curve in each network"),
        "\\subsection{Faults that cost accuracy}\n\\label{sec:harm}\n\n"
        + block(s, "about 0.3 page",
                ["HarmSmall", "HarmSmallLHD", "HarmSmallAcc", "HarmMid", "HarmMidLHD", "HarmMidAcc", "HarmBig",
                 "HarmBigLHD", "HarmBigAcc", "CnnFrozenDrop", "CnnHalfDeadDrop", "OtherMaxDrop"],
                ["Cost is the final accuracy below the healthy twin with the same seed",
                 f"Compare with your GTSRB model, where the frozen convolution cost {int(after) - int(before)} points"]),
        "\\subsection{Dead units and fixed thresholds}\n\\label{sec:fixed}\n\n"
        + block(s, "about 0.8 page, including Figure 2",
                ["ZeroFrozenRuns", "ZeroFrozenOneEpoch", "ZeroEpoch", "LhdFrozenEpoch", "ZeroLead", "ZeroFlagged",
                 "LhdFrozenRight", "LhdDeadRight", "LhdDeadAny", "FalseAlarms", "QuarterDead", "QuarterFlagged",
                 "SageDead", "SageFlagged", "DeepDead", "DeepFlagged", "WandbDead", "WandbFlagged", "AnyDeadDead",
                 "AnyDeadFlagged", "CnnDenseLow", "CnnDenseHigh", "TransformerDeadMax"],
                ["Every fixed rule is held to p epochs and counted only when it names the planted layer; the full "
                 "table is Table A1 in the appendix",
                 "A frozen layer's weight change is exactly 0, so a check for an exact 0 needs no range and no patience"])
        + figure_float("detectors_tradeoff", "dead unit faults caught at the planted layer against healthy runs "
                                             "flagged, for LHD and each fixed rule"),
        "\\subsection{False alarms}\n\\label{sec:alarms}\n\n"
        + block(s, "about 0.3 page",
                ["FalseAlarms", "HealthyRuns", "FalseAlarmRate", "FalseAlarmInterval", "AccFalseAlarms",
                 "AccFalseAlarmInterval"]
                + [m for m in s.facts if m.startswith("Alarm")]
                + ["Floor", "SmallLayerUnits", "OneUnitShare", "DraftCommit", "DraftFresh", "DraftAlarms", "LaterFresh",
                   "LaterAlarms"],
                ["When no calibration run has a dead unit in a layer, the top of its range is the floor, so one dead "
                 "unit can cross it",
                 "An alarm for too few dead units points to no fault; checking only the top of the range would "
                 "remove it"]),
    ])
    limitations = ("\\section{Limitations}\n\\label{sec:limits}\n\n"
                   + block(s, "0.5 page. Every item below must stay in.",
                           ["ZeroFrozenOneEpoch", "ZeroEpoch", "ZeroLead", "LhdDeadRight", "CalibrationPerNet"]
                           + [m for m in s.facts if m.startswith("Cal") and m[3] in LETTERS and m[4:] in
                              ("Runs", "Worst", "Net")]
                           + ["CnnDenseLow", "CnnDenseHigh", "CnnDenseACaught", "CnnDenseBCaught", "CnnDenseCCaught",
                              "HarmSmall", "BrokenRuns", "VanishingFired", "RunawayFired", "DivergenceFired", "Floor",
                              "SmallLayerUnits", "OneUnitShare", "EpochRange", "TransformerAcc", "MlpParams",
                              "CnnParams", "TransformerParams", "Threads", "FreshThirdSet", "OtherRuns",
                              "WrongCauseRuns", "WrongCauseList"],
                           limitation_items(s)))
    conclusion = ("\\section{Conclusion}\n\\label{sec:conclusion}\n\n"
                  + block(s, "3 sentences", ["Runs", "CaughtRate", "LayerRight", "FalseAlarms", "HealthyRuns",
                                             "LhdDeadRight"], []))
    statement = ai_statement(tex)
    back = ("% MAIN TEXT ENDS HERE. Everything above must fit in 6 pages.\n\\FloatBarrier\n\n"
            "\\section*{Use of generative AI}\n\\label{sec:ai}\n\n"
            + block(s, "not counted", [], ["The long paper's statement, for reference  [paper/lhd.tex]",
                                           statement])
            + "\n\\section*{Code and data}\n\\label{sec:code}\n\n"
            + block(s, "not counted", ["PlanCommit", "DraftCommit", "SettingsCommit"],
                    [f"Repository {long.REPO}", "The long version is tagged v0-long"])
            + "\n\\input{../references}\n")
    end = "\n\\end{document}\n"
    return "\n".join([preamble(), abstract, intro, related, method, setup, results, limitations, conclusion, back,
                      appendix(s), end])


def appendix(s: Sheet) -> str:
    from experiments.evaluate import clustered_tests

    draws = inspect.signature(clustered_tests).parameters["draws"].default
    parts = [
        "\\section{Networks and runs in detail}\n\\label{app:runs}\n\n"
        + block(s, "not counted",
                ["MlpWidths", "CnnFilters", "CnnDense", "TransformerWidth", "TransformerHeads", "TransformerBlocks",
                 "TransformerHidden", "CnnValImages", "TextWindows", "TextContext", "TextHeldOut", "CalibrationSeeds",
                 "ScoredSeedStart", "SeedsPerFault", "UnitSeedOffset", "DeadBias", "FreshFirstSet", "FreshThirdSet",
                 "PlanCommit", "PlanDate"],
                layer_notes()
                + ["Killed units are drawn without replacement, their number is the share times the layer's width "
                   "rounded with at least one unit, and Adam's running averages for them are cleared  [experiments/train.py]",
                   f"Statistics: one sided Wilcoxon signed rank test on the gap averaged within each seed and within "
                   f"each fault condition, a bootstrap with {draws:,} draws that resamples whole seeds, and Wilson "
                   f"score intervals for rates  [experiments/evaluate.py]"]),
        "\\section{Detection over time}\n\\label{app:time}\n\n"
        + block(s, "not counted", [m for m in s.facts if m.startswith("Time")], [])
        + figure_float("detection_over_time", "the share of broken runs flagged by each epoch, LHD against the "
                                              "accuracy curve, one panel per network"),
        "\\section{Every detector}\n\\label{app:detectors}\n\n"
        + block(s, "not counted",
                ["PatienceAlarmChange", "PatienceDeadChange", "SageAnyTen", "SageRightTen", "SageAnyTwentyFive",
                 "SageRightTwentyFive", "AccOneCaught", "AccOneFlagged", "LossCaught", "LossFlagged"],
                ["Requiring three epochs instead of one changed little, since a layer's dead unit fraction rarely "
                 "drops back once it crosses a threshold",
                 "Counted at any layer, the 50% rule caught more runs with 10% dead units than with 25%, because it "
                 "fired on units that died on their own in other layers"])
        + table_float("detectors", "detectors", "every detector held to three epochs, frozen layers and dead units "
                                                "caught at the planted layer, and healthy runs flagged"),
        "\\section{Each layer and how many units must die}\n\\label{app:layers}\n\n"
        + block(s, "not counted",
                [m for m in s.facts if m.startswith("Dose")]
                + ["WrongLayerRuns", "WrongLayerList", "WrongCauseRuns", "WrongCauseList"],
                ["In each wrong layer run, the layer LHD named had its own signal out of range",
                 "A frozen attention projection also lowers the gradient that reaches it, so its pattern can match "
                 "a vanishing gradient"])
        + figure_float("per_layer", "for every layer and fault, the share of broken runs where LHD caught the fault "
                                    "and named both the planted layer and the right cause"),
        "\\section{What each signal adds}\n\\label{app:ablation}\n\n"
        + block(s, "not counted", [m for m in s.facts if m.startswith("Ablation")],
                ["None of the planted faults was designed to move the weight norm or the gradient norm"])
        + table_float("ablation", "ablation", "the same runs replayed with each signal removed or used alone"),
        "\\section{Margin and calibration size}\n\\label{app:settings}\n\n"
        + block(s, "not counted",
                ["Margin", "EdgeFlagged", "EdgeCaught", "MarginFlagged", "MarginCaught"]
                + [m for m in s.facts if m.startswith("Cal") and m[3] in LETTERS],
                ["These checks replay the same saved runs after the main results; the settings were fixed before "
                 "them  [experiments/sweep.py]"]),
        "\\section{Cost}\n\\label{app:cost}\n\n"
        + block(s, "not counted", ["MlpOverhead", "TransformerOverhead", "CnnOverhead", "ComputeSaved", "TrainingHours"],
                ["Measured on fresh healthy runs on one CPU thread; most of the overhead is the gradient norm, read "
                 "after every batch, and the probe batch, run once per epoch"]),
        "\\section{Faults beyond frozen layers and dead units}\n\\label{app:extra}\n\n"
        + block(s, "not counted",
                ["LateOnset"] + [m for m in s.facts if any(m.startswith(k) for k in EXTRA_KEYS.values())]
                + ["LrNormal", "LrTimesTen", "LrTimesHundred", "LateZeroEpoch", "LateZeroTies", "CutZero",
                   "LateDeadSage", "SageFlagged", "DivergenceFired"],
                ["Late faults are the main faults started at the epoch above instead of epoch 1",
                 "A gradient cut detaches one layer's input from the first epoch, so no gradient reaches the layers "
                 "before it  [experiments/grid.py DETACH_CUTS]",
                 "A learning rate jump multiplies Adam's rate at the start of that epoch, and the cosine schedule "
                 "keeps shrinking it from the raised value",
                 "This third set was trained after the main results were in"])
        + table_float("extra", "extra", "each extra fault, caught by LHD, the accuracy curve, the zero update check, "
                                        "and the 50% rule, with healthy runs flagged"),
        "\\section{Statistics for each seed and condition}\n\\label{app:stats}\n\n"
        + block(s, "not counted",
                ["SeedGroups", "SeedGroupsLHD", "SeedP", "Conditions", "ConditionsLHD", "ConditionsAcc",
                 "ConditionsSame", "ConditionsNeither", "ConditionP", "MeanGapAll", "MeanGapInterval"],
                ["The mean gap over all runs leans on the runs the accuracy curve missed, so the gap on runs both "
                 "caught is the fairer measure"]),
    ]
    return ("\\clearpage\n\\appendix\n\\setcounter{figure}{0}\n\\setcounter{table}{0}\n"
            "\\renewcommand{\\thefigure}{A\\arabic{figure}}\n\\renewcommand{\\thetable}{A\\arabic{table}}\n\n"
            + "\n".join(parts))


MAIN_END = "% MAIN TEXT ENDS HERE"
MAIN_PAGES = 6


def compile_pdf(folder: Path, name: str, out: Path) -> str:
    """Compiles with tectonic into out and returns the log, stopping on any undefined reference."""
    result = subprocess.run([str(long.TECTONIC), "-X", "compile", "--keep-logs", "--keep-intermediates",
                             "--outdir", str(out), f"{name}.tex"], cwd=folder, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit(1)
    log = (out / f"{name}.log").read_text(errors="ignore")
    undefined = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
    if undefined:
        raise SystemExit(f"undefined references in {name}.pdf: {', '.join(undefined)}")
    return log


def headings(text: str) -> list[tuple[str, str, float | None]]:
    """Each heading of the main text with its level and the page budget written under it."""
    found = []
    for level, title, rest in re.findall(r"^\\(section|subsection)\{([^}]*)\}\n(.*?)(?=^\\(?:sub)?section\{|\Z)",
                                         text, re.S | re.M):
        budget = re.search(r"% BUDGET (?:about )?([\d.]+) pages?", rest)
        found.append((level, title, float(budget.group(1)) if budget else None))
    return found


def measure(tex: str, folder: Path) -> list[tuple[str, float, float | None]]:
    """Pages each part of the main text takes, from positions saved while typesetting."""
    import shutil
    import tempfile

    main, rest = tex.split(MAIN_END, 1)
    count = iter(range(1000))

    def mark(match) -> str:
        n = next(count)
        return f"\\par\\zsavepos{{lhdpos{n}}}\\label{{lhdpage{n}}}\n{match.group(0)}"

    marked = re.sub(r"^\\(?:sub)?section\{", mark, main, flags=re.M)
    end = next(count)
    marked += f"\\par\\zsavepos{{lhdpos{end}}}\\label{{lhdpage{end}}}\n"
    marked = marked.replace("\\begin{document}", "\\usepackage{zref-savepos}\n\\begin{document}", 1)
    marked = marked.replace("\\maketitle", "\\maketitle\\typeout{LHDTEXT \\the\\textheight\\space "
                            "\\the\\dimexpr\\paperheight-1in-\\voffset-\\topmargin-\\headheight-\\headsep\\relax}", 1)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "short"
        shutil.copytree(folder, work)
        shutil.copy(long.PAPER / "references.tex", Path(tmp) / "references.tex")
        (work / "measure.tex").write_text(marked + MAIN_END + rest)
        log = compile_pdf(work, "measure", work)
        aux = (work / "measure.aux").read_text()
    height, top = (float(v) * 65536 for v in re.search(r"LHDTEXT ([\d.]+)pt ([\d.]+)pt", log).groups())
    ys = {int(n): int(y) for n, y in re.findall(r"\\zref@newlabel\{lhdpos(\d+)\}\{.*?\\posy\{(-?\d+)\}", aux)}
    pages = {int(n): int(p) for n, p in re.findall(r"\\newlabel\{lhdpage(\d+)\}\{\{[^}]*\}\{(\d+)\}", aux)}
    spot = [pages[i] - 1 + (top - ys[i]) / height for i in range(end + 1)]
    parts = headings(main)
    out = [("Title and abstract", spot[0], None)]
    out += [("  " + title if level == "subsection" else title, spot[i + 1] - spot[i], budget)
            for i, (level, title, budget) in enumerate(parts)]
    starts = [i for i, (level, _, _) in enumerate(parts) if level == "section"] + [end]
    for a, b in zip(starts, starts[1:]):
        if b - a > 1:
            title, _, budget = out[a + 1]
            out[a + 1] = (title, spot[b] - spot[a], budget)
    out.append(("Main text in all", spot[end], float(MAIN_PAGES)))
    return out


def words(text: str) -> int:
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
    body = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", body)
    return len(re.findall(r"[A-Za-z0-9\\][^\s]*", body))


def report(tex: str) -> None:
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    conclusion = re.search(r"\\section\{Conclusion\}(.*?)" + re.escape(MAIN_END), tex, re.S).group(1)
    prose = "\n".join(line for line in conclusion.splitlines() if not line.lstrip().startswith(("%", "\\label")))
    sentences = len(re.findall(r"[.!?](?:\s|$)", prose))
    text = "\n".join(line.split("%")[0] for line in tex.splitlines())
    dashes = len(re.findall("\u2014|\u2013|(?<!-)--(?!-)|---", text))
    print(f"abstract {words(abstract)} words of 120, conclusion {sentences} sentences of 3, "
          f"dashes outside comments {dashes}")
    print(f"{'part':44} {'pages':>6} {'budget':>7}")
    for title, used, budget in measure(tex, SHORT):
        print(f"{title:44} {used:6.2f} {'' if budget is None else f'{budget:7.2f}'}")


def build_pdf() -> None:
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        compile_pdf(SHORT, "lhd_short", Path(tmp))
        shutil.copy(Path(tmp) / "lhd_short.pdf", SHORT / "lhd_short.pdf")
    print(f"wrote {SHORT / 'lhd_short.pdf'} with no undefined references", flush=True)


def sheet(d: long.Data) -> Sheet:
    s = Sheet()
    for part in (headline, harm, detectors, false_alarms, method_facts, setup_facts, limitation_facts,
                 appendix_facts):
        part(s, d)
    return s


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--facts", action="store_true", help="print every number with its source and stop")
    parser.add_argument("--outline", action="store_true", help="write paper/short/lhd_short.tex if it is missing")
    args = parser.parse_args(argv)
    d = long.Data()
    drawn = write_figures()
    s = sheet(d)
    figure_facts(s, drawn)
    if args.facts:
        for f in s.facts.values():
            print(f"{f.macro:24} {f.shown():34} {f.what}  [{f.source}]")
        return
    SHORT.mkdir(parents=True, exist_ok=True)
    (SHORT / "numbers.tex").write_text(s.tex())
    print(f"wrote {SHORT / 'numbers.tex'} with {len(s.facts)} numbers", flush=True)
    write_tables(d)
    if args.outline:
        target = SHORT / "lhd_short.tex"
        if target.exists():
            raise SystemExit(f"{target} exists and holds your prose, so it is left alone")
        target.write_text(outline(s, d))
        print(f"wrote {target}", flush=True)
    if not (SHORT / "lhd_short.tex").exists():
        raise SystemExit("no paper/short/lhd_short.tex yet; run with --outline to write it")
    build_pdf()
    report((SHORT / "lhd_short.tex").read_text())


if __name__ == "__main__":
    main()
