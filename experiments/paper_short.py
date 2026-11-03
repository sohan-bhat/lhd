"""Builds the short LHD paper's numbers, tables, and figures, and compiles it.

Usage:
    python -m experiments.paper_short             numbers, tables, figures, and the PDF
    python -m experiments.paper_short --facts     print every number with its source and stop

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
            "Network & Dataset & Parameters & Epochs & Calibration & Fresh healthy & Frozen or dead & Other faults \\\\\n"
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


if __name__ == "__main__":
    main()
