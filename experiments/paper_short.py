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
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime

from experiments import models
from experiments import paper as long
from experiments.evaluate import SETTINGS
from experiments.grid import CALIBRATION_SEEDS
from experiments.train import ARCHS

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


def sheet(d: long.Data) -> Sheet:
    s = Sheet()
    for part in (headline, harm, detectors, false_alarms):
        part(s, d)
    return s


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--facts", action="store_true", help="print every number with its source and stop")
    args = parser.parse_args(argv)
    s = sheet(long.Data())
    if args.facts:
        for f in s.facts.values():
            print(f"{f.macro:24} {f.shown():34} {f.what}  [{f.source}]")
        return
    SHORT.mkdir(parents=True, exist_ok=True)
    (SHORT / "numbers.tex").write_text(s.tex())
    print(f"wrote {SHORT / 'numbers.tex'} with {len(s.facts)} numbers", flush=True)


if __name__ == "__main__":
    main()
